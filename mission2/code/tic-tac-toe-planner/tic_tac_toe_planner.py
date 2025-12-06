import argparse
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Dict, Literal

import google.generativeai as genai
from obsws_python import ReqClient
from obsws_python.error import OBSSDKError
from google.ai.generativelanguage import Schema, Type

BoardMark = Literal["◯", "×", "□"]


SYSTEM_INSTRUCTIONS = """
You extract tic-tac-toe state from an image and propose the next move for player "×".
- Board cells are labeled 1-9 left-to-right, top-to-bottom.
- Markers: "◯", "×", "□" ("□" means empty).
- You always act as player "×". Only propose an empty cell as the next move.
- Return JSON only with:
  - current_status: mapping of cell ids ("1"-"9") to one of the markers.
  - next_action: integer 1-9 indicating the best move for the side to play.
- Choose next_action that wins if possible, otherwise blocks opponent, otherwise any empty cell.
- Do not include explanations or extra keys.
"""

DEFAULT_MODEL = "gemini-2.0-flash-lite-preview-02-05"

RESPONSE_SCHEMA = Schema(
    type_=Type.OBJECT,
    properties={
        "current_status": Schema(
            type_=Type.OBJECT,
            properties={
                str(i): Schema(type_=Type.STRING, enum=["◯", "×", "□"])
                for i in range(1, 10)
            },
            required=[str(i) for i in range(1, 10)],
        ),
        # Gemini does not support integer enums; leave unconstrained and validate client-side.
        "next_action": Schema(type_=Type.INTEGER, description="Cell id (1-9) where × should play."),
    },
    required=["current_status", "next_action"],
)

OBS_SCENE_NAME = "シーン"
CELL_SOURCE_NAMES = [str(i) for i in range(1, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer tic-tac-toe board state from an image using Gemini Flash Lite 2.5 "
            "and propose the next move."
        )
    )
    parser.add_argument(
        "-i",
        "--image",
        required=False,
        help="Path to the tic-tac-toe board image.",
    )
    parser.add_argument(
        "--stream-url",
        help="Video stream URL or device path to capture frames from when using --think-interval.",
    )
    parser.add_argument(
        "--think-interval",
        type=float,
        help="Seconds between captures from --stream-url. When set, the tool keeps thinking in a loop.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GOOGLE_API_KEY"),
        help="Google API key. Defaults to GOOGLE_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Upper bound for response tokens.",
    )
    parser.add_argument(
        "--obs-host",
        default="localhost",
        help="OBS websocket host to apply the suggested move (if omitted, OBS is not updated).",
    )
    parser.add_argument(
        "--obs-port",
        type=int,
        default=4455,
        help="OBS websocket port (default: 4455).",
    )
    parser.add_argument(
        "--obs-timeout",
        type=float,
        default=5.0,
        help="OBS websocket timeout seconds (default: 5).",
    )
    return parser.parse_args()


def build_prompt() -> str:
    return SYSTEM_INSTRUCTIONS.strip()


def load_image_part(image_path: Path):
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    data = image_path.read_bytes()
    # gemini image input can be passed as a blob dict
    return {"mime_type": mime_type, "data": data}


def build_image_part_from_bytes(image_bytes: bytes, mime_type: str = "image/png") -> Dict[str, object]:
    if not image_bytes:
        raise ValueError("Image data is empty.")
    return {"mime_type": mime_type, "data": image_bytes}


def validate_response(payload: Dict[str, object]) -> None:
    if "current_status" not in payload or "next_action" not in payload:
        raise ValueError("Response missing required keys.")
    current_status = payload["current_status"]
    if not isinstance(current_status, dict):
        raise ValueError("current_status must be an object mapping.")
    for key, value in current_status.items():
        if key not in {str(i) for i in range(1, 10)}:
            raise ValueError(f"Invalid cell id: {key}")
        if value not in {"◯", "×", "□"}:
            raise ValueError(f"Invalid board marker: {value}")
    next_action = payload["next_action"]
    if not isinstance(next_action, int) or not (1 <= next_action <= 9):
        raise ValueError("next_action must be an integer between 1 and 9.")


def generate_plan(
    image_part: Dict[str, object], api_key: str, model_name: str, max_output_tokens: int
) -> Dict[str, object]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
            "max_output_tokens": max_output_tokens,
        },
    )
    prompt = build_prompt()
    response = model.generate_content([prompt, image_part])
    text = response.text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON response: {exc}") from exc
    validate_response(parsed)
    return parsed


def apply_obs_next_action(next_action: int, host: str, port: int, timeout: float) -> None:
    target_source = str(next_action)
    try:
        with ReqClient(host=host, port=port, password="", timeout=timeout) as client:
            for source_name in CELL_SOURCE_NAMES:
                item = client.get_scene_item_id(OBS_SCENE_NAME, source_name)
                client.set_scene_item_enabled(
                    OBS_SCENE_NAME,
                    item.scene_item_id,
                    source_name == target_source,
                )
    except (OBSSDKError, ConnectionRefusedError, TimeoutError) as exc:
        raise RuntimeError(f"OBS update failed: {exc}") from exc


def capture_frame_from_stream(stream_url: str) -> bytes:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Capturing from a stream requires opencv-python-headless (install via uv/pip)."
        ) from exc

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video stream: {stream_url}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to read a frame from the video stream.")
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame to PNG.")
    return buffer.tobytes()


def stream_think_loop(
    stream_url: str,
    think_interval: float,
    api_key: str,
    model_name: str,
    max_output_tokens: int,
    obs_host: str,
    obs_port: int,
    obs_timeout: float,
) -> int:
    while True:
        loop_started = time.time()
        try:
            frame_bytes = capture_frame_from_stream(stream_url)
            plan = generate_plan(
                image_part=build_image_part_from_bytes(frame_bytes),
                api_key=api_key,
                model_name=model_name,
                max_output_tokens=max_output_tokens,
            )
            sys.stdout.write(json.dumps(plan, ensure_ascii=False, indent=2))
            sys.stdout.write("\n")
            if obs_host:
                apply_obs_next_action(plan["next_action"], obs_host, obs_port, obs_timeout)
                sys.stdout.write(
                    f"OBS updated: scene '{OBS_SCENE_NAME}' shows source '{plan['next_action']}' "
                    "and hides the other cells.\n"
                )
            sys.stdout.flush()
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            sys.stderr.write(f"[think-loop] Error: {exc}\n")
            sys.stderr.flush()
        elapsed = time.time() - loop_started
        sleep_for = max(0.0, think_interval - elapsed)
        if sleep_for:
            time.sleep(sleep_for)


def main() -> int:
    args = parse_args()
    if not args.api_key:
        sys.stderr.write("Google API key is required (use --api-key or set GOOGLE_API_KEY).\n")
        return 1
    if args.think_interval is not None:
        if args.think_interval <= 0:
            sys.stderr.write("--think-interval must be positive seconds.\n")
            return 1
        if not args.stream_url:
            sys.stderr.write("--think-interval requires --stream-url to capture frames.\n")
            return 1
        return stream_think_loop(
            stream_url=args.stream_url,
            think_interval=args.think_interval,
            api_key=args.api_key,
            model_name=args.model,
            max_output_tokens=args.max_output_tokens,
            obs_host=args.obs_host,
            obs_port=args.obs_port,
            obs_timeout=args.obs_timeout,
        )
    if not args.image:
        sys.stderr.write("--image is required when --think-interval is not set.\n")
        return 1
    try:
        plan = generate_plan(
            image_part=load_image_part(Path(args.image)),
            api_key=args.api_key,
            model_name=args.model,
            max_output_tokens=args.max_output_tokens,
        )
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(plan, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")
    if args.obs_host:
        try:
            apply_obs_next_action(plan["next_action"], args.obs_host, args.obs_port, args.obs_timeout)
            sys.stdout.write(
                f"OBS updated: scene '{OBS_SCENE_NAME}' shows source '{plan['next_action']}' "
                "and hides the other cells.\n"
            )
        except Exception as exc:
            sys.stderr.write(f"Failed to apply OBS changes: {exc}\n")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
