import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Dict, Literal

import google.generativeai as genai
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
        required=True,
        help="Path to the tic-tac-toe board image.",
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
    image_path: Path, api_key: str, model_name: str, max_output_tokens: int
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
    image_part = load_image_part(image_path)
    response = model.generate_content([prompt, image_part])
    text = response.text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON response: {exc}") from exc
    validate_response(parsed)
    return parsed


def main() -> int:
    args = parse_args()
    if not args.api_key:
        sys.stderr.write("Google API key is required (use --api-key or set GOOGLE_API_KEY).\n")
        return 1
    try:
        plan = generate_plan(
            image_path=Path(args.image),
            api_key=args.api_key,
            model_name=args.model,
            max_output_tokens=args.max_output_tokens,
        )
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(plan, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
