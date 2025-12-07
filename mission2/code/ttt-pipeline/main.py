"""Pipeline that stitches overlay + planner to mask the next tic-tac-toe move on a live stream."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Add sibling projects to path for reuse
CODE_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = CODE_ROOT / "tic_tac_toe_overlay"
PLANNER_ROOT = CODE_ROOT / "tic-tac-toe-planner"
# Add project roots so we can import the overlay package and planner module
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(PLANNER_ROOT))

from tic_tac_toe_overlay.main import (  # type: ignore  # noqa: E402
    BoardState,
    detect_board_state,
    save_board_state,
)
from tic_tac_toe_planner import (  # type: ignore  # noqa: E402
    PlannerResult,
    generate_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture frames from a video source, infer board state with tic_tac_toe_overlay, "
            "plan the next move with tic-tac-toe-planner, and mask the target cell in red."
        )
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source (camera index like 0/1 or file path).",
    )
    parser.add_argument(
        "--output",
        default="pipeline_output.mp4",
        help="Where to write the masked video (mp4).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between board polling/inference (default: 5).",
    )
    parser.add_argument(
        "--weights",
        default=str(OVERLAY_ROOT / "models" / "cell_grid.pt"),
        help="YOLO weights path for tic_tac_toe_overlay.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detection.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device (e.g. cpu, cuda:0, mps).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=30,
        help="Max detections after NMS (train/infer).",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Apply grayscale before inference (matches overlay options).",
    )
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE before inference.")
    parser.add_argument(
        "--contrast-alpha",
        type=float,
        default=None,
        help="alpha for preprocessing (cv2.convertScaleAbs).",
    )
    parser.add_argument(
        "--contrast-beta",
        type=float,
        default=None,
        help="beta for preprocessing (cv2.convertScaleAbs).",
    )
    parser.add_argument(
        "--planner-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name for planner Hugging Face inference (default: Qwen/Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--planner-temperature",
        type=float,
        default=0.0,
        help="Temperature for planner LLM.",
    )
    parser.add_argument(
        "--planner-max-tokens",
        type=int,
        default=256,
        help="Max tokens for planner LLM.",
    )
    parser.add_argument(
        "--planner-hf-token",
        default=None,
        help="Hugging Face token for planner (defaults to env HF_TOKEN/HUGGINGFACEHUB_API_TOKEN).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a preview window (press q to quit).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after writing this many frames (for debugging).",
    )
    parser.add_argument(
        "--save-state",
        default=None,
        help="Optional path to persist the latest detected board state JSON.",
    )
    return parser.parse_args()


def parse_source(source: str) -> str | int:
    try:
        return int(source)
    except ValueError:
        return source


def overlay_cell(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), thickness=-1)
    return cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)


def log_board_change(board_state: BoardState, planner_result: PlannerResult | None) -> None:
    state_str = board_state.compact_string()
    if planner_result:
        print(
            f"[info] Board changed -> planning move {planner_result.next_action} (engine={planner_result.engine}) | {state_str}"
        )
    else:
        print(f"[info] Board state updated: {state_str}")


def main() -> int:
    args = parse_args()
    video_source = parse_source(args.source)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        sys.stderr.write(f"Failed to open video source: {args.source}\n")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        width, height = 640, 480

    writer = None
    output_path = Path(args.output) if args.output else None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    model = YOLO(str(args.weights))
    last_state_map = None
    target_bbox: Tuple[int, int, int, int] | None = None
    last_inference_time = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.monotonic()

            if now - last_inference_time >= args.interval:
                board_state, ordered_cells = detect_board_state(
                    frame,
                    model,
                    conf=args.conf,
                    device=args.device,
                    max_det=args.max_det,
                    grayscale=args.grayscale,
                    clahe=args.clahe,
                    contrast_alpha=args.contrast_alpha,
                    contrast_beta=args.contrast_beta,
                )
                last_inference_time = now
                if board_state and ordered_cells:
                    state_map = board_state.state_map
                    if state_map != last_state_map:
                        try:
                            planner_result = generate_plan(
                                state_map=state_map,
                                model=args.planner_model,
                                temperature=args.planner_temperature,
                                max_output_tokens=args.planner_max_tokens,
                                hf_token=args.planner_hf_token,
                            )
                            target_bbox = board_state.bbox_for_cell(planner_result.next_action)
                            log_board_change(board_state, planner_result)
                        except Exception as exc:
                            target_bbox = None
                            sys.stderr.write(f"[warn] Planning failed: {exc}\n")
                        last_state_map = state_map
                        if args.save_state:
                            save_board_state(board_state, json_path=Path(args.save_state), text_path=None)
                else:
                    sys.stderr.write("[warn] Board detection failed; skipping this interval.\n")

            output_frame = frame
            if target_bbox:
                output_frame = overlay_cell(frame, target_bbox)

            if writer:
                writer.write(output_frame)
            if args.display:
                cv2.imshow("tic-tac-toe-pipeline", output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if output_path:
        print(f"[info] Saved masked stream to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
