# Pipeline

Capture a video stream, detect the tic-tac-toe board with `tic_tac_toe_overlay`, call `tic-tac-toe-planner` to pick the next move for player **X**, and mask the chosen cell in red while streaming the result to a camera device (e.g., a virtual camera).

## Setup
- Python 3.10+
- Install dependencies (uv example):
  ```bash
  cd mission2/code/ttt-pipeline
  uv sync
  ```
- Prepare overlay weights (`cell_grid.pt`) with `mission2/code/tic_tac_toe_overlay/main.py` (training is triggered automatically there when weights are missing).

## Usage
Run the ttt-pipeline against a camera (e.g., index 0) and stream the masked frames to another camera device (e.g., `/dev/video2` created by `v4l2loopback`):
```bash
cd mission2/code/ttt-pipeline
uv run python3 main.py \
  --source 0 \
  --weights ../tic_tac_toe_overlay/models/cell_grid.pt \
  --planner-model Qwen/Qwen2.5-7B-Instruct \
  --output-camera /dev/video2 \
  --interval 5 \
  --display \
  --save-state latest_board.json
```
Notes:
- `--output-camera` defaults to `1`. Point it at a virtual/loopback device so another app (OBS, browser, etc.) can subscribe to the masked feed. Set it to an empty string to disable camera streaming.
- Use `--display` for a local preview window; press `q` to stop.
- Ensure the target camera device exists and is writable before starting the pipeline; the script will warn and continue without streaming if it cannot open the device.
Options (important ones):
- `--source`: camera index or video file path.
- `--output-camera`: camera index or device path to publish the masked stream (default 1).
- `--interval`: seconds between board polls (default 5s).
- `--planner-model`, `--planner-temperature`, `--planner-max-tokens`, `--planner-hf-token`: Hugging Face LLM controls for the planner.
- `--display`: show a preview window (`q` to quit).
- `--save-state`: save the most recent detected board JSON (cell labels + pixel boxes).
- Preprocessing knobs mirror the overlay script (`--grayscale`, `--clahe`, `--contrast-alpha/beta`, `--conf`, `--max-det`).

The pipeline caches the latest board state. When the state changes at a polling tick, it calls the planner to choose the next move and masks that cell in red on every frame until the next change.
