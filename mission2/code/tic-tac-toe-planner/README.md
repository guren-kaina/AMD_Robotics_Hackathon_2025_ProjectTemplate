## Overview
CLI that reads a tic-tac-toe board image and uses Gemini Flash Lite 2.5 to infer the board state and recommend the next move for player `"×"`. The model returns JSON with each cell's status (`"◯"`, `"×"`, or `"□"`) and the suggested action `1-9`.

## Setup (with uv)
- Python 3.12+ recommended.
- Install [uv](https://github.com/astral-sh/uv) if not present:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Sync dependencies from `pyproject.toml`:
  ```bash
  cd mission2/code/tic-tac-toe-planner
  uv sync
  ```
- Provide your Google API key via `GOOGLE_API_KEY` or `--api-key`.

## Usage
Run the CLI against a board image (e.g., `image_sample.png` in this directory):
```bash
cd mission2/code/tic-tac-toe-planner
uv run tic_tac_toe_planner.py --image image_sample.png
```
Or via the console script:
```bash
cd mission2/code/tic-tac-toe-planner
uv run ttt-planner --image image_sample.png
```
Options:
- `--api-key`: overrides `GOOGLE_API_KEY`.
- `--model`: defaults to `gemini-2.0-flash-lite-preview-02-05`.
- `--max-output-tokens`: cap response length (default 512).
- `--stream-url`: video stream URL/device path to capture frames from (used with `--think-interval`).
- `--think-interval`: when set (seconds), the tool repeatedly captures frames from `--stream-url`, runs Gemini on each frame, and applies OBS updates if configured.

The tool prints JSON like:
```json
{
  "current_status": {"1": "◯", "2": "□", "3": "□", "4": "×", "5": "□", "6": "◯", "7": "×", "8": "□", "9": "◯"},
  "next_action": 3
}
```

### Apply the suggested move to OBS
If you pass `--obs-host`, the tool connects to OBS websocket (default port `4455`, no password) and updates scene `シーン`: it shows only the suggested cell source (`"1"`–`"9"`) and hides the other cell sources.
```bash
uv run tic_tac_toe_planner.py --image image_sample.png --obs-host 100.76.113.92
```
Assumptions for OBS:
- Scene name is fixed to `シーン`.
- Cell sources are named `"1"`〜`"9"` (string). The tool turns on only the suggested source; the others are turned off.

### Continuous thinking from a stream
Install dependencies (includes OpenCV headless) with `uv sync`, then run:
```bash
uv run tic_tac_toe_planner.py \
  --stream-url rtsp://example.com/stream \
  --think-interval 2.0 \
  --api-key $GOOGLE_API_KEY \
  --obs-host localhost
```
The CLI will capture a frame every `--think-interval` seconds, infer the next move, and push changes to OBS each cycle. Stop with `Ctrl+C`.
