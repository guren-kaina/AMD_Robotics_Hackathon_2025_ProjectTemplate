# Pipeline

Capture a video stream, detect the tic-tac-toe board with `tic_tac_toe_overlay`, call `tic-tac-toe-planner` to pick the next move for player **X**, and mask the chosen cell in red on the output stream.

## Setup
- Python 3.10+
- Install dependencies (uv example):
  ```bash
  cd mission2/code/pipeline
  uv pip install -r pyproject.toml
  ```
- Prepare overlay weights (`cell_grid.pt`) with `mission2/code/tic_tac_toe_overlay/main.py` (training is triggered automatically there when weights are missing).

## Usage
Run the pipeline against a camera (e.g., index 0) and save `pipeline_output.mp4`:
```bash
cd mission2/code/pipeline
python main.py \
  --source 0 \
  --weights ../tic_tac_toe_overlay/models/cell_grid.pt \
  --planner-model Qwen/Qwen2.5-7B-Instruct \
  --interval 5 \
  --output pipeline_output.mp4 \
  --save-state latest_board.json
```
If you want to stream to a file-like sink (e.g., named pipe) instead of a local mp4, point `--output` to that path. For example:
```bash
mkfifo /tmp/ttt_stream
python main.py --source 0 --output /tmp/ttt_stream ... &
# Then another process can read the stream, e.g.:
ffplay -i /tmp/ttt_stream
```
Ensure the consumer is ready before starting the pipeline so frames are not blocked.
Options (important ones):
- `--source`: camera index or video file path.
- `--interval`: seconds between board polls (default 5s).
- `--planner-model`, `--planner-temperature`, `--planner-max-tokens`, `--planner-hf-token`: Hugging Face LLM controls for the planner.
- `--display`: show a preview window (`q` to quit).
- `--save-state`: save the most recent detected board JSON (cell labels + pixel boxes).
- Preprocessing knobs mirror the overlay script (`--grayscale`, `--clahe`, `--contrast-alpha/beta`, `--conf`, `--max-det`).

The pipeline caches the latest board state. When the state changes at a polling tick, it calls the planner to choose the next move and masks that cell in red on every frame until the next change.
