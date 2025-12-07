## Purpose
Detect the 3x3 tic-tac-toe cells in an input frame, classify each cell as empty/O/X, and optionally overlay numbers/labels. The script auto-generates synthetic data, trains a lightweight YOLO model, runs inference on a single image, and exports both an overlay image and structured board state (JSON/text) for the pipeline/planner.

## Setup
Requires Python 3.10.

```bash
# 依存インストール（uv利用例）
uv pip install -r pyproject.toml
```

If offline, preinstall `ultralytics` and `torch`.

## Usage
```bash
python main.py \
  --image input.jpg \        # 任意の単一画像。pipeline では各フレームに同等処理を適用
  --output overlay.jpg \     # オーバーレイ不要なら --skip-overlay
  --state-json board_state.json \
  --print-state \
  --device mps        # 例: Apple Silicon の場合
  --grayscale         # Grayscale for both synth/inference
  --clahe             # Grayscale+CLAHE before inference
  --contrast-alpha 1.2 --contrast-beta -5  # Extra contrast tweak before inference
  --save-preprocessed tmp/pre.jpg           # Save preprocessed image
  --preprocess-train                        # Apply preprocessing to train/val images too
  --max-det 20        # Max detections after NMS (train/infer, default 30)
  --train-nms-time 3  # NMS time limit for train/val (seconds)
  --num-train 1000    # Synthetic train images
  --num-val 200       # Synthetic val images
  --real-data real    # Copy real/images+labels into train for training
```

- First run builds synthetic data in `data/synth_grid`, trains YOLO weights to `models/cell_grid.pt`, then runs inference.
- `--force-regen` regenerates data, `--force-train` retrains.
- Output saved to `overlay.jpg` (unless `--skip-overlay`).
- `--real-data` copies `<real-data>/images` and `labels` into train for training (beware overwrites).
- `--state-json/--state-text/--print-state` export the detected board (cell labels + pixel boxes) as text/JSON for the planner/pipeline. No VLM/Gemini is used—only the YOLO detector runs here.

### Simple GUI for real image labeling
`label_tool.py` (OpenCV) is included. Drag to create boxes, number keys to switch class, n/p to navigate, s to save, d to delete last, u/Ctrl+Z to undo, q/Esc to quit.

```bash
python label_tool.py --images real/images --labels real/labels
# Key binds:
# 1:cell, 2:white_circle, 3:black_cross
# n/p: next/prev, s: save, d: delete last, u/Ctrl+Z: undo, q/Esc: quit
```

## How it works
- Synthetic board: gray background, white outer/inner grid (line width ~20% of cell), labeled as occupancy classes (empty_cell, white_circle_cell, black_cross_cell). `--grayscale` can be applied to synth/infer.
- Each cell max 1 token. Board has 0–5 white circles, black crosses are same count or minus one. Crosses have handle heads with jittered color/edge; add a few dummy O/X outside board to learn ignoring; include low-contrast, blur, noise cases.
- Train YOLOv8n (default 20 epochs) to directly detect 9 occupancy cells, then overlay cell index and class label.
- Default synthetic count: train 1000 / val 200. Background/lines vary in contrast, brightness, noise, blur for diversity.
