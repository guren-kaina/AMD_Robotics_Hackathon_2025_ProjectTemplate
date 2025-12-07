## Purpose
Detect the 3x3 tic-tac-toe cells in an input frame, classify each cell as empty/O/X, and optionally overlay numbers/labels. The script auto-generates synthetic data, trains a lightweight YOLO model, runs inference on a single image, and exports both an overlay image and structured board state (JSON/text) for the pipeline/planner.

## Setup
Requires Python 3.10.

```bash
# Install dependencies (example with uv)
uv pip install -r pyproject.toml
```

If offline, preinstall `ultralytics` and `torch`.

## Usage
```bash
python main.py \
  --image input.jpg \        # Single image input; pipelines can apply the same per-frame.
  --output overlay.jpg \     # Use --skip-overlay if you only want detections.
  --state-json board_state.json \
  --print-state \
  --device mps        # example: Apple Silicon
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

## Release to Hugging Face
- Code license: AGPL-3.0 (aligned to Ultralytics dependency)
- Model weights: CC-BY-4.0 (e.g., `models/train/weights/best.pt`)
- Dataset: CC-BY-4.0 (real images + synthetic generation recipe)

Steps (manual):
1) Login: `uvx hf login`
2) Create repos: `uvx hf repo create <org>/tic-tac-toe-cell-detector --repo-type model` (use `--repo-type dataset` for the dataset)
3) Clone, place README/MODEL_CARD/DATASET_CARD, add `models/train/weights/best.pt` and `real/`
4) `git add . && git commit -m "Add model and dataset (CC-BY-4.0)" && git push`

Document the license and usage in the model card / data card before push.

### Publish only model & dataset (minimal package, without full repo)
Pre-req: `uvx hf login`, `git lfs install` (and ensure LFS hooks active), optional speed-up: `uvx pip install -U "huggingface_hub[hf_transfer]"` then set `HF_HUB_ENABLE_HF_TRANSFER=1` when pushing. HF repos must already exist with correct `.gitattributes` (e.g., `*.pt` in LFS for model, images for dataset); the Make tasks clone and update them.
```
# run inside tic_tac_toe_overlay (or add -C tic_tac_toe_overlay)
make HF_MODEL_REPO=your-org/tic-tac-toe-model hf-model-push
make HF_DATASET_REPO=your-org/tic-tac-toe-dataset hf-dataset-push
```
What it does:
- Model: stages `best.pt`, README from `MODEL_CARD.md`, LICENSE=CC-BY-4.0, then `git push` to the HF model repo.
- Dataset: stages `real/images` + `real/labels`, README from `DATASET_CARD.md`, LICENSE=CC-BY-4.0, then `git push` to the HF dataset repo.
Codebase itself stays private; only the minimal artifacts are published.

Note: these targets assume repos already exist with correct LFS configuration and `.gitattributes`. They simply clone, replace artifacts, and push (commit message: update). To speed up pushes, install `huggingface_hub[hf_transfer]` and set `HF_HUB_ENABLE_HF_TRANSFER=1`.

### Dataset parquet
- `make hf-dataset-stage` also generates `dataset.parquet` from `real/labels` (YOLO txt) with columns: `image`, `label_file`, `class_id`, `cx`, `cy`, `w`, `h`. One row per bbox (empty labels produce a single row with null class/coords). Requires `pyarrow`.
