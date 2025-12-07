## Context
- Project: tic-tac-toe 3x3 cell detection with numbering overlay.
- Task: Detect 3x3 cells (empty/O/X) in top-camera images and overlay labels using YOLO.
- Stack: Python 3.10, OpenCV, Ultralytics YOLO. Network restricted environment.

## Implementation Notes
- `main.py`: end-to-end synthetic generation → YOLO training → inference → overlay. Options for preprocessing (grayscale/CLAHE/contrast).
- Synthetic data: gray background with white outer/inner grid (line width ~20% of cell). Occupancy classes (empty_cell, white_circle_cell, black_cross_cell). Max one token per cell, 0–5 white circles per board, crosses same count or minus one. Tokens vary size; low contrast/blur/noise; crosses have handle heads with jittered outline color. Dummy O/X outside board to learn ignoring. Background/lines vary in contrast, brightness noise, blur. Default counts: train 1000 / val 200.
- Training: YOLOv8n (default 20 epochs) saved to `models/cell_grid.pt`. `--force-train` retrains even if weights exist. `--preprocess-train` multiplies images per contrast_alpha (0.8x/1.0x/1.2x) for training.
- Inference: detect occupancy 3 classes directly and draw cell index + label to `overlay.jpg`. Preprocessing uses the specified alpha/beta only.
- Device selectable (cpu/cuda:0/mps). Preprocessing controls: `--grayscale`/`--clahe`/`contrast-alpha`/`contrast-beta`. `--max-det` for NMS cap, `--train-nms-time` for NMS time limit during train/val.
- Options: `--num-train`, `--num-val` adjust synthetic counts; `--real-data` copies real images/labels into train automatically.
- Notebook `train_rocm.ipynb`: ROCm training/inference plus full real-image inference and label consistency check (warn if IoU<0.3 per class).

## Next Steps
- Improve real-image accuracy: add more real labels; tweak synthetic color/contrast/occlusion.
- Tune thresholds/epochs based on saved inference outputs.
- Use `label_tool.py` (OpenCV GUI) for efficient labeling; includes Undo/navigation.
- Hugging Face publication: if sharing model (`models/train/weights/best.pt`) and data (`real/`, synthetic generation scripts), confirm license/PII, optionally note heavy artifacts, and describe training setup/class schema/inference steps/permissions in README. Publish with `uvx hf repo create` → `git push`.

## License Notes
- Library: Ultralytics YOLOv8 is AGPL-3.0 (SaaS/distribution requires AGPL source disclosure; commercial closed use needs a commercial license).
- If publishing this repo to Hugging Face: keep code under AGPL-3.0 unless you hold a commercial Ultralytics license (state it if so).
- Model weights/dataset licenses are independent: open use → CC-BY-4.0; if you want restrictions → CC-BY-NC-4.0, etc. Real images confirmed no PII/portraits.
- In README/model card/data card, state permissions/restrictions, credit, training settings, class schema, and AGPL obligation from Ultralytics (or note commercial license).

## Hugging Face tasks (minimal model/data only)
- Vars: `HF_STAGE=/tmp/hf_release` (staging), `HF_MODEL_REPO`, `HF_DATASET_REPO`. Run via `tic_tac_toe_overlay/Makefile` (or `make -C tic_tac_toe_overlay ...`).
- Steps:
  - Model: `make hf-model-stage` clones existing HF repo, uses `MODEL_CARD.md` as README, copies `best.pt`, sets LICENSE to CC-BY-4.0. `.gitattributes` is assumed pre-configured (LFS/Xet as needed).
  - Data: `make hf-dataset-stage` clones existing dataset repo, uses `DATASET_CARD.md` as README, copies `real/images` and `real/labels`, generates `dataset.parquet` via `export_parquet.py` (needs pyarrow), sets LICENSE to CC-BY-4.0. `.gitattributes` is assumed pre-configured.
  - Push: `make hf-model-push` / `make hf-dataset-push` (requires `uvx hf login`; assumes repos already exist and LFS hooks installed). Targets run `git lfs install --local || true` then commit/push with `HF_HUB_ENABLE_HF_TRANSFER=1 git push ...` for faster uploads (per HF hub guide). Code stays private; only model/data repos are published.
