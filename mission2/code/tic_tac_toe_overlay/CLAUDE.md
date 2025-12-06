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
