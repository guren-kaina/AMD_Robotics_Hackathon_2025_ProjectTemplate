---
license: cc-by-4.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - yolo
  - object-detection
  - tic-tac-toe
---

# Tic-tac-toe Cell Detector (YOLOv8n)

## Overview
- YOLOv8n model that detects occupancy per 3x3 cell (empty / white_circle / black_cross).
- Output: bounding boxes and classes for 9 cells. The included script can overlay cell indices and labels on the image.
- Intended input: top-down tic-tac-toe board images (matching this repo's synthetic/real data distribution).

## License
- Model weights: CC-BY-4.0
- Code: AGPL-3.0 (per Ultralytics dependency)
- Source code for data generation/training/inference: https://github.com/guren-kaina/AMD_Robotics_Hackathon_2025_ProjectTemplate/tree/main/mission2/code/tic_tac_toe_overlay

## Usage
```bash
pip install ultralytics
python - <<'PY'
from ultralytics import YOLO
model = YOLO("models/train/weights/best.pt")  # weights from this repo
res = model("your_input.jpg", imgsz=640, conf=0.25)
print(res[0].boxes.cls, res[0].boxes.xyxy)
res[0].save(filename="overlay.jpg")  # visualization
PY
```
- Class IDs: 0=empty_cell, 1=white_circle_cell, 2=black_cross_cell
- The included `main.py` runs preprocessing, cell index drawing, and JSON export.

## Training data
- Synthetic: gray background + white grid, includes low contrast/blur/noise and O/X distractors. Default generation train 1000 / val 200.
- Real: `real/images` and YOLO-format labels `real/labels` (no PII).

## Training setup
- Base: Ultralytics YOLOv8n
- Image size: default 640
- Epochs: default 20
- Options: `--preprocess-train` for contrast augmentation, `--real-data` to mix real data into training
- Weights saved to `models/train/weights/best.pt`

## Limitations and notes
- Accuracy may drop with oblique views or extreme lighting.
- Only 3 classes; out-of-board objects or different token shapes are unsupported.
- For new domains, re-label and retrain.
