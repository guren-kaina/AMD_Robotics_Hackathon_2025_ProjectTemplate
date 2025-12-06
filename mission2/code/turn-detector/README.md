# turn-detector

CLI utility that watches a tic-tac-toe video stream, checks detections to see when the turn belongs to player `×`, and saves one frame per turn to a chosen directory.

## How it works
- Supports three detection engines: standard YOLO with explicit classes, open-vocabulary zero-shot detector (OWL-ViT by default), or CLIP-based grid classifier that warps the board and zero-shot classifies each of the 9 cells as {O, X, Empty}.
- Uses detections to count `◯` and `×` pieces plus optional obstacle classes (hands, people, blockers).
- Declares "our turn" only if the board is clear of obstacles and the number of `◯` pieces is greater than `×`.
- Captures exactly one frame for each of your turns, then waits for the next turn transition.

## Setup
1. Ensure Python 3.12+ is available.
2. Install dependencies (example with uv):  
   ```bash
   cd mission2/code/turn-detector
   uv pip install .
   ```  
   Or with pip: `pip install -e .`
3. Prepare a YOLO model that emits class names for your `◯`, `×`, and obstacle labels.

## Usage
```bash
turn-detector \
  --model path/to/model.pt \
  --source 0 \
  --output-dir ./captures \
  --o-label o --x-label x \
  --obstruction-label hand --obstruction-label person
```

To load weights directly from Hugging Face:
```bash
turn-detector \
  --hf-model-id your-org/your-yolo-repo \
  --hf-model-file model.pt \  # optional if the repo has exactly one *.pt
  --source 0 \
  --output-dir ./captures
```

For quick testing with a single image instead of a video stream:
```bash
turn-detector \
  --model path/to/model.pt \
  --image path/to/frame.jpg \
  --output-dir ./captures
```

To try zero-shot detection (open-vocabulary, OWL-ViT default) without custom training:
```bash
turn-detector \
  --detector-engine open-vocab \
  --ov-model-id google/owlvit-base-patch32 \
  --o-prompt "tic tac toe circle mark" \
  --x-prompt "tic tac toe cross mark" \
  --obstruction-prompt hand --obstruction-prompt person \
  --image path/to/frame.jpg \
  --output-dir ./captures
```

To try CLIP-based grid zero-shot classification (no custom training; detects grid, warps, and classifies each cell):
```bash
turn-detector \
  --detector-engine clip-grid \
  --clip-model-name ViT-B-32 \
  --clip-pretrained laion2b_s34b_b79k \
  --image path/to/frame.jpg \
  --output-dir ./captures
```

Key flags:
- `--source`: webcam index (`0`) or video path/URL (ignored if `--image` is set).
- `--image`: run once on a still image (helpful for testing labels and turn logic).
- `--output-dir`: where captured frames are written.
- `--o-label` / `--x-label`: YOLO class names for each side (case-insensitive).
- `--obstruction-label`: classes treated as obstacles; repeat to add more.
- `--min-gap-sec`: minimum time between captures to avoid duplicates in one turn window.
- `--hf-model-id`: download YOLO weights from Hugging Face Hub (ignores `--model`).
- `--hf-model-file`: relative path to the weights file inside the repo (defaults to the only `*.pt` if unambiguous).
- `--detector-engine`: choose `yolo` (class-based), `open-vocab` (zero-shot detector), or `clip-grid` (CLIP zero-shot per-cell).
- `--ov-model-id`: open-vocabulary model id (default OWL-ViT base).
- `--o-prompt` / `--x-prompt` / `--obstruction-prompt`: text prompts the zero-shot detector uses to find your pieces/obstacles.
- `--detector-engine clip-grid` options: `--clip-model-name`, `--clip-pretrained`, `--clip-prompts-*`, `--clip-tau-low`, `--clip-tau-empty`, `--clip-margin`.

Stop the program with `Ctrl+C`. Captured images are named `turn_<timestamp>_<counter>.jpg`.
