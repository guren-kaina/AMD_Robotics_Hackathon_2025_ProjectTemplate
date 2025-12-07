## Overview
Text-based planner that receives tic-tac-toe board state (JSON or inline text) exported by the overlay, then returns the best next move for player `X`. It calls a Hugging Face-hosted/open LLM via `huggingface_hub` (chat fallback included) and can optionally update OBS. No image/VLM input is used—board state is text-only.

## Setup (with uv)
```bash
cd mission2/code/tic-tac-toe-planner
uv sync
# set HF token if required by your chosen model
export HF_TOKEN=xxxx
```

## Usage
Run against the JSON exported by `tic_tac_toe_overlay`:
```bash
cd mission2/code/tic-tac-toe-planner
uv run tic_tac_toe_planner.py \
  --state-json ../tic_tac_toe_overlay/board_state.json \
  --model Qwen/Qwen2.5-7B-Instruct
```
Or pass an inline string:
```bash
uv run tic_tac_toe_planner.py --state "1=O,2=empty,3=X,4=empty,5=empty,6=empty,7=empty,8=empty,9=empty"
```

Key options:
- `--model`, `--temperature`, `--max-output-tokens`: Hugging Face LLM controls (default model: `Qwen/Qwen2.5-7B-Instruct`).
- `--hf-token`: token for HF Inference (defaults to `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN`).
- `--state-json` / `--state`: board input formats. The JSON format matches the overlay export (`state_map` plus per-cell boxes).
- `--obs-host`: apply the suggested move to OBS scene `シーン` with sources `"1"`〜`"9"` (omit to skip).

The tool prints JSON like:
```json
{
  "current_status": {"1": "O", "2": "empty", "3": "X", "4": "empty", "5": "empty", "6": "empty", "7": "empty", "8": "empty", "9": "empty"},
  "next_action": 5,
  "engine": "huggingface"
}
```
