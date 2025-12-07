## Overview
Text-based planner that receives tic-tac-toe board state (JSON or inline text) exported by the overlay, then returns the best next move for player `X`. It runs a local LLM via `transformers` (default: `Qwen/Qwen3-4B-Instruct-2507`) and can optionally update OBS. No image/VLM input is used—board state is text-only.

## Setup (with uv)
```bash
cd mission2/code/tic-tac-toe-planner
uv sync
# download your chosen model locally (e.g. with `huggingface-cli download`) if not already cached
```

## Usage
Run against the JSON exported by `tic_tac_toe_overlay`:
```bash
cd mission2/code/tic-tac-toe-planner
uv run tic_tac_toe_planner.py \
  --state-json ../tic_tac_toe_overlay/board_state.json \
  --model Qwen/Qwen3-4B-Instruct-2507
```
Or pass an inline string:
```bash
uv run tic_tac_toe_planner.py --state "1=O,2=empty,3=X,4=empty,5=empty,6=empty,7=empty,8=empty,9=empty"
```
Call a remote Hugging Face Inference Endpoint instead of a local model:
```bash
HF_TOKEN=your_token uv run tic_tac_toe_planner.py \
  --state "1=O,2=empty,3=X,4=empty,5=empty,6=empty,7=empty,8=empty,9=empty" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --use-hf-remote
```
You can also pass other model ids (e.g. `Qwen/Qwen2.5-7B-Instruct`) with `--use-hf-remote`; the client first tries `text_generation` and falls back to `chat_completion`.

Key options:
- `--model`, `--temperature`, `--max-output-tokens`: local LLM controls (default model: `Qwen/Qwen3-4B-Instruct-2507`).
- `--use-hf-remote`, `--hf-api-token`/`--hf-token`, `--hf-endpoint-url`: call a hosted model via the Hugging Face Inference API.
- `--state-json` / `--state`: board input formats. The JSON format matches the overlay export (`state_map` plus per-cell boxes).
- `--obs-host`: apply the suggested move to OBS scene `シーン` with sources `"1"`〜`"9"` (omit to skip).

The tool prints JSON like:
```json
{
  "current_status": {"1": "O", "2": "empty", "3": "X", "4": "empty", "5": "empty", "6": "empty", "7": "empty", "8": "empty", "9": "empty"},
  "next_action": 5,
  "engine": "transformers:local"
}
```
