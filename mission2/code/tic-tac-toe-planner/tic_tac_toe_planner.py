import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

import torch
from obsws_python import ReqClient
from obsws_python.error import OBSSDKError
from transformers import AutoModelForCausalLM, AutoTokenizer

BoardMark = Literal["O", "X", "empty"]

SYSTEM_PROMPT = """
You plan tic-tac-toe for player "X" using ONLY the provided cell states.
- Cells are 1-9, left-to-right then top-to-bottom.
- States are one of: O, X, empty.
- Always choose an empty cell. Prefer a winning move, else block, else any empty cell.
- Respond with minified JSON only: {"current_status": {"1": "...", ...}, "next_action": <int>}.
""".strip()

OBS_SCENE_NAME = "シーン"
CELL_SOURCE_NAMES = [str(i) for i in range(1, 10)]


@dataclass
class PlannerResult:
    current_status: Dict[str, str]
    next_action: int
    engine: str
    raw_response: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the next tic-tac-toe move for player 'X' from textual board state using a local LLM."
        )
    )
    parser.add_argument(
        "--state-json",
        help="Path to board state JSON exported by tic_tac_toe_overlay (cells + bbox).",
    )
    parser.add_argument(
        "--state",
        help=(
            "Inline board state string, e.g. '1=O,2=empty,3=X,4=empty,5=empty,6=empty,7=empty,8=empty,9=empty'."
        ),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Local model id/path for text generation (default: Qwen/Qwen3-4B-Instruct-2507).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the LLM (defaults to 0.0).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Max tokens to request from the LLM.",
    )
    parser.add_argument(
        "--obs-host",
        default=None,
        help="OBS websocket host to apply the suggested move (omit to skip OBS update).",
    )
    parser.add_argument(
        "--obs-port",
        type=int,
        default=4455,
        help="OBS websocket port (default: 4455).",
    )
    parser.add_argument(
        "--obs-timeout",
        type=float,
        default=5.0,
        help="OBS websocket timeout seconds (default: 5).",
    )
    return parser.parse_args()


def normalize_mark(mark: str) -> BoardMark:
    m = mark.strip().lower()
    mapping = {
        "◯": "o",
        "○": "o",
        "o": "o",
        "0": "o",
        "×": "x",
        "✕": "x",
        "x": "x",
        "□": "empty",
        "empty": "empty",
        "": "empty",
        "-": "empty",
        "_": "empty",
    }
    mapped = mapping.get(m, m)
    if mapped not in {"o", "x", "empty"}:
        raise ValueError(f"Unsupported marker: {mark}")
    if mapped == "o":
        return "O"
    if mapped == "x":
        return "X"
    return "empty"


def normalize_state_map(state_map: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for idx in range(1, 10):
        key = str(idx)
        raw_val = state_map.get(key)
        if raw_val is None and str(idx) not in state_map and idx in state_map:  # type: ignore
            raw_val = state_map[idx]  # type: ignore
        if raw_val is None:
            raise ValueError(f"Missing cell {key} in board state.")
        normalized[key] = normalize_mark(str(raw_val))
    return normalized


def load_state_from_json(json_path: Path) -> Dict[str, str]:
    data = json.loads(json_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Board state JSON must be an object.")
    if "state_map" in data:
        return normalize_state_map({str(k): v for k, v in data["state_map"].items()})
    if "cells" in data:
        cell_map: Dict[str, str] = {}
        for cell in data["cells"]:
            cid = cell.get("id") or cell.get("cell_id") or cell.get("index")
            if cid is None:
                continue
            label = cell.get("label") or cell.get("state")
            if label is None:
                continue
            cell_map[str(cid)] = label
        if cell_map:
            return normalize_state_map(cell_map)
    if all(str(k) in {str(i) for i in range(1, 10)} for k in data.keys()):
        return normalize_state_map({str(k): v for k, v in data.items()})
    raise ValueError("Unsupported JSON format for board state.")


def parse_state_string(text: str) -> Dict[str, str]:
    pieces = re.split(r"[\n,]", text)
    state_map: Dict[str, str] = {}
    for piece in pieces:
        chunk = piece.strip()
        if not chunk:
            continue
        if ":" in chunk:
            key, val = chunk.split(":", 1)
        elif "=" in chunk:
            key, val = chunk.split("=", 1)
        else:
            parts = chunk.split()
            if len(parts) != 2:
                raise ValueError(f"Cannot parse chunk '{chunk}' in board state string.")
            key, val = parts
        state_map[str(int(key))] = val.strip()
    return normalize_state_map(state_map)


def state_to_prompt_text(state_map: Dict[str, str]) -> str:
    lines = ["Board state:"]
    for idx in range(1, 10):
        lines.append(f"{idx}: {state_map[str(idx)]}")
    lines.append("You are X. Respond with JSON only.")
    return "\n".join(lines)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_plan_response_text(text: str, fallback_state: Dict[str, str], engine_label: str) -> PlannerResult:
    cleaned = text.strip()
    candidates = [cleaned]
    if cleaned.startswith("```"):
        candidates.append(cleaned.strip("`"))
        candidates.append(cleaned.replace("```json", "").replace("```", "").strip())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            status = payload.get("current_status", fallback_state)
            next_action = payload.get("next_action")
            status_map = normalize_state_map(status)
            if not isinstance(next_action, int) or not (1 <= next_action <= 9):
                raise ValueError("next_action must be int 1-9")
            return PlannerResult(current_status=status_map, next_action=next_action, engine=engine_label, raw_response=text)
        except Exception:
            continue
    digit_match = re.search(r"([1-9])", cleaned)
    if digit_match:
        return PlannerResult(
            current_status=fallback_state,
            next_action=int(digit_match.group(1)),
            engine=engine_label,
            raw_response=text,
        )
    raise ValueError("Failed to parse LLM response")


_MODEL_CACHE: dict[str, tuple[AutoModelForCausalLM, AutoTokenizer]] = {}


def get_local_model(model_id: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    device = resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto" if device.type != "cpu" else None)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    _MODEL_CACHE[model_id] = (model, tokenizer)
    return model, tokenizer


def build_prompt(tokenizer: AutoTokenizer, state_map: Dict[str, str]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state_to_prompt_text(state_map)},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{state_to_prompt_text(state_map)}\n"


def suggest_with_transformers(
    state_map: Dict[str, str],
    model_id: str,
    temperature: float,
    max_output_tokens: int,
) -> PlannerResult:
    model, tokenizer = get_local_model(model_id)
    prompt_text = build_prompt(tokenizer, state_map)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_output_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return parse_plan_response_text(generated_text, fallback_state=state_map, engine_label="transformers:local")


def generate_plan(
    state_map: Dict[str, str],
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> PlannerResult:
    normalized = normalize_state_map(state_map)
    return suggest_with_transformers(normalized, model, temperature, max_output_tokens)


def apply_obs_next_action(next_action: int, host: str, port: int, timeout: float) -> None:
    target_source = str(next_action)
    try:
        with ReqClient(host=host, port=port, password="", timeout=timeout) as client:
            for source_name in CELL_SOURCE_NAMES:
                item = client.get_scene_item_id(OBS_SCENE_NAME, source_name)
                client.set_scene_item_enabled(
                    OBS_SCENE_NAME,
                    item.scene_item_id,
                    source_name == target_source,
                )
    except (OBSSDKError, ConnectionRefusedError, TimeoutError) as exc:
        raise RuntimeError(f"OBS update failed: {exc}") from exc


def main() -> int:
    args = parse_args()
    if not args.state_json and not args.state:
        sys.stderr.write("Please provide --state-json exported by overlay or --state string.\n")
        return 1

    try:
        if args.state_json:
            state_map = load_state_from_json(Path(args.state_json))
        else:
            state_map = parse_state_string(args.state)
    except Exception as exc:
        sys.stderr.write(f"Failed to load board state: {exc}\n")
        return 1

    try:
        result = generate_plan(
            state_map=state_map,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    payload = {
        "current_status": result.current_status,
        "next_action": result.next_action,
        "engine": result.engine,
    }
    if result.raw_response:
        payload["raw_response"] = result.raw_response
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")

    if args.obs_host:
        try:
            apply_obs_next_action(result.next_action, args.obs_host, args.obs_port, args.obs_timeout)
            sys.stdout.write(
                f"OBS updated: scene '{OBS_SCENE_NAME}' shows source '{result.next_action}' "
                "and hides the other cells.\n"
            )
        except Exception as exc:
            sys.stderr.write(f"Failed to apply OBS changes: {exc}\n")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
