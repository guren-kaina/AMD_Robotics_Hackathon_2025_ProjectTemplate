import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import pipeline
from ultralytics import YOLO
import open_clip
import math
import csv


# =========================
# Data structures
# =========================

@dataclass
class FrameDecision:
    """Summary of a processed frame."""
    o_count: int
    x_count: int
    obstruction_hits: List[str]
    board_clear: bool
    our_turn: bool
    saved_path: Optional[Path]


# =========================
# Utilities
# =========================

def parse_source(value: str) -> Union[int, str]:
    if value.isdigit():
        return int(value)
    return value


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


# =========================
# (A) FIXED ROI: hard-coded board & cells
# =========================
#
# 画像 sample（640x480）から計測した値:
#   board_bbox_px = (x=202, y=129, w=212, h=211)
# これを比率に直して保存し、実行時の画像サイズに応じて復元します。
#
# 「構図が同じ（カメラ位置、盤の置き方、ズームが固定）」であれば、
# 解像度変更にも耐える設計です。微調整したくなったら下の定数だけ
# いじればOKです。

# 640x480基準（width=640, height=480）の相対比率
FIXED_BOARD_BBOX_FRAC = dict(
    x=202 / 640.0,
    y=129 / 480.0,
    w=212 / 640.0,
    h=211 / 480.0,
)

def _fixed_board_bbox(img_h: int, img_w: int) -> Tuple[int, int, int, int]:
    """固定画角用：比率から盤の外接矩形(px)を復元"""
    bx = int(round(FIXED_BOARD_BBOX_FRAC["x"] * img_w))
    by = int(round(FIXED_BOARD_BBOX_FRAC["y"] * img_h))
    bw = int(round(FIXED_BOARD_BBOX_FRAC["w"] * img_w))
    bh = int(round(FIXED_BOARD_BBOX_FRAC["h"] * img_h))
    # 画像内にクリップ
    bx = max(0, min(img_w - 1, bx))
    by = max(0, min(img_h - 1, by))
    bw = max(2, min(img_w - bx, bw))
    bh = max(2, min(img_h - by, bh))
    return bx, by, bw, bh


def split_cells_fixed_roi(frame_bgr: np.ndarray, inner_margin: float) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    盤の外接矩形を固定比率から復元し、3x3に等分して9セルを切り出す。
    inner_margin は各セル辺に対する相対マージン（0〜0.2程度）。
    戻り値: (cells_pil, cells_bgr)
    """
    h, w = frame_bgr.shape[:2]
    bx, by, bw, bh = _fixed_board_bbox(h, w)

    # セルサイズ（floatで計算）
    sh, sw = bh / 3.0, bw / 3.0
    m = float(np.clip(inner_margin, 0.0, 0.2))

    cells_pil: List[Image.Image] = []
    cells_bgr: List[np.ndarray] = []
    for r in range(3):
        for c in range(3):
            y0 = by + r * sh + m * sh
            x0 = bx + c * sw + m * sw
            y1 = by + (r + 1) * sh - m * sh
            x1 = bx + (c + 1) * sw - m * sw
            iy0, ix0 = int(round(y0)), int(round(x0))
            iy1, ix1 = int(round(y1)), int(round(x1))
            iy0 = max(0, min(h - 1, iy0)); iy1 = max(iy0 + 1, min(h, iy1))
            ix0 = max(0, min(w - 1, ix0)); ix1 = max(ix0 + 1, min(w, ix1))
            crop = frame_bgr[iy0:iy1, ix0:ix1]
            cells_bgr.append(crop)
            cells_pil.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    return cells_pil, cells_bgr


# =========================
# (B) AUTO ROI: 従来の自動検出（必要なら使えるように残す）
# =========================

def _prep_for_lines(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    binimg = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binimg


def _cluster_lines_by_angle(lines: Optional[np.ndarray], angle_thresh_deg: float = 20.0):
    if lines is None or len(lines) == 0:
        return [], []
    vertical, horizontal = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        a = abs((angle + 180) % 180)
        if a > 90: a = 180 - a
        if a <= angle_thresh_deg:
            horizontal.append((x1, y1, x2, y2))
        elif (90 - a) <= angle_thresh_deg:
            vertical.append((x1, y1, x2, y2))
    return vertical, horizontal


def detect_board_bbox_auto(image: np.ndarray) -> Tuple[int, int, int, int]:
    """（参考）Houghから適当に最大領域を求める緩いフォールバック"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binimg = _prep_for_lines(gray)
    linesP = cv2.HoughLinesP(
        binimg, rho=1, theta=np.pi / 180, threshold=60,
        minLineLength=int(0.25 * min(h, w)), maxLineGap=10
    )
    vlines, hlines = _cluster_lines_by_angle(linesP, 20.0)
    # 線分のバウンディングボックス
    xs, ys = [], []
    for x1,y1,x2,y2 in (vlines + hlines):
        xs += [x1, x2]; ys += [y1, y2]
    if not xs or not ys:
        # 画像中央にフォールバック
        side = int(0.9 * min(h, w))
        x0 = (w - side) // 2; y0 = (h - side) // 2
        return x0, y0, side, side
    x0, x1 = max(0, min(xs)), min(w-1, max(xs))
    y0, y1 = max(0, min(ys)), min(h-1, max(ys))
    bw, bh = max(2, x1 - x0 + 1), max(2, y1 - y0 + 1)
    return x0, y0, bw, bh


# =========================
# Classical priors: O（白い円チップ）
# =========================

def _circle_confidence(cell_bgr: np.ndarray) -> float:
    """円チップ（明るい円）らしさ: 0..1"""
    h, w = cell_bgr.shape[:2]
    if h < 8 or w < 8:
        return 0.0
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    A = cv2.contourArea(cnt)
    if A < 1.0:
        return 0.0
    P = cv2.arcLength(cnt, True)
    circ = 4.0 * np.pi * A / max(1.0, P * P)  # 0..1
    area_ratio = A / float(h * w)
    ar_score = 0.0
    if 0.03 <= area_ratio <= 0.40:
        center = 0.15
        ar_score = max(0.0, 1.0 - abs(area_ratio - center) / center)
    return float(max(0.0, min(1.0, circ))) * float(max(0.0, min(1.0, ar_score)))


# =========================
# Argparse
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tic-tac-toe detector with fixed-ROI grid cropping and zero-shot classification."
    )
    parser.add_argument("--detector-engine", choices=["yolo", "open-vocab", "clip-grid"], default="clip-grid")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--source", default="0")
    source_group.add_argument("--image", default=None)

    # ROIモード: fixed（推奨）/ auto
    parser.add_argument("--roi-mode", choices=["fixed", "auto"], default="fixed",
                        help="Use 'fixed' to crop cells by hard-coded coordinates (recommended for fixed camera).")

    # CLIP 設定
    parser.add_argument("--clip-model-name", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--clip-tau-low", type=float, default=0.50)
    parser.add_argument("--clip-tau-empty", type=float, default=0.45)
    parser.add_argument("--clip-margin", type=float, default=0.06)  # ← grid線を避け気味

    parser.add_argument("--clip-prompts-o", action="append", default=[
        "a white round game piece (chip/coin/token) placed in a tic-tac-toe cell",
        "a white circular token on a tic-tac-toe board",
        "a round white checker piece in a tic-tac-toe grid",
        "solid white round disc on a tic-tac-toe cell",
        "a white circle mark, an O on a tic-tac-toe cell",
    ])
    parser.add_argument("--clip-prompts-x", action="append", default=[
        "a black cross mark, an X on a tic-tac-toe cell",
        "black cross mark in the tic tac toe board",
        "X shaped mark on a tic-tac-toe board",
        "chalk-drawn X mark",
        "thin pencil cross",
    ])
    parser.add_argument("--clip-prompts-empty", action="append", default=[
        "an empty tic-tac-toe cell with no marks or pieces",
        "blank square with just grid lines, no token",
        "empty cell of a tic-tac-toe board, no O or X, no chip",
    ])

    # YOLO / open-vocab は一応残しておく
    parser.add_argument("--model", default=None)
    parser.add_argument("--hf-model-id", default=None)
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--hf-model-file", default=None)
    parser.add_argument("--ov-model-id", default="google/owlvit-base-patch32")
    parser.add_argument("--ov-revision", default=None)
    parser.add_argument("--ov-score-threshold", type=float, default=0.2)
    parser.add_argument("--o-prompt", default="white circle mark in the tic tac toe board")
    parser.add_argument("--x-prompt", default="black cross mark in the tic tac toe board")
    parser.add_argument("--obstruction-prompt", action="append", default=["hand", "person", "arm over board"])

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--o-label", default="o")
    parser.add_argument("--x-label", default="x")
    parser.add_argument("--obstruction-label", action="append", default=["hand", "person", "obstruction", "blocker"])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--min-gap-sec", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug-dir", default=None, help="Dump cropped cells and scores.")
    return parser.parse_args()


# =========================
# YOLO weights resolver（必要時）
# =========================

def resolve_model_path(args: argparse.Namespace) -> Union[str, Path]:
    if args.hf_model_id:
        repo_dir = Path(
            snapshot_download(repo_id=args.hf_model_id, revision=args.hf_revision, cache_dir=args.hf_cache_dir)
        )
        if args.hf_model_file:
            weights = repo_dir / args.hf_model_file
            if not weights.is_file():
                raise FileNotFoundError(f"Hugging Face model file not found: {weights}")
            return weights
        pt_files = list(repo_dir.glob("*.pt"))
        if len(pt_files) == 1:
            return pt_files[0]
        if not pt_files:
            raise FileNotFoundError("No *.pt weights found in Hugging Face repo. Use --hf-model-file.")
        raise FileNotFoundError("Multiple *.pt files found. Use --hf-model-file to specify one.")
    if args.model:
        return args.model
    raise ValueError("Specify --model or --hf-model-id to load YOLO weights.")


# =========================
# Main detector
# =========================

class TurnDetector:
    def __init__(
        self,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        o_label: str,
        x_label: str,
        obstruction_labels: Iterable[str],
        confidence: float,
        min_gap_sec: float,
        detector_engine: str = "clip-grid",
        ov_model_id: Optional[str] = None,
        ov_revision: Optional[str] = None,
        ov_score_threshold: float = 0.1,
        o_prompt: Optional[str] = None,
        x_prompt: Optional[str] = None,
        obstruction_prompts: Optional[Iterable[str]] = None,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        clip_tau_low: float = 0.50,
        clip_tau_empty: float = 0.45,
        clip_margin: float = 0.06,
        clip_prompts_o: Optional[Iterable[str]] = None,
        clip_prompts_x: Optional[Iterable[str]] = None,
        clip_prompts_empty: Optional[Iterable[str]] = None,
        roi_mode: str = "fixed",
        verbose: bool = False,
        debug_dir: Optional[str] = None,
    ) -> None:
        self.engine = detector_engine
        self.roi_mode = roi_mode
        if self.engine == "yolo":
            self.model = YOLO(str(model_path))
        elif self.engine == "open-vocab":
            if not ov_model_id:
                raise ValueError("ov_model_id is required when using open-vocab detector.")
            self.ov_pipeline = pipeline(
                task="zero-shot-object-detection",
                model=ov_model_id,
                revision=ov_revision,
                device="cpu",
            )
            self.ov_score_threshold = ov_score_threshold
            self.o_prompt = (o_prompt or "").strip()
            self.x_prompt = (x_prompt or "").strip()
            obs_prompts = list(obstruction_prompts or [])
            if not self.o_prompt or not self.x_prompt:
                raise ValueError("o_prompt and x_prompt are required for open-vocab detection.")
            self.obstruction_prompts = [p.strip() for p in obs_prompts if p.strip()] or ["hand", "person", "arm over board"]
            self.prompt_to_label = {
                self.o_prompt.lower(): o_label.lower(),
                self.x_prompt.lower(): x_label.lower(),
            }
            for prompt in self.obstruction_prompts:
                self.prompt_to_label[prompt.lower()] = prompt.lower()
        else:
            # clip-grid
            self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
            self.clip_model = model.to(self.clip_device)
            self.clip_model.eval()
            self.clip_preprocess = preprocess
            tokenizer = open_clip.get_tokenizer(clip_model_name)
            prompts_o = list(clip_prompts_o or [])
            prompts_x = list(clip_prompts_x or [])
            prompts_empty = list(clip_prompts_empty or [])
            if not prompts_o or not prompts_x or not prompts_empty:
                raise ValueError("clip-prompts for O/X/Empty are required in clip-grid mode.")
            with torch.no_grad():
                self.clip_text_features = self._encode_prompts(tokenizer, prompts_o, prompts_x, prompts_empty)
            self.clip_tau_low = clip_tau_low
            self.clip_tau_empty = clip_tau_empty
            self.clip_margin = clip_margin

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.o_label = o_label.lower()
        self.x_label = x_label.lower()
        self.obstruction_labels = {label.lower() for label in obstruction_labels}
        if self.engine == "open-vocab":
            self.obstruction_labels.update({p.lower() for p in self.obstruction_prompts})
        self.confidence = confidence
        self.min_gap_sec = min_gap_sec
        self.verbose = verbose
        self.turn_active = False
        self.last_capture = 0.0
        self.capture_idx = 0
        self._last_frame: Optional["cv2.typing.MatLike"] = None
        self.empty_label = "□"

        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    # ----- CLIP helpers -----

    def _encode_prompts(self, tokenizer, prompts_o: List[str], prompts_x: List[str], prompts_empty: List[str]) -> torch.Tensor:
        def encode_group(prompts: List[str]) -> torch.Tensor:
            tokens = tokenizer(prompts)
            with torch.no_grad():
                feats = self.clip_model.encode_text(tokens.to(self.clip_device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.mean(dim=0, keepdim=True)
        o_feat = encode_group([p for p in prompts_o if p.strip()])
        x_feat = encode_group([p for p in prompts_x if p.strip()])
        empty_feat = encode_group([p for p in prompts_empty if p.strip()])
        return torch.cat([o_feat, x_feat, empty_feat], dim=0)

    def _clip_classify_cells(self, cells: List[Image.Image]) -> torch.Tensor:
        imgs: List[torch.Tensor] = [self.clip_preprocess(c).unsqueeze(0) for c in cells]
        batch = torch.cat(imgs, dim=0).to(self.clip_device)
        use_autocast = torch.cuda.is_available()
        if use_autocast:
            amp_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
            autocast_cm = torch.cuda.amp.autocast(dtype=amp_dtype)
        else:
            autocast_cm = nullcontext()
        with torch.no_grad(), autocast_cm:
            img_feats = self.clip_model.encode_image(batch)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = (100.0 * img_feats @ self.clip_text_features.T).to(torch.float32)
        return logits.softmax(dim=-1).cpu()

    # ----- Priors -----

    def _apply_priors(self, probs_np: np.ndarray, cells_bgr: List[np.ndarray]) -> np.ndarray:
        # probs_np shape: (9, 3) for [O, X, Empty]
        for i, cell_bgr in enumerate(cells_bgr):
            oconf = _circle_confidence(cell_bgr)
            if oconf > 0.0:
                probs_np[i, 0] += 0.35 * oconf
                probs_np[i, 2] -= 0.25 * oconf

            # 「ほぼエッジなし」はEmptyを微増
            edges = cv2.Canny(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY), 40, 120)
            ed = float((edges > 0).mean())
            if ed < 0.02 and oconf < 0.05:
                probs_np[i, 2] += 0.15

            probs_np[i] = np.clip(probs_np[i], 1e-6, 1.0)
            probs_np[i] = probs_np[i] / probs_np[i].sum()
        return probs_np

    # ----- Detection path -----

    def _detect_labels_clip_fixed(self, frame: np.ndarray) -> List[str]:
        cells_pil, cells_bgr = split_cells_fixed_roi(frame, inner_margin=self.clip_margin)
        probs = self._clip_classify_cells(cells_pil)
        probs_np = self._apply_priors(probs.numpy().copy(), cells_bgr)

        labels: List[str] = []
        classes = ["o", "x", "empty"]
        for i in range(9):
            p = probs_np[i]
            max_idx = int(np.argmax(p))
            max_prob = float(p[max_idx])
            cls = classes[max_idx]
            if max_prob < self.clip_tau_low:
                label = "uncertain"
            elif cls == "empty" and max_prob < self.clip_tau_empty:
                label = "uncertain"
            elif cls == "o":
                label = self.o_label
            elif cls == "x":
                label = self.x_label
            else:
                label = self.empty_label
            labels.append(label)

        if self.debug_dir is not None:
            ts = time.strftime("%Y%m%d-%H%M%S")
            for i, cell in enumerate(cells_bgr):
                r, c = divmod(i, 3)
                cv2.imwrite(str(self.debug_dir / f"cell_{ts}_{r}{c}.png"), cell)
            with open(self.debug_dir / f"scores_{ts}.csv", "w", newline="") as f:
                wtr = csv.writer(f)
                wtr.writerow(["cell", "p_O", "p_X", "p_Empty", "label"])
                for i, p in enumerate(probs_np):
                    r, c = divmod(i, 3)
                    wtr.writerow([f"{r},{c}", f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}", labels[i]])
        return labels

    def _detect_labels(self, frame) -> List[str]:
        if self.engine == "yolo":
            results = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
            names = getattr(results, "names", None) or getattr(self.model, "names", {}) or {}
            cls_indices: List[int] = []
            if getattr(results, "boxes", None) is not None and getattr(results.boxes, "cls", None) is not None:
                cls_indices = [int(x) for x in results.boxes.cls.cpu().tolist()]
            return [str(names.get(idx, idx)).lower() for idx in cls_indices]

        if self.engine == "clip-grid":
            if self.roi_mode == "fixed":
                return self._detect_labels_clip_fixed(frame)
            else:
                # AUTO（必要な場合のみ使用）
                x, y, w, h = detect_board_bbox_auto(frame)
                board = frame[y:y+h, x:x+w].copy()
                # 固定処理と同じ切り出しロジックを流用
                cells_pil, cells_bgr = split_cells_fixed_roi(
                    # 一時的にFIXEDに見せるため、擬似的なframe=board、比率も入れ替えたいが
                    # シンプルに3x3分割だけ行う
                    board, inner_margin=self.clip_margin
                )
                probs = self._clip_classify_cells(cells_pil)
                probs_np = self._apply_priors(probs.numpy().copy(), cells_bgr)
                labels: List[str] = []
                classes = ["o", "x", "empty"]
                for i in range(9):
                    p = probs_np[i]
                    max_idx = int(np.argmax(p))
                    max_prob = float(p[max_idx])
                    cls = classes[max_idx]
                    if max_prob < self.clip_tau_low:
                        label = "uncertain"
                    elif cls == "empty" and max_prob < self.clip_tau_empty:
                        label = "uncertain"
                    elif cls == "o":
                        label = self.o_label
                    elif cls == "x":
                        label = self.x_label
                    else:
                        label = self.empty_label
                    labels.append(label)
                return labels

        # open-vocab
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        detections = self.ov_pipeline(
            pil_image,
            candidate_labels=[self.o_prompt, self.x_prompt, *self.obstruction_prompts],
        )
        labels: List[str] = []
        for det in detections:
            score = float(det.get("score", 0.0))
            if score < self.ov_score_threshold:
                continue
            label = str(det.get("label", "")).lower()
            mapped = self.prompt_to_label.get(label)
            if mapped:
                labels.append(mapped)
        return labels

    # ----- Turn logic -----

    def _assess_frame(self, labels: List[str]) -> FrameDecision:
        o_count = sum(1 for label in labels if label == self.o_label)
        x_count = sum(1 for label in labels if label == self.x_label)
        obstruction_hits = [label for label in labels if label in self.obstruction_labels]
        board_clear = len(obstruction_hits) == 0
        our_turn = board_clear and o_count > x_count
        should_capture = self._should_capture(our_turn)
        saved_path = self._save_frame() if should_capture else None
        if self.verbose:
            print(f"detected o={o_count}, x={x_count}, board_clear={board_clear}, our_turn={our_turn}, capture={saved_path}")
        self.turn_active = our_turn
        return FrameDecision(o_count, x_count, obstruction_hits, board_clear, our_turn, saved_path)

    def _should_capture(self, our_turn: bool) -> bool:
        if not our_turn or self.turn_active:
            return False
        now = time.monotonic()
        if now - self.last_capture < self.min_gap_sec:
            return False
        self.last_capture = now
        return True

    def _save_frame(self) -> Path:
        if self._last_frame is None:
            raise RuntimeError("No frame available to save.")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.capture_idx += 1
        path = self.output_dir / f"turn_{timestamp}_{self.capture_idx:04d}.jpg"
        if not cv2.imwrite(str(path), self._last_frame):
            raise RuntimeError(f"Failed to save frame to {path}")
        return path

    # ----- IO -----

    def process_stream(self, source: Union[int, str], max_frames: Optional[int] = None) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        frame_count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                self._last_frame = frame
                labels = self._detect_labels(frame)
                decision = self._assess_frame(labels)
                if decision.saved_path:
                    print(f"Captured your turn: {decision.saved_path}")
                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            cap.release()

    def process_image(self, image_path: Union[str, Path]) -> FrameDecision:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        self._last_frame = frame
        labels = self._detect_labels(frame)
        decision = self._assess_frame(labels)
        print(
            f"image={image_path}, o={decision.o_count}, x={decision.x_count}, "
            f"board_clear={decision.board_clear}, our_turn={decision.our_turn}, captured={decision.saved_path}"
        )
        return decision


# =========================
# Entrypoint
# =========================

def main() -> int:
    args = parse_args()
    try:
        if args.detector_engine == "yolo":
            model_path = resolve_model_path(args)
        elif args.detector_engine == "open-vocab":
            model_path = args.model or args.hf_model_id or args.ov_model_id
        else:
            model_path = args.clip_model_name

        detector = TurnDetector(
            model_path=model_path,
            output_dir=args.output_dir,
            o_label=args.o_label,
            x_label=args.x_label,
            obstruction_labels=args.obstruction_label,
            confidence=args.confidence,
            min_gap_sec=args.min_gap_sec,
            detector_engine=args.detector_engine,
            ov_model_id=args.ov_model_id,
            ov_revision=args.ov_revision,
            ov_score_threshold=args.ov_score_threshold,
            o_prompt=args.o_prompt,
            x_prompt=args.x_prompt,
            obstruction_prompts=args.obstruction_prompt,
            clip_model_name=args.clip_model_name,
            clip_pretrained=args.clip_pretrained,
            clip_tau_low=args.clip_tau_low,
            clip_tau_empty=args.clip_tau_empty,
            clip_margin=args.clip_margin,
            clip_prompts_o=args.clip_prompts_o,
            clip_prompts_x=args.clip_prompts_x,
            clip_prompts_empty=args.clip_prompts_empty,
            roi_mode=args.roi_mode,
            verbose=args.verbose,
            debug_dir=args.debug_dir,
        )
        if args.image:
            detector.process_image(args.image)
        else:
            detector.process_stream(source=parse_source(args.source), max_frames=args.max_frames)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
