#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class GridBox:
    cls: int
    xyxy: np.ndarray
    conf: float

    @property
    def xyxy_int(self) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.xyxy
        return int(x0), int(y0), int(x1), int(y1)

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.xyxy
        return (x0 + x1) / 2, (y0 + y1) / 2


CLS_TO_LABEL = {0: "empty", 1: "O", 2: "X"}


@dataclass
class CellState:
    id: int
    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0


@dataclass
class BoardState:
    cells: List[CellState]

    @property
    def state_map(self) -> Dict[str, str]:
        return {str(cell.id): cell.label for cell in self.cells}

    def compact_string(self) -> str:
        entries = [
            f"{idx}={self.state_map.get(str(idx), 'unknown')}" for idx in range(1, 10)
        ]
        return ",".join(entries)

    def pretty_text(self) -> str:
        rows = []
        for r in range(3):
            labels = [self.state_map.get(str(r * 3 + c + 1), "?") for c in range(3)]
            rows.append(" | ".join(labels))
        lines = ["Board state:"] + rows + [f"Raw: {self.compact_string()}"]
        return "\n".join(lines)

    def bbox_for_cell(self, cell_id: int) -> Tuple[int, int, int, int] | None:
        for cell in self.cells:
            if cell.id == cell_id:
                return cell.bbox
        return None

    def to_jsonable(self) -> Dict[str, object]:
        cells_payload = []
        for cell in self.cells:
            x0, y0, x1, y1 = cell.bbox
            cells_payload.append(
                {
                    "id": cell.id,
                    "label": cell.label,
                    "bbox": [x0, y0, x1, y1],
                    "center": [cell.center[0], cell.center[1]],
                    "confidence": cell.confidence,
                }
            )
        return {
            "cells": cells_payload,
            "state_map": self.state_map,
            "state_string": self.compact_string(),
        }


def _transform_box(
    x0: float, y0: float, x1: float, y1: float, matrix: np.ndarray
) -> np.ndarray:
    """Return xyxy after affine transform"""
    corners = np.array(
        [[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], dtype=np.float32
    )
    transformed = (matrix @ corners.T).T
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    return np.array([x0, y0, x1, y1])


def _write_yolo_label(
    label_path: Path, boxes: Iterable[Tuple[int, np.ndarray]], width: int, height: int
) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for cls, box in boxes:
            x0, y0, x1, y1 = box
            xc = (x0 + x1) / 2 / width
            yc = (y0 + y1) / 2 / height
            w = (x1 - x0) / width
            h = (y1 - y0) / height
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def generate_synthetic_sample(
    width: int, height: int
) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
    """Generate one synthetic board image and YOLO boxes"""
    # Background variation (including some low-contrast cases)
    bg = random.randint(120, 210)
    base = np.full((height, width, 3), bg, dtype=np.uint8)
    noise = np.random.normal(0, 8, (height, width, 3)).astype(np.int16)
    image = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    margin_edge = 30
    min_dim = min(width, height) - 2 * margin_edge
    board_size = random.randint(int(min_dim * 0.45), int(min_dim * 0.65))
    board_size = max(40, board_size)
    cx = random.randint(
        margin_edge + board_size // 2, width - margin_edge - board_size // 2
    )
    cy = random.randint(
        margin_edge + board_size // 2, height - margin_edge - board_size // 2
    )
    x0 = cx - board_size // 2
    y0 = cy - board_size // 2
    cell = board_size / 3.0
    # Line width ~20% of cell (approx 1cm if cell is 5cm)
    frame_th = max(2, int(cell * random.uniform(0.18, 0.22)))
    grid_th = max(2, int(cell * random.uniform(0.18, 0.22)))
    low_contrast = random.random() < 0.2
    if low_contrast:
        frame_val = int(random.uniform(bg + 15, min(255, bg + 40)))
        grid_val = int(random.uniform(bg + 10, min(255, bg + 35)))
    else:
        frame_val = int(random.uniform(210, 255))
        grid_val = int(random.uniform(200, 245))  # include some lower-contrast lines
    frame_color = (frame_val, frame_val, frame_val)
    grid_color = (grid_val, grid_val, grid_val)

    # outer frame (rectangle border)
    cv2.rectangle(
        image, (x0, y0), (x0 + board_size, y0 + board_size), frame_color, frame_th
    )

    # inner grid lines
    for i in range(1, 3):
        x = int(x0 + cell * i)
        cv2.line(image, (x, y0), (x, y0 + board_size), grid_color, grid_th)
    for i in range(1, 3):
        y = int(y0 + cell * i)
        cv2.line(image, (x0, y), (x0 + board_size, y), grid_color, grid_th)

    # board-level token placement (max 1 token per cell)
    cell_boxes: List[np.ndarray] = []
    n_white = random.randint(0, min(5, 9))
    n_black = random.randint(max(0, n_white - 1), n_white)
    available_indices = list(range(9))
    white_indices = random.sample(available_indices, n_white)
    available_indices = [i for i in available_indices if i not in white_indices]
    n_black = min(n_black, len(available_indices))
    black_indices = random.sample(available_indices, n_black)

    # Padding so tokens do not overlap grid lines
    line_pad_base = max(frame_th, grid_th) + 2.0

    radius_range = (cell * 0.25, cell * 0.30)  # O radius
    size_range = (cell * 0.12, cell * 0.18)  # X half-size
    thickness_cross_range = (
        max(1, int(round(cell * 0.15))),
        max(1, int(round(cell * 0.22))),
    )

    # Cross base color and center head (handle)
    def clip_color(v: float) -> int:
        return int(np.clip(v, 0, 255))

    if low_contrast:
        cross_base_color = clip_color(bg + random.uniform(-10, 25))
    else:
        cross_base_color = clip_color(bg + random.uniform(-20, 40))
    cross_center_radius = (cell * 0.05, cell * 0.10)

    def sample_center(
        left: float,
        top: float,
        right: float,
        bottom: float,
        size: float,
        pad_scale: float = 0.5,
    ):
        pad = line_pad_base * pad_scale + size
        if right - left <= 2 * pad or bottom - top <= 2 * pad:
            return None
        cx = random.uniform(left + pad, right - pad)
        cy = random.uniform(top + pad, bottom - pad)
        return cx, cy

    boxes: List[Tuple[int, np.ndarray]] = []
    for idx in range(9):
        c_idx = idx % 3
        r_idx = idx // 3
        left = x0 + cell * c_idx
        right = x0 + cell * (c_idx + 1)
        top = y0 + cell * r_idx
        bottom = y0 + cell * (r_idx + 1)
        cell_boxes.append(np.array([left, top, right, bottom], dtype=np.float32))

    for idx in white_indices:
        left, top, right, bottom = cell_boxes[idx]
        sampled = None
        for _ in range(5):
            sampled = sample_center(left, top, right, bottom, radius_range[1])
            if sampled is not None:
                break
        if sampled is None:
            continue
        cx, cy = sampled
        r = random.uniform(*radius_range)
        cv2.circle(image, (int(cx), int(cy)), int(r), (245, 245, 245), -1)
        boxes.append((1, np.array([cx - r, cy - r, cx + r, cy + r], dtype=np.float32)))

    for idx in black_indices:
        left, top, right, bottom = cell_boxes[idx]
        sampled = None
        for _ in range(5):
            sampled = sample_center(
                left, top, right, bottom, size_range[1], pad_scale=1.0
            )  # extra padding for X
            if sampled is not None:
                break
        if sampled is None:
            continue
        cx, cy = sampled
        s = random.uniform(*size_range)
        thickness_cross = random.randint(*thickness_cross_range)
        p1 = (int(cx - s), int(cy - s))
        p2 = (int(cx + s), int(cy + s))
        p3 = (int(cx - s), int(cy + s))
        p4 = (int(cx + s), int(cy - s))
        cross_color = (cross_base_color, cross_base_color, cross_base_color)
        cv2.line(image, p1, p2, cross_color, thickness_cross)
        cv2.line(image, p3, p4, cross_color, thickness_cross)
        # Add cylindrical head with slightly darker outline
        head_r = random.uniform(*cross_center_radius)
        head_outline_val = clip_color(cross_base_color - random.uniform(20, 80))
        head_outline = (head_outline_val, head_outline_val, head_outline_val)
        head_outline_th = random.randint(2, 4)
        cv2.circle(image, (int(cx), int(cy)), int(head_r), cross_color, -1)
        cv2.circle(
            image, (int(cx), int(cy)), int(head_r), head_outline, head_outline_th
        )
        boxes.append((2, np.array([cx - s, cy - s, cx + s, cy + s], dtype=np.float32)))

    # Place a few dummy tokens outside the board to learn ignoring them
    def place_distractors(cls_id: int, count: int):
        for _ in range(count):
            # place outside board
            margin = 10
            cx = random.uniform(margin, width - margin)
            cy = random.uniform(margin, height - margin)
            # skip if overlaps board region (simple check)
            if (
                x0 - margin <= cx <= x0 + board_size + margin
                and y0 - margin <= cy <= y0 + board_size + margin
            ):
                continue
            if cls_id == 1:
                r = random.uniform(*radius_range)
                cv2.circle(image, (int(cx), int(cy)), int(r), (245, 245, 245), -1)
                boxes.append(
                    (
                        cls_id,
                        np.array([cx - r, cy - r, cx + r, cy + r], dtype=np.float32),
                    )
                )
            else:
                s = random.uniform(*size_range)
                thickness_cross = random.randint(*thickness_cross_range)
                p1 = (int(cx - s), int(cy - s))
                p2 = (int(cx + s), int(cy + s))
                p3 = (int(cx - s), int(cy + s))
                p4 = (int(cx + s), int(cy - s))
                cross_color = (cross_base_color, cross_base_color, cross_base_color)
                cv2.line(image, p1, p2, cross_color, thickness_cross)
                cv2.line(image, p3, p4, cross_color, thickness_cross)
                head_r = random.uniform(*cross_center_radius)
                head_outline_val = clip_color(cross_base_color - random.uniform(20, 80))
                head_outline = (head_outline_val, head_outline_val, head_outline_val)
                head_outline_th = random.randint(2, 4)
                cv2.circle(image, (int(cx), int(cy)), int(head_r), cross_color, -1)
                cv2.circle(
                    image,
                    (int(cx), int(cy)),
                    int(head_r),
                    head_outline,
                    head_outline_th,
                )
                boxes.append(
                    (
                        cls_id,
                        np.array([cx - s, cy - s, cx + s, cy + s], dtype=np.float32),
                    )
                )

    # Occasionally add dummy tokens
    if random.random() < 0.5:
        place_distractors(1, random.randint(0, 2))
    if random.random() < 0.5:
        place_distractors(2, random.randint(0, 2))

    # Label cells by occupancy (0: empty, 1: white, 2: black)
    for idx, box in enumerate(cell_boxes):
        if idx in white_indices:
            cls_id = 1
        elif idx in black_indices:
            cls_id = 2
        else:
            cls_id = 0
        boxes.append((cls_id, box))

    # Keep board inside image: no rotation, fixed scale
    angle = 0.0
    scale = 1.0
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    image = cv2.warpAffine(image, matrix, (width, height), borderValue=(bg, bg, bg))
    # Brightness/contrast jitter
    alpha = random.uniform(0.8, 1.05)
    beta = random.uniform(-10, 10)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if random.random() < 0.4:
        k = random.choice([3, 5, 7])
        image = cv2.GaussianBlur(image, (k, k), 0)
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(2, 6), image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    rot_boxes: List[Tuple[int, np.ndarray]] = []
    for cls, b in boxes:
        rb = _clip_box(_transform_box(*b, matrix), width, height)
        rot_boxes.append((cls, rb))
    return image, rot_boxes


def write_dataset_yaml(root: Path) -> Path:
    """Write YOLO dataset YAML with absolute paths"""
    root_abs = root.resolve()
    yaml_path = root / "synth_grid.yaml"
    names = ["empty_cell", "white_circle_cell", "black_cross_cell"]
    yaml_content = "\n".join(
        [
            f"path: {root_abs}",
            f"train: {root_abs / 'images/train'}",
            f"val: {root_abs / 'images/val'}",
            "nc: 3",
            f"names: {names}",
            "",
        ]
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def generate_dataset(
    root: Path,
    n_train: int = 1000,
    n_val: int = 200,
    img_size: Tuple[int, int] = (640, 480),
    grayscale: bool = False,
) -> Path:
    """Create synthetic dataset and YAML"""
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    (root / "images/train").mkdir(parents=True, exist_ok=True)
    (root / "images/val").mkdir(parents=True, exist_ok=True)
    (root / "labels/train").mkdir(parents=True, exist_ok=True)
    (root / "labels/val").mkdir(parents=True, exist_ok=True)

    w, h = img_size
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            img, boxes = generate_synthetic_sample(w, h)
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            stem = f"{split}_{i:04d}"
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), img)
            _write_yolo_label(root / "labels" / split / f"{stem}.txt", boxes, w, h)

    return write_dataset_yaml(root)


def ensure_weights(
    weights_path: Path,
    data_yaml: Path,
    epochs: int = 6,
    force: bool = False,
    device: str = "cpu",
    train_nms_time: float | None = None,
    max_det: int = 300,
) -> Path:
    """Train a lightweight YOLO model if weights are missing (or force=True)"""
    if weights_path.exists() and not force:
        return weights_path
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.yaml")
    # Optionally override NMS time limit during train/val
    old_defaults = None
    if train_nms_time is not None:
        try:
            import ultralytics.utils.nms as nms

            fn = nms.non_max_suppression
            defaults = list(fn.__defaults__)
            old_defaults = fn.__defaults__
            # max_time_img is the 9th default arg (0-based)
            defaults[8] = train_nms_time
            fn.__defaults__ = tuple(defaults)
        except Exception as e:  # pragma: no cover
            print(f"[warn] failed to set train_nms_time: {e}")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=device,
        max_det=max_det,
        project=str(weights_path.parent),
        name="train",
        exist_ok=True,
        verbose=True,
    )
    if old_defaults is not None:
        import ultralytics.utils.nms as nms

        nms.non_max_suppression.__defaults__ = old_defaults
    best = Path(model.trainer.best)
    shutil.copy(best, weights_path)
    return weights_path


def copy_real_data(real_root: Path, dataset_root: Path) -> None:
    """Copy real images/labels into train set"""
    img_dir = real_root / "images"
    label_dir = real_root / "labels"
    if not img_dir.exists() or not label_dir.exists():
        print(f"[warn] real data not found under {real_root} (images/labels)")
        return
    dest_img = dataset_root / "images/train"
    dest_lbl = dataset_root / "labels/train"
    copied = 0
    for img in img_dir.glob("**/*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        rel = img.relative_to(img_dir)
        dest_img_path = dest_img / rel
        dest_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dest_img_path)
        lbl_src = label_dir / rel.with_suffix(".txt")
        if lbl_src.exists():
            dest_lbl_path = dest_lbl / rel.with_suffix(".txt")
            dest_lbl_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(lbl_src, dest_lbl_path)
        else:
            print(f"[warn] label not found for {img.name}, skipped label")
        copied += 1
    print(f"[info] copied {copied} real images into train set")


def preprocess_dataset_images(
    dataset_root: Path,
    splits: Tuple[str, ...],
    grayscale: bool,
    clahe: bool,
    contrast_alpha: float | None,
    contrast_beta: float | None,
) -> None:
    """Apply preprocessing to train/val images and augment contrast alpha (0.8/1.0/1.2) for training."""
    if contrast_alpha is None:
        alphas = [1.0]
    else:
        alphas = [contrast_alpha, contrast_alpha * 0.8, contrast_alpha * 1.2]
    for split in splits:
        img_dir = dataset_root / "images" / split
        if not img_dir.exists():
            continue
        count = 0
        for img_path in img_dir.rglob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            # Skip already augmented files (__aXX)
            if "__a" in img_path.stem:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            label_src = (
                dataset_root / "labels" / split / img_path.relative_to(img_dir)
            ).with_suffix(".txt")
            label_text = label_src.read_text() if label_src.exists() else None
            for idx, a in enumerate(alphas):
                target_img_path = (
                    img_path
                    if idx == 0
                    else img_path.with_name(
                        f"{img_path.stem}__a{int(round(a * 100))}{img_path.suffix}"
                    )
                )
                target_label_path = (
                    label_src
                    if idx == 0
                    else label_src.with_name(
                        f"{label_src.stem}__a{int(round(a * 100))}{label_src.suffix}"
                    )
                )
                img_proc = apply_preprocess(
                    img,
                    grayscale=grayscale,
                    clahe=clahe,
                    contrast_alpha=a,
                    contrast_beta=contrast_beta,
                )
                target_img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target_img_path), img_proc)
                if label_text is not None:
                    target_label_path.parent.mkdir(parents=True, exist_ok=True)
                    target_label_path.write_text(label_text)
                count += 1
        print(
            f"[info] preprocessed {count} images in {img_dir} (including augmented alpha variants)"
        )


def predict_cells(
    image: Path | np.ndarray,
    model: Path | YOLO,
    conf: float = 0.5,
    device: str = "cpu",
    max_det: int = 300,
    grayscale: bool = False,
    clahe: bool = False,
    contrast_alpha: float | None = None,
    contrast_beta: float | None = None,
    save_preprocessed: Path | None = None,
) -> List[GridBox]:
    model_obj = model if isinstance(model, YOLO) else YOLO(str(model))
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image}")
    else:
        img = image.copy()
    img = apply_preprocess(
        img,
        grayscale=grayscale,
        clahe=clahe,
        contrast_alpha=contrast_alpha,
        contrast_beta=contrast_beta,
    )
    if save_preprocessed:
        save_preprocessed.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_preprocessed), img)
    predict_kwargs = {
        "source": img,
        "conf": conf,
        "imgsz": 640,
        "device": device,
        "max_det": max_det,
        "verbose": False,
    }
    results = model_obj.predict(**predict_kwargs)
    boxes: List[GridBox] = []
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        cls = int(box.cls.item())
        boxes.append(GridBox(cls=cls, xyxy=xyxy, conf=float(box.conf.item())))
    boxes.sort(key=lambda b: b.conf, reverse=True)
    return boxes


def order_cells(boxes: List[GridBox]) -> List[np.ndarray]:
    """Sort by y then x and return 9 cells in top-left to bottom-right order"""
    if len(boxes) < 9:
        return []
    top9 = sorted(boxes, key=lambda b: b.conf, reverse=True)[:9]
    sorted_by_y = sorted(top9, key=lambda b: b.center[1])
    rows = [sorted_by_y[i : i + 3] for i in range(0, 9, 3)]
    ordered: List[GridBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b.center[0]))
    return ordered


def cells_to_board_state(cells: List[GridBox]) -> BoardState:
    states: List[CellState] = []
    for idx, cell in enumerate(cells):
        states.append(
            CellState(
                id=idx + 1,
                label=CLS_TO_LABEL.get(cell.cls, "unknown"),
                bbox=cell.xyxy_int,
                confidence=cell.conf,
            )
        )
    return BoardState(states)


def board_state_to_text(board_state: BoardState) -> str:
    return board_state.pretty_text()


def save_board_state(
    board_state: BoardState,
    json_path: Path | None,
    text_path: Path | None,
    print_state: bool = False,
) -> None:
    if json_path:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(board_state.to_jsonable(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    text_content = board_state_to_text(board_state)
    if text_path:
        text_path = Path(text_path)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text_content, encoding="utf-8")
    if print_state:
        print(text_content)


def detect_board_state(
    image: Path | np.ndarray,
    model: Path | YOLO,
    conf: float = 0.5,
    device: str = "cpu",
    max_det: int = 300,
    grayscale: bool = False,
    clahe: bool = False,
    contrast_alpha: float | None = None,
    contrast_beta: float | None = None,
    save_preprocessed: Path | None = None,
) -> tuple[BoardState | None, List[GridBox]]:
    boxes = predict_cells(
        image,
        model,
        conf=conf,
        device=device,
        max_det=max_det,
        grayscale=grayscale,
        clahe=clahe,
        contrast_alpha=contrast_alpha,
        contrast_beta=contrast_beta,
        save_preprocessed=save_preprocessed,
    )
    ordered = order_cells(boxes)
    if not ordered:
        return None, boxes
    return cells_to_board_state(ordered), ordered


def apply_preprocess(
    img: np.ndarray,
    grayscale: bool = False,
    clahe: bool = False,
    contrast_alpha: float | None = None,
    contrast_beta: float | None = None,
) -> np.ndarray:
    out = img
    if grayscale:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        if clahe:
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            out = clahe_obj.apply(out)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if contrast_alpha is not None or contrast_beta is not None:
        a = contrast_alpha if contrast_alpha is not None else 1.0
        b = contrast_beta if contrast_beta is not None else 0.0
        out = cv2.convertScaleAbs(out, alpha=a, beta=b)
    return out


def draw_overlay(image_path: Path, cells: List[GridBox], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    cls_colors = [(0, 200, 0), (255, 255, 255), (0, 0, 0)]
    cls_names = ["empty", "O", "X"]
    for idx, cell in enumerate(cells):
        x0, y0, x1, y1 = cell.xyxy_int
        color = cls_colors[cell.cls]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        label = f"{idx + 1}:{cls_names[cell.cls]}"
        cv2.putText(
            image,
            label,
            (int((x0 + x1) / 2), int((y0 + y1) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color if cell.cls != 0 else (0, 0, 255),
            2 if cell.cls != 0 else 1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    image_path = Path(args.image)
    output_path = Path(args.output)
    weights_path = Path(args.weights)
    dataset_root = Path(args.dataset)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    need_regen = args.force_regen or not dataset_root.exists()
    required_dirs = [
        dataset_root / "images/train",
        dataset_root / "images/val",
        dataset_root / "labels/train",
        dataset_root / "labels/val",
    ]
    data_yaml = dataset_root / "synth_grid.yaml"
    if not need_regen:
        if not data_yaml.exists() or any(not d.exists() for d in required_dirs):
            need_regen = True
    if need_regen:
        print(f"[info] Generating synthetic dataset at {dataset_root}")
        data_yaml = generate_dataset(
            dataset_root,
            n_train=args.num_train,
            n_val=args.num_val,
            grayscale=args.grayscale,
        )
    else:
        # rewrite YAML with abs paths if dataset exists
        data_yaml = write_dataset_yaml(dataset_root)
    if args.real_data:
        copy_real_data(Path(args.real_data), dataset_root)
    if args.preprocess_train:
        preprocess_dataset_images(
            dataset_root,
            splits=("train", "val"),
            grayscale=args.grayscale,
            clahe=args.clahe,
            contrast_alpha=args.contrast_alpha,
            contrast_beta=args.contrast_beta,
        )

    if args.force_train or not weights_path.exists():
        print(f"[info] Training lightweight YOLO model to detect grid: {weights_path}")
        ensure_weights(
            weights_path,
            data_yaml,
            epochs=args.epochs,
            force=args.force_train,
            device=args.device,
            train_nms_time=args.train_nms_time,
            max_det=args.max_det,
        )

    print(f"[info] Running detection on {image_path}")
    model = YOLO(str(weights_path))
    board_state, ordered_cells = detect_board_state(
        image_path,
        model,
        conf=args.conf,
        device=args.device,
        max_det=args.max_det,
        grayscale=args.grayscale,
        clahe=args.clahe,
        contrast_alpha=args.contrast_alpha,
        contrast_beta=args.contrast_beta,
        save_preprocessed=Path(args.save_preprocessed)
        if args.save_preprocessed
        else None,
    )
    if not board_state or not ordered_cells:
        print("[error] Failed to detect 9 cells by YOLO.")
        return
    print(f"[info] Detected {len(ordered_cells)} cells")

    save_board_state(
        board_state,
        json_path=Path(args.state_json) if args.state_json else None,
        text_path=Path(args.state_text) if args.state_text else None,
        print_state=args.print_state,
    )

    if not args.skip_overlay:
        draw_overlay(image_path, ordered_cells, output_path)
        print(f"[info] Overlay saved to {output_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect tic-tac-toe cells with YOLO and overlay numbers."
    )
    parser.add_argument("--image", default="top-camera.jpg", help="Input image path")
    parser.add_argument("--output", default="overlay.jpg", help="Output image path")
    parser.add_argument(
        "--skip-overlay",
        action="store_true",
        help="Skip writing overlay image (useful when only the state text/JSON is needed)",
    )
    parser.add_argument(
        "--state-json",
        default=None,
        help="Path to save board state (cell labels + bbox) as JSON",
    )
    parser.add_argument(
        "--state-text",
        default=None,
        help="Path to save board state as readable text",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="Print the board state summary to stdout",
    )
    parser.add_argument(
        "--weights", default="models/cell_grid.pt", help="YOLO weights path"
    )
    parser.add_argument(
        "--dataset", default="data/synth_grid", help="Synthetic dataset root"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold for detection"
    )
    parser.add_argument("--epochs", type=int, default=20, help="YOLO training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device (e.g. cpu, cuda:0, mps)"
    )
    parser.add_argument(
        "--max-det", type=int, default=30, help="Max detections after NMS (train/infer)"
    )
    parser.add_argument(
        "--num-train", type=int, default=1000, help="Number of synthetic train images"
    )
    parser.add_argument(
        "--num-val", type=int, default=200, help="Number of synthetic val images"
    )
    parser.add_argument(
        "--real-data",
        default=None,
        help="Root path of real data (copy images/labels into train)",
    )
    parser.add_argument(
        "--train-nms-time",
        type=float,
        default=None,
        help="Override max_time_img (NMS time limit) during train/val (seconds)",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply grayscale+CLAHE before inference",
    )
    parser.add_argument(
        "--contrast-alpha",
        type=float,
        default=None,
        help="alpha for preprocessing (cv2.convertScaleAbs)",
    )
    parser.add_argument(
        "--contrast-beta",
        type=float,
        default=None,
        help="beta for preprocessing (cv2.convertScaleAbs)",
    )
    parser.add_argument(
        "--save-preprocessed", default=None, help="Path to save preprocessed image"
    )
    parser.add_argument(
        "--preprocess-train",
        action="store_true",
        help="Apply preprocessing (grayscale/clahe/contrast) to train/val images after synth/real copy",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Use grayscale for synth/inference",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        dest="force_train",
        help="Retrain even if weights exist",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        dest="force_regen",
        help="Regenerate synthetic dataset",
    )
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
