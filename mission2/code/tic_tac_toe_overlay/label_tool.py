#!/usr/bin/env python
"""
OpenCV YOLO rectangle labeler
- Mouse drag to create a box
- Keys: 1/2/3 (classes), n/p (next/prev), s (save), d (delete last), q/Esc (quit)
- Labels saved as YOLO format (.txt): cls xc yc w h (normalized) under labels_dir
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

CLS_NAMES = ["empty_cell", "white_circle_cell", "black_cross_cell"]
CLS_COLORS = [(34, 204, 34), (255, 51, 51), (51, 51, 255)]  # BGR


@dataclass
class Box:
    cls: int
    xyxy: Tuple[float, float, float, float]

    def to_yolo(self, w: int, h: int) -> str:
        x0, y0, x1, y1 = self.xyxy
        xc = (x0 + x1) / 2 / w
        yc = (y0 + y1) / 2 / h
        bw = (x1 - x0) / w
        bh = (y1 - y0) / h
        return f"{self.cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

    @staticmethod
    def from_yolo(line: str, w: int, h: int) -> "Box":
        cls, xc, yc, bw, bh = map(float, line.split())
        x0 = (xc - bw / 2) * w
        y0 = (yc - bh / 2) * h
        x1 = (xc + bw / 2) * w
        y1 = (yc + bh / 2) * h
        return Box(int(cls), (x0, y0, x1, y1))


class LabelTool:
    def __init__(self, images: List[Path], labels_dir: Path):
        if not images:
            raise ValueError("No images found.")
        self.images = images
        self.labels_dir = labels_dir
        self.index = 0
        self.boxes: List[Box] = []
        self.history: List[List[Box]] = []
        self.current_cls = 0
        self.drag_start = None
        self.preview = None
        self.window = "label_tool"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)
        self.load_image()

    def load_image(self):
        img_path = self.images[self.index]
        self.image = cv2.imread(str(img_path))
        if self.image is None:
            raise FileNotFoundError(img_path)
        h, w, _ = self.image.shape
        self.boxes = self.read_labels(img_path, w, h)
        self.draw()

    def read_labels(self, img_path: Path, w: int, h: int) -> List[Box]:
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            return []
        boxes = []
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            boxes.append(Box.from_yolo(line, w, h))
        return boxes

    def save_labels(self):
        img_path = self.images[self.index]
        h, w, _ = self.image.shape
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with label_path.open("w", encoding="utf-8") as f:
            for b in self.boxes:
                f.write(b.to_yolo(w, h) + "\n")
        print(f"[info] saved {label_path}")

    def push_history(self):
        # 浅いコピーで履歴保存（Boxは不変データクラス）
        self.history.append(self.boxes.copy())
        if len(self.history) > 50:  # メモリ保護
            self.history.pop(0)

    def undo(self):
        if not self.history:
            return
        self.boxes = self.history.pop()
        self.draw()

    def draw(self):
        vis = self.image.copy()
        # boxes
        for b in self.boxes:
            x0, y0, x1, y1 = map(int, b.xyxy)
            color = CLS_COLORS[b.cls]
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv2.putText(vis, CLS_NAMES[b.cls], (x0 + 3, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # preview
        if self.preview is not None:
            x0, y0, x1, y1 = self.preview
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1, cv2.LINE_AA)
        # HUD
        cv2.putText(
            vis,
            f"img {self.index+1}/{len(self.images)}  cls:{self.current_cls+1}-{CLS_NAMES[self.current_cls]}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window, vis)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.preview = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            x0, y0 = self.drag_start
            self.preview = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            self.draw()
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x0, y0 = self.drag_start
            x1, y1 = x, y
            self.drag_start = None
            self.preview = None
            if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
                self.draw()
                return
            self.push_history()
            box = Box(self.current_cls, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
            self.boxes.append(box)
            self.draw()

    def keyloop(self):
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k == 255:
                continue
            if k in (ord("q"), 27):  # Esc
                self.save_labels()
                break
            elif k in (ord("1"), ord("2"), ord("3")):
                self.current_cls = int(chr(k)) - 1
                self.draw()
            elif k == ord("n"):
                self.save_labels()
                self.index = (self.index + 1) % len(self.images)
                self.load_image()
            elif k == ord("p"):
                self.save_labels()
                self.index = (self.index - 1) % len(self.images)
                self.load_image()
            elif k == ord("s"):
                self.save_labels()
            elif k == ord("d"):
                if self.boxes:
                    self.push_history()
                    self.boxes.pop()
                    self.draw()
            elif k == ord("u") or (k == 26):  # 'u' or Ctrl+Z
                self.undo()


def main():
    parser = argparse.ArgumentParser(description="OpenCV YOLO label tool")
    parser.add_argument("--images", default="real/images", help="画像フォルダ (jpg/png)")
    parser.add_argument("--labels", default="real/labels", help="ラベル保存先フォルダ")
    args = parser.parse_args()

    img_dir = Path(args.images)
    images = sorted([p for p in img_dir.glob("**/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not images:
        raise FileNotFoundError(f"No images found under {img_dir}")
    labels_dir = Path(args.labels)
    tool = LabelTool(images, labels_dir)
    tool.keyloop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
