import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    """Return the first image path that matches a label stem."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_rows(images_dir: Path, labels_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = find_image_for_label(images_dir, stem)
        if image_path is None:
            print(f"[warn] image not found for label: {label_path.name}", file=sys.stderr)
            continue

        with label_path.open() as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            rows.append(
                {
                    "image": str(image_path),
                    "label_file": str(label_path),
                    "class_id": None,
                    "cx": None,
                    "cy": None,
                    "w": None,
                    "h": None,
                }
            )
            continue

        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                print(f"[warn] malformed line in {label_path.name}: {ln}", file=sys.stderr)
                continue
            class_id, cx, cy, w, h = parts
            rows.append(
                {
                    "image": str(image_path),
                    "label_file": str(label_path),
                    "class_id": int(class_id),
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO labels to Parquet")
    parser.add_argument("--images", type=Path, required=True, help="Path to images directory")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels directory")
    parser.add_argument("--out", type=Path, required=True, help="Parquet output path")
    args = parser.parse_args()

    rows = load_rows(args.images, args.labels)
    if not rows:
        print("[warn] no rows found; Parquet not written", file=sys.stderr)
        return

    table = pa.Table.from_pylist(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    print(f"[info] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
