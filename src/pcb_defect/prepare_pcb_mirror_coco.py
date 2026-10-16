#!/usr/bin/env python3
"""Prepare YOLO PCB splits as mirror-padded COCO, preserving original resolution.

Example:
python src/pcb_defect/prepare_pcb_mirror_coco.py \
  --input data/raw/pcb_clean_split \
  --output data/processed/pcb_mirror_rtdetr_1024_v2 \
  --size 1024 \
  --classes mouse_bite spur missing_hole short open_circuit spurious_copper
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Create COCO dataset with reflection padding and valid masks")
    p.add_argument("--input", default="data/raw/pcb_clean_split")
    p.add_argument("--output", default="data/processed/pcb_mirror_rtdetr_1024")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--classes", nargs="+", required=True, help="Class names in YOLO ID order")
    p.add_argument("--image-format", choices=["jpg", "png"], default="jpg", help="JPG uses much less disk")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--png-compression", type=int, default=3)
    p.add_argument("--min-free-gb", type=float, default=1.0, help="Stop before writing if available disk space drops below this")
    return p.parse_args()


def load_splits(source, nclasses):
    splits = {}
    for split in ("train", "val", "test"):
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"Missing image or label directory for {split}")
        entries = []
        for path in sorted(image_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTS:
                continue
            relative = path.relative_to(image_dir)
            label_path = label_dir / relative.with_suffix(".txt")
            if not label_path.is_file():
                raise ValueError(f"Missing label: {label_path}")
            objects = []
            for lineno, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Expected YOLO class cx cy w h at {label_path}:{lineno}")
                try:
                    cls = int(fields[0])
                    cx, cy, bw, bh = map(float, fields[1:])
                except ValueError as exc:
                    raise ValueError(f"Invalid YOLO row at {label_path}:{lineno}") from exc
                if not (0 <= cls < nclasses and all(np.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, bw, bh))):
                    raise ValueError(f"Invalid class or normalized bbox at {label_path}:{lineno}")
                objects.append((cls, cx, cy, bw, bh))
            entries.append((path, relative, objects))
        if not entries:
            raise ValueError(f"No images in {image_dir}")
        splits[split] = entries
    return splits


def checked_write(path, image, params=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = cv2.imwrite(str(path), image, params or [])
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV could not write {path}. Check disk space and permissions.") from exc
    if not ok or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Could not write {path}. Check disk space (df -h .), permissions and file system.")


def main():
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Input directory not found: {source}")
    if args.size < 1 or not (1 <= args.jpeg_quality <= 100) or not (0 <= args.png_compression <= 9):
        raise SystemExit("Invalid size, JPEG quality, or PNG compression")
    if output == source or source in output.parents:
        raise SystemExit("Output must not be inside the input directory")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output is not empty: {output}\nUse a fresh --output path (or manually remove only a confirmed partial output).")

    splits = load_splits(source, len(args.classes))
    categories = [{"id": i + 1, "name": name, "supercategory": "pcb_defect"} for i, name in enumerate(args.classes)]
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotations").mkdir(exist_ok=True)

    for split, entries in splits.items():
        images = []
        annotations = []
        ann_id = 1
        for image_id, (path, relative, objects) in enumerate(entries, 1):
            if image_id == 1 or image_id % 100 == 0:
                free_gb = shutil.disk_usage(output).free / (1024 ** 3)
                if free_gb < args.min_free_gb:
                    raise RuntimeError(f"Low disk space ({free_gb:.2f} GiB free). Stopped at {split} image {image_id}.")
            original = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if original is None:
                raise RuntimeError(f"Cannot read image: {path}")
            height, width = original.shape[:2]
            # Preserve original pixels unless a dimension is larger than the canvas.
            scale = min(1.0, args.size / width, args.size / height)
            new_width = min(args.size, max(1, round(width * scale)))
            new_height = min(args.size, max(1, round(height * scale)))
            if (new_width, new_height) == (width, height):
                resized = original
            else:
                resized = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_AREA)
            left = (args.size - new_width) // 2
            right = args.size - new_width - left
            top = (args.size - new_height) // 2
            bottom = args.size - new_height - top
            mirrored = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)
            mask = np.zeros((args.size, args.size), dtype=np.uint8)
            mask[top:top + new_height, left:left + new_width] = 255

            # Relative path is preserved to avoid filename collisions across nested directories.
            image_rel = Path(split) / "images" / relative.with_suffix("." + args.image_format)
            mask_rel = Path(split) / "valid_masks" / relative.with_suffix(".png")
            image_path = output / image_rel
            mask_path = output / mask_rel
            if args.image_format == "jpg":
                image_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            else:
                image_params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
            checked_write(image_path, mirrored, image_params)
            checked_write(mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression])

            images.append({
                "id": image_id,
                "file_name": image_rel.as_posix(),
                "width": args.size,
                "height": args.size,
                "valid_rect_xywh": [left, top, new_width, new_height],
                "valid_mask_file": mask_rel.as_posix(),
            })
            for cls, cx, cy, bw, bh in objects:
                x1 = max(0.0, (cx - bw / 2) * width)
                y1 = max(0.0, (cy - bh / 2) * height)
                x2 = min(float(width), (cx + bw / 2) * width)
                y2 = min(float(height), (cy + bh / 2) * height)
                if x2 <= x1 or y2 <= y1:
                    continue
                sx = new_width / width
                sy = new_height / height
                box = [left + x1 * sx, top + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [round(v, 4) for v in box],
                    "area": round(box[2] * box[3], 4),
                    "iscrowd": 0,
                })
                ann_id += 1
            if image_id % 500 == 0 or image_id == len(entries):
                print(f"{split}: {image_id}/{len(entries)} images", flush=True)

        coco = {"images": images, "annotations": annotations, "categories": categories}
        ann_path = output / "annotations" / f"instances_{split}.json"
        ann_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        print(f"{split}: {len(images)} images, {len(annotations)} boxes", flush=True)

    print(f"Saved: {output}")
    print("NOTE: Valid masks are saved, but standard RT-DETR loss does not use them automatically.")
    print("NOTE: COCO file_name is relative to the processed dataset root; configure the RT-DETR image folder accordingly.")


if __name__ == "__main__":
    main()
