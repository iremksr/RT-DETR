
import json
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = DATASETS_DIR / "visdrone_coco"

CATEGORIES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

EXPECTED = {
    "train": {
        "images": 6471,
        "annotations": 343204,
        "special_annotations": 10345,
    },
    "val": {
        "images": 548,
        "annotations": 38759,
        "special_annotations": 1410,
    },
}


def convert_split(split):
    dataset_dir = DATASETS_DIR / f"VisDrone2019-DET-{split}"
    image_dir = dataset_dir / "images"
    annotation_dir = dataset_dir / "annotations"

    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    if not annotation_dir.is_dir():
        raise FileNotFoundError(annotation_dir)

    images = []
    annotations = []
    special_annotations = []

    annotation_id = 1

    image_paths = sorted(image_dir.glob("*.jpg"))

    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as img:
            width, height = img.size

        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        txt_path = annotation_dir / f"{image_path.stem}.txt"

        if not txt_path.exists():
            raise FileNotFoundError(txt_path)

        with open(txt_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                # VisDrone TXT satırlarında sondaki virgülü temizle
                values = [
                    int(value.strip())
                    for value in line.rstrip(",").split(",")
                ]

                if len(values) != 8:
                    raise ValueError(
                        f"Invalid annotation at "
                        f"{txt_path}:{line_number}: {line}"
                    )

                (
                    x,
                    y,
                    w,
                    h,
                    score,
                    category,
                    truncation,
                    occlusion,
                ) = values

                if category not in range(12):
                    raise ValueError(
                        f"Unknown category {category} at "
                        f"{txt_path}:{line_number}"
                    )

                record = {
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "category_id": category,
                    "score": score,
                    "truncation": truncation,
                    "occlusion": occlusion,
                }

                # Ignored regions (0) ve others (11)
                # Eğitim hedeflerine dahil edilmiyor.
                if category in (0, 11):
                    special_annotations.append(record)
                    continue

                # Geçersiz annotation'ları atla
                if score != 1 or w <= 0 or h <= 0:
                    continue

                # VisDrone 1-10 -> RT-DETR 0-9
                mapped_category_id = category - 1

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": mapped_category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })

                annotation_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": i,
                "name": name,
                "supercategory": "object",
            }
            for i, name in enumerate(CATEGORIES)
        ],
    }

    expected = EXPECTED[split]

    assert len(images) == expected["images"]
    assert len(annotations) == expected["annotations"]
    assert len(special_annotations) == expected["special_annotations"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{split}.json"
    special_path = OUTPUT_DIR / f"{split}_special.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    with open(special_path, "w", encoding="utf-8") as f:
        json.dump(special_annotations, f)

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Valid annotations:", len(annotations))
    print("Special annotations:", len(special_annotations))
    print("Category IDs:", list(range(10)))
    print("Saved:", output_path)
    print("Special saved:", special_path)


if __name__ == "__main__":
    convert_split("train")
    convert_split("val")
