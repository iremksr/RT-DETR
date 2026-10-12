
import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "visdrone_coco"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

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


def validate_split(split):
    json_path = DATA_DIR / f"{split}.json"
    special_path = DATA_DIR / f"{split}_special.json"

    image_dir = (
        RAW_DIR / f"VisDrone2019-DET-{split}" / "images"
    )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(special_path, "r", encoding="utf-8") as f:
        special_annotations = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    expected = EXPECTED[split]

    assert len(images) == expected["images"]
    assert len(annotations) == expected["annotations"]
    assert len(special_annotations) == expected["special_annotations"]

    image_ids = [img["id"] for img in images]
    annotation_ids = [ann["id"] for ann in annotations]
    category_ids = [cat["id"] for cat in categories]

    assert len(image_ids) == len(set(image_ids))
    assert len(annotation_ids) == len(set(annotation_ids))

    # RT-DETR 10 sınıf: 0-9
    assert set(category_ids) == set(range(10))

    image_lookup = {img["id"]: img for img in images}

    for img in images:
        assert (image_dir / img["file_name"]).is_file(), (
            f"Missing image: {img['file_name']}"
        )

    invalid_boxes = []
    out_of_bounds = []

    for ann in annotations:
        assert ann["image_id"] in image_lookup
        assert ann["category_id"] in range(10)
        assert ann["iscrowd"] == 0

        x, y, w, h = ann["bbox"]

        if w <= 0 or h <= 0:
            invalid_boxes.append(ann["id"])

        img = image_lookup[ann["image_id"]]

        if (
            x < 0
            or y < 0
            or x + w > img["width"]
            or y + h > img["height"]
        ):
            out_of_bounds.append(ann["id"])

        assert ann["area"] == w * h

    # Özel annotation'ların orijinal kategori ID'leri korunmalı
    for ann in special_annotations:
        assert ann["image_id"] in image_lookup
        assert ann["category_id"] in (0, 11)

    class_counts = Counter(
        ann["category_id"] for ann in annotations
    )

    print(f"\n{split.upper()} VALIDATION")
    print("Images:", len(images))
    print("Annotations:", len(annotations))
    print("Special annotations:", len(special_annotations))
    print("Categories:", len(categories))
    print("Category IDs:", sorted(category_ids))
    print("Invalid boxes:", len(invalid_boxes))
    print("Out-of-bounds boxes:", len(out_of_bounds))
    print("Class counts:", dict(sorted(class_counts.items())))

    assert not invalid_boxes, "Invalid bounding boxes found!"
    assert not out_of_bounds, "Out-of-bounds boxes found!"

    print("Validation: PASSED")


if __name__ == "__main__":
    validate_split("train")
    validate_split("val")
