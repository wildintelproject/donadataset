#!/usr/bin/env python3
"""
DonaDataset — validation script
Checks dataset integrity:
  - Every image has a matching label file
  - Every label has a matching image file
  - Label files are valid YOLO format
  - Class ids are within the declared range

Usage:
    python scripts/validate.py
    python scripts/validate.py --data ./data
    python scripts/validate.py --split train

Requirements:
    uv sync --group scripts
    or: pip install pyyaml pillow tqdm
"""

import argparse
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METADATA         = Path("metadata/dataset.yaml")
SPLITS           = ["train", "val", "test"]


def load_num_classes() -> int:
    with open(METADATA) as f:
        cfg = yaml.safe_load(f)
    return int(cfg["nc"])


def validate_split(split_dir: Path, nc: int) -> list[str]:
    errors: list[str] = []
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        return [f"Missing directory: {images_dir}"]
    if not labels_dir.exists():
        return [f"Missing directory: {labels_dir}"]

    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    labels = {p.stem: p for p in labels_dir.glob("*.txt")}

    # Images without labels
    for stem in sorted(set(images) - set(labels)):
        errors.append(f"No label for image: {images[stem].name}")

    # Labels without images
    for stem in sorted(set(labels) - set(images)):
        errors.append(f"No image for label: {labels[stem].name}")

    # Validate label content
    for stem, label_path in tqdm(labels.items(), desc=f"  {split_dir.name}", leave=False):
        with open(label_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path.name}:{lineno} — expected 5 fields, got {len(parts)}")
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path.name}:{lineno} — non-numeric values")
                    continue
                if cls_id < 0 or cls_id >= nc:
                    errors.append(f"{label_path.name}:{lineno} — class id {cls_id} out of range [0, {nc-1}]")
                if not all(0.0 <= c <= 1.0 for c in coords):
                    errors.append(f"{label_path.name}:{lineno} — coordinates out of [0, 1] range")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DonaDataset integrity")
    parser.add_argument("--data",  type=Path, default=Path("data"), help="Dataset root (default: ./data)")
    parser.add_argument("--split", choices=SPLITS, default=None,   help="Validate a single split")
    args = parser.parse_args()

    nc = load_num_classes()
    splits = [args.split] if args.split else SPLITS

    all_errors: list[str] = []
    for split in splits:
        split_dir = args.data / split
        print(f"Validating {split} …")
        errors = validate_split(split_dir, nc)
        all_errors.extend(f"[{split}] {e}" for e in errors)

    if all_errors:
        print(f"\n❌  {len(all_errors)} error(s) found:\n")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"\n✅  Dataset is valid ({', '.join(splits)}).")


if __name__ == "__main__":
    main()