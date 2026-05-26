#!/usr/bin/env python3
"""
DonaDataset — download script
Downloads images and labels from HuggingFace Hub into ./data/

Usage:
    python scripts/download.py
    python scripts/download.py --split train
    python scripts/download.py --output /path/to/data

Requirements:
    uv sync --group scripts
    or: pip install huggingface-hub tqdm
"""

import argparse
from pathlib import Path

HF_REPO = "wildintelproject/donadataset"
SPLITS  = ["train", "val", "test"]


def download(split: str | None, output: Path) -> None:
    from huggingface_hub import snapshot_download

    splits = [split] if split else SPLITS

    for s in splits:
        print(f"Downloading split: {s} …")
        snapshot_download(
            repo_id   = HF_REPO,
            repo_type = "dataset",
            allow_patterns = [f"{s}/*"],
            local_dir = output,
        )
        print(f"  → {output / s}")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DonaDataset from HuggingFace Hub")
    parser.add_argument(
        "--split", choices=SPLITS, default=None,
        help="Download a single split (default: all)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data"),
        help="Local destination directory (default: ./data)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    download(args.split, args.output)


if __name__ == "__main__":
    main()