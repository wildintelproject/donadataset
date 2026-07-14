#!/usr/bin/env python3
"""
DonaDataset — download script
Downloads images and labels from HuggingFace Hub into ./data/

The dataset is published as .tar shards (data/<split>/*.tar, one or more
per split) rather than loose files, to keep the number of individual
objects in the HuggingFace Hub repo manageable — this script downloads
just those shards and extracts them, leaving you with a plain
images/<split>/ + labels/<split>/ layout ready for training.

Usage:
    python scripts/download.py
    python scripts/download.py --split train
    python scripts/download.py --output /path/to/data
    python scripts/download.py --repo-id myuser/my-fork

Requirements:
    uv sync --group scripts
    or: pip install huggingface-hub tqdm
"""

import argparse
import tarfile
import tempfile
from pathlib import Path

HF_REPO = "wildintelproject/donadataset"
SPLITS  = ["train", "val", "test"]


def download(split: str | None, output: Path, repo_id: str) -> None:
    from huggingface_hub import snapshot_download

    splits = [split] if split else SPLITS
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="donadataset-shards-") as tmp:
        tmp_path = Path(tmp)
        print(f"Downloading shards for: {', '.join(splits)} …")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f"data/{s}/*.tar" for s in splits],
            local_dir=tmp_path,
        )

        for s in splits:
            shard_dir = tmp_path / "data" / s
            shards = sorted(shard_dir.glob("*.tar")) if shard_dir.is_dir() else []
            if not shards:
                print(f"  ! No shards found for split '{s}' — skipping.")
                continue

            print(f"Extracting split: {s} ({len(shards)} shard(s)) …")
            for shard in shards:
                with tarfile.open(shard) as tar:
                    tar.extractall(output)
            print(f"  → {output / 'images' / s}, {output / 'labels' / s}")

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
    parser.add_argument(
        "--repo-id", default=HF_REPO,
        help=f"HuggingFace Hub dataset repo to download from (default: {HF_REPO})",
    )
    args = parser.parse_args()
    download(args.split, args.output, args.repo_id)


if __name__ == "__main__":
    main()
