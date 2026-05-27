#!/usr/bin/env python3
"""
DonaDataset — upload script
Uploads images and labels from ./data/ to HuggingFace Hub.

Usage:
    python scripts/upload.py
    python scripts/upload.py --split train
    python scripts/upload.py --input /path/to/data
    python scripts/upload.py --dry-run

Requirements:
    uv sync --group scripts
    or: pip install huggingface-hub tqdm

Authentication:
    huggingface-cli login
    or set the HF_TOKEN environment variable.
"""

import argparse
import os
import sys
from pathlib import Path

HF_REPO = "wildintelproject/donadataset"
SPLITS  = ["train", "val", "test"]


def check_split(split_dir: Path) -> bool:
    """Return True if the split directory has at least images and labels."""
    images = split_dir / "images"
    labels = split_dir / "labels"
    if not images.exists():
        print(f"  ✗ Missing {images}", file=sys.stderr)
        return False
    if not labels.exists():
        print(f"  ✗ Missing {labels}", file=sys.stderr)
        return False
    n_images = len(list(images.glob("*")))
    n_labels = len(list(labels.glob("*")))
    print(f"  ✓ {n_images} images · {n_labels} labels")
    return True


def upload(split: str | None, data_dir: Path, dry_run: bool) -> None:
    from huggingface_hub import HfApi, login

    # ── Authentication ────────────────────────────────────────────────────
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        print("HF_TOKEN not set — attempting cached credentials …")
        # huggingface-cli login must have been run previously
        login(add_to_git_credential=False)

    api = HfApi()
    splits = [split] if split else SPLITS

    # ── Validate local data before uploading ─────────────────────────────
    print("\nValidating local data …")
    errors = []
    for s in splits:
        split_dir = data_dir / s
        print(f"  [{s}]")
        if not split_dir.exists():
            print(f"  ✗ Directory not found: {split_dir}", file=sys.stderr)
            errors.append(s)
        elif not check_split(split_dir):
            errors.append(s)

    if errors:
        print(f"\n✗ Aborting: missing data for splits: {errors}", file=sys.stderr)
        sys.exit(1)

    print()

    # ── Upload ────────────────────────────────────────────────────────────
    for s in splits:
        split_dir = data_dir / s
        print(f"Uploading split: {s} …")

        if dry_run:
            print(f"  [dry-run] would upload {split_dir} → {HF_REPO}/{s}/")
            continue

        api.upload_folder(
            repo_id    = HF_REPO,
            repo_type  = "dataset",
            folder_path= str(split_dir),
            path_in_repo= s,
            commit_message=f"Upload split: {s}",
        )
        print(f"  ✓ {split_dir} → {HF_REPO}/{s}/")

    if dry_run:
        print("\n[dry-run] No files were uploaded.")
    else:
        print("\nDone. Dataset updated at:")
        print(f"  https://huggingface.co/datasets/{HF_REPO}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload DonaDataset splits to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/upload.py                     # upload all splits
  python scripts/upload.py --split train       # upload only train
  python scripts/upload.py --dry-run           # check data without uploading

Authentication:
  Set the HF_TOKEN environment variable, or run 'huggingface-cli login' first.
        """,
    )
    parser.add_argument(
        "--split", choices=SPLITS, default=None,
        help="Upload a single split (default: all)",
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data"),
        help="Local data directory (default: ./data)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate local data without uploading anything",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"✗ Data directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    upload(args.split, args.input, args.dry_run)


if __name__ == "__main__":
    main()
