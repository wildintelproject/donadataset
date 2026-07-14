# User Guide

This guide is for anyone who wants to **use** DonaDataset — download it and train a
model with it. If you're looking to publish or update the dataset itself, see the
[Publishing Guide](publishing-guide.md) instead.

---

## 1. Downloading the dataset

DonaDataset is published across several public repositories — see the
[Publishing Guide](publishing-guide.md) for the full list. This guide covers getting it
from the **main repository, HuggingFace Hub**, where it's published as `.tar` shards
(`data/<split>/*.tar`, one or more per split) rather than loose files, to keep the
number of individual objects in the repository manageable. You don't need to deal with
that directly — pick one of the two methods below.

### The easy way: `scripts/download.py`

```bash
# Set up the environment once
./setup.sh
source .venv/bin/activate

# Download everything into ./data/
python scripts/download.py

# Or just one split
python scripts/download.py --split train

# Or into a custom location
python scripts/download.py --output /path/to/data
```

This downloads the shards for the split(s) you asked for and extracts them for you,
leaving a plain `images/<split>/` + `labels/<split>/` layout underneath `--output`
(default `./data`) — see [section 2](#2-what-youll-find) for exactly what that looks
like. Nothing is left behind afterward except the extracted files; the downloaded
`.tar` shards themselves are removed once extraction finishes.

### The manual way: HuggingFace CLI / Python

If you'd rather not use the script (e.g. you only want the raw archive, or you're
downloading from a fork under a different `--repo-id`):

```bash
pip install huggingface-hub

huggingface-cli download wildintelproject/donadataset \
  --repo-type dataset --local-dir ./donadataset-raw
```

This gets you the whole repository as published — the `.tar` shards under `data/`,
plus `README.md`, `LICENSE`, `CITATION.cff`, and the metadata/manifest files described
in the [Publishing Guide](publishing-huggingface.md#4-how-we-upload-it-every-file-explained).
Extract the shards for the split(s) you want yourself:

```bash
mkdir -p data
for shard in donadataset-raw/data/train/*.tar; do
  tar -xf "$shard" -C data
done
```

Each shard's internal paths already start with `images/<split>/...` and
`labels/<split>/...`, so extracting straight into `data/` reconstructs the same layout
`scripts/download.py` produces automatically.

!!! note "Not to be confused with `donadataset publish huggingface download`"
    That command belongs to the [Publishing Guide](publishing-huggingface.md) — it's a
    maintainer tool that re-downloads the whole repo to verify checksums after an
    upload, requires an `HF_TOKEN`, and doesn't extract anything for training. If you
    just want the data, use one of the two methods above instead.

## 2. What you'll find

### Directory layout

```
data/
├── train/
│   ├── images/   ← camera-trap images (.jpg)
│   └── labels/   ← YOLO annotations (.txt)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

See [Dataset splits](dataset-description.md#dataset-splits) for what each split is for
and how they were built.

### Annotation format

Each label file contains one row per detected animal:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalised to `[0, 1]` relative to the image dimensions. See
[Classes](dataset-description.md#classes) for what each `class_id` maps to.

### Metadata files

Also included in this repository (not part of the downloaded `data/` — these live
alongside it in the project checkout):

| File | Description |
|------|-------------|
| `metadata/classes.yaml` | Maps class ids to common and scientific species names |
| `metadata/dataset.yaml` | Ultralytics YOLO config — use directly with `donanet train` |

### Training on it

`metadata/dataset.yaml` is a ready-to-use
[Ultralytics](https://docs.ultralytics.com/datasets/detect/) YOLO dataset config, so any
Ultralytics-compatible YOLO training setup can use it directly once you've downloaded
the data into the `path` it expects (`./data` by default).

With a generic Ultralytics YOLO install (`pip install ultralytics`):

```bash
yolo detect train data=metadata/dataset.yaml model=yolov8n.pt epochs=100
```

Or with [DonaNet](https://github.com/wildintelproject/donanet), the model this dataset
was purpose-built for:

```bash
donanet train --data metadata/dataset.yaml
```
