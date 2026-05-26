# Dataset Description

## Overview

DonaDataset is a collection of camera-trap images annotated in
[YOLO format](https://docs.ultralytics.com/datasets/detect/) for object detection.
It covers the mammal species present in
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/)
(Huelva, Spain), one of the most important wetland ecosystems in Europe.

## Storage

Images and labels are hosted on HuggingFace Hub at
[wildintelproject/donadataset](https://huggingface.co/datasets/wildintelproject/donadataset).

## Dataset splits

| Split   | Purpose                          |
|---------|----------------------------------|
| `train` | Model training                   |
| `val`   | Hyperparameter tuning / monitoring |
| `test`  | Final evaluation (held-out)      |

## Directory layout (after download)

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

## Annotation format

Each label file contains one row per detected animal:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalised to `[0, 1]` relative to the image dimensions.

## Metadata files

| File | Description |
|------|-------------|
| `metadata/classes.yaml` | Maps class ids to common and scientific species names |
| `metadata/dataset.yaml` | Ultralytics YOLO config — use directly with `donanet train` |
