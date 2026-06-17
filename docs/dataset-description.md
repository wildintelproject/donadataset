# Dataset Description

## Overview

DonaDataset is a collection of camera-trap images annotated in
[YOLO format](https://docs.ultralytics.com/datasets/detect/) for object detection.
It covers the mammal species present in
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/)
(Huelva, Spain), one of the most important wetland ecosystems in Europe.
The DonaDataset label space contains 16 object categories plus the `Empty` category used for negative examples.
The object categories include mammal species, birds grouped under `Ave`, and humans and vehicles grouped under
the `Homo sapiens` label.
Empty images are included as negative examples during training. These images do not contain annotated objects,
and their YOLO `.txt` label files should contain no bounding-box rows.

## Storage

Images and labels are hosted on HuggingFace Hub at
[wildintelproject/donadataset](https://huggingface.co/datasets/wildintelproject/donadataset),
published in the [Arias Montano](https://rabida.uhu.es/) institutional repository of the
[University of Huelva](https://www.uhu.es/), on [Zenodo](https://zenodo.org/), and on
[Dataverse](https://dataverse.harvard.edu/).

## Dataset splits

| Split   | Purpose                          |
|---------|----------------------------------|
| `train` | Model training                   |
| `val`   | Hyperparameter tuning / monitoring |
| `test`  | Final evaluation (held-out)      |

The dataset is divided into `train`, `val` and `test` partitions.

| Split | Purpose |
|---|---|
| `train` | Model training |
| `val` | Validation during training and monitoring |
| `test` | Final held-out evaluation |

The split was performed at the camera-location and temporal-sequence level, not at the individual image level.
A temporal sequence was defined as a group of consecutive camera-trap images captured less than 90 seconds apart.

All images from the same camera location and temporal sequence were assigned to the same split. This reduces
spatial and temporal leakage between the training, validation and test sets.

---

## Directory layout after download

After download, DonaDataset should follow the DonaNet-compatible dataset layout:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
└── annotations.csv
```
The `dataset/` directory can be placed directly inside the DonaNet repository.

---

## Annotation format

Each YOLO label file contains one row per annotated object:
```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `[0, 1]` relative to the image dimensions.
Each image should have a corresponding `.txt` label file with the same base filename.

Example:
```text
dataset/images/train/image_001.jpg
dataset/labels/train/image_001.txt
```
For empty images, the `.txt` label file should be empty or should contain no bounding-box rows.

---

## Metadata files

| File | Description |
|---|---|
| `dataset/data.yaml` | Ultralytics YOLO dataset configuration used by DonaNet |
| `dataset/annotations.csv` | Tabular annotation file used for DonaNet evaluation statistics |
| `metadata/classes.yaml` | Source class mapping used by the dataset repository |
| `metadata/dataset.yaml` | Source dataset metadata used by the dataset repository |

The class IDs in `dataset/data.yaml`, `metadata/classes.yaml`, YOLO `.txt` label files and `annotations.csv`
must remain consistent.

For DonaNet evaluation, `annotations.csv` must contain at least the following columns:
```text
file_name, category, bbox_x_center, bbox_y_center, bbox_width, bbox_height
```
Additional columns such as `label`, `path`, `group`, contributor or source information can also be included.

---

## GDPR note

Due to GDPR restrictions, images containing `Homo sapiens` are not included in the public image release.
The `Homo sapiens` label is retained in the DonaDataset label space because DonaNet uses this label for humans and vehicles. If users want to train a model including this label, compatible images must be added from another permitted source.
When images are added from another source, the labels, YOLO `.txt` files, `data.yaml` and `annotations.csv` must remain consistent.
