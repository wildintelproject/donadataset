# Dataset Description

## Overview

DonaDataset is a collection of camera-trap images annotated in
[YOLO format](https://docs.ultralytics.com/datasets/detect/) for object detection.
It covers the mammal species present in
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/)
(Huelva, Spain), one of the most important wetland ecosystems in Europe. It is the
training dataset behind
[DonaNet](https://github.com/wildintelproject/donanet), a YOLO-based neural network
developed as part of the [WildINTEL project](https://wildintel.eu/) for the automated
monitoring of wildlife through camera-trap imagery.

## Classes

The dataset covers the following mammal species present in Doñana National Park.
Class ids match the order in `metadata/classes.yaml` and `metadata/dataset.yaml`.

| ID | Common name       | Scientific name         |
|----|-------------------|-------------------------|
| 0  | Red deer          | *Cervus elaphus*        |
| 1  | Fallow deer       | *Dama dama*             |
| 2  | Wild boar         | *Sus scrofa*            |
| 3  | Iberian lynx      | *Lynx pardinus*         |
| 4  | Red fox           | *Vulpes vulpes*         |
| 5  | Egyptian mongoose | *Herpestes ichneumon*   |
| 6  | European rabbit   | *Oryctolagus cuniculus* |
| 7  | Badger            | *Meles meles*           |

!!! note "Adding new classes"
    To extend the dataset with new species, add entries to `metadata/classes.yaml`
    keeping ids contiguous, update `metadata/dataset.yaml` accordingly, and open a
    pull request. The CI workflow will verify consistency automatically.

## Collection protocol

### Camera traps

Images were collected using camera traps deployed across Doñana National Park.
Cameras were positioned at known wildlife corridors, water points and feeding areas
to maximise species coverage.

### Image processing

Raw images are exported in JPEG format. No colour correction or resizing is applied
before annotation — the model receives images at their original resolution.

### Annotation workflow

Annotations were produced using the YOLO bounding-box format. Each detected animal
is labelled with:

- **Class id** — species identifier (see [Classes](#classes) above)
- **Bounding box** — normalised centre coordinates and dimensions

Ambiguous detections (partial occlusion, low confidence) are excluded from the dataset.

## Dataset splits

Images are split into `train / val / test` partitions using the
[DonaNet](https://github.com/wildintelproject/donanet) `prepare-dataset` command
with the following default ratios:

| Split   | Ratio | Purpose                            |
|---------|-------|-------------------------------------|
| `train` | 70 %  | Model training                     |
| `val`   | 20 %  | Hyperparameter tuning / monitoring |
| `test`  | 10 %  | Final evaluation (held-out)        |

Stratified sampling is applied to ensure each species is represented in all splits.

## Storage

Images and labels are published across several external repositories — see the
[Publishing Guide](publishing-guide.md) for the full list and details. For how to
actually download the data and what you'll find once you do (directory layout,
annotation format, metadata files), see the [User Guide](user-guide.md).
