# <img src="docs/img/wildIntel_logo.webp" alt="WildINTEL Logo" height="60"> DonaDataset

![License](https://img.shields.io/badge/license-CC--BY--4.0-blue.svg)
[![WildINTEL](https://img.shields.io/badge/WildINTEL-v1.0-blue)](https://wildintel.eu/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-dataset-orange)](https://huggingface.co/datasets/wildintelproject/donadataset)

<hr>

## Camera-trap mammal dataset from Doñana National Park

**DonaDataset** is the annotated camera-trap image dataset used to train
[DonaNet](https://github.com/wildintelproject/donanet), a YOLO-based neural network for detecting
and classifying the mammals that inhabit
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/)
(Spain).

DonaDataset follows the YOLO detection dataset format expected by DonaNet. It contains camera-trap
images, YOLO `.txt` label files, a YOLO dataset configuration file and a tabular annotation file used
for evaluation statistics.

Images and labels are hosted on **[HuggingFace Hub](https://huggingface.co/datasets/wildintelproject/donadataset)**
and also published in the **[Arias Montano](https://rabida.uhu.es/)** institutional repository of the
[University of Huelva](https://www.uhu.es/), on **[Zenodo](https://zenodo.org/)**, and on
**[Dataverse](https://dataverse.harvard.edu/)**.
This repository contains the metadata, class definitions, and utility scripts.
---

## Download

```bash
# 1. Set up the environment
./setup.sh

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Download all splits
python scripts/download.py

# Download a single split
python scripts/download.py --split train
```
After downloading, the dataset should be available in the following DonaNet-compatible structure:
```
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

## Repository structure

```
donadataset/
├── metadata/
│   ├── classes.yaml    ← class id → species name mapping
│   └── dataset.yaml    ← source dataset metadata
├── scripts/
│   ├── download.py     ← download images + labels from HuggingFace
│   ├── upload.py       ← upload images + labels to HuggingFace
│   └── validate.py     ← check dataset integrity
└── docs/               ← MkDocs documentation
```
## Dataset files

### `data.yaml`

The required YOLO dataset configuration file is:
```text
dataset/data.yaml
```

The current DonaDataset configuration contains 16 object categories plus the `Empty` category used for negative examples.
```yaml
path: .

train: images/train
val: images/val
test: images/test

nc: 17

names:
  0: 'Ave'
  1: 'Felis catus'
  2: 'Bos taurus'
  3: 'Canis familiaris'
  4: 'Herpestes ichneumon'
  5: 'Meles meles'
  6: 'Oryctolagus cuniculus'
  7: 'Dama dama'
  8: 'Genetta genetta'
  9: 'Equus sp'
  10: 'Homo sapiens'
  11: 'Lepus granatensis'
  12: 'Lynx pardinus'
  13: 'Cervus elaphus'
  14: 'Vulpes vulpes'
  15: 'Sus scrofa'
  16: 'Empty'
```

The class order in `data.yaml` must match the numerical class IDs used in the YOLO `.txt` label files and in the `label` column of `annotations.csv`, when that column is present.

Although `Empty` is listed in `data.yaml`, it is not used as a bounding-box object class. It identifies images without annotated objects. Empty images are included during training as negative examples, and their corresponding YOLO `.txt` label files should contain no bounding-box rows.


---
### YOLO `.txt` label files

YOLO training uses compact `.txt` label files stored under:

```text
dataset/labels/train/
dataset/labels/val/
dataset/labels/test/
```

Each image should have a corresponding `.txt` label file with the same base filename.

Example:
```text
dataset/images/train/image_001.jpg
dataset/labels/train/image_001.txt
```

Each YOLO label file contains one row per annotated object:
```text
<class_id> <x_center> <y_center> <width> <height>
```

All bounding-box values are normalized relative to image width and height.
For empty images, the `.txt` label file should be empty or should contain no bounding-box rows.

---

### `annotations.csv`

The YOLO `.txt` label files use the compact YOLO format, while `annotations.csv` stores the same annotation information in tabular form for evaluation and summary statistics.

For DonaNet evaluation, `annotations.csv` must contain at least the following columns:
```text
file_name, category, bbox_x_center, bbox_y_center, bbox_width, bbox_height
```
Additional columns such as `label`, `path`, `group`, contributor or source information can also be included.
If `annotations.csv` is missing or does not follow the expected format, DonaNet can still run inference and generate predictions, but it cannot generate the full evaluation statistics.

---

## GDPR note

Due to GDPR restrictions, images containing `Homo sapiens` are not included in the public image release.
The `Homo sapiens` label is retained in the DonaDataset label space because DonaNet uses this label for humans and vehicles. If users want to train a model including this label, compatible images must be added from another permitted source.
When images are added from another source, the labels, YOLO `.txt` files, `data.yaml` and `annotations.csv` must remain consistent.

---

## Documentation

Full documentation available at:
**https://wildintelproject.github.io/donadataset/**

For maintainers — how to publish and update the dataset across all external repositories:
**[Publishing Guide](docs/publishing-guide.md)**

---

## Funding

This work is part of the [WildINTEL project](https://wildintel.eu/), funded by the [Biodiversa+](https://www.biodiversa.eu/) Joint Research Call 2022-2023 "Improved
transnational monitoring of biodiversity and ecosystem change for science and society (BiodivMon)". Biodiversa+ is the
European co-funded biodiversity partnership supporting excellent research on biodiversity with an impact for policy and
society. Biodiversa+ is part of the European Biodiversity Strategy for 2030 that aims to put Europe's biodiversity on a
path to recovery by 2030 and is co-funded by the European Commission.

**WildINTEL has been co-funded by the [European Commission](https://commission.europa.eu/) (GA No. 101052342) and the following funding organisations: [Agencia Estatal de Investigación](https://www.aei.gob.es/) (Spain, PCI2023-145963-2, PCI2024-153489), [National Science Centre](https://www.ncn.gov.pl/?language=en) (Poland, UMO-2023/05/Y/NZ8/00104), the [Research Council of Norway](https://www.forskningsradet.no/en/) (Norway, NFR350962) and the [German Research Foundation](https://www.dfg.de/en/) (Germany).**
---

## License

Dataset: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
Code: [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)
