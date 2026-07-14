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

Images and labels are published across several external repositories — see the
**[Publishing Guide](docs/publishing-guide.md)** for the full list and details.
This repository contains the metadata, class definitions, and utility scripts.

## 📥 Download

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

See the [User Guide](docs/user-guide.md) for alternative download methods and exactly
what you'll find afterward (directory layout, annotation format, metadata files).

## 📂 Repository structure

```
donadataset/
├── metadata/
│   ├── classes.yaml    ← class id → species name mapping
│   └── dataset.yaml    ← Ultralytics YOLO dataset config
├── scripts/
│   ├── download.py     ← download images + labels from HuggingFace
│   ├── upload.py       ← upload images + labels to HuggingFace
│   └── validate.py     ← check dataset integrity
└── docs/               ← MkDocs documentation
```

## 📚 Documentation

Full documentation available at:
**https://wildintelproject.github.io/donadataset/**

For maintainers — how to publish and update the dataset across all external repositories:
**[Publishing Guide](docs/publishing-guide.md)**

## 🏛️ Funding

This work is part of the [WildINTEL project](https://wildintel.eu/), funded by the
[Biodiversa+](https://www.biodiversa.eu/) Joint Research Call 2022–2023
*"Improved transnational monitoring of biodiversity and ecosystem change for science and society (BiodivMon)"*.

## 📝 License

Dataset: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
Code: [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)
