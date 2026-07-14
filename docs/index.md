# DonaDataset

![WildINTEL](img/wildIntel_logo.webp){ style="display: block; margin: 0 auto;" }

**DonaDataset** is the annotated camera-trap image dataset used to train
[DonaNet](https://github.com/wildintelproject/donanet), a YOLO-based neural network for detecting
and classifying the mammals that inhabit
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/) (Spain).

Images and labels are published across several external repositories — see the
**[Publishing Guide](publishing-guide.md)** for the full list and details.
This repository contains the metadata, class definitions, and utility scripts.

---

## Documentation Map

**[Dataset Description](dataset-description.md)**

What DonaDataset is, the mammal species covered, how the data was collected and
annotated, and dataset splits.

**[User Guide](user-guide.md)**

How to download the dataset and what you'll find once you do: directory layout,
annotation format, and metadata files.

**[About](about.md)**

Background on DonaDataset, the WildINTEL project, and funding.

**[Publishing Guide](publishing-guide.md)** — for maintainers

Step-by-step guide to publishing and keeping the dataset synchronized across HuggingFace
Hub, Zenodo, B2SHARE, GBIF, and the other external repositories.

---

## Quick start

```bash
# 1. Set up the environment
./setup.sh

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Download all splits
python scripts/download.py
```

See the [User Guide](user-guide.md) for alternative download methods and exactly what
you'll find in `./data` once it's done.
