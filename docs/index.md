# DonaDataset

![WildINTEL](img/wildIntel_logo.webp){ style="display: block; margin: 0 auto;" }

**DonaDataset** is the annotated camera-trap image dataset used to train
[DonaNet](https://github.com/wildintelproject/donanet), a YOLO-based neural network for detecting
and classifying the mammals that inhabit
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/) (Spain).

Images and labels are hosted on **[HuggingFace Hub](https://huggingface.co/datasets/wildintelproject/donadataset)**.
This repository contains the metadata, class definitions, and utility scripts.

---

## Documentation Map

- [Dataset Description](dataset-description.md)
- [Classes](classes.md)
- [Collection Protocol](collection-protocol.md)
- [About](about.md)

---

## Quick download

```bash
python scripts/download.py
```

Or with the HuggingFace CLI:

```bash
huggingface-cli download wildintelproject/donadataset \
  --repo-type dataset --local-dir ./data
```
