# Example source dataset

Mock camera-trap dataset with the same layout that
`donadataset generate real --source <this dir>` expects:

```
source_dataset/
├── images/
│   ├── train/  val/  test/
└── labels/
    ├── train/  val/  test/
```

Labels use the **original 18-class** annotation scheme
(`metadata/source_classes.yaml`), not the public 8-class scheme.

It intentionally exercises every branch of the generation pipeline:

| File(s) | Case |
|---|---|
| `images/train/img_001.jpg` | Normal image, kept as-is |
| `images/train/img_002.jpg` | Label is class `10` (Homo sapiens) → whole image dropped |
| `images/train/img_003.jpg` + `img_003.png` | Duplicate stem, different extension → only the `.jpg` is kept (`EXTENSION_PRIORITY`) |
| `images/train/img_004.jpg` | No matching `.txt` label → reported and skipped |
| `images/train/img_005.jpg` | Label mixes a kept class (`6`) and a removed class (`17`, Vehicle) → whole image dropped |
| `images/train/img_006.jpg` | Two bounding boxes, both kept classes → kept, both lines remapped |
| `images/val/*`, `images/test/*` | Plain kept images across the other splits |

## Try it

```bash
uv run donadataset generate real \
  --source examples/source_dataset \
  --output /tmp/donadataset-demo \
  --split train --split val --split test
```

Inspect the result in `/tmp/donadataset-demo/` — check the printed stats
against the table above (e.g. `train` should report
`kept_images: 3`, `removed_images: 2`, `missing_labels: 1`,
`duplicated_images_removed: 1`, `total_control: 7`, matching `total_images: 7`).
