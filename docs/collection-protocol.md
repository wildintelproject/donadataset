# Collection Protocol

## Camera traps

Images were collected using camera traps deployed across Doñana National Park.
Cameras were positioned at known wildlife corridors, water points and feeding areas
to maximise species coverage.

## Image processing

Raw images are exported in JPEG format. No colour correction or resizing is applied
before annotation — the model receives images at their original resolution.

## Annotation workflow

Annotations were produced using the YOLO bounding-box format. Each detected animal
is labelled with:

- **Class id** — species identifier (see [Classes](classes.md))
- **Bounding box** — normalised centre coordinates and dimensions

Ambiguous detections (partial occlusion, low confidence) are excluded from the dataset.

## Dataset splits

Images are split into `train / val / test` partitions using the
[DonaNet](https://github.com/wildintelproject/donanet) `prepare-dataset` command
with the following default ratios:

| Split   | Ratio |
|---------|-------|
| `train` | 70 %  |
| `val`   | 20 %  |
| `test`  | 10 %  |

Stratified sampling is applied to ensure each species is represented in all splits.
