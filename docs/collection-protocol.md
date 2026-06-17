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

Images were split into `train`, `val` and `test` partitions using a sequence-based strategy designed to reduce spatial and temporal leakage.

Camera-trap images are often captured in bursts, where consecutive images from the same camera can be nearly identical. To avoid placing near-duplicate images in different dataset partitions, images were first grouped by camera location and temporal sequence.
A temporal sequence was defined as a group of consecutive images captured less than 90 seconds apart. All images belonging to the same camera location and temporal sequence were assigned to the same split.
This means that a full sequence was assigned entirely to one of the following partitions:

| Split | Approximate ratio |
|---|---:|
| `train` | 80 % |
| `val` | 10 % |
| `test` | 10 % |

The split was therefore performed at the sequence level, not at the individual image level.
This strategy helps ensure that the training, validation and test sets are more independent in both time and space.
