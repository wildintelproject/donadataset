# Classes


DonaDataset uses the following label space for DonaNet training and evaluation.
Class IDs match the order in `dataset/data.yaml`, `metadata/classes.yaml` and the numerical labels used in the YOLO `.txt` annotation files.
The dataset contains 16 object categories plus the `Empty` category used for negative examples.

| ID | Label | Common name / meaning |
|---:|---|---|
| 0 | `Ave` | Bird |
| 1 | `Felis catus` | Feral cat |
| 2 | `Bos taurus` | Cattle |
| 3 | `Canis familiaris` | Dog |
| 4 | `Herpestes ichneumon` | Egyptian mongoose |
| 5 | `Meles meles` | European badger |
| 6 | `Oryctolagus cuniculus` | European rabbit |
| 7 | `Dama dama` | Fallow deer |
| 8 | `Genetta genetta` | Common genet |
| 9 | `Equus sp` | Horse / donkey |
| 10 | `Homo sapiens` | Human and vehicle |
| 11 | `Lepus granatensis` | Iberian hare |
| 12 | `Lynx pardinus` | Iberian lynx |
| 13 | `Cervus elaphus` | Red deer |
| 14 | `Vulpes vulpes` | Red fox |
| 15 | `Sus scrofa` | Wild boar |
| 16 | `Empty` | Image without annotated objects |

Labels `0` to `15` are object categories used in bounding-box annotations.

Label `16`, `Empty`, identifies images without annotated objects. Empty images are included during training as negative examples. In YOLO format, empty images should have an empty `.txt` label file or no bounding-box rows.

!!! note "Important note about `Empty`"
    Although `Empty` is listed in `data.yaml`, it is not used as a bounding-box object class.
    It is used to keep track of images without annotated objects and to include them as negative examples during training.

!!! note "GDPR note"
    Due to GDPR restrictions, images containing `Homo sapiens` are not included in the public image release.
    The `Homo sapiens` label is retained because DonaNet uses this label for humans and vehicles.

!!! note "Updating classes"
    If the label space is updated, the class order must remain consistent across `dataset/data.yaml`, `metadata/classes.yaml`, `metadata/dataset.yaml`, YOLO `.txt` label files and `annotations.csv`.