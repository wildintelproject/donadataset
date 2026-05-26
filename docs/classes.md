# Classes

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
