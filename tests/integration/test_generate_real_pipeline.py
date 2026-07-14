"""Integration tests for the full 'generate real' pipeline (process_dataset)."""
from pathlib import Path

import yaml

from donadataset.commands.generate import DuplicateKeyMode, load_original_names, process_dataset


def _run_real(source: Path, output: Path, splits=("train", "val", "test")) -> None:
    process_dataset(
        source=source,
        output=output,
        splits=list(splits),
        remove_class_ids={10, 17},
        duplicate_key_mode=DuplicateKeyMode.stem,
        original_names=load_original_names(
            Path(__file__).resolve().parent.parent.parent / "metadata" / "source_classes.yaml"
        ),
    )


def test_process_dataset_against_example_source_dataset(tmp_path: Path, example_source_dataset: Path):
    output = tmp_path / "out"

    _run_real(example_source_dataset, output)

    yaml_path = output / "donana_filtered.yaml"
    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text())

    # Homo sapiens (10) and Vehicle (17) removed, ids left contiguous.
    assert data["nc"] == 16
    assert "Homo sapiens" not in data["names"].values()
    assert "Vehicle" not in data["names"].values()

    # Matches the case table documented in examples/source_dataset/README.md.
    train_images = sorted(p.name for p in (output / "images" / "train").iterdir())
    assert train_images == ["img_001.jpg", "img_003.jpg", "img_006.jpg"]

    for split in ("val", "test"):
        images = list((output / "images" / split).glob("*"))
        labels = list((output / "labels" / split).glob("*"))
        assert len(images) == 3
        assert len(labels) == 3


def test_process_dataset_drops_removed_class_image(make_source_dataset, tmp_path: Path):
    source = make_source_dataset("train", {
        "kept.jpg": [0],
        "human.jpg": [10],
    })
    output = tmp_path / "out"

    process_dataset(
        source=source,
        output=output,
        splits=["train"],
        remove_class_ids={10, 17},
        duplicate_key_mode=DuplicateKeyMode.stem,
        original_names={0: "Ave", 10: "Homo sapiens"},
    )

    kept = [p.name for p in (output / "images" / "train").iterdir()]
    assert kept == ["kept.jpg"]


def test_process_dataset_drops_image_with_missing_label(make_source_dataset, tmp_path: Path):
    source = make_source_dataset("train", {
        "kept.jpg": [0],
        "no_label.jpg": None,
    })
    output = tmp_path / "out"

    process_dataset(
        source=source,
        output=output,
        splits=["train"],
        remove_class_ids=set(),
        duplicate_key_mode=DuplicateKeyMode.stem,
        original_names={0: "Ave"},
    )

    kept = [p.name for p in (output / "images" / "train").iterdir()]
    assert kept == ["kept.jpg"]


def test_process_dataset_remaps_class_ids_in_output_labels(make_source_dataset, tmp_path: Path):
    source = make_source_dataset("train", {"a.jpg": [5]})
    output = tmp_path / "out"

    process_dataset(
        source=source,
        output=output,
        splits=["train"],
        remove_class_ids={0, 1, 2, 3, 4},  # classes 0-4 removed, so old id 5 -> new id 0
        duplicate_key_mode=DuplicateKeyMode.stem,
        original_names={i: f"class_{i}" for i in range(6)},
    )

    label_text = (output / "labels" / "train" / "a.txt").read_text().strip()
    assert label_text.startswith("0 ")


def test_process_dataset_wipes_previous_output(make_source_dataset, tmp_path: Path):
    source = make_source_dataset("train", {"a.jpg": [0]})
    output = tmp_path / "out"
    stale_file = output / "leftover.txt"
    output.mkdir(parents=True)
    stale_file.write_text("stale")

    process_dataset(
        source=source,
        output=output,
        splits=["train"],
        remove_class_ids=set(),
        duplicate_key_mode=DuplicateKeyMode.stem,
        original_names={0: "Ave"},
    )

    assert not stale_file.exists()
