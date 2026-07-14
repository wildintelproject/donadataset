"""Integration tests chaining 'generate real' -> 'generate toy' (process_toy_dataset)."""
from pathlib import Path

import yaml

from donadataset.commands.generate import (
    DuplicateKeyMode,
    load_original_names,
    process_dataset,
    process_toy_dataset,
)

SOURCE_CLASSES_MAP = Path(__file__).resolve().parent.parent.parent / "metadata" / "source_classes.yaml"


def _run_real(source: Path, output: Path) -> None:
    process_dataset(
        source=source,
        output=output,
        splits=["train", "val", "test"],
        remove_class_ids={10, 17},
        duplicate_key_mode=DuplicateKeyMode.stem,
        original_names=load_original_names(SOURCE_CLASSES_MAP),
    )


def test_toy_dataset_built_from_real_output(tmp_path: Path, example_source_dataset: Path):
    real_output = tmp_path / "real"
    toy_output = tmp_path / "toy"
    _run_real(example_source_dataset, real_output)

    process_toy_dataset(
        source=real_output,
        output=toy_output,
        splits=["train", "val", "test"],
        samples_per_class={"train": 1, "val": 1, "test": 1},
        random_seed=42,
    )

    toy_yaml = toy_output / "donadatasetToy.yaml"
    assert toy_yaml.exists()
    data = yaml.safe_load(toy_yaml.read_text())
    assert data["nc"] == 16

    for split in ("train", "val", "test"):
        images = list((toy_output / "images" / split).glob("*"))
        labels = list((toy_output / "labels" / split).glob("*"))
        assert len(images) > 0
        assert len(images) == len(labels)


def test_toy_dataset_is_reproducible_with_same_seed(tmp_path: Path, example_source_dataset: Path):
    real_output = tmp_path / "real"
    _run_real(example_source_dataset, real_output)

    def _selected_train_images(output: Path) -> list[str]:
        process_toy_dataset(
            source=real_output,
            output=output,
            splits=["train"],
            samples_per_class={"train": 1, "val": 1, "test": 1},
            random_seed=123,
        )
        return sorted(p.name for p in (output / "images" / "train").glob("*"))

    result_a = _selected_train_images(tmp_path / "toy_a")
    result_b = _selected_train_images(tmp_path / "toy_b")

    assert result_a == result_b


def test_toy_dataset_samples_at_most_n_per_class(make_source_dataset, tmp_path: Path):
    source = make_source_dataset("train", {
        f"img_{i}.jpg": [0] for i in range(5)
    })
    # process_toy_dataset reads its own source YAML, so write one matching this fixture.
    (source / "donana_filtered.yaml").write_text(
        yaml.safe_dump({"nc": 1, "names": {0: "Ave"}})
    )
    output = tmp_path / "toy"

    process_toy_dataset(
        source=source,
        output=output,
        splits=["train"],
        samples_per_class={"train": 2, "val": 2, "test": 2},
        random_seed=1,
    )

    images = list((output / "images" / "train").glob("*"))
    assert len(images) == 2


def test_toy_dataset_keeps_all_images_when_fewer_than_requested(make_source_dataset, tmp_path: Path):
    source = make_source_dataset("train", {"only.jpg": [0]})
    (source / "donana_filtered.yaml").write_text(
        yaml.safe_dump({"nc": 1, "names": {0: "Ave"}})
    )
    output = tmp_path / "toy"

    process_toy_dataset(
        source=source,
        output=output,
        splits=["train"],
        samples_per_class={"train": 25, "val": 25, "test": 25},
        random_seed=1,
    )

    images = list((output / "images" / "train").glob("*"))
    assert len(images) == 1
