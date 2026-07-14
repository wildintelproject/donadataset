"""Unit tests for the toy-dataset helper functions in donadataset.generate."""
from pathlib import Path

import pytest
import yaml

from donadataset.commands.generate import build_class_to_images_index, create_toy_yaml, load_yaml


def test_load_yaml_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")


def test_load_yaml_reads_content(tmp_path: Path):
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("nc: 2\nnames:\n  0: a\n  1: b\n")

    data = load_yaml(yaml_path)

    assert data == {"nc": 2, "names": {0: "a", 1: "b"}}


def test_build_class_to_images_index_groups_by_class(make_source_dataset):
    source = make_source_dataset("train", {
        "a.jpg": [0],
        "b.jpg": [0, 1],
        "c.jpg": [1],
        "d.jpg": None,  # no label -> contributes to no class
    })

    index = build_class_to_images_index(source, "train")

    assert {p.name for p in index[0]} == {"a.jpg", "b.jpg"}
    assert {p.name for p in index[1]} == {"b.jpg", "c.jpg"}
    assert 2 not in index


def test_create_toy_yaml_rewrites_split_paths_and_keeps_other_keys(tmp_path: Path):
    output = tmp_path / "toy"
    output_yaml = output / "toy.yaml"
    output.mkdir()
    source_yaml_data = {
        "train": "/old/train/path",
        "val": "/old/val/path",
        "test": "/old/test/path",
        "nc": 2,
        "names": {0: "a", 1: "b"},
    }

    create_toy_yaml(source_yaml_data, output, ["train", "val", "test"], output_yaml)

    written = yaml.safe_load(output_yaml.read_text())
    assert written["nc"] == 2
    assert written["names"] == {0: "a", 1: "b"}
    assert written["train"] == str((output / "images" / "train").resolve())
    assert written["val"] == str((output / "images" / "val").resolve())
    assert written["test"] == str((output / "images" / "test").resolve())
