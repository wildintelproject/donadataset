"""Unit tests for YOLO label parsing/filtering/remapping in donadataset.generate."""
from pathlib import Path

import pytest

from donadataset.commands.generate import (
    label_contains_removed_class,
    read_yolo_label,
    remap_label_lines,
)


def test_read_yolo_label_missing_file_returns_empty_list(tmp_path: Path):
    assert read_yolo_label(tmp_path / "missing.txt") == []


def test_read_yolo_label_parses_lines_and_skips_blank(tmp_path: Path):
    label_path = tmp_path / "label.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2\n\n5 0.1 0.1 0.05 0.05\n")

    assert read_yolo_label(label_path) == [
        ["0", "0.5", "0.5", "0.2", "0.2"],
        ["5", "0.1", "0.1", "0.05", "0.05"],
    ]


def test_label_contains_removed_class_true_when_present():
    lines = [["10", "0.5", "0.5", "0.2", "0.2"]]
    assert label_contains_removed_class(lines, {10, 17}) is True


def test_label_contains_removed_class_false_when_absent():
    lines = [["0", "0.5", "0.5", "0.2", "0.2"]]
    assert label_contains_removed_class(lines, {10, 17}) is False


def test_label_contains_removed_class_true_for_mixed_labels():
    """A single removed-class line is enough to flag the whole label."""
    lines = [["6", "0.3", "0.3", "0.1", "0.1"], ["17", "0.6", "0.6", "0.2", "0.2"]]
    assert label_contains_removed_class(lines, {10, 17}) is True


def test_remap_label_lines_rewrites_class_id_only():
    lines = [["5", "0.5", "0.5", "0.2", "0.2"]]
    assert remap_label_lines(lines, {5: 2}) == ["2 0.5 0.5 0.2 0.2"]


def test_remap_label_lines_preserves_multiple_lines_order():
    lines = [["5", "0.1", "0.1", "0.1", "0.1"], ["0", "0.2", "0.2", "0.2", "0.2"]]
    assert remap_label_lines(lines, {5: 1, 0: 0}) == [
        "1 0.1 0.1 0.1 0.1",
        "0 0.2 0.2 0.2 0.2",
    ]


def test_remap_label_lines_raises_for_unmapped_class():
    lines = [["99", "0.5", "0.5", "0.2", "0.2"]]
    with pytest.raises(ValueError):
        remap_label_lines(lines, {5: 2})
