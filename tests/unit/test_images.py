"""Unit tests for image discovery / duplicate resolution in donadataset.generate."""
from pathlib import Path

from donadataset.commands.generate import (
    EXTENSION_PRIORITY,
    DuplicateKeyMode,
    extension_rank,
    find_images,
    get_duplicate_key,
    get_label_path_for_image,
    select_unique_images,
)


def test_find_images_filters_by_extension_and_sorts(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in ["b.jpg", "a.png", "c.txt", "d.JPG"]:
        (images_dir / name).touch()

    found = find_images(images_dir)

    assert [p.name for p in found] == ["a.png", "b.jpg", "d.JPG"]


def test_find_images_recurses_into_subdirectories(tmp_path: Path):
    images_dir = tmp_path / "images"
    (images_dir / "sub").mkdir(parents=True)
    (images_dir / "top.jpg").touch()
    (images_dir / "sub" / "nested.jpg").touch()

    found = find_images(images_dir)

    assert len(found) == 2


def test_find_images_missing_dir_returns_empty_list(tmp_path: Path):
    assert find_images(tmp_path / "does-not-exist") == []


def test_get_label_path_for_image_mirrors_relative_path(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    image_path = images_dir / "sub" / "a.jpg"

    label_path = get_label_path_for_image(image_path, images_dir, labels_dir)

    assert label_path == labels_dir / "sub" / "a.txt"


def test_get_duplicate_key_stem_mode_ignores_subdirectory(tmp_path: Path):
    images_dir = tmp_path / "images"
    img1 = images_dir / "dirA" / "A.JPG"
    img2 = images_dir / "dirB" / "a.png"

    assert get_duplicate_key(img1, images_dir, DuplicateKeyMode.stem) == "a"
    assert get_duplicate_key(img1, images_dir, DuplicateKeyMode.stem) == get_duplicate_key(
        img2, images_dir, DuplicateKeyMode.stem
    )


def test_get_duplicate_key_relative_stem_mode_distinguishes_subdirs(tmp_path: Path):
    images_dir = tmp_path / "images"
    img1 = images_dir / "dirA" / "a.jpg"
    img2 = images_dir / "dirB" / "a.jpg"

    key1 = get_duplicate_key(img1, images_dir, DuplicateKeyMode.relative_stem)
    key2 = get_duplicate_key(img2, images_dir, DuplicateKeyMode.relative_stem)

    assert key1 != key2


def test_extension_rank_prefers_jpg_over_png():
    assert extension_rank(Path("a.jpg")) < extension_rank(Path("a.png"))


def test_extension_rank_unknown_extension_ranks_last():
    assert extension_rank(Path("a.gif")) == len(EXTENSION_PRIORITY)


def test_select_unique_images_prefers_extension_priority_on_duplicate(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    (images_dir / "a.jpg").touch()
    (images_dir / "a.png").touch()

    images = [images_dir / "a.png", images_dir / "a.jpg"]
    selected, dup_groups, dup_removed = select_unique_images(
        images, images_dir, labels_dir, "train", DuplicateKeyMode.stem,
    )

    assert dup_groups == 1
    assert dup_removed == 1
    assert selected == [images_dir / "a.jpg"]


def test_select_unique_images_prefers_image_with_existing_label(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Same stem, different subdirectories -> only duplicates under "stem" mode.
    (images_dir / "sub1").mkdir()
    (images_dir / "sub2").mkdir()
    img_without_label = images_dir / "sub1" / "a.jpg"
    img_with_label = images_dir / "sub2" / "a.jpg"
    img_without_label.touch()
    img_with_label.touch()
    (labels_dir / "sub2").mkdir()
    (labels_dir / "sub2" / "a.txt").touch()

    selected, _, _ = select_unique_images(
        [img_without_label, img_with_label], images_dir, labels_dir, "train", DuplicateKeyMode.stem,
    )

    assert selected == [img_with_label]


def test_select_unique_images_no_duplicates_keeps_all(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (images_dir / "a.jpg").touch()
    (images_dir / "b.jpg").touch()

    images = [images_dir / "a.jpg", images_dir / "b.jpg"]
    selected, dup_groups, dup_removed = select_unique_images(
        images, images_dir, labels_dir, "train", DuplicateKeyMode.stem,
    )

    assert dup_groups == 0
    assert dup_removed == 0
    assert set(selected) == set(images)


def test_select_unique_images_default_is_verbose(tmp_path: Path, capsys):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (images_dir / "a.jpg").touch()
    (images_dir / "a.png").touch()

    select_unique_images(
        [images_dir / "a.png", images_dir / "a.jpg"], images_dir, labels_dir, "train", DuplicateKeyMode.stem,
    )

    assert "duplicadas" in capsys.readouterr().out


def test_select_unique_images_quiet_suppresses_output_but_keeps_result(tmp_path: Path, capsys):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (images_dir / "a.jpg").touch()
    (images_dir / "a.png").touch()

    selected, dup_groups, dup_removed = select_unique_images(
        [images_dir / "a.png", images_dir / "a.jpg"], images_dir, labels_dir, "train", DuplicateKeyMode.stem,
        quiet=True,
    )

    assert capsys.readouterr().out == ""
    assert dup_groups == 1
    assert dup_removed == 1
    assert selected == [images_dir / "a.jpg"]
