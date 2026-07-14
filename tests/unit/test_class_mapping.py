"""Unit tests for donadataset.generate.build_class_mapping."""
from donadataset.commands.generate import build_class_mapping


def test_removes_ids_and_renumbers_consecutively():
    original = {0: "Ave", 1: "Felis catus", 2: "Bos taurus"}

    old_to_new, new_names = build_class_mapping(original, remove_ids={1})

    assert old_to_new == {0: 0, 2: 1}
    assert new_names == {0: "Ave", 1: "Bos taurus"}


def test_no_removals_is_identity_mapping():
    original = {0: "a", 1: "b", 2: "c"}

    old_to_new, new_names = build_class_mapping(original, remove_ids=set())

    assert old_to_new == {0: 0, 1: 1, 2: 2}
    assert new_names == original


def test_removes_multiple_ids_like_homo_sapiens_and_vehicle():
    original = {i: f"class_{i}" for i in range(18)}

    old_to_new, new_names = build_class_mapping(original, remove_ids={10, 17})

    assert 10 not in old_to_new
    assert 17 not in old_to_new
    assert len(old_to_new) == 16
    # ids stay contiguous starting at 0
    assert sorted(new_names.keys()) == list(range(16))
