"""Unit tests for services.zenodo's Zenodo staging directory.

'zenodo prepare' must copy every evidence file into a directory separate
from the HuggingFace export dir, so that directory ends up containing
exactly what gets uploaded to Zenodo.
"""
from pathlib import Path

from donadataset.services.zenodo import get_output_filename, get_zenodo_output_dir, stage_files_for_zenodo


def test_default_output_dir_is_sibling_of_hfh_export_dir(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    config = {"paths": {"output_dir": str(hfh_export_dir)}, "project": {"dataset_slug": "mytest"}}

    output_dir = get_zenodo_output_dir(config)

    assert output_dir == hfh_export_dir.parent / "Zenodo_mytest"


def test_output_dir_override_wins(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    custom_dir = tmp_path / "wherever-i-want"
    config = {
        "paths": {"output_dir": str(hfh_export_dir)},
        "project": {"dataset_slug": "mytest"},
        "zenodo": {"output_dir": str(custom_dir)},
    }

    assert get_zenodo_output_dir(config) == custom_dir


def test_stage_files_copies_into_output_dir(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    hfh_export_dir.mkdir()
    readme = hfh_export_dir / "README.md"
    readme.write_text("hello")
    citation = hfh_export_dir / "CITATION.cff"
    citation.write_text("cff content")

    output_dir = tmp_path / "Zenodo_mytest"
    staged = stage_files_for_zenodo([readme, citation], output_dir)

    assert staged == [output_dir / "README.md", output_dir / "CITATION.cff"]
    assert (output_dir / "README.md").read_text() == "hello"
    assert (output_dir / "CITATION.cff").read_text() == "cff content"
    # originals must be untouched, not moved
    assert readme.exists()
    assert citation.exists()


def test_stage_files_is_a_no_op_when_source_already_inside_output_dir(tmp_path: Path):
    output_dir = tmp_path / "Zenodo_mytest"
    output_dir.mkdir()
    already_there = output_dir / "README.md"
    already_there.write_text("hello")

    staged = stage_files_for_zenodo([already_there], output_dir)

    assert staged == [already_there]
    assert already_there.read_text() == "hello"


def test_get_output_filename_defaults_inside_zenodo_output_dir(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    config = {"paths": {"output_dir": str(hfh_export_dir)}, "project": {"dataset_slug": "mytest"}}

    path = get_output_filename(config, "linked_record_filename", "zenodo_linked_dataset_record.json")

    assert path == hfh_export_dir.parent / "Zenodo_mytest" / "zenodo_linked_dataset_record.json"


def test_get_output_filename_respects_explicit_override(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    custom_path = tmp_path / "somewhere-else.json"
    config = {
        "paths": {"output_dir": str(hfh_export_dir)},
        "project": {"dataset_slug": "mytest"},
        "zenodo": {"output": {"linked_record_filename": str(custom_path)}},
    }

    path = get_output_filename(config, "linked_record_filename", "zenodo_linked_dataset_record.json")

    assert path == custom_path
