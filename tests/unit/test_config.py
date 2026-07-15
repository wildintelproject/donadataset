"""Unit tests for donadataset.config (Dynaconf + Pydantic settings)."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from donadataset.config import (
    GenerateSettings,
    GenerateToySettings,
    Settings,
    get_app_documents_dir,
    get_hfh_output_dir,
    load_settings,
)


def test_settings_defaults_validate():
    settings = Settings()
    assert settings.GENERATE.splits == ["train", "val", "test"]
    assert settings.GENERATE.remove_class_ids == [10, 17]
    assert settings.GENERATE.duplicate_key_mode == "stem"
    assert settings.GENERATE_TOY.samples_per_class == {"train": 25, "val": 10, "test": 10}
    assert settings.GENERATE_TOY.random_seed == 42


@pytest.mark.parametrize(
    "field,value",
    [
        ("duplicate_key_mode", "bogus"),
        ("splits", ["train", "bogus"]),
    ],
)
def test_generate_settings_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        GenerateSettings(**{field: value})


def test_generate_toy_settings_rejects_invalid_samples_per_class_key():
    with pytest.raises(ValidationError):
        GenerateToySettings(samples_per_class={"train": 1, "bogus": 2})


def test_generate_toy_settings_rejects_incomplete_samples_per_class():
    """A partial dict must be rejected here, not surface as a KeyError later.

    donadataset.generate indexes DEFAULT_SAMPLES["train"/"val"/"test"] at
    import time, so a config missing one of these keys would otherwise crash
    the whole CLI on startup instead of failing validation up front.
    """
    with pytest.raises(ValidationError):
        GenerateToySettings(samples_per_class={"train": 5})


def test_load_settings_creates_file_with_defaults(tmp_path: Path):
    config_file = tmp_path / "settings.toml"
    assert not config_file.exists()

    settings = load_settings(config_file)

    assert config_file.exists()
    assert settings.GENERATE.duplicate_key_mode == "stem"
    assert settings.GENERATE_TOY.samples_per_class["train"] == 25


def test_load_settings_does_not_overwrite_existing_file(tmp_path: Path):
    config_file = tmp_path / "settings.toml"
    config_file.write_text('[GENERATE]\nduplicate_key_mode = "relative_stem"\n')

    settings = load_settings(config_file)

    assert settings.GENERATE.duplicate_key_mode == "relative_stem"


def test_load_settings_rejects_invalid_toml_value(tmp_path: Path):
    config_file = tmp_path / "settings.toml"
    config_file.write_text('[GENERATE]\nduplicate_key_mode = "bogus"\n')

    with pytest.raises(ValidationError):
        load_settings(config_file)


# ── get_hfh_output_dir ───────────────────────────────────────────────────────

def test_get_hfh_output_dir_nests_under_repo_id_when_configured():
    result = get_hfh_output_dir("wildintelproject/donadataset")
    assert result == get_app_documents_dir() / "HFH" / "wildintelproject/donadataset"


def test_get_hfh_output_dir_falls_back_to_flat_hfh_when_repo_id_unset():
    assert get_hfh_output_dir(None) == get_app_documents_dir() / "HFH"
    assert get_hfh_output_dir("") == get_app_documents_dir() / "HFH"
