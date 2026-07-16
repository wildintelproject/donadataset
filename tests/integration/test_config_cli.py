"""Integration tests for 'donadataset generate config show/get/set'."""
import pytest
from typer.testing import CliRunner

from donadataset.config import DEFAULT_CONFIG_FILE, load_settings
from donadataset.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _restore_config_file():
    """Snapshot settings.toml before each test and restore it after.

    config_commands always reads/writes the single app-wide settings file
    (isolated to a fake HOME by tests/conftest.py), so tests that call
    ``config set`` must not leak state into the next test.
    """
    load_settings()  # make sure it exists before snapshotting
    original = DEFAULT_CONFIG_FILE.read_text(encoding="utf-8")
    yield
    DEFAULT_CONFIG_FILE.write_text(original, encoding="utf-8")


def test_config_show_prints_file_path_and_contents():
    result = runner.invoke(app, ["generate", "config", "show"])
    assert result.exit_code == 0
    assert str(DEFAULT_CONFIG_FILE) in result.output
    assert "[GENERATE]" in result.output
    assert "[GENERATE_TOY]" in result.output


def test_config_get_returns_scalar_value():
    result = runner.invoke(app, ["generate", "config", "get", "GENERATE.duplicate_key_mode"])
    assert result.exit_code == 0
    assert result.output.strip() == "stem"


def test_config_get_unknown_section_errors():
    result = runner.invoke(app, ["generate", "config", "get", "BOGUS.x"])
    assert result.exit_code != 0
    assert "Sección desconocida" in result.output


def test_config_get_unknown_field_errors():
    result = runner.invoke(app, ["generate", "config", "get", "GENERATE.bogus_field"])
    assert result.exit_code != 0
    assert "Parámetro desconocido" in result.output


def test_config_get_requires_dotted_key():
    result = runner.invoke(app, ["generate", "config", "get", "source"])
    assert result.exit_code != 0
    assert "SECCION.CLAVE" in result.output


def test_config_set_scalar_value_persists():
    set_result = runner.invoke(app, ["generate", "config", "set", "GENERATE.duplicate_key_mode=relative_stem"])
    assert set_result.exit_code == 0

    get_result = runner.invoke(app, ["generate", "config", "get", "GENERATE.duplicate_key_mode"])
    assert get_result.output.strip() == "relative_stem"


def test_config_set_int_value_persists():
    set_result = runner.invoke(app, ["generate", "config", "set", "GENERATE_TOY.random_seed=7"])
    assert set_result.exit_code == 0

    get_result = runner.invoke(app, ["generate", "config", "get", "GENERATE_TOY.random_seed"])
    assert get_result.output.strip() == "7"


def test_config_set_list_value_parses_csv():
    set_result = runner.invoke(app, ["generate", "config", "set", "GENERATE.remove_class_ids=10,16,17"])
    assert set_result.exit_code == 0

    get_result = runner.invoke(app, ["generate", "config", "get", "GENERATE.remove_class_ids"])
    assert get_result.output.strip() == "[10, 16, 17]"


def test_config_set_dict_value_parses_csv_pairs():
    set_result = runner.invoke(
        app, ["generate", "config", "set", "GENERATE_TOY.samples_per_class=train=5,val=2,test=2"],
    )
    assert set_result.exit_code == 0

    get_result = runner.invoke(app, ["generate", "config", "get", "GENERATE_TOY.samples_per_class"])
    assert get_result.output.strip() == "{'train': 5, 'val': 2, 'test': 2}"


def test_config_set_invalid_value_rejected_and_not_persisted():
    result = runner.invoke(app, ["generate", "config", "set", "GENERATE.duplicate_key_mode=bogus"])
    assert result.exit_code == 1
    assert "Valor inválido" in result.output

    get_result = runner.invoke(app, ["generate", "config", "get", "GENERATE.duplicate_key_mode"])
    assert get_result.output.strip() == "stem"


def test_config_set_missing_equals_sign_errors():
    result = runner.invoke(app, ["generate", "config", "set", "GENERATE.duplicate_key_mode"])
    assert result.exit_code != 0
    assert "SECCION.CLAVE=VALOR" in result.output


def test_config_set_unknown_field_errors():
    result = runner.invoke(app, ["generate", "config", "set", "GENERATE.bogus_field=1"])
    assert result.exit_code != 0
    assert "Parámetro desconocido" in result.output


# GENERATE has 8 fields, GENERATE_TOY has 5, HUGGINGFACE has 13, ZENODO has 4,
# B2SHARE has 4, GBIF has 8 -> 42 prompts in total, in model declaration order
# (duplicate_key_mode is GENERATE's 5th field).
GENERATE_FIELD_COUNT      = 8
GENERATE_TOY_FIELD_COUNT  = 5
HUGGINGFACE_FIELD_COUNT   = 13
ZENODO_FIELD_COUNT        = 4
B2SHARE_FIELD_COUNT       = 4
GBIF_FIELD_COUNT          = 8
DUPLICATE_KEY_MODE_INDEX  = 4  # 0-based position of duplicate_key_mode within GENERATE


def test_config_wizard_keeps_defaults_when_all_answers_blank():
    blank_answers = "\n" * (
        GENERATE_FIELD_COUNT + GENERATE_TOY_FIELD_COUNT + HUGGINGFACE_FIELD_COUNT
        + ZENODO_FIELD_COUNT + B2SHARE_FIELD_COUNT + GBIF_FIELD_COUNT
    )

    result = runner.invoke(app, ["generate", "config", "wizard"], input=blank_answers)

    assert result.exit_code == 0
    assert "No se ha cambiado ningún valor" in result.output


def test_config_wizard_saves_a_changed_value():
    answers = (
        "\n" * (GENERATE_FIELD_COUNT + GENERATE_TOY_FIELD_COUNT - 1)
        + "7\n"    # random_seed, GENERATE_TOY's last prompt
        + "\n" * HUGGINGFACE_FIELD_COUNT
        + "\n" * ZENODO_FIELD_COUNT
        + "\n" * B2SHARE_FIELD_COUNT
        + "\n" * GBIF_FIELD_COUNT
        + "y\n"    # confirm save
    )

    result = runner.invoke(app, ["generate", "config", "wizard"], input=answers)

    assert result.exit_code == 0, result.output
    assert "GENERATE_TOY.random_seed" in result.output

    get_result = runner.invoke(app, ["generate", "config", "get", "GENERATE_TOY.random_seed"])
    assert get_result.output.strip() == "7"


def test_config_wizard_retries_after_invalid_value_and_cancel_does_not_persist():
    answers = (
        "\n" * DUPLICATE_KEY_MODE_INDEX
        + "bogus\n"           # invalid -> retry
        + "relative_stem\n"   # valid
        + "\n" * (GENERATE_FIELD_COUNT - DUPLICATE_KEY_MODE_INDEX - 1)
        + "\n" * GENERATE_TOY_FIELD_COUNT
        + "\n" * HUGGINGFACE_FIELD_COUNT
        + "\n" * ZENODO_FIELD_COUNT
        + "\n" * B2SHARE_FIELD_COUNT
        + "\n" * GBIF_FIELD_COUNT
        + "n\n"               # cancel at the confirmation step
    )

    result = runner.invoke(app, ["generate", "config", "wizard"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Valor inválido" in result.output
    assert "Cancelado" in result.output

    get_result = runner.invoke(app, ["generate", "config", "get", "GENERATE.duplicate_key_mode"])
    assert get_result.output.strip() == "stem"
