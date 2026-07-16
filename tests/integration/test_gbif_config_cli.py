"""Integration tests for 'donadataset publish gbif config show/get/set/wizard'.

Same engine as tests/integration/test_config_cli.py's generic 'donadataset
config' (settings.toml's GBIF section), just without the 'GBIF.' prefix.
Since it's the single shared app-wide settings file, every test must restore
it afterward.
"""
import pytest
from typer.testing import CliRunner

from donadataset.config import DEFAULT_CONFIG_FILE, load_settings
from donadataset.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _restore_config_file():
    load_settings()  # make sure it exists before snapshotting
    original = DEFAULT_CONFIG_FILE.read_text(encoding="utf-8")
    yield
    DEFAULT_CONFIG_FILE.write_text(original, encoding="utf-8")


def test_show_prints_gbif_section():
    result = runner.invoke(app, ["publish", "gbif", "config", "show"])
    assert result.exit_code == 0
    assert "GBIF" in result.output
    assert "institution_code" in result.output


def test_get_returns_scalar_value():
    result = runner.invoke(app, ["publish", "gbif", "config", "get", "institution_code"])
    assert result.exit_code == 0
    assert result.output.strip() == "UHU"


def test_get_unknown_field_errors():
    result = runner.invoke(app, ["publish", "gbif", "config", "get", "bogus_field"])
    assert result.exit_code != 0
    assert "Parámetro desconocido" in result.output


def test_set_scalar_value_persists():
    set_result = runner.invoke(app, ["publish", "gbif", "config", "set", "contact_email=you@example.org"])
    assert set_result.exit_code == 0, set_result.output

    get_result = runner.invoke(app, ["publish", "gbif", "config", "get", "contact_email"])
    assert get_result.output.strip() == "you@example.org"

    # unrelated existing fields are untouched
    still_there = runner.invoke(app, ["publish", "gbif", "config", "get", "institution_code"])
    assert still_there.output.strip() == "UHU"


def test_set_missing_equals_sign_errors():
    result = runner.invoke(app, ["publish", "gbif", "config", "set", "contact_email"])
    assert result.exit_code != 0
    assert "CAMPO=VALOR" in result.output


def test_set_unknown_field_errors():
    result = runner.invoke(app, ["publish", "gbif", "config", "set", "bogus_field=1"])
    assert result.exit_code != 0
    assert "Parámetro desconocido" in result.output


# GBIFSettings has 8 fields, in model declaration order.
FIELD_COUNT = 8


def test_wizard_keeps_defaults_when_all_answers_blank():
    result = runner.invoke(app, ["publish", "gbif", "config", "wizard"], input="\n" * FIELD_COUNT)

    assert result.exit_code == 0
    assert "No se ha cambiado ningún valor" in result.output


CONTACT_EMAIL_INDEX = 5  # 0-based position of contact_email within GBIFSettings (6th field)


def test_wizard_saves_a_changed_value():
    answers = (
        "\n" * CONTACT_EMAIL_INDEX
        + "you@example.org\n"
        + "\n" * (FIELD_COUNT - CONTACT_EMAIL_INDEX - 1)
        + "y\n"
    )

    result = runner.invoke(app, ["publish", "gbif", "config", "wizard"], input=answers)

    assert result.exit_code == 0, result.output
    assert "contact_email" in result.output

    get_result = runner.invoke(app, ["publish", "gbif", "config", "get", "contact_email"])
    assert get_result.output.strip() == "you@example.org"


def test_wizard_cancel_does_not_persist():
    answers = (
        "\n" * CONTACT_EMAIL_INDEX
        + "you@example.org\n"
        + "\n" * (FIELD_COUNT - CONTACT_EMAIL_INDEX - 1)
        + "n\n"
    )

    result = runner.invoke(app, ["publish", "gbif", "config", "wizard"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Cancelado" in result.output

    get_result = runner.invoke(app, ["publish", "gbif", "config", "get", "contact_email"])
    assert get_result.output.strip() in ("None", "")
