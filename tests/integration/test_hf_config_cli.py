"""Integration tests for 'donadataset publish huggingface config show/get/set/wizard'.

Unlike the old free-form HuggingFaceHub.yaml, this now operates directly on
settings.toml's HUGGINGFACE section (same engine as
tests/integration/test_config_cli.py's generic 'donadataset config'), just
without the 'HUGGINGFACE.' prefix. Since it's the single shared app-wide
settings file, every test must restore it afterward.
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


def test_show_prints_huggingface_section():
    result = runner.invoke(app, ["publish", "huggingface", "config", "show"])
    assert result.exit_code == 0
    assert "HUGGINGFACE" in result.output
    assert "dataset_slug" in result.output


def test_get_returns_scalar_value():
    result = runner.invoke(app, ["publish", "huggingface", "config", "get", "license_id"])
    assert result.exit_code == 0
    assert result.output.strip() == "CC-BY-4.0"


def test_get_unknown_field_errors():
    result = runner.invoke(app, ["publish", "huggingface", "config", "get", "bogus_field"])
    assert result.exit_code != 0
    assert "Parámetro desconocido" in result.output


def test_set_scalar_value_persists():
    set_result = runner.invoke(app, ["publish", "huggingface", "config", "set", "repo_id=myuser/donadataset"])
    assert set_result.exit_code == 0, set_result.output

    get_result = runner.invoke(app, ["publish", "huggingface", "config", "get", "repo_id"])
    assert get_result.output.strip() == "myuser/donadataset"

    # unrelated existing fields are untouched
    still_there = runner.invoke(app, ["publish", "huggingface", "config", "get", "license_id"])
    assert still_there.output.strip() == "CC-BY-4.0"


def test_set_missing_equals_sign_errors():
    result = runner.invoke(app, ["publish", "huggingface", "config", "set", "repo_id"])
    assert result.exit_code != 0
    assert "CAMPO=VALOR" in result.output


def test_set_unknown_field_errors():
    result = runner.invoke(app, ["publish", "huggingface", "config", "set", "bogus_field=1"])
    assert result.exit_code != 0
    assert "Parámetro desconocido" in result.output


def test_set_secret_field_without_value_prompts_hidden():
    """'config set token' (no '=value') must not require an inline value —
    it prompts for it with hidden input instead, so the secret never has to
    be typed in the clear as part of the command itself."""
    result = runner.invoke(app, ["publish", "huggingface", "config", "set", "token"], input="s3cr3t-hf-token\n")

    assert result.exit_code == 0, result.output
    assert "s3cr3t-hf-token" not in result.output
    assert "••••••••" in result.output

    get_result = runner.invoke(app, ["publish", "huggingface", "config", "get", "token"])
    assert get_result.output.strip() == "s3cr3t-hf-token"


def test_show_masks_token_but_get_reveals_it():
    set_result = runner.invoke(app, ["publish", "huggingface", "config", "set", "token"], input="s3cr3t-hf-token\n")
    assert set_result.exit_code == 0, set_result.output

    show_result = runner.invoke(app, ["publish", "huggingface", "config", "show"])
    assert show_result.exit_code == 0
    assert "s3cr3t-hf-token" not in show_result.output
    assert "••••••••" in show_result.output

    get_result = runner.invoke(app, ["publish", "huggingface", "config", "get", "token"])
    assert get_result.output.strip() == "s3cr3t-hf-token"


# HuggingFaceSettings has 13 fields, in model declaration order.
FIELD_COUNT = 13
REPO_ID_INDEX = 11  # 0-based position of repo_id (12th field; token is the 13th/last)


def test_wizard_keeps_defaults_when_all_answers_blank():
    result = runner.invoke(app, ["publish", "huggingface", "config", "wizard"], input="\n" * FIELD_COUNT)

    assert result.exit_code == 0
    assert "No se ha cambiado ningún valor" in result.output


def test_wizard_saves_a_changed_value():
    answers = (
        "\n" * REPO_ID_INDEX
        + "someuser/somedataset\n"
        + "\n" * (FIELD_COUNT - REPO_ID_INDEX - 1)  # token, left blank
        + "y\n"
    )

    result = runner.invoke(app, ["publish", "huggingface", "config", "wizard"], input=answers)

    assert result.exit_code == 0, result.output
    assert "repo_id" in result.output

    get_result = runner.invoke(app, ["publish", "huggingface", "config", "get", "repo_id"])
    assert get_result.output.strip() == "someuser/somedataset"


def test_wizard_cancel_does_not_persist():
    answers = (
        "\n" * REPO_ID_INDEX
        + "someuser/somedataset\n"
        + "\n" * (FIELD_COUNT - REPO_ID_INDEX - 1)
        + "n\n"
    )

    result = runner.invoke(app, ["publish", "huggingface", "config", "wizard"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Cancelado" in result.output

    get_result = runner.invoke(app, ["publish", "huggingface", "config", "get", "repo_id"])
    assert get_result.output.strip() in ("None", "")


def test_wizard_prompts_hidden_for_token_and_keeps_it_when_blank():
    """token is a secret field: blank input must keep the current value
    (unset here) instead of it appearing as a plain-text default, and the
    wizard must never print the raw value anywhere in its output."""
    answers = "\n" * REPO_ID_INDEX + "someuser/somedataset\n" + "s3cr3t-hf-token\n" + "y\n"

    result = runner.invoke(app, ["publish", "huggingface", "config", "wizard"], input=answers)

    assert result.exit_code == 0, result.output
    assert "s3cr3t-hf-token" not in result.output
    assert "••••••••" in result.output

    get_result = runner.invoke(app, ["publish", "huggingface", "config", "get", "token"])
    assert get_result.output.strip() == "s3cr3t-hf-token"
