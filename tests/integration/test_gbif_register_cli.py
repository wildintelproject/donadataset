"""Integration tests for 'publish gbif register'.

'register' talks to the real GBIF Registry API (create/update a dataset,
replace its DWC_ARCHIVE endpoint) — nothing here calls that live API. Every
test either exercises a validation failure (bad URL, missing
publishing_organization_key/installation_key, missing GBIF_USERNAME/
GBIF_PASSWORD) or the --dry-run path, which logs what it would do and
returns before any network call in donadataset.services.gbif.run_register.
"""
from pathlib import Path

from typer.testing import CliRunner

from donadataset.main import app

runner = CliRunner()

VALID_ARGS = [
    "publish", "gbif", "register",
    "--archive-url", "https://example.org/donadataset-camtrap-dp.zip",
    "--publishing-organization-key", "00000000-0000-0000-0000-000000000001",
    "--installation-key", "00000000-0000-0000-0000-000000000002",
]


def test_register_fails_on_invalid_archive_url(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GBIF_USERNAME", "user")
    monkeypatch.setenv("GBIF_PASSWORD", "pass")

    result = runner.invoke(app, [
        "publish", "gbif", "register",
        "--archive-url", "not-a-url",
        "--publishing-organization-key", "00000000-0000-0000-0000-000000000001",
        "--installation-key", "00000000-0000-0000-0000-000000000002",
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 1
    assert "http(s) url" in result.output.lower()


def test_register_fails_without_publishing_organization_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GBIF_USERNAME", "user")
    monkeypatch.setenv("GBIF_PASSWORD", "pass")

    result = runner.invoke(app, [
        "publish", "gbif", "register",
        "--archive-url", "https://example.org/donadataset-camtrap-dp.zip",
        "--installation-key", "00000000-0000-0000-0000-000000000002",
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 1
    assert "publishing_organization_key" in result.output.lower()


def test_register_fails_without_installation_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GBIF_USERNAME", "user")
    monkeypatch.setenv("GBIF_PASSWORD", "pass")

    result = runner.invoke(app, [
        "publish", "gbif", "register",
        "--archive-url", "https://example.org/donadataset-camtrap-dp.zip",
        "--publishing-organization-key", "00000000-0000-0000-0000-000000000001",
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 1
    assert "installation_key" in result.output.lower()


def test_register_fails_without_credentials(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GBIF_USERNAME", raising=False)
    monkeypatch.delenv("GBIF_PASSWORD", raising=False)

    result = runner.invoke(app, VALID_ARGS + ["--output-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "gbif_username" in result.output.lower()
    assert "gbif_password" in result.output.lower()


def test_register_fails_on_invalid_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GBIF_USERNAME", "user")
    monkeypatch.setenv("GBIF_PASSWORD", "pass")

    result = runner.invoke(app, VALID_ARGS + ["--output-dir", str(tmp_path), "--environment", "bogus"])

    assert result.exit_code == 1
    assert "sandbox" in result.output.lower()


def test_register_dry_run_succeeds_and_makes_no_network_call(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GBIF_USERNAME", "user")
    monkeypatch.setenv("GBIF_PASSWORD", "pass")

    result = runner.invoke(app, VALID_ARGS + ["--output-dir", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output.lower()
    # No linked-record file is written on a dry run — nothing was actually registered.
    assert not (tmp_path / "gbif_linked_dataset_record.json").exists()
