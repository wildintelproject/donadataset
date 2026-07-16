"""Integration tests for 'publish zenodo wizard'.

Every network-touching service call (run_zenodo_prepare, run_zenodo_upload,
run_zenodo_sync_doi, run_check_release_readiness, run_release, and the
HuggingFace re-upload) is monkeypatched, so nothing here reaches the real
Zenodo or HuggingFace Hub APIs. These tests cover the wizard's own
orchestration logic: the ZENODO_TOKEN precondition, prompting for and
persisting a missing repo_id, detecting/offering to sync an existing linked
draft, the "publish now?" confirmation gate before the irreversible release
step, and re-using donadataset.commands.huggingface._run_wizard_step's
retry/skip/abort loop (already covered in isolation by
test_huggingface_wizard_cli.py).
"""
import json
from pathlib import Path

import pytest
from dynaconf import loaders
from typer.testing import CliRunner

from donadataset.commands.config_commands import _apply_update
from donadataset.config import DEFAULT_CONFIG_FILE, load_settings
from donadataset.main import app
from donadataset.services import huggingface as hf_service
from donadataset.services import zenodo as zenodo_service

runner = CliRunner()


@pytest.fixture(autouse=True)
def _restore_config_file():
    """Same rationale as test_huggingface_wizard_cli.py's own fixture:
    settings.toml is one shared file for the whole test session, so another
    test file's 'config set' calls can leave HUGGINGFACE.repo_id populated,
    desyncing this file's scripted --input answers. Force a known-clean
    slate before each test, and restore whatever was there afterward."""
    load_settings()
    original = DEFAULT_CONFIG_FILE.read_text(encoding="utf-8")

    cleared = _apply_update(load_settings(), "HUGGINGFACE", "repo_id", None)
    loaders.toml_loader.write(str(DEFAULT_CONFIG_FILE), cleared.model_dump(mode="json"), merge=False)

    yield

    DEFAULT_CONFIG_FILE.write_text(original, encoding="utf-8")


def _patch_happy_path(monkeypatch):
    """Stubs every service call the wizard's phases 1-4 make, all succeeding
    trivially. The 'prepare' stub also writes/updates the linked-record JSON
    the wizard re-reads after phase 1, mirroring what the real 'prepare'
    does."""
    calls = {
        "prepare_new": 0, "prepare_sync": 0, "upload": 0, "sync_doi": 0,
        "hf_upload": 0, "readiness": 0, "release": 0,
    }

    def _fake_prepare(config_path, dry_run=False, template_context=None, verify_data=False, sync_existing_draft=False):
        if sync_existing_draft:
            calls["prepare_sync"] += 1
            return
        calls["prepare_new"] += 1
        config = zenodo_service.load_config_source(config_path, **(template_context or {}))
        record_path = zenodo_service.get_linked_record_path(config)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps({"deposition_id": 42, "reserved_doi": "10.5281/zenodo.42"}), encoding="utf-8",
        )

    def _fake_upload(config_path, dry_run=False, template_context=None):
        calls["upload"] += 1

    def _fake_sync_doi(config_path, dry_run=False, template_context=None):
        calls["sync_doi"] += 1

    def _fake_hf_upload(config_path, dry_run=False, allow_patterns=None):
        calls["hf_upload"] += 1

    def _fake_readiness(config_path, template_context=None):
        calls["readiness"] += 1

    def _fake_release(config_path, dry_run=False, skip_readiness_check=False, no_config_update=False, template_context=None):
        calls["release"] += 1
        config = zenodo_service.load_config_source(config_path, **(template_context or {}))
        report_path = zenodo_service.get_publication_report_path(config)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"status": "passed", "record_url": "https://zenodo.org/records/42"}), encoding="utf-8",
        )

    monkeypatch.setattr(zenodo_service, "run_zenodo_prepare", _fake_prepare)
    monkeypatch.setattr(zenodo_service, "run_zenodo_upload", _fake_upload)
    monkeypatch.setattr(zenodo_service, "run_zenodo_sync_doi", _fake_sync_doi)
    monkeypatch.setattr(zenodo_service, "run_check_release_readiness", _fake_readiness)
    monkeypatch.setattr(zenodo_service, "run_release", _fake_release)
    monkeypatch.setattr(hf_service, "run_upload", _fake_hf_upload)
    return calls


def test_wizard_fails_without_zenodo_token(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    monkeypatch.delenv("ZENODO_SANDBOX_TOKEN", raising=False)

    result = runner.invoke(app, [
        "publish", "zenodo", "wizard",
        "--output-dir", str(tmp_path / "Zenodo"),
    ], input="someuser/somedataset\ny\n")

    assert result.exit_code == 1
    assert "zenodo" in result.output.lower()
    assert "token" in result.output.lower()


def test_wizard_prompts_for_missing_repo_id_and_persists_it(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ZENODO_TOKEN", "zenodo_dummy")
    calls = _patch_happy_path(monkeypatch)

    result = runner.invoke(app, [
        "publish", "zenodo", "wizard",
        "--hfh-output-dir", str(tmp_path / "HFH"),
        "--output-dir", str(tmp_path / "Zenodo"),
    ], input="someuser/somedataset\ny\nn\n")  # repo_id, save it, decline publishing

    assert result.exit_code == 0, result.output
    assert "someuser/somedataset" in result.output
    assert load_settings().HUGGINGFACE.repo_id == "someuser/somedataset"
    assert calls["prepare_new"] == 1
    assert calls["release"] == 0


def test_wizard_declining_to_publish_stops_before_release(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ZENODO_TOKEN", "zenodo_dummy")
    calls = _patch_happy_path(monkeypatch)

    result = runner.invoke(app, [
        "publish", "zenodo", "wizard",
        "--hfh-output-dir", str(tmp_path / "HFH"),
        "--output-dir", str(tmp_path / "Zenodo"),
    ], input="someuser/somedataset\ny\nn\n")  # repo_id, save it, decline publishing

    assert result.exit_code == 0, result.output
    assert calls["prepare_new"] == 1
    assert calls["upload"] == 1
    assert calls["sync_doi"] == 1
    assert calls["hf_upload"] == 1
    assert calls["readiness"] == 1
    assert calls["release"] == 0


def test_wizard_full_happy_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ZENODO_TOKEN", "zenodo_dummy")
    calls = _patch_happy_path(monkeypatch)

    result = runner.invoke(app, [
        "publish", "zenodo", "wizard",
        "--hfh-output-dir", str(tmp_path / "HFH"),
        "--output-dir", str(tmp_path / "Zenodo"),
    ], input="someuser/somedataset\ny\ny\n")  # repo_id, save it, DO publish

    assert result.exit_code == 0, result.output
    assert calls["prepare_new"] == 1
    assert calls["upload"] == 1
    assert calls["sync_doi"] == 1
    assert calls["hf_upload"] == 1
    assert calls["readiness"] == 1
    assert calls["release"] == 1
    assert "10.5281/zenodo.42" in result.output
    assert "https://zenodo.org/records/42" in result.output
    assert "publicación en zenodo completada" in result.output.lower()


def test_wizard_offers_to_sync_existing_linked_draft(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ZENODO_TOKEN", "zenodo_dummy")
    calls = _patch_happy_path(monkeypatch)

    output_dir = tmp_path / "Zenodo"
    output_dir.mkdir(parents=True)
    (output_dir / "zenodo_linked_dataset_record.json").write_text(
        json.dumps({"deposition_id": 7, "reserved_doi": "10.5281/zenodo.7"}), encoding="utf-8",
    )

    result = runner.invoke(app, [
        "publish", "zenodo", "wizard",
        "--hfh-output-dir", str(tmp_path / "HFH"),
        "--output-dir", str(output_dir),
    ], input="someuser/somedataset\ny\ny\nn\n")  # repo_id, save it, sync existing draft, decline publishing

    assert result.exit_code == 0, result.output
    assert calls["prepare_sync"] == 1
    assert calls["prepare_new"] == 0
    assert "deposition_id=7" in result.output
