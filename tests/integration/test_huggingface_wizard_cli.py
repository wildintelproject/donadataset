"""Integration tests for 'publish huggingface wizard'.

Every network-touching service call (run_export, run_upload, run_release,
get_repo_doi, run_sync_hfh_doi) is monkeypatched, so nothing here reaches
the real HuggingFace Hub API. These tests cover the wizard's own
orchestration logic: the HF_TOKEN precondition, prompting for and
persisting a missing repo_id, the "make it public?" confirmation gate, and
the retry/skip/abort loop (`_run_wizard_step`) in isolation.
"""
from pathlib import Path

import pytest
from dynaconf import loaders
from typer.testing import CliRunner

from donadataset.commands import huggingface as hf_cmd
from donadataset.commands.config_commands import _apply_update
from donadataset.config import DEFAULT_CONFIG_FILE, load_settings
from donadataset.main import app
from donadataset.services import huggingface as hf_service

runner = CliRunner()


@pytest.fixture(autouse=True)
def _restore_config_file():
    """settings.toml is one shared file for the whole test session (see
    tests/conftest.py) — other test files' own 'config set' calls can leave
    HUGGINGFACE.repo_id populated by the time this file's tests run, which
    would make the wizard skip its 'ask for repo_id' prompt and silently
    desync every scripted --input answer that follows. Force a known-clean
    slate (repo_id unset) before each test, and restore whatever was there
    afterward, same pattern as tests/integration/test_hf_config_cli.py."""
    load_settings()  # make sure it exists before snapshotting
    original = DEFAULT_CONFIG_FILE.read_text(encoding="utf-8")

    cleared = _apply_update(load_settings(), "HUGGINGFACE", "repo_id", None)
    loaders.toml_loader.write(str(DEFAULT_CONFIG_FILE), cleared.model_dump(mode="json"), merge=False)

    yield

    DEFAULT_CONFIG_FILE.write_text(original, encoding="utf-8")


def _fake_run_export(config_path, *, output_dir, repo_id=None, **kwargs):
    """Stands in for the real 'prepare': just writes the minimal resolved
    YAML downstream wizard steps read (get_private/get_repo_id/get_token),
    without touching the real dataset or filesystem shards."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / hf_service.INTERNAL_CONFIG_FILENAME).write_text(
        f"huggingface:\n  repo_id: {repo_id}\n  private: true\n", encoding="utf-8",
    )


def test_wizard_fails_without_hf_token(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = runner.invoke(app, [
        "publish", "huggingface", "wizard",
        "--output-dir", str(tmp_path / "HFH"),
    ])

    assert result.exit_code == 1
    assert "hf_token" in result.output.lower()


def test_wizard_prompts_for_missing_repo_id_and_persists_it(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr(hf_service, "run_export", _fake_run_export)
    monkeypatch.setattr(hf_service, "run_upload", lambda config, dry_run=False: None)

    result = runner.invoke(app, [
        "publish", "huggingface", "wizard",
        "--output-dir", str(tmp_path / "HFH"),
    ], input="someuser/somedataset\ny\nn\n")  # repo_id, save it, decline going public

    assert result.exit_code == 0, result.output
    assert "someuser/somedataset" in result.output
    assert load_settings().HUGGINGFACE.repo_id == "someuser/somedataset"


def test_wizard_declining_to_go_public_stops_before_release(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr(hf_service, "run_export", _fake_run_export)
    monkeypatch.setattr(hf_service, "run_upload", lambda config, dry_run=False: None)

    release_calls = []
    monkeypatch.setattr(hf_service, "run_release", lambda *a, **k: release_calls.append(1))
    doi_calls = []
    monkeypatch.setattr(hf_service, "get_repo_doi", lambda *a, **k: doi_calls.append(1))

    result = runner.invoke(app, [
        "publish", "huggingface", "wizard",
        "--output-dir", str(tmp_path / "HFH"),
    ], input="someuser/somedataset\ny\nn\n")  # repo_id, save it, decline going public

    assert result.exit_code == 0, result.output
    assert not release_calls
    assert not doi_calls


def test_wizard_full_happy_path_with_doi_already_present(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr(hf_service, "run_export", _fake_run_export)

    upload_calls = []
    monkeypatch.setattr(hf_service, "run_upload", lambda config, dry_run=False: upload_calls.append(1))
    release_calls = []
    monkeypatch.setattr(hf_service, "run_release", lambda *a, **k: release_calls.append(1))
    monkeypatch.setattr(hf_service, "get_repo_doi", lambda *a, **k: "10.1234/fake-doi")
    sync_calls = []
    monkeypatch.setattr(hf_service, "run_sync_hfh_doi", lambda config, dry_run=False: sync_calls.append(1))

    result = runner.invoke(app, [
        "publish", "huggingface", "wizard",
        "--output-dir", str(tmp_path / "HFH"),
    ], input="someuser/somedataset\ny\ny\n")  # repo_id, save it, DO go public

    assert result.exit_code == 0, result.output
    assert "10.1234/fake-doi" in result.output
    assert "ya existe un doi generado" in result.output.lower()
    assert len(release_calls) == 1
    assert len(sync_calls) == 1
    # upload runs twice: once before release, once more after sync-doi.
    assert len(upload_calls) == 2
    assert "publicación en huggingface hub completada" in result.output.lower()


# ── _run_wizard_step retry/skip/abort loop, in isolation ────────────────────────

def test_run_wizard_step_retries_until_success(monkeypatch):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient failure")
        return "ok"

    # _run_wizard_step uses typer.prompt/console directly — exercise it via typer's own runner by
    # wrapping it in a throwaway command instead of calling it as a bare function (so input= works).
    import typer as _typer

    probe_app = _typer.Typer()

    @probe_app.command()
    def probe():
        value = hf_cmd._run_wizard_step("Probe", flaky)
        hf_cmd.console.print(f"RESULT={value}")

    probe_result = runner.invoke(probe_app, [], input="r\nr\n")
    assert probe_result.exit_code == 0, probe_result.output
    assert "RESULT=ok" in probe_result.output
    assert attempts["n"] == 3


def test_run_wizard_step_skip_returns_none(monkeypatch):
    import typer as _typer

    probe_app = _typer.Typer()

    @probe_app.command()
    def probe():
        value = hf_cmd._run_wizard_step("Probe", lambda: (_ for _ in ()).throw(RuntimeError("boom")), allow_skip=True)
        hf_cmd.console.print(f"RESULT={value}")

    probe_result = runner.invoke(probe_app, [], input="s\n")
    assert probe_result.exit_code == 0, probe_result.output
    assert "RESULT=None" in probe_result.output


def test_run_wizard_step_abort_exits_nonzero():
    import typer as _typer

    probe_app = _typer.Typer()

    @probe_app.command()
    def probe():
        hf_cmd._run_wizard_step("Probe", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    probe_result = runner.invoke(probe_app, [], input="a\n")
    assert probe_result.exit_code == 1
