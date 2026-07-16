"""Integration tests for 'publish all'.

`_run_step` shells out to a fresh `python -m donadataset.main ...`
subprocess for each step, so every test here monkeypatches
`donadataset.commands.publish_all.subprocess.run` instead of letting it
spawn a real subprocess (which would need every integration's config/tokens
to actually succeed) — these tests only cover the orchestration: fixed
step order, --include/--exclude resolution, --dry-run printing without
executing, and stopping on the first failing step.
"""
from typing import List

import pytest
from typer.testing import CliRunner

from donadataset.commands import publish_all as publish_all_cmd
from donadataset.main import app

runner = CliRunner()


class _FakeCompletedProcess:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode


def _record_calls(monkeypatch, returncode: int = 0):
    calls: List[List[str]] = []

    def _fake_run(command, *args, **kwargs):
        calls.append(command)
        return _FakeCompletedProcess(returncode)

    monkeypatch.setattr(publish_all_cmd.subprocess, "run", _fake_run)
    return calls


def _step_args(calls: List[List[str]]) -> List[List[str]]:
    """Strips the fixed '[sys.executable, -m, donadataset.main]' prefix,
    leaving just the CLI args each step actually invoked."""
    return [call[3:] for call in calls]


def test_resolve_repos_defaults_to_all_in_fixed_order():
    assert publish_all_cmd._resolve_repos(None, None) == publish_all_cmd.ALL_REPOS


def test_resolve_repos_exclude_removes_repo():
    assert publish_all_cmd._resolve_repos(None, "b2share") == [
        "huggingface", "zenodo", "gbif",
    ]


def test_resolve_repos_include_always_wins_over_exclude():
    assert publish_all_cmd._resolve_repos("b2share", "b2share") == publish_all_cmd.ALL_REPOS


def test_resolve_repos_rejects_unknown_repo():
    with pytest.raises(Exception):
        publish_all_cmd._resolve_repos(None, "not-a-real-repo")


def test_publish_all_runs_every_integration_in_fixed_order(monkeypatch):
    calls = _record_calls(monkeypatch)

    result = runner.invoke(app, ["publish", "all"])

    assert result.exit_code == 0, result.output
    steps = _step_args(calls)
    assert steps[0] == ["publish", "huggingface", "pipeline"]
    assert steps[-1] == ["publish", "gbif", "pipeline"]
    assert ["publish", "zenodo", "prepare"] in steps
    assert ["publish", "zenodo", "upload"] in steps
    # sync-doi re-uploads to HuggingFace Hub by itself — no separate step needed for Zenodo.
    assert ["publish", "zenodo", "sync-doi"] in steps
    assert ["publish", "zenodo", "release"] in steps
    assert ["publish", "b2share", "pipeline"] in steps
    # B2SHARE's own PID gap is still closed with a separate, explicit re-upload.
    assert ["publish", "huggingface", "upload"] in steps


def test_publish_all_exclude_skips_that_integration(monkeypatch):
    calls = _record_calls(monkeypatch)

    result = runner.invoke(app, ["publish", "all", "--exclude", "b2share"])

    assert result.exit_code == 0, result.output
    steps = _step_args(calls)
    assert not any(step[:2] == ["publish", "b2share"] for step in steps)
    assert any(step[:2] == ["publish", "gbif"] for step in steps)


def test_publish_all_dry_run_prints_plan_without_running(monkeypatch):
    calls = _record_calls(monkeypatch)

    result = runner.invoke(app, ["publish", "all", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert calls == []
    assert "publish huggingface pipeline" in result.output
    assert "publish gbif pipeline" in result.output


def test_publish_all_stops_on_first_failing_step(monkeypatch):
    calls = _record_calls(monkeypatch, returncode=1)

    result = runner.invoke(app, ["publish", "all"])

    assert result.exit_code == 1
    # Only the very first step (HuggingFace Hub pipeline) should have run.
    assert len(calls) == 1
