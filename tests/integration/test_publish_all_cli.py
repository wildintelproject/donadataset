"""Integration tests for 'publish all'.

'all' orchestrates the other integrations by shelling out to
`python -m donadataset.main publish <repo> ...` for each step — actually
running that pipeline needs four real external services (HuggingFace Hub,
Zenodo, B2SHARE, GBIF) with real credentials, so that's out of scope here.
These tests instead cover the parts that are pure Python or safely
observable without a real subprocess: the --include/--exclude resolution
logic, and the --dry-run path (which prints the planned commands and
never calls subprocess.run at all).
"""
from typer.testing import CliRunner

from donadataset.commands import publish_all as publish_all_cmd
from donadataset.main import app

runner = CliRunner()


# ── _resolve_repos ────────────────────────────────────────────────────────────

def test_resolve_repos_defaults_to_all_in_canonical_order():
    assert publish_all_cmd._resolve_repos(None, None) == ["huggingface", "zenodo", "b2share", "gbif"]


def test_resolve_repos_exclude_removes_from_the_list():
    assert publish_all_cmd._resolve_repos(None, "gbif,b2share") == ["huggingface", "zenodo"]


def test_resolve_repos_include_alone_does_not_shrink_the_default_all():
    # Every repo is already included by default, so a whitelist alone changes nothing.
    assert publish_all_cmd._resolve_repos("gbif", None) == ["huggingface", "zenodo", "b2share", "gbif"]


def test_resolve_repos_include_overrides_exclude_for_the_same_repo():
    assert publish_all_cmd._resolve_repos("gbif", "gbif") == ["huggingface", "zenodo", "b2share", "gbif"]
    assert publish_all_cmd._resolve_repos("gbif", "gbif,zenodo") == ["huggingface", "b2share", "gbif"]


def test_resolve_repos_can_exclude_huggingface_itself():
    assert publish_all_cmd._resolve_repos(None, "huggingface") == ["zenodo", "b2share", "gbif"]


# ── CLI: --dry-run and validation ──────────────────────────────────────────────

def test_publish_all_dry_run_prints_full_plan_and_calls_no_subprocess(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called during --dry-run")

    monkeypatch.setattr(publish_all_cmd.subprocess, "run", _fail_if_called)

    result = runner.invoke(app, ["publish", "all", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "huggingface -> zenodo -> b2share -> gbif" in result.output
    for step in [
        "publish huggingface pipeline",
        "publish zenodo prepare",
        "publish zenodo upload",
        "publish huggingface upload",
        "publish zenodo check-readiness",
        "publish zenodo release",
        "publish b2share pipeline",
        "publish gbif pipeline",
    ]:
        assert step in result.output


def test_publish_all_dry_run_respects_exclude(monkeypatch):
    monkeypatch.setattr(publish_all_cmd.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    result = runner.invoke(app, ["publish", "all", "--dry-run", "--exclude", "zenodo,b2share,gbif"])

    assert result.exit_code == 0, result.output
    assert "Orden de publicación:[/bold] huggingface" in result.output.replace("\n", " ") or "huggingface" in result.output
    assert "zenodo" not in result.output.lower()
    assert "b2share" not in result.output.lower()
    assert "gbif" not in result.output.lower()


def test_publish_all_rejects_unknown_repo_name():
    result = runner.invoke(app, ["publish", "all", "--dry-run", "--exclude", "bogus"])

    assert result.exit_code != 0
    assert "desconocido" in result.output.lower()


def test_publish_all_with_empty_selection_exits_cleanly():
    result = runner.invoke(app, [
        "publish", "all", "--dry-run",
        "--exclude", "huggingface,zenodo,b2share,gbif",
    ])

    assert result.exit_code == 0, result.output
    assert "no hay ningún repositorio seleccionado" in result.output.lower()


# ── CLI: a failing step stops the pipeline ──────────────────────────────────────

def test_publish_all_stops_on_first_failing_step(monkeypatch):
    calls = []

    class _FakeCompletedProcess:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        # Fail on the very first step (huggingface pipeline).
        return _FakeCompletedProcess(returncode=1)

    monkeypatch.setattr(publish_all_cmd.subprocess, "run", fake_run)

    result = runner.invoke(app, ["publish", "all"])

    assert result.exit_code == 1
    assert len(calls) == 1  # never got to zenodo/b2share/gbif
