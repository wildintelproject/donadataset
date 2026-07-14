"""Integration tests for 'publish b2share prepare/check-readiness/release/sync-pid'.

These only exercise validation and --dry-run paths. Nothing here calls the
real B2SHARE API (no community_id is available for live testing yet) — every
scenario here is reachable purely through local file/token/repo_id
validation, mirroring tests/integration/test_zenodo_cli.py.

There is no --config flag: every command derives its resolved config path
from --repo-id/--output-dir/--community-id (rendering the single bundled
templates/B2SHARE.yaml.j2). token_env_var is fixed to B2SHARE_TOKEN in that
template. Files like b2share_linked_dataset_record.json and
b2share_public_release_readiness_report.json always live at fixed names
inside --output-dir, so tests place hand-written JSON there directly.
"""
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from donadataset.main import app
from donadataset.services import b2share as b2share_service

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
B2SHARE_JINJA_TEMPLATE_PATH = REPO_ROOT / "templates" / "B2SHARE.yaml.j2"


def _generate_prepare_and_fake_download(
    tmp_path: Path, example_source_dataset: Path, repo_id: str = "myuser/donadataset-test",
) -> tuple[Path, Path]:
    """generate real -> huggingface prepare -> a hand-written 'passed'
    verification_report_downloaded.json placed where 'b2share prepare' itself
    would write it. Returns (hf_output_dir, b2share_output_dir).
    """
    real_output = tmp_path / "real"
    real_result = runner.invoke(app, [
        "generate", "real", "--source", str(example_source_dataset), "--output", str(real_output),
    ])
    assert real_result.exit_code == 0, real_result.output

    hf_output_dir = tmp_path / "HFH"
    prepare_result = runner.invoke(app, [
        "publish", "huggingface", "prepare",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(hf_output_dir),
        "--repo-id", repo_id,
    ])
    assert prepare_result.exit_code == 0, prepare_result.output

    b2share_output_dir = tmp_path / "B2SHARE"
    downloaded_report_path = b2share_output_dir / "verification_report_downloaded.json"
    downloaded_report_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_report_path.write_text(json.dumps({
        "status": "passed", "repo_id": repo_id, "repo_type": "dataset",
        "global_files_verified": 5, "internal_tar_members_verified": 3, "num_errors": 0,
    }))

    return hf_output_dir, b2share_output_dir


# ── prepare ───────────────────────────────────────────────────────────────────

def test_prepare_requires_b2share_enabled(tmp_path: Path):
    """b2share.enabled is always true in the bundled template (no CLI flag
    for it) — exercise the service-layer guard directly instead."""
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(yaml.safe_dump({"b2share": {"enabled": False}}))

    with pytest.raises(RuntimeError, match="b2share.enabled"):
        b2share_service.run_b2share_linked_dataset_creation(config_path, dry_run=True)


def test_prepare_dry_run_succeeds_with_valid_evidence(tmp_path: Path, example_source_dataset: Path):
    _, b2share_output_dir = _generate_prepare_and_fake_download(tmp_path, example_source_dataset)

    result = runner.invoke(app, [
        "publish", "b2share", "prepare",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(b2share_output_dir),
        "--community-id", "11111111-1111-1111-1111-111111111111",
        "--dry-run",
    ])

    assert result.exit_code == 0, result.output
    assert "Dry run enabled" in result.output


def test_prepare_dry_run_does_not_depend_on_any_local_hfh_export(tmp_path: Path):
    """Evidence files are fetched from a live HuggingFace Hub download — so
    --dry-run must succeed even without ever having run 'huggingface
    prepare' at all."""
    result = runner.invoke(app, [
        "publish", "b2share", "prepare",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(tmp_path / "B2SHARE"),
        "--community-id", "11111111-1111-1111-1111-111111111111",
        "--dry-run",
    ])

    assert result.exit_code == 0, result.output
    assert "no b2share record will be created" in result.output.lower()


def test_prepare_sync_existing_draft_requires_linked_record(tmp_path: Path, example_source_dataset: Path):
    _, b2share_output_dir = _generate_prepare_and_fake_download(tmp_path, example_source_dataset)

    result = runner.invoke(app, [
        "publish", "b2share", "prepare",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(b2share_output_dir),
        "--community-id", "11111111-1111-1111-1111-111111111111",
        "--dry-run", "--sync-existing-draft",
    ])

    assert result.exit_code == 1
    assert "linked dataset record not found" in result.output.lower()


def test_prepare_missing_hf_token_fails_before_community_check(tmp_path: Path, monkeypatch):
    """The live HuggingFace download happens before B2SHARE is ever
    contacted, so a missing HF_TOKEN must surface first, even without a
    community_id configured."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = runner.invoke(app, [
        "publish", "b2share", "prepare",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(tmp_path / "B2SHARE"),
    ])

    assert result.exit_code == 1
    assert "HF_TOKEN" in result.output


# ── templates/B2SHARE.yaml.j2 (Jinja2 template rendered by 'b2share prepare') ─

def test_prepare_renders_jinja_template_using_only_repo_id_and_output_dir(tmp_path: Path):
    """Only --repo-id, --output-dir, and --community-id should be needed to
    get a working render — no local HuggingFace export required at all,
    since evidence files come from a live download."""
    result = runner.invoke(app, [
        "publish", "b2share", "prepare",
        "--output-dir", str(tmp_path / "B2SHARE_jinja-demo"),
        "--repo-id", "myuser/jinja-b2share-demo",
        "--community-id", "11111111-1111-1111-1111-111111111111",
        "--dry-run",
    ])

    assert result.exit_code == 0, result.output
    assert "Dry run enabled" in result.output


def test_prepare_jinja_template_keeps_placeholder_when_repo_id_missing(tmp_path: Path):
    """Without --repo-id (and no settings.toml default in the isolated test
    HOME), the rendered related fields must stay an obvious placeholder
    instead of silently producing a broken URL."""
    from donadataset.services.b2share import build_b2share_template_context
    from donadataset.services.huggingface import load_config_source

    ctx = build_b2share_template_context(
        hfh_output_dir=str(tmp_path / "HFH"), b2share_output_dir=str(tmp_path / "B2SHARE"), repo_id=None,
    )
    config = load_config_source(B2SHARE_JINJA_TEMPLATE_PATH, **ctx)

    assert "REPLACE_WITH_HF_USER" in config["huggingface"]["repo_id"]
    assert "REPLACE_WITH_HF_USER" in config["b2share"]["alternate_identifier"]


# ── check-readiness ────────────────────────────────────────────────────────────

def test_check_readiness_rejects_invalid_repo_id(tmp_path: Path):
    result = runner.invoke(app, [
        "publish", "b2share", "check-readiness",
        "--repo-id", "no-slash-here", "--output-dir", str(tmp_path / "B2SHARE"),
    ])

    assert result.exit_code == 1
    assert "repo_id" in result.output


def test_check_readiness_fails_when_linked_record_missing(tmp_path: Path):
    result = runner.invoke(app, [
        "publish", "b2share", "check-readiness",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(tmp_path / "B2SHARE"),
    ])

    assert result.exit_code == 1
    assert "linked dataset record not found" in result.output.lower()


def test_check_readiness_fails_on_malformed_linked_record(tmp_path: Path):
    b2share_output_dir = tmp_path / "B2SHARE"
    b2share_output_dir.mkdir(parents=True)
    # b2share_environment present, but record_id missing -> should fail on that check
    (b2share_output_dir / "b2share_linked_dataset_record.json").write_text(
        json.dumps({"b2share_environment": "sandbox"}),
    )

    result = runner.invoke(app, [
        "publish", "b2share", "check-readiness",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(b2share_output_dir),
    ])

    assert result.exit_code == 1
    assert "record_id" in result.output


# ── release ───────────────────────────────────────────────────────────────────

VALID_LINKED_RECORD = {
    "record_id": "abc123",
    "pid": None,
    "pid_url": None,
    "record_url": "https://trng-b2share.eudat.eu/records/abc123",
    "b2share_environment": "sandbox",
}


def _write_release_files(b2share_output_dir: Path, linked_record: dict, readiness_report: dict | None) -> None:
    b2share_output_dir.mkdir(parents=True, exist_ok=True)
    (b2share_output_dir / "b2share_linked_dataset_record.json").write_text(json.dumps(linked_record))
    if readiness_report is not None:
        (b2share_output_dir / "b2share_public_release_readiness_report.json").write_text(json.dumps(readiness_report))


def test_release_fails_when_linked_record_missing(tmp_path: Path):
    result = runner.invoke(app, [
        "publish", "b2share", "release",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(tmp_path / "B2SHARE"),
    ])

    assert result.exit_code == 1
    assert "linked dataset record not found" in result.output.lower()


def test_release_requires_readiness_report_by_default(tmp_path: Path):
    b2share_output_dir = tmp_path / "B2SHARE"
    _write_release_files(b2share_output_dir, VALID_LINKED_RECORD, readiness_report=None)

    result = runner.invoke(app, [
        "publish", "b2share", "release",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(b2share_output_dir),
    ])

    assert result.exit_code == 1
    assert "readiness report not found" in result.output.lower()


def test_release_fails_when_readiness_report_not_passed(tmp_path: Path):
    b2share_output_dir = tmp_path / "B2SHARE"
    _write_release_files(
        b2share_output_dir, VALID_LINKED_RECORD, readiness_report={"status": "failed"},
    )

    result = runner.invoke(app, [
        "publish", "b2share", "release",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(b2share_output_dir),
    ])

    assert result.exit_code == 1
    assert "does not indicate a passed" in result.output.lower()


def test_release_missing_token_env_var_fails_after_readiness_passes(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("B2SHARE_TOKEN", raising=False)
    b2share_output_dir = tmp_path / "B2SHARE"
    _write_release_files(
        b2share_output_dir, VALID_LINKED_RECORD, readiness_report={"status": "passed"},
    )

    result = runner.invoke(app, [
        "publish", "b2share", "release",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(b2share_output_dir),
    ])

    assert result.exit_code == 1
    assert "B2SHARE_TOKEN" in result.output


def test_release_skip_readiness_check_still_requires_token(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("B2SHARE_TOKEN", raising=False)
    b2share_output_dir = tmp_path / "B2SHARE"
    _write_release_files(b2share_output_dir, VALID_LINKED_RECORD, readiness_report=None)

    result = runner.invoke(app, [
        "publish", "b2share", "release",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(b2share_output_dir),
        "--skip-readiness-check",
    ])

    assert result.exit_code == 1
    assert "skipping public release readiness check" in result.output.lower()
    assert "B2SHARE_TOKEN" in result.output


def test_release_dry_run_still_requires_token(tmp_path: Path, monkeypatch):
    """--dry-run still reaches (and fails cleanly on) the token check, same
    as Zenodo's release — it validates as much as possible without
    publishing."""
    monkeypatch.delenv("B2SHARE_TOKEN", raising=False)
    b2share_output_dir = tmp_path / "B2SHARE"
    _write_release_files(
        b2share_output_dir, VALID_LINKED_RECORD, readiness_report={"status": "passed"},
    )

    result = runner.invoke(app, [
        "publish", "b2share", "release",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(b2share_output_dir),
        "--dry-run",
    ])

    assert result.exit_code == 1
    assert "B2SHARE_TOKEN" in result.output


# ── sync-pid ──────────────────────────────────────────────────────────────────

def test_sync_pid_fails_when_linked_record_missing(tmp_path: Path):
    result = runner.invoke(app, [
        "publish", "b2share", "sync-pid",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(tmp_path / "B2SHARE"),
        "--hfh-output-dir", str(tmp_path / "HFH"),
    ])

    assert result.exit_code == 1
    assert "linked dataset record not found" in result.output.lower()


def test_sync_pid_warns_when_no_pid_assigned_yet(tmp_path: Path, example_source_dataset: Path):
    """Before a moderator approves the record (or right after 'release' if
    no PID/DOI came back yet), sync-pid must not fail — it just reports
    nothing to do."""
    hf_output_dir, b2share_output_dir = _generate_prepare_and_fake_download(tmp_path, example_source_dataset)
    _write_release_files(b2share_output_dir, VALID_LINKED_RECORD, readiness_report=None)

    result = runner.invoke(app, [
        "publish", "b2share", "sync-pid",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(b2share_output_dir),
        "--hfh-output-dir", str(hf_output_dir),
    ])

    assert result.exit_code == 0, result.output
    assert "no pid/doi found yet" in result.output.lower()


def test_sync_pid_writes_pid_into_citation_cff(tmp_path: Path, example_source_dataset: Path):
    hf_output_dir, b2share_output_dir = _generate_prepare_and_fake_download(tmp_path, example_source_dataset)
    published_record = {
        **VALID_LINKED_RECORD,
        "pid": "10.5281/b2share.abc123",
        "pid_url": "https://doi.org/10.5281/b2share.abc123",
    }
    _write_release_files(b2share_output_dir, published_record, readiness_report=None)

    result = runner.invoke(app, [
        "publish", "b2share", "sync-pid",
        "--repo-id", "myuser/donadataset-test", "--output-dir", str(b2share_output_dir),
        "--hfh-output-dir", str(hf_output_dir),
    ])

    assert result.exit_code == 0, result.output
    citation = yaml.safe_load((hf_output_dir / "CITATION.cff").read_text())
    assert citation["doi"] == "10.5281/b2share.abc123"
