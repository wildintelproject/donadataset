"""Integration tests for 'publish huggingface upload/download/release'.

These only exercise validation and --dry-run paths (config parsing, local
folder checks, token presence). Nothing here calls the real Hugging Face Hub
network API — dry-run returns before authenticate()/upload_folder()/
snapshot_download() in donadataset.services.huggingface.

There is no --config flag: every command derives its resolved config path
from --output-dir (<output-dir>/HuggingFaceHub.yaml, written by
'prepare'). token_env_var is fixed to HF_TOKEN in the bundled template (it's
not a settings.toml field or a flag), so these tests monkeypatch HF_TOKEN
directly instead of a per-test fake variable name.
"""
from pathlib import Path

from typer.testing import CliRunner

from donadataset.main import app
from donadataset.services import huggingface as hf_service

runner = CliRunner()


def _generate_and_prepare(
    tmp_path: Path, example_source_dataset: Path, repo_id: str = "myuser/donadataset-test",
) -> Path:
    """generate real -> publish huggingface prepare, returns the HFH output dir."""
    real_output = tmp_path / "real"
    real_result = runner.invoke(app, [
        "generate", "real",
        "--source", str(example_source_dataset),
        "--output", str(real_output),
    ])
    assert real_result.exit_code == 0, real_result.output

    hf_output_dir = tmp_path / "HFH"
    prepare_result = runner.invoke(app, [
        "publish", "huggingface", "prepare",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(hf_output_dir),
        "--dataset-slug", "test-donadataset",
        "--dataset-name", "Test DonaDataset",
        "--version", "0.0.1",
        "--repo-id", repo_id,
    ])
    assert prepare_result.exit_code == 0, prepare_result.output

    return hf_output_dir


# ── upload ────────────────────────────────────────────────────────────────────

def test_upload_dry_run_succeeds_after_prepare(tmp_path, example_source_dataset, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset)

    result = runner.invoke(app, ["publish", "huggingface", "upload", "--output-dir", str(hf_output_dir), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run enabled" in result.output


def test_upload_missing_token_env_var_fails(tmp_path, example_source_dataset, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset)

    result = runner.invoke(app, ["publish", "huggingface", "upload", "--output-dir", str(hf_output_dir), "--dry-run"])

    assert result.exit_code == 1
    assert "HF_TOKEN" in result.output


def test_upload_rejects_placeholder_repo_id(tmp_path, example_source_dataset, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset, repo_id="REPLACE_WITH_HF_USER/donadataset")

    result = runner.invoke(app, ["publish", "huggingface", "upload", "--output-dir", str(hf_output_dir), "--dry-run"])

    assert result.exit_code == 1
    assert "repo_id" in result.output


def test_upload_rejects_repo_id_without_slash(tmp_path, example_source_dataset, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset, repo_id="no-slash-here")

    result = runner.invoke(app, ["publish", "huggingface", "upload", "--output-dir", str(hf_output_dir), "--dry-run"])

    assert result.exit_code == 1


def test_upload_fails_when_output_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")

    result = runner.invoke(app, [
        "publish", "huggingface", "upload", "--output-dir", str(tmp_path / "does-not-exist"),
    ])

    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_upload_fails_when_local_verification_did_not_pass(tmp_path, example_source_dataset, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset)

    # Corrupt the report to simulate a failed local verification.
    report_path = hf_output_dir / "verification_report_local.json"
    report_path.write_text('{"status": "failed", "num_errors": 1, "errors": ["boom"]}')

    result = runner.invoke(app, ["publish", "huggingface", "upload", "--output-dir", str(hf_output_dir)])

    assert result.exit_code == 1
    assert "did not pass" in result.output


# ── download ──────────────────────────────────────────────────────────────────

def test_download_dry_run_with_valid_config(tmp_path, example_source_dataset, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset)

    result = runner.invoke(app, ["publish", "huggingface", "download", "--output-dir", str(hf_output_dir), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run enabled" in result.output


def test_download_missing_token_env_var_fails(tmp_path, example_source_dataset, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset)

    result = runner.invoke(app, ["publish", "huggingface", "download", "--output-dir", str(hf_output_dir), "--dry-run"])

    assert result.exit_code == 1
    assert "HF_TOKEN" in result.output


def test_download_missing_output_dir_errors(tmp_path):
    result = runner.invoke(app, [
        "publish", "huggingface", "download", "--output-dir", str(tmp_path / "does-not-exist"),
    ])

    assert result.exit_code != 0


# ── release ──────────────────────────────────────────────────────────────────
#
# Unlike upload/download, release authenticates against the real API even
# in --dry-run (it only skips the mutating call) — so these tests only cover
# validation that happens before any network call: incompatible flags,
# missing output dir, invalid repo_id, missing token.

def test_release_rejects_dry_run_and_verify_only_together(tmp_path, example_source_dataset, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset)

    result = runner.invoke(app, [
        "publish", "huggingface", "release", "--output-dir", str(hf_output_dir), "--dry-run", "--verify-only",
    ])

    assert result.exit_code == 1
    assert "no ambos" in result.output.lower()


def test_release_rejects_placeholder_repo_id(tmp_path, example_source_dataset, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    hf_output_dir = _generate_and_prepare(
        tmp_path, example_source_dataset, repo_id="REPLACE_WITH_HF_USER/donadataset",
    )

    result = runner.invoke(app, ["publish", "huggingface", "release", "--output-dir", str(hf_output_dir), "--verify-only"])

    assert result.exit_code == 1
    assert "repo_id" in result.output


def test_release_missing_token_env_var_fails(tmp_path, example_source_dataset, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset)

    result = runner.invoke(app, ["publish", "huggingface", "release", "--output-dir", str(hf_output_dir), "--verify-only"])

    assert result.exit_code == 1
    assert "HF_TOKEN" in result.output


def test_release_missing_output_dir_errors(tmp_path):
    result = runner.invoke(app, [
        "publish", "huggingface", "release", "--output-dir", str(tmp_path / "does-not-exist"),
    ])

    assert result.exit_code != 0


def test_release_report_is_written_inside_output_dir_not_cwd(tmp_path, example_source_dataset, monkeypatch):
    """Regression test: hfh_publication_report.json used to be written
    relative to the process cwd instead of --output-dir (get_output_dir()
    was computed but never joined onto get_public_visibility_report_path()),
    so it could land in whatever directory the command happened to be run
    from — e.g. the repo root — instead of next to the rest of the export."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset)

    monkeypatch.setattr(hf_service, "authenticate", lambda token: {"name": "tester"})
    monkeypatch.setattr(hf_service, "HfApi", lambda *a, **k: object())
    monkeypatch.setattr(
        hf_service, "get_dataset_visibility",
        lambda api, repo_id, token: {"repo_id": repo_id, "private": True, "public": False},
    )
    monkeypatch.setattr(
        hf_service, "check_public_url",
        lambda url, timeout_seconds: {"url": url, "status": "passed", "http_status": 200, "reason": None},
    )

    # A clean, empty cwd — so the assertion below can't be confounded by a
    # report that happens to already exist in whatever directory pytest was
    # invoked from (e.g. a leftover from a real, pre-fix run).
    clean_cwd = tmp_path / "clean_cwd"
    clean_cwd.mkdir()
    monkeypatch.chdir(clean_cwd)

    result = runner.invoke(app, [
        "publish", "huggingface", "release", "--output-dir", str(hf_output_dir), "--dry-run",
    ])

    assert result.exit_code == 0, result.output
    assert (hf_output_dir / "hfh_publication_report.json").is_file()
    assert not (clean_cwd / "hfh_publication_report.json").exists()


def test_download_report_path_stays_inside_output_dir(tmp_path, example_source_dataset, monkeypatch):
    """Same regression as the release report, for verification_report_downloaded.json."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    hf_output_dir = _generate_and_prepare(tmp_path, example_source_dataset)

    captured = {}

    def fake_download_and_verify_hfh(config, token, download_dir, report_path, delete_after_success):
        captured["report_path"] = report_path
        return {"status": "passed"}

    monkeypatch.setattr(hf_service, "download_and_verify_hfh", fake_download_and_verify_hfh)

    result = runner.invoke(app, ["publish", "huggingface", "download", "--output-dir", str(hf_output_dir)])

    assert result.exit_code == 0, result.output
    assert captured["report_path"] == hf_output_dir / "verification_report_downloaded.json"
