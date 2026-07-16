"""Integration tests for 'publish gbif upload'.

'upload' copies the .zip 'gbif prepare' already built into the local
HuggingFace Hub export (--hfh-output-dir), regenerates that export's
checksums-sha256.txt, and pushes both files with a scoped 'huggingface
upload' (allow_patterns) — the same "sync local metadata, then re-upload
just the changed files" pattern 'zenodo sync-doi' uses. The local staging
step is exercised for real (real filesystem, real checksum computation);
the actual push to HuggingFace Hub is always monkeypatched, so nothing here
touches the real API.
"""
import json
from pathlib import Path

from typer.testing import CliRunner

from donadataset.main import app
from donadataset.services import huggingface as hf_service

runner = CliRunner()

REPO_ID = "someuser/somedataset"


def _generate_real_dataset_and_hfh_export(tmp_path: Path, example_source_dataset: Path) -> tuple[Path, Path]:
    """generate real -> huggingface prepare (a real local export, so
    'gbif upload's local staging runs for real). Returns (real_output,
    hfh_output_dir)."""
    real_output = tmp_path / "real"
    result = runner.invoke(app, [
        "generate", "real", "--source", str(example_source_dataset), "--output", str(real_output),
    ])
    assert result.exit_code == 0, result.output

    hfh_output_dir = tmp_path / "HFH"
    hfh_result = runner.invoke(app, [
        "publish", "huggingface", "prepare",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(hfh_output_dir),
        "--repo-id", REPO_ID,
    ])
    assert hfh_result.exit_code == 0, hfh_result.output
    (real_output / "manifest.csv").write_bytes((hfh_output_dir / "manifest.csv").read_bytes())

    return real_output, hfh_output_dir


def _run_gbif_prepare(real_output: Path, output_dir: Path) -> None:
    result = runner.invoke(app, [
        "publish", "gbif", "prepare",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(output_dir),
        "--hf-repo-id", REPO_ID,
    ])
    assert result.exit_code == 0, result.output


def test_upload_fails_without_repo_id(tmp_path: Path):
    result = runner.invoke(app, [
        "publish", "gbif", "upload",
        "--output-dir", str(tmp_path / "gbif_out"),
        "--hfh-output-dir", str(tmp_path / "HFH"),
        "--hf-repo-id", "",
    ])

    assert result.exit_code == 1
    assert "repo_id" in result.output.lower()


def test_upload_fails_when_archive_missing(tmp_path: Path):
    output_dir = tmp_path / "gbif_out"
    output_dir.mkdir()

    result = runner.invoke(app, [
        "publish", "gbif", "upload",
        "--output-dir", str(output_dir),
        "--hfh-output-dir", str(tmp_path / "HFH"),
        "--hf-repo-id", REPO_ID,
    ])

    assert result.exit_code == 1
    assert "no camtrap dp package found" in result.output.lower()


def test_upload_fails_when_multiple_archives_found(tmp_path: Path):
    output_dir = tmp_path / "gbif_out"
    output_dir.mkdir()
    (output_dir / "a-camtrap-dp.zip").write_bytes(b"fake")
    (output_dir / "b-camtrap-dp.zip").write_bytes(b"fake")

    result = runner.invoke(app, [
        "publish", "gbif", "upload",
        "--output-dir", str(output_dir),
        "--hfh-output-dir", str(tmp_path / "HFH"),
        "--hf-repo-id", REPO_ID,
    ])

    assert result.exit_code == 1
    assert "several camtrap dp packages" in result.output.lower()


def test_upload_fails_when_hfh_export_missing(tmp_path: Path, example_source_dataset: Path):
    real_output = tmp_path / "real"
    result = runner.invoke(app, [
        "generate", "real", "--source", str(example_source_dataset), "--output", str(real_output),
    ])
    assert result.exit_code == 0, result.output

    # No manifest.csv either, but repo_id/archive validation happens first —
    # media.filePath resolution never gets a chance to fail on that here.
    output_dir = tmp_path / "gbif_out"
    output_dir.mkdir()
    (output_dir / "somedataset-camtrap-dp.zip").write_bytes(b"fake")

    result = runner.invoke(app, [
        "publish", "gbif", "upload",
        "--output-dir", str(output_dir),
        "--hfh-output-dir", str(tmp_path / "HFH"),
        "--hf-repo-id", REPO_ID,
    ])

    assert result.exit_code == 1
    assert "huggingface hub export folder not found" in result.output.lower()


def test_upload_dry_run_does_not_copy_or_push_anything(tmp_path: Path, example_source_dataset: Path, monkeypatch):
    real_output, hfh_output_dir = _generate_real_dataset_and_hfh_export(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"
    _run_gbif_prepare(real_output, output_dir)

    hf_upload_calls: list = []
    monkeypatch.setattr(
        hf_service, "run_upload",
        lambda config_path, dry_run=False, allow_patterns=None: hf_upload_calls.append(1),
    )

    result = runner.invoke(app, [
        "publish", "gbif", "upload",
        "--output-dir", str(output_dir),
        "--hfh-output-dir", str(hfh_output_dir),
        "--hf-repo-id", REPO_ID,
        "--dry-run",
    ])

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output.lower()
    assert not (hfh_output_dir / "somedataset-camtrap-dp.zip").exists()
    assert hf_upload_calls == []


def test_upload_copies_archive_regenerates_checksums_and_pushes_scoped(
    tmp_path: Path, example_source_dataset: Path, monkeypatch,
):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    real_output, hfh_output_dir = _generate_real_dataset_and_hfh_export(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"
    _run_gbif_prepare(real_output, output_dir)

    hf_upload_calls: list = []
    monkeypatch.setattr(
        hf_service, "run_upload",
        lambda config_path, dry_run=False, allow_patterns=None: hf_upload_calls.append(
            (Path(config_path).parent, dry_run, allow_patterns),
        ),
    )

    result = runner.invoke(app, [
        "publish", "gbif", "upload",
        "--output-dir", str(output_dir),
        "--hfh-output-dir", str(hfh_output_dir),
        "--hf-repo-id", REPO_ID,
    ])

    assert result.exit_code == 0, result.output

    assert (hfh_output_dir / "somedataset-camtrap-dp.zip").exists()
    checksums_text = (hfh_output_dir / "checksums-sha256.txt").read_text()
    assert "somedataset-camtrap-dp.zip" in checksums_text

    assert hf_upload_calls == [
        (hfh_output_dir, False, ["somedataset-camtrap-dp.zip", "checksums-sha256.txt"]),
    ]

    # Rich may wrap the long URL across lines in the captured output, so
    # check for its distinctive parts rather than the exact substring.
    output_no_newlines = result.output.replace("\n", "")
    assert f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/somedataset-camtrap-dp.zip" in output_no_newlines
    assert "gbif register --archive-url" in output_no_newlines


def test_upload_fails_without_token(tmp_path: Path, example_source_dataset: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    real_output, hfh_output_dir = _generate_real_dataset_and_hfh_export(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"
    _run_gbif_prepare(real_output, output_dir)

    result = runner.invoke(app, [
        "publish", "gbif", "upload",
        "--output-dir", str(output_dir),
        "--hfh-output-dir", str(hfh_output_dir),
        "--hf-repo-id", REPO_ID,
    ])

    assert result.exit_code == 1
    assert "token" in result.output.lower()
    # The archive was still staged locally (and checksums regenerated)
    # before the token check for the actual push.
    assert (hfh_output_dir / "somedataset-camtrap-dp.zip").exists()
