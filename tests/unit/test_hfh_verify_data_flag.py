"""Unit tests for the verify_data=False path added to the HFH live-download
helpers (services/huggingface.py), which 'zenodo prepare' now uses by
default (services/zenodo.py's ensure_fresh_hfh_download_report) to skip
downloading/verifying the data/<split>/*.tar shards — Zenodo never uploads
them anyway, only the small evidence files. --verify-data restores the old
full-download-and-verify behaviour.
"""
from pathlib import Path

from donadataset.services import huggingface as hf_service
from donadataset.services.common import sha256_file


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_list_missing_required_files_skips_data_dirs_when_verify_data_false(tmp_path: Path):
    config = {}  # every filename falls back to its default, none exist yet

    missing_full = hf_service.list_missing_required_files(tmp_path, config, verify_data=True)
    missing_light = hf_service.list_missing_required_files(tmp_path, config, verify_data=False)

    assert "data/train" in missing_full
    assert "data/train" not in missing_light
    # the small evidence files are still required either way
    assert "README.md" in missing_full
    assert "README.md" in missing_light


def test_verify_global_checksums_skips_data_entries_when_verify_data_false(tmp_path: Path):
    _write(tmp_path / "README.md")
    readme_digest = sha256_file(tmp_path / "README.md")

    checksums_path = tmp_path / "checksums-sha256.txt"
    checksums_path.write_text(
        f"{readme_digest}  README.md\n"
        f"deadbeef  data/train/train-00000.tar\n",
        encoding="utf-8",
    )
    config = {}

    # Full mode: the missing shard is a real error.
    verified_full, errors_full = hf_service.verify_global_checksums(tmp_path, config, verify_data=True)
    assert verified_full == 1
    assert any("data/train/train-00000.tar" in e for e in errors_full)

    # Light mode: the shard entry is skipped entirely, no error.
    verified_light, errors_light = hf_service.verify_global_checksums(tmp_path, config, verify_data=False)
    assert verified_light == 1
    assert errors_light == []


def test_download_repository_passes_ignore_patterns_when_verify_data_false(tmp_path: Path, monkeypatch):
    captured = {}

    def _fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(hf_service, "snapshot_download", _fake_snapshot_download)
    monkeypatch.setattr(hf_service, "ensure_clean_dir", lambda path: None)

    hf_service.download_repository("someuser/somedataset", "dataset", "hf_dummy", tmp_path, verify_data=False)
    assert captured["ignore_patterns"] == ["data/**", "data/*"]

    captured.clear()
    hf_service.download_repository("someuser/somedataset", "dataset", "hf_dummy", tmp_path, verify_data=True)
    assert captured["ignore_patterns"] is None


def test_create_downloaded_verification_report_records_data_verified_flag(tmp_path: Path):
    report_path = tmp_path / "verification_report_downloaded.json"
    downloaded_dir = tmp_path / "download"
    downloaded_dir.mkdir()

    report = hf_service.create_downloaded_verification_report(
        report_path=report_path,
        repo_id="someuser/somedataset",
        repo_type="dataset",
        downloaded_dir=downloaded_dir,
        structural_errors=[],
        checksum_verified_count=5,
        checksum_errors=[],
        internal_verified_count=0,
        internal_errors=[],
        deleted_download_dir=False,
        data_verified=False,
    )

    assert report["data_verified"] is False
    assert report["status"] == "passed"
