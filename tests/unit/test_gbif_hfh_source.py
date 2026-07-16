"""Unit tests for services.gbif's HFH-sourced dataset resolution.

'gbif prepare' needs full images+labels (EXIF + box counts), not just small
metadata files, so it can't do a live "fetch what's needed" like Zenodo does
— it reuses whatever 'huggingface prepare' already left on disk locally, and
only downloads from HuggingFace Hub (and extracts once) when that's missing.
"""
import shutil
import tarfile
from pathlib import Path

import pytest
import yaml

from donadataset.services import gbif as gbif_service


def _write_fake_hfh_export(hfh_dir: Path) -> None:
    """Builds a minimal directory shaped like a real 'huggingface prepare'
    output: a donana.yaml with a 'names' mapping, a manifest.csv, and one
    packed data/train/*.tar shard containing one image + one label."""
    hfh_dir.mkdir(parents=True, exist_ok=True)
    (hfh_dir / "donana.yaml").write_text(yaml.safe_dump({"names": {0: "Empty", 1: "Deer"}}))
    (hfh_dir / "manifest.csv").write_text(
        "image_id,split,shard\nimg1,train,data/train/train-00000.tar\n"
    )

    staging_dir = hfh_dir / "_staging"
    (staging_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (staging_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (staging_dir / "images" / "train" / "img1.jpg").write_bytes(b"fake-image-bytes")
    (staging_dir / "labels" / "train" / "img1.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    shard_dir = hfh_dir / "data" / "train"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_dir / "train-00000.tar", "w") as tar:
        tar.add(staging_dir / "images" / "train" / "img1.jpg", arcname="images/train/img1.jpg")
        tar.add(staging_dir / "labels" / "train" / "img1.txt", arcname="labels/train/img1.txt")

    shutil.rmtree(staging_dir)


# ── _looks_like_hfh_export / _looks_like_extracted_yolo_dataset ─────────────

def test_looks_like_hfh_export_true_when_shards_and_names_yaml_present(tmp_path: Path):
    hfh_dir = tmp_path / "HFH"
    _write_fake_hfh_export(hfh_dir)
    assert gbif_service._looks_like_hfh_export(hfh_dir) is True


def test_looks_like_hfh_export_false_when_directory_missing(tmp_path: Path):
    assert gbif_service._looks_like_hfh_export(tmp_path / "nope") is False


def test_looks_like_hfh_export_false_without_tar_shards(tmp_path: Path):
    hfh_dir = tmp_path / "HFH"
    hfh_dir.mkdir()
    (hfh_dir / "donana.yaml").write_text(yaml.safe_dump({"names": {0: "Empty"}}))
    assert gbif_service._looks_like_hfh_export(hfh_dir) is False


def test_looks_like_extracted_yolo_dataset_true_after_extraction(tmp_path: Path):
    hfh_dir = tmp_path / "HFH"
    _write_fake_hfh_export(hfh_dir)
    extracted_dir = tmp_path / "extracted"
    gbif_service.extract_hfh_export_to_yolo_source(hfh_dir, extracted_dir)
    assert gbif_service._looks_like_extracted_yolo_dataset(extracted_dir) is True


def test_looks_like_extracted_yolo_dataset_false_without_manifest(tmp_path: Path):
    """A cache extracted before manifest.csv started traveling alongside the
    images/labels tree must not be mistaken for a complete, reusable cache."""
    extracted_dir = tmp_path / "extracted"
    (extracted_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (extracted_dir / "images" / "train" / "img1.jpg").write_bytes(b"fake-image-bytes")
    (extracted_dir / "donana.yaml").write_text(yaml.safe_dump({"names": {0: "Empty", 1: "Deer"}}))
    assert gbif_service._looks_like_extracted_yolo_dataset(extracted_dir) is False


# ── extract_hfh_export_to_yolo_source ────────────────────────────────────────

def test_extract_writes_yolo_tree_and_copies_names_yaml(tmp_path: Path):
    hfh_dir = tmp_path / "HFH"
    _write_fake_hfh_export(hfh_dir)
    extracted_dir = tmp_path / "extracted"

    result = gbif_service.extract_hfh_export_to_yolo_source(hfh_dir, extracted_dir)

    assert result == extracted_dir
    assert (extracted_dir / "images" / "train" / "img1.jpg").is_file()
    assert (extracted_dir / "labels" / "train" / "img1.txt").is_file()
    assert (extracted_dir / "donana.yaml").is_file()
    # media.filePath always links to HuggingFace Hub — manifest.csv has to
    # travel with the extracted images, not just the names YAML.
    assert (extracted_dir / "manifest.csv").is_file()


def test_extract_reuses_already_extracted_dataset_instead_of_re_extracting(tmp_path: Path, monkeypatch):
    hfh_dir = tmp_path / "HFH"
    _write_fake_hfh_export(hfh_dir)
    extracted_dir = tmp_path / "extracted"
    gbif_service.extract_hfh_export_to_yolo_source(hfh_dir, extracted_dir)

    calls = []
    monkeypatch.setattr(gbif_service, "find_tar_files", lambda *a, **k: calls.append(1) or [])

    gbif_service.extract_hfh_export_to_yolo_source(hfh_dir, extracted_dir)

    assert calls == []


def test_extract_re_syncs_manifest_when_previously_extracted_without_one(tmp_path: Path):
    """Regression test: a cache extracted before manifest.csv started
    traveling with it (images/labels present, manifest.csv missing) must be
    self-healed instead of being reused half-broken forever."""
    hfh_dir = tmp_path / "HFH"
    _write_fake_hfh_export(hfh_dir)
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    for tar_path in gbif_service.find_tar_files(hfh_dir):
        gbif_service.safe_extract_tar(tar_path, extracted_dir)
    shutil.copy2(hfh_dir / "donana.yaml", extracted_dir / "donana.yaml")
    assert not (extracted_dir / "manifest.csv").exists()

    gbif_service.extract_hfh_export_to_yolo_source(hfh_dir, extracted_dir)

    assert (extracted_dir / "manifest.csv").is_file()


# ── resolve_gbif_source_dataset_dir ──────────────────────────────────────────

def test_resolve_returns_explicit_source_dataset_dir_unchanged(tmp_path: Path):
    explicit = tmp_path / "my-local-dataset"
    result = gbif_service.resolve_gbif_source_dataset_dir("someuser/somerepo", explicit)
    assert result == explicit


def test_resolve_fails_without_source_dataset_dir_or_repo_id():
    with pytest.raises(RuntimeError, match="repo-id"):
        gbif_service.resolve_gbif_source_dataset_dir(None, None)


def test_resolve_fails_with_placeholder_repo_id():
    with pytest.raises(RuntimeError, match="repo-id"):
        gbif_service.resolve_gbif_source_dataset_dir("REPLACE_WITH_HF_USER/REPLACE_WITH_DATASET_SLUG", None)


def test_resolve_uses_local_hfh_export_without_downloading(tmp_path: Path, monkeypatch):
    repo_id = "someuser/somerepo"
    hfh_dir = tmp_path / "HFH" / repo_id
    _write_fake_hfh_export(hfh_dir)
    download_dir = tmp_path / "GBIF" / repo_id / "hfh_download"

    monkeypatch.setattr(gbif_service, "get_hfh_output_dir", lambda rid: hfh_dir)
    monkeypatch.setattr(gbif_service, "get_gbif_hfh_download_dir", lambda rid: download_dir)

    def _fail_if_called(**kwargs):
        raise AssertionError("download_repository should not have been called")

    monkeypatch.setattr(gbif_service, "download_repository", _fail_if_called)

    result = gbif_service.resolve_gbif_source_dataset_dir(repo_id, None)

    assert result == download_dir
    assert (download_dir / "images" / "train" / "img1.jpg").is_file()
    # the original, un-extracted HFH export must never be touched
    assert not (hfh_dir / "images").exists()


def test_resolve_downloads_from_hfh_when_local_export_missing(tmp_path: Path, monkeypatch):
    repo_id = "someuser/somerepo"
    hfh_dir = tmp_path / "HFH" / repo_id  # deliberately never created
    download_dir = tmp_path / "GBIF" / repo_id / "hfh_download"

    monkeypatch.setattr(gbif_service, "get_hfh_output_dir", lambda rid: hfh_dir)
    monkeypatch.setattr(gbif_service, "get_gbif_hfh_download_dir", lambda rid: download_dir)

    calls = {}

    def _fake_download_repository(*, repo_id, repo_type, token, download_dir, verify_data):
        calls["repo_id"] = repo_id
        calls["repo_type"] = repo_type
        calls["verify_data"] = verify_data
        _write_fake_hfh_export(download_dir)
        return download_dir

    monkeypatch.setattr(gbif_service, "download_repository", _fake_download_repository)

    result = gbif_service.resolve_gbif_source_dataset_dir(repo_id, None)

    assert calls["repo_id"] == repo_id
    assert calls["repo_type"] == "dataset"
    assert calls["verify_data"] is True
    assert result == download_dir
    assert (download_dir / "images" / "train" / "img1.jpg").is_file()


def test_resolve_reuses_previously_extracted_download_without_recontacting_hfh(
    tmp_path: Path, monkeypatch,
):
    """Once hfh_download/ already has an extracted YOLO tree, a later run
    must not even check the local HFH export or hit the network again."""
    repo_id = "someuser/somerepo"
    download_dir = tmp_path / "GBIF" / repo_id / "hfh_download"
    _write_fake_hfh_export(download_dir)
    gbif_service.extract_hfh_export_to_yolo_source(download_dir, download_dir)

    monkeypatch.setattr(gbif_service, "get_gbif_hfh_download_dir", lambda rid: download_dir)

    def _fail_if_called(*a, **k):
        raise AssertionError("should not be called once already extracted")

    monkeypatch.setattr(gbif_service, "get_hfh_output_dir", _fail_if_called)
    monkeypatch.setattr(gbif_service, "download_repository", _fail_if_called)

    result = gbif_service.resolve_gbif_source_dataset_dir(repo_id, None)

    assert result == download_dir
    assert (download_dir / "images" / "train" / "img1.jpg").is_file()


# ── read_local_shard_urls (media.filePath -> HuggingFace Hub, unconditional) ─

def test_read_local_shard_urls_builds_urls_from_local_manifest(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "manifest.csv").write_text(
        "image_id,split,shard\n"
        "img1,train,data/train/train-00000.tar\n"
        "img2,val,data/val/val-00000.tar\n"
    )

    urls = gbif_service.read_local_shard_urls(source_dir, "someuser/somerepo")

    assert urls == {
        "img1": "https://huggingface.co/datasets/someuser/somerepo/resolve/main/data/train/train-00000.tar",
        "img2": "https://huggingface.co/datasets/someuser/somerepo/resolve/main/data/val/val-00000.tar",
    }


def test_read_local_shard_urls_fails_when_manifest_missing(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    with pytest.raises(RuntimeError, match="manifest.csv"):
        gbif_service.read_local_shard_urls(source_dir, "someuser/somerepo")


# ── resolve_gbif_hfh_output_dir ('gbif upload's own local-first resolution) ─

def test_resolve_hfh_output_dir_returns_explicit_dir_unchanged(tmp_path: Path):
    explicit = tmp_path / "my-local-export"
    result = gbif_service.resolve_gbif_hfh_output_dir("someuser/somerepo", explicit)
    assert result == explicit


def test_resolve_hfh_output_dir_uses_local_hfh_export_without_downloading(tmp_path: Path, monkeypatch):
    repo_id = "someuser/somerepo"
    hfh_dir = tmp_path / "HFH" / repo_id
    _write_fake_hfh_export(hfh_dir)
    download_dir = tmp_path / "GBIF" / repo_id / "hfh_download"

    monkeypatch.setattr(gbif_service, "get_hfh_output_dir", lambda rid: hfh_dir)
    monkeypatch.setattr(gbif_service, "get_gbif_hfh_download_dir", lambda rid: download_dir)

    def _fail_if_called(**kwargs):
        raise AssertionError("download_repository should not have been called")

    monkeypatch.setattr(gbif_service, "download_repository", _fail_if_called)

    result = gbif_service.resolve_gbif_hfh_output_dir(repo_id, None)

    assert result == hfh_dir


def test_resolve_hfh_output_dir_reuses_gbif_cache_without_recontacting_hfh(tmp_path: Path, monkeypatch):
    """If a previous 'gbif prepare' or 'gbif upload' run already downloaded
    the export into hfh_download/, reuse it as-is instead of re-checking
    the local HFH export path or hitting the network again — even though
    prepare may have also extracted images/labels/ into that same
    directory (see resolve_gbif_hfh_output_dir's own docstring)."""
    repo_id = "someuser/somerepo"
    download_dir = tmp_path / "GBIF" / repo_id / "hfh_download"
    _write_fake_hfh_export(download_dir)

    monkeypatch.setattr(gbif_service, "get_hfh_output_dir", lambda rid: tmp_path / "HFH" / repo_id)
    monkeypatch.setattr(gbif_service, "get_gbif_hfh_download_dir", lambda rid: download_dir)

    def _fail_if_called(*a, **k):
        raise AssertionError("should not be called once already cached")

    monkeypatch.setattr(gbif_service, "download_repository", _fail_if_called)

    result = gbif_service.resolve_gbif_hfh_output_dir(repo_id, None)

    assert result == download_dir


def test_resolve_hfh_output_dir_downloads_into_cache_when_neither_present(tmp_path: Path, monkeypatch):
    repo_id = "someuser/somerepo"
    hfh_dir = tmp_path / "HFH" / repo_id  # deliberately never created
    download_dir = tmp_path / "GBIF" / repo_id / "hfh_download"

    monkeypatch.setattr(gbif_service, "get_hfh_output_dir", lambda rid: hfh_dir)
    monkeypatch.setattr(gbif_service, "get_gbif_hfh_download_dir", lambda rid: download_dir)

    calls = {}

    def _fake_download_repository(*, repo_id, repo_type, token, download_dir, verify_data):
        calls["repo_id"] = repo_id
        _write_fake_hfh_export(download_dir)
        return download_dir

    monkeypatch.setattr(gbif_service, "download_repository", _fake_download_repository)

    result = gbif_service.resolve_gbif_hfh_output_dir(repo_id, None)

    assert calls["repo_id"] == repo_id
    assert result == download_dir
    assert (download_dir / "manifest.csv").is_file()


def test_resolve_hfh_output_dir_dry_run_never_downloads(tmp_path: Path, monkeypatch):
    repo_id = "someuser/somerepo"
    monkeypatch.setattr(gbif_service, "get_hfh_output_dir", lambda rid: tmp_path / "HFH" / repo_id)
    monkeypatch.setattr(gbif_service, "get_gbif_hfh_download_dir", lambda rid: tmp_path / "GBIF" / repo_id / "hfh_download")

    def _fail_if_called(**kwargs):
        raise AssertionError("dry_run must never download")

    monkeypatch.setattr(gbif_service, "download_repository", _fail_if_called)

    result = gbif_service.resolve_gbif_hfh_output_dir(repo_id, None, dry_run=True)

    assert result is None


# ── regenerate_hfh_checksums ─────────────────────────────────────────────────

def test_regenerate_hfh_checksums_keeps_existing_entries_and_adds_new_file(tmp_path: Path):
    hfh_dir = tmp_path / "hfh"
    hfh_dir.mkdir()
    (hfh_dir / "README.md").write_text("hello")
    (hfh_dir / "checksums-sha256.txt").write_text(
        f"{gbif_service.sha256_file(hfh_dir / 'README.md')}  README.md\n"
    )
    (hfh_dir / "new-archive.zip").write_bytes(b"fake-zip-bytes")

    gbif_service.regenerate_hfh_checksums(hfh_dir, ["new-archive.zip"])

    entries = gbif_service.read_checksums(hfh_dir / "checksums-sha256.txt")
    assert set(entries) == {"README.md", "new-archive.zip"}
    assert entries["new-archive.zip"] == gbif_service.sha256_file(hfh_dir / "new-archive.zip")


def test_regenerate_hfh_checksums_ignores_untracked_clutter(tmp_path: Path):
    """The whole point of not doing a fresh directory scan: local-only
    files that were never part of the published HuggingFace Hub repo (e.g.
    'gbif prepare's own extracted images/labels/ inside a shared
    hfh_download/ cache) must never leak into the regenerated checksums."""
    hfh_dir = tmp_path / "hfh"
    hfh_dir.mkdir()
    (hfh_dir / "README.md").write_text("hello")
    (hfh_dir / "checksums-sha256.txt").write_text(
        f"{gbif_service.sha256_file(hfh_dir / 'README.md')}  README.md\n"
    )
    (hfh_dir / "new-archive.zip").write_bytes(b"fake-zip-bytes")
    (hfh_dir / "images").mkdir()
    (hfh_dir / "images" / "img1.jpg").write_bytes(b"local-only-clutter")

    gbif_service.regenerate_hfh_checksums(hfh_dir, ["new-archive.zip"])

    entries = gbif_service.read_checksums(hfh_dir / "checksums-sha256.txt")
    assert set(entries) == {"README.md", "new-archive.zip"}


def test_regenerate_hfh_checksums_fails_when_checksums_file_missing(tmp_path: Path):
    hfh_dir = tmp_path / "hfh"
    hfh_dir.mkdir()
    (hfh_dir / "new-archive.zip").write_bytes(b"fake-zip-bytes")

    with pytest.raises(RuntimeError, match="checksums-sha256.txt"):
        gbif_service.regenerate_hfh_checksums(hfh_dir, ["new-archive.zip"])
