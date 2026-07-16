"""Unit tests for services.zenodo's Zenodo staging directory.

'zenodo prepare' must copy every evidence file into a directory separate
from the HuggingFace export dir, so that directory ends up containing
exactly what gets uploaded to Zenodo.
"""
import hashlib
from pathlib import Path

from donadataset.services.zenodo import (
    find_optional_camtrap_dp_archive,
    get_output_filename,
    get_zenodo_output_dir,
    get_zenodo_staged_files_to_upload,
    stage_and_patch_files_for_zenodo,
    stage_files_for_zenodo,
    verify_local_reports,
)


def test_default_output_dir_is_sibling_of_hfh_export_dir(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    config = {"paths": {"output_dir": str(hfh_export_dir)}, "project": {"dataset_slug": "mytest"}}

    output_dir = get_zenodo_output_dir(config)

    assert output_dir == hfh_export_dir.parent / "Zenodo_mytest"


def test_output_dir_override_wins(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    custom_dir = tmp_path / "wherever-i-want"
    config = {
        "paths": {"output_dir": str(hfh_export_dir)},
        "project": {"dataset_slug": "mytest"},
        "zenodo": {"output_dir": str(custom_dir)},
    }

    assert get_zenodo_output_dir(config) == custom_dir


def test_stage_files_copies_into_output_dir(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    hfh_export_dir.mkdir()
    readme = hfh_export_dir / "README.md"
    readme.write_text("hello")
    citation = hfh_export_dir / "CITATION.cff"
    citation.write_text("cff content")

    output_dir = tmp_path / "Zenodo_mytest"
    staged = stage_files_for_zenodo([readme, citation], output_dir)

    assert staged == [output_dir / "README.md", output_dir / "CITATION.cff"]
    assert (output_dir / "README.md").read_text() == "hello"
    assert (output_dir / "CITATION.cff").read_text() == "cff content"
    # originals must be untouched, not moved
    assert readme.exists()
    assert citation.exists()


def test_stage_files_is_a_no_op_when_source_already_inside_output_dir(tmp_path: Path):
    output_dir = tmp_path / "Zenodo_mytest"
    output_dir.mkdir()
    already_there = output_dir / "README.md"
    already_there.write_text("hello")

    staged = stage_files_for_zenodo([already_there], output_dir)

    assert staged == [already_there]
    assert already_there.read_text() == "hello"


def test_get_output_filename_defaults_inside_zenodo_output_dir(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    config = {"paths": {"output_dir": str(hfh_export_dir)}, "project": {"dataset_slug": "mytest"}}

    path = get_output_filename(config, "linked_record_filename", "zenodo_linked_dataset_record.json")

    assert path == hfh_export_dir.parent / "Zenodo_mytest" / "zenodo_linked_dataset_record.json"


def test_get_output_filename_respects_explicit_override(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    custom_path = tmp_path / "somewhere-else.json"
    config = {
        "paths": {"output_dir": str(hfh_export_dir)},
        "project": {"dataset_slug": "mytest"},
        "zenodo": {"output": {"linked_record_filename": str(custom_path)}},
    }

    path = get_output_filename(config, "linked_record_filename", "zenodo_linked_dataset_record.json")

    assert path == custom_path


# ── stage_and_patch_files_for_zenodo ("zenodo prepare" local staging) ───────

def _fake_deposition(doi: str = "10.5072/zenodo.42") -> dict:
    return {
        "id": 42, "record_id": 42,
        "links": {"record_html": "https://sandbox.zenodo.org/records/42"},
        "metadata": {"prereserve_doi": {"doi": doi}},
    }


def test_stage_and_patch_injects_reserved_doi_into_staged_citation(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    hfh_export_dir.mkdir()
    citation = hfh_export_dir / "CITATION.cff"
    citation.write_text("cff-version: 1.2.0\nmessage: Cite this dataset\ntitle: Test\n")
    readme = hfh_export_dir / "README.md"
    readme.write_text("hello")

    config = {"paths": {"output_dir": str(hfh_export_dir)}, "project": {"dataset_slug": "mytest"}}
    output_dir = get_zenodo_output_dir(config)

    linked_record = stage_and_patch_files_for_zenodo(
        config, _fake_deposition(), [citation, readme], {"status": "passed"},
    )

    # The original, unstaged files must never be touched.
    assert "10.5072/zenodo.42" not in citation.read_text()
    assert "10.5072/zenodo.42" not in readme.read_text()

    staged_citation = output_dir / "CITATION.cff"
    assert "10.5072/zenodo.42" in staged_citation.read_text()

    staged_readme = output_dir / "README.md"
    assert "## Zenodo Sandbox DOI" in staged_readme.read_text()
    assert "10.5072/zenodo.42" in staged_readme.read_text()

    assert linked_record["reserved_doi"] == "10.5072/zenodo.42"


def test_stage_and_patch_regenerates_checksums_for_the_patched_files(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    hfh_export_dir.mkdir()
    citation = hfh_export_dir / "CITATION.cff"
    citation.write_text("cff-version: 1.2.0\nmessage: Cite this dataset\ntitle: Test\n")
    readme = hfh_export_dir / "README.md"
    readme.write_text("hello")

    config = {"paths": {"output_dir": str(hfh_export_dir)}, "project": {"dataset_slug": "mytest"}}
    output_dir = get_zenodo_output_dir(config)

    stage_and_patch_files_for_zenodo(config, _fake_deposition(), [citation, readme], {"status": "passed"})

    checksums_text = (output_dir / "checksums-sha256.txt").read_text()
    expected_citation_hash = hashlib.sha256((output_dir / "CITATION.cff").read_bytes()).hexdigest()
    expected_readme_hash = hashlib.sha256((output_dir / "README.md").read_bytes()).hexdigest()

    assert expected_citation_hash in checksums_text
    assert expected_readme_hash in checksums_text
    assert "CITATION.cff" in checksums_text
    assert "README.md" in checksums_text


def test_stage_and_patch_does_not_fail_when_doi_not_yet_returned(tmp_path: Path):
    """Zenodo's prereserve_doi should always come back synchronously, but if
    it somehow doesn't, staging must still succeed (just skip the patch)
    instead of crashing 'zenodo prepare'."""
    hfh_export_dir = tmp_path / "HFH"
    hfh_export_dir.mkdir()
    citation = hfh_export_dir / "CITATION.cff"
    citation.write_text("cff-version: 1.2.0\nmessage: Cite this dataset\ntitle: Test\n")
    readme = hfh_export_dir / "README.md"
    readme.write_text("hello")

    config = {"paths": {"output_dir": str(hfh_export_dir)}, "project": {"dataset_slug": "mytest"}}
    output_dir = get_zenodo_output_dir(config)
    deposition_without_doi = {"id": 42, "record_id": 42, "links": {}, "metadata": {}}

    linked_record = stage_and_patch_files_for_zenodo(
        config, deposition_without_doi, [citation, readme], {"status": "passed"},
    )

    assert linked_record["reserved_doi"] is None
    assert (output_dir / "README.md").read_text() == readme.read_text()
    assert (output_dir / "CITATION.cff").read_text() == citation.read_text()


# ── verify_local_reports ("zenodo check-readiness") ──────────────────────────

def test_verify_local_reports_finds_hfh_publication_report_inside_hfh_output_dir(tmp_path: Path):
    """Regression test: hfh_publication_report.json is written by
    'huggingface release' inside the HFH export dir (paths.output_dir), not
    Zenodo's own output dir — verify_local_reports used to look for it as a
    bare filename relative to the process cwd instead, so it always reported
    it missing even when it existed."""
    hfh_export_dir = tmp_path / "HFH"
    hfh_export_dir.mkdir()
    (hfh_export_dir / "hfh_publication_report.json").write_text('{"status": "passed"}')

    zenodo_output_dir = tmp_path / "Zenodo"
    zenodo_output_dir.mkdir()
    (zenodo_output_dir / "verification_report_downloaded.json").write_text('{"status": "passed"}')
    (zenodo_output_dir / "zenodo_file_verification_report.json").write_text('{"status": "passed"}')

    config = {
        "paths": {"output_dir": str(hfh_export_dir)},
        "zenodo": {"output_dir": str(zenodo_output_dir)},
        "project": {"dataset_slug": "mytest"},
    }

    result = verify_local_reports(config)

    assert result["status"] == "passed", result["errors"]
    assert not any("hfh_publication_report.json" in error for error in result["errors"])


def test_verify_local_reports_reports_hfh_publication_report_missing_when_absent(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    hfh_export_dir.mkdir()

    zenodo_output_dir = tmp_path / "Zenodo"
    zenodo_output_dir.mkdir()
    (zenodo_output_dir / "verification_report_downloaded.json").write_text('{"status": "passed"}')
    (zenodo_output_dir / "zenodo_file_verification_report.json").write_text('{"status": "passed"}')

    config = {
        "paths": {"output_dir": str(hfh_export_dir)},
        "zenodo": {"output_dir": str(zenodo_output_dir)},
        "project": {"dataset_slug": "mytest"},
    }

    result = verify_local_reports(config)

    assert result["status"] == "failed"
    assert any(
        "hfh_publication_report.json" in error and str(hfh_export_dir) in error
        for error in result["errors"]
    )


# ── find_optional_camtrap_dp_archive (GBIF's optional evidence file) ────────

def test_find_optional_camtrap_dp_archive_finds_matching_zip(tmp_path: Path):
    (tmp_path / "donadataset-camtrap-dp.zip").write_bytes(b"fake zip")
    (tmp_path / "README.md").write_text("hello")

    result = find_optional_camtrap_dp_archive(tmp_path)

    assert result == [tmp_path / "donadataset-camtrap-dp.zip"]


def test_find_optional_camtrap_dp_archive_returns_empty_list_when_absent(tmp_path: Path):
    (tmp_path / "README.md").write_text("hello")
    assert find_optional_camtrap_dp_archive(tmp_path) == []


def test_find_optional_camtrap_dp_archive_returns_empty_list_when_dir_missing(tmp_path: Path):
    assert find_optional_camtrap_dp_archive(tmp_path / "does-not-exist") == []


def test_get_zenodo_staged_files_to_upload_includes_camtrap_dp_when_present(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    config = {"paths": {"output_dir": str(hfh_export_dir)}, "project": {"dataset_slug": "mytest"}}
    output_dir = get_zenodo_output_dir(config)
    output_dir.mkdir(parents=True)
    (output_dir / "donadataset-camtrap-dp.zip").write_bytes(b"fake zip")

    files = get_zenodo_staged_files_to_upload(config)

    assert output_dir / "donadataset-camtrap-dp.zip" in files


def test_get_zenodo_staged_files_to_upload_omits_camtrap_dp_when_absent(tmp_path: Path):
    hfh_export_dir = tmp_path / "HFH"
    config = {"paths": {"output_dir": str(hfh_export_dir)}, "project": {"dataset_slug": "mytest"}}

    files = get_zenodo_staged_files_to_upload(config)

    assert not any(f.name.endswith("-camtrap-dp.zip") for f in files)


# ── run_zenodo_prepare end-to-end: picks up the Camtrap DP archive ─────────

def _write_fake_downloaded_evidence(hfh_download_dir: Path, include_camtrap_dp: bool) -> None:
    """Mimics what a real 'huggingface prepare' export (and thus a live HFH
    download) contains — the minimum set run_zenodo_prepare requires to get
    past validate_files_to_upload(), plus optionally the GBIF Camtrap DP zip."""
    hfh_download_dir.mkdir(parents=True, exist_ok=True)
    (hfh_download_dir / "README.md").write_text("hello")
    (hfh_download_dir / "LICENSE").write_text("license text")
    (hfh_download_dir / "CITATION.cff").write_text(
        "cff-version: 1.2.0\nmessage: Cite this dataset\ntitle: Test\n"
    )
    (hfh_download_dir / "HuggingFaceHub.yaml").write_text("repo: {}\n")
    (hfh_download_dir / "donana.yaml").write_text("names:\n  0: Empty\n")
    (hfh_download_dir / "dataset_info.json").write_text("{}")
    (hfh_download_dir / "metadata.csv").write_text("")
    (hfh_download_dir / "manifest.csv").write_text("")
    (hfh_download_dir / "manifest-files-sha256.csv").write_text("")
    (hfh_download_dir / "checksums-sha256.txt").write_text("")
    (hfh_download_dir / "validation_report.json").write_text('{"status": "passed"}')
    (hfh_download_dir / "verification_report_local.json").write_text('{"status": "passed"}')
    if include_camtrap_dp:
        (hfh_download_dir / "donadataset-camtrap-dp.zip").write_bytes(b"fake zip")


def _run_zenodo_prepare_with_mocked_network(tmp_path: Path, monkeypatch, include_camtrap_dp: bool) -> Path:
    from donadataset.services import zenodo as zenodo_service

    hfh_export_dir = tmp_path / "HFH"
    config = {
        "paths": {"output_dir": str(hfh_export_dir)},
        "project": {"dataset_slug": "mytest"},
        "huggingface": {"repo_id": "someuser/somerepo"},
        "zenodo": {
            "enabled": True,
            "environment": "sandbox",
            "creators": [{"name": "Test Author"}],
        },
    }
    zenodo_output_dir = get_zenodo_output_dir(config)

    def _fake_ensure_fresh_hfh_download_report(cfg, verify_data=False):
        _write_fake_downloaded_evidence(
            zenodo_service.get_zenodo_hfh_download_dir(cfg), include_camtrap_dp,
        )
        report_path = zenodo_service.get_zenodo_downloaded_report_path(cfg)
        zenodo_service.write_json(report_path, {"status": "passed"})
        return {"status": "passed"}

    fake_deposition = {
        "id": 42, "record_id": 42,
        "links": {"record_html": "https://sandbox.zenodo.org/records/42", "bucket": "https://fake/bucket"},
        "metadata": {"prereserve_doi": {"doi": "10.5072/zenodo.42"}},
    }

    monkeypatch.setattr(zenodo_service, "ensure_fresh_hfh_download_report", _fake_ensure_fresh_hfh_download_report)
    monkeypatch.setattr(zenodo_service, "get_zenodo_token", lambda cfg: "dummy-token")
    monkeypatch.setattr(zenodo_service, "create_deposition", lambda api_base_url, token: fake_deposition)
    monkeypatch.setattr(
        zenodo_service, "update_deposition_metadata",
        lambda api_base_url, token, deposition_id, metadata: fake_deposition,
    )

    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        import yaml as _yaml
        _yaml.safe_dump(config, f)

    zenodo_service.run_zenodo_prepare(config_path, dry_run=False)
    return zenodo_output_dir


def test_run_zenodo_prepare_includes_camtrap_dp_archive_when_present(tmp_path: Path, monkeypatch):
    zenodo_output_dir = _run_zenodo_prepare_with_mocked_network(tmp_path, monkeypatch, include_camtrap_dp=True)
    assert (zenodo_output_dir / "donadataset-camtrap-dp.zip").is_file()


def test_run_zenodo_prepare_succeeds_without_camtrap_dp_archive(tmp_path: Path, monkeypatch):
    zenodo_output_dir = _run_zenodo_prepare_with_mocked_network(tmp_path, monkeypatch, include_camtrap_dp=False)
    assert not any(zenodo_output_dir.glob("*-camtrap-dp.zip"))
