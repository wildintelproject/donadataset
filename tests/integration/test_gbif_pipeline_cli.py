"""Integration tests for 'publish gbif pipeline' (prepare -> upload -> register;
media.filePath always links to HuggingFace Hub in both), the command
donadataset.commands.publish_all shells out to for the GBIF phase of 'publish all'.

Every network-touching call (the actual push to HuggingFace Hub, GBIF Registry
API) is monkeypatched, so nothing here reaches a real external service. The
local staging step (copying the .zip into the HFH export and regenerating its
checksums-sha256.txt, see gbif_service.run_upload) is exercised for real
against a real local 'huggingface prepare' export, since it's all local
filesystem work.
"""
import json
from pathlib import Path
from typing import Any, Dict

from typer.testing import CliRunner

from donadataset.main import app
from donadataset.services import gbif as gbif_service
from donadataset.services import huggingface as hf_service

runner = CliRunner()

REPO_ID = "someuser/somedataset"
FAKE_ARCHIVE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/somedataset-camtrap-dp.zip"


class _FakeResponse:
    def __init__(self, payload: Any):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def _generate_real_dataset_and_hfh_export(tmp_path: Path, example_source_dataset: Path) -> tuple[Path, Path]:
    """generate real -> huggingface prepare (a real local export, so
    'gbif upload's local staging — copy the .zip in, regenerate
    checksums-sha256.txt — runs for real, not mocked) -> copies manifest.csv
    into the loose YOLO output, same as test_gbif_cli.py's
    _generate_real_dataset. Returns (real_output, hfh_output_dir)."""
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


def _patch_registry_api(monkeypatch, calls: list):
    """Fakes the GBIF Registry API: dataset creation returns a fixed key,
    endpoint listing is empty (so no delete happens), endpoint creation
    just records the call."""
    def fake_post(url: str, json: Dict[str, Any] = None, auth=None, **kwargs):  # noqa: A002
        calls.append(("POST", url, json))
        if url.endswith("/v1/dataset"):
            return _FakeResponse("11111111-1111-1111-1111-111111111111")
        return _FakeResponse({})

    def fake_get(url: str, auth=None, **kwargs):
        calls.append(("GET", url, None))
        return _FakeResponse([])

    monkeypatch.setattr(gbif_service.requests, "post", fake_post)
    monkeypatch.setattr(gbif_service.requests, "get", fake_get)


def test_pipeline_chains_prepare_upload_url_straight_into_register(
    tmp_path: Path, example_source_dataset: Path, monkeypatch,
):
    monkeypatch.setenv("GBIF_USERNAME", "user")
    monkeypatch.setenv("GBIF_PASSWORD", "pass")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    real_output, hfh_output_dir = _generate_real_dataset_and_hfh_export(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"

    hf_upload_calls: list = []
    monkeypatch.setattr(
        hf_service, "run_upload",
        lambda config_path, dry_run=False, allow_patterns=None: hf_upload_calls.append(allow_patterns),
    )
    registry_calls: list = []
    _patch_registry_api(monkeypatch, registry_calls)

    result = runner.invoke(app, [
        "publish", "gbif", "pipeline",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(output_dir),
        "--hf-repo-id", REPO_ID,
        "--hfh-output-dir", str(hfh_output_dir),
        "--publishing-organization-key", "00000000-0000-0000-0000-000000000001",
        "--installation-key", "00000000-0000-0000-0000-000000000002",
    ])

    assert result.exit_code == 0, result.output
    assert (output_dir / "somedataset-camtrap-dp.zip").exists()

    # 'upload' staged the .zip into the HFH export and regenerated its
    # checksums, then pushed exactly those two files (never the rest of the
    # already-published export).
    assert (hfh_output_dir / "somedataset-camtrap-dp.zip").exists()
    assert hf_upload_calls == [["somedataset-camtrap-dp.zip", "checksums-sha256.txt"]]
    checksums_text = (hfh_output_dir / "checksums-sha256.txt").read_text()
    assert "somedataset-camtrap-dp.zip" in checksums_text

    # The URL 'upload' produced is exactly what got registered as the
    # CAMTRAP_DP endpoint — no manual copy-paste.
    endpoint_post_calls = [call for call in registry_calls if call[0] == "POST" and call[1].endswith("/endpoint")]
    assert endpoint_post_calls, registry_calls
    assert endpoint_post_calls[0][2] == {"type": "CAMTRAP_DP", "url": FAKE_ARCHIVE_URL}

    record = json.loads((output_dir / "gbif_linked_dataset_record.json").read_text())
    assert record["archive_url"] == FAKE_ARCHIVE_URL
    assert record["dataset_key"] == "11111111-1111-1111-1111-111111111111"


def test_pipeline_fails_without_gbif_credentials_after_prepare_and_upload_succeed(
    tmp_path: Path, example_source_dataset: Path, monkeypatch,
):
    monkeypatch.delenv("GBIF_USERNAME", raising=False)
    monkeypatch.delenv("GBIF_PASSWORD", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    real_output, hfh_output_dir = _generate_real_dataset_and_hfh_export(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"

    monkeypatch.setattr(
        hf_service, "run_upload", lambda config_path, dry_run=False, allow_patterns=None: None,
    )

    result = runner.invoke(app, [
        "publish", "gbif", "pipeline",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(output_dir),
        "--hf-repo-id", REPO_ID,
        "--hfh-output-dir", str(hfh_output_dir),
        "--publishing-organization-key", "00000000-0000-0000-0000-000000000001",
        "--installation-key", "00000000-0000-0000-0000-000000000002",
    ])

    assert result.exit_code == 1
    assert "gbif_username" in result.output.lower()
    # prepare's and upload's own artifacts (built before register ever runs) are still there.
    assert (output_dir / "somedataset-camtrap-dp.zip").exists()
    assert (hfh_output_dir / "somedataset-camtrap-dp.zip").exists()
