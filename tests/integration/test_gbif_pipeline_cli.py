"""Integration tests for 'publish gbif pipeline' (prepare --upload-to-huggingface
--link-media-to-huggingface -> register), the command donadataset.commands.publish_all
shells out to for the GBIF phase of 'publish all'.

Every network-touching call (HuggingFace Hub upload, GBIF Registry API) is
monkeypatched, so nothing here reaches a real external service.
"""
import json
from pathlib import Path
from typing import Any, Dict

from typer.testing import CliRunner

from donadataset.main import app
from donadataset.services import gbif as gbif_service

runner = CliRunner()

FAKE_ARCHIVE_URL = "https://huggingface.co/datasets/someuser/somedataset/resolve/main/donadataset-camtrap-dp.zip"


class _FakeResponse:
    def __init__(self, payload: Any):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def _generate_real_dataset(tmp_path: Path, example_source_dataset: Path) -> Path:
    real_output = tmp_path / "real"
    result = runner.invoke(app, [
        "generate", "real", "--source", str(example_source_dataset), "--output", str(real_output),
    ])
    assert result.exit_code == 0, result.output
    return real_output


def _fake_shard_urls(real_output: Path) -> Dict[str, str]:
    """'pipeline' always runs with --link-media-to-huggingface too, so it
    also needs fetch_hfh_shard_urls mocked — one fake shard URL per split,
    same shape as test_gbif_cli.py's equivalent helper."""
    urls: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        shard_url = f"https://huggingface.co/datasets/someuser/somedataset/resolve/main/data/{split}/{split}-00000.tar"
        for label_path in (real_output / "labels" / split).glob("*.txt"):
            urls[label_path.stem] = shard_url
    return urls


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


def test_pipeline_chains_prepare_url_straight_into_register(
    tmp_path: Path, example_source_dataset: Path, monkeypatch,
):
    monkeypatch.setenv("GBIF_USERNAME", "user")
    monkeypatch.setenv("GBIF_PASSWORD", "pass")
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"

    monkeypatch.setattr(gbif_service, "upload_archive_to_huggingface", lambda archive_path, repo_id: FAKE_ARCHIVE_URL)
    monkeypatch.setattr(gbif_service, "fetch_hfh_shard_urls", lambda repo_id: _fake_shard_urls(real_output))
    registry_calls: list = []
    _patch_registry_api(monkeypatch, registry_calls)

    result = runner.invoke(app, [
        "publish", "gbif", "pipeline",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(output_dir),
        "--hf-repo-id", "someuser/somedataset",
        "--publishing-organization-key", "00000000-0000-0000-0000-000000000001",
        "--installation-key", "00000000-0000-0000-0000-000000000002",
    ])

    assert result.exit_code == 0, result.output
    assert (output_dir / "donadataset-camtrap-dp.zip").exists()

    # The URL run_prepare returned (from the monkeypatched upload) is exactly
    # what got registered as the CAMTRAP_DP endpoint — no manual copy-paste.
    endpoint_post_calls = [call for call in registry_calls if call[0] == "POST" and call[1].endswith("/endpoint")]
    assert endpoint_post_calls, registry_calls
    assert endpoint_post_calls[0][2] == {"type": "CAMTRAP_DP", "url": FAKE_ARCHIVE_URL}

    record = json.loads((output_dir / "gbif_linked_dataset_record.json").read_text())
    assert record["archive_url"] == FAKE_ARCHIVE_URL
    assert record["dataset_key"] == "11111111-1111-1111-1111-111111111111"


def test_pipeline_fails_without_gbif_credentials_after_prepare_succeeds(
    tmp_path: Path, example_source_dataset: Path, monkeypatch,
):
    monkeypatch.delenv("GBIF_USERNAME", raising=False)
    monkeypatch.delenv("GBIF_PASSWORD", raising=False)
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"

    monkeypatch.setattr(gbif_service, "upload_archive_to_huggingface", lambda archive_path, repo_id: FAKE_ARCHIVE_URL)
    monkeypatch.setattr(gbif_service, "fetch_hfh_shard_urls", lambda repo_id: _fake_shard_urls(real_output))

    result = runner.invoke(app, [
        "publish", "gbif", "pipeline",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(output_dir),
        "--hf-repo-id", "someuser/somedataset",
        "--publishing-organization-key", "00000000-0000-0000-0000-000000000001",
        "--installation-key", "00000000-0000-0000-0000-000000000002",
    ])

    assert result.exit_code == 1
    assert "gbif_username" in result.output.lower()
    # prepare's own artifacts (built before register ever runs) are still there.
    assert (output_dir / "donadataset-camtrap-dp.zip").exists()
