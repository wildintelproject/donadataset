"""Integration tests for 'publish gbif prepare'.

GBIF publishes camera-trap data as Camtrap DP (https://camtrap-dp.tdwg.org/)
— 'prepare' is purely local/offline: it scans a clean YOLO dataset and
generates the whole package itself (deployments.csv, media.csv,
observations.csv, datapackage.json, zipped) with no CSV to fill in by hand.

Since this pipeline tracks no real GPS or per-camera deployment dates
anywhere, 'prepare' assumes one synthetic deployment per split
(train/val/test, see SPLIT_DEPLOYMENT_COORDINATES in
donadataset.services.gbif) and reads each image's capture date from its
EXIF metadata when present, falling back to a placeholder date otherwise.
The bundled example dataset's images carry no EXIF data at all, so most
tests here exercise the placeholder-date path; test_prepare_uses_real_exif_*
injects EXIF into a copy of one image to exercise the real-date path.
"""
import csv
import json
import zipfile
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from donadataset.main import app
from donadataset.services import gbif as gbif_service

runner = CliRunner()


def _generate_real_dataset(tmp_path: Path, example_source_dataset: Path) -> Path:
    real_output = tmp_path / "real"
    result = runner.invoke(app, [
        "generate", "real", "--source", str(example_source_dataset), "--output", str(real_output),
    ])
    assert result.exit_code == 0, result.output
    return real_output


def _run_prepare(real_output: Path, output_dir: Path, extra_args: list[str] | None = None):
    return runner.invoke(app, [
        "publish", "gbif", "prepare",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(output_dir),
        *(extra_args or []),
    ])


def test_prepare_generates_valid_camtrap_dp_package(tmp_path: Path, example_source_dataset: Path):
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"

    result = _run_prepare(real_output, output_dir, ["--dataset-slug", "testset", "--contact-email", "you@example.org"])
    assert result.exit_code == 0, result.output

    deployments_csv = output_dir / "deployments.csv"
    media_csv = output_dir / "media.csv"
    observations_csv = output_dir / "observations.csv"
    datapackage_json = output_dir / "datapackage.json"
    archive_path = output_dir / "testset-camtrap-dp.zip"
    for path in (deployments_csv, media_csv, observations_csv, datapackage_json, archive_path):
        assert path.exists(), path

    with zipfile.ZipFile(archive_path) as zf:
        assert set(zf.namelist()) == {"datapackage.json", "deployments.csv", "media.csv", "observations.csv"}

    datapackage = json.loads(datapackage_json.read_text())
    assert datapackage["name"] == "testset"
    assert {resource["name"] for resource in datapackage["resources"]} == {"deployments", "media", "observations"}
    assert datapackage["contributors"][0]["email"] == "you@example.org"

    with deployments_csv.open(newline="") as f:
        deployment_rows = list(csv.DictReader(f))
    # example_source_dataset produces images in all three splits after 'generate real'.
    assert {row["deploymentID"] for row in deployment_rows} == {"train", "val", "test"}
    for row in deployment_rows:
        assert -90 <= float(row["latitude"]) <= 90
        assert -180 <= float(row["longitude"]) <= 180
        assert row["deploymentStart"] <= row["deploymentEnd"]

    with media_csv.open(newline="") as f:
        media_rows = list(csv.DictReader(f))
    assert len(media_rows) == 9  # 3 train + 3 val + 3 test, matching generate real's output
    media_ids = {row["mediaID"] for row in media_rows}
    assert len(media_ids) == len(media_rows)  # unique
    for row in media_rows:
        assert row["fileMediatype"] == "image/jpeg"
        assert row["filePath"].startswith(f"images/{row['deploymentID']}/")
        # No image in the bundled fixture carries EXIF, so every row should be flagged as estimated.
        assert "estimated" in row["mediaComments"]

    with observations_csv.open(newline="") as f:
        observation_rows = list(csv.DictReader(f))
    observation_ids = [row["observationID"] for row in observation_rows]
    assert len(observation_ids) == len(set(observation_ids))  # unique
    for row in observation_rows:
        assert row["scientificName"].strip().lower() != "empty"
        assert row["observationType"] in {"animal", "blank"}
        if row["observationType"] == "animal":
            assert row["scientificName"]
            assert int(row["count"]) >= 1
        else:
            assert row["scientificName"] == ""
            assert row["count"] == ""


def test_prepare_marks_image_detected_as_empty_as_blank_observation(tmp_path: Path, example_source_dataset: Path):
    """img_103 (val split) is labelled with the source dataset's 'Empty' class only."""
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"

    result = _run_prepare(real_output, output_dir)
    assert result.exit_code == 0, result.output

    with (output_dir / "observations.csv").open(newline="") as f:
        rows = {row["mediaID"]: row for row in csv.DictReader(f)}

    assert rows["img_103"]["observationType"] == "blank"
    assert rows["img_103"]["scientificName"] == ""


def test_prepare_aggregates_repeated_boxes_of_the_same_species(tmp_path: Path, example_source_dataset: Path):
    """img_006 (train split) has two boxes of the same class -> count=2, one row."""
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"

    result = _run_prepare(real_output, output_dir)
    assert result.exit_code == 0, result.output

    with (output_dir / "observations.csv").open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["mediaID"] == "img_006"]

    assert len(rows) == 1
    assert rows[0]["count"] == "2"


def test_prepare_uses_real_exif_datetime_when_available(tmp_path: Path, example_source_dataset: Path):
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)

    image_path = real_output / "images" / "train" / "img_001.jpg"
    img = Image.open(image_path)
    exif = img.getexif()
    exif[0x0132] = "2021:06:15 08:30:00"  # DateTime
    img.save(image_path, exif=exif.tobytes())

    output_dir = tmp_path / "gbif_out"
    result = _run_prepare(real_output, output_dir)
    assert result.exit_code == 0, result.output

    with (output_dir / "media.csv").open(newline="") as f:
        rows = {row["mediaID"]: row for row in csv.DictReader(f)}

    assert rows["img_001"]["timestamp"] == "2021-06-15T08:30:00Z"
    assert rows["img_001"]["mediaComments"] == ""

    with (output_dir / "deployments.csv").open(newline="") as f:
        deployment_rows = {row["deploymentID"]: row for row in csv.DictReader(f)}
    # img_001 is the only image in 'train' with real EXIF -> it alone anchors the range.
    assert deployment_rows["train"]["deploymentStart"] == "2021-06-15T08:30:00Z"
    assert deployment_rows["train"]["deploymentEnd"] == "2021-06-15T08:30:00Z"


def test_prepare_fails_on_missing_source_dataset_dir(tmp_path: Path):
    result = runner.invoke(app, [
        "publish", "gbif", "prepare",
        "--source-dataset-dir", str(tmp_path / "does-not-exist"),
        "--output-dir", str(tmp_path / "out"),
    ])

    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_prepare_fails_when_output_dir_exists_without_overwrite(tmp_path: Path, example_source_dataset: Path):
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"

    first = _run_prepare(real_output, output_dir)
    assert first.exit_code == 0, first.output

    second = _run_prepare(real_output, output_dir)
    assert second.exit_code == 1
    assert "already exists" in second.output.lower()

    third = _run_prepare(real_output, output_dir, ["--overwrite"])
    assert third.exit_code == 0, third.output


def test_read_exif_datetime_returns_none_for_image_without_exif(example_source_dataset: Path):
    assert gbif_service.read_exif_datetime(example_source_dataset / "images" / "train" / "img_001.jpg") is None


# ── --upload-to-huggingface ──────────────────────────────────────────────────
#
# Only the validation paths are exercised here (missing HF_TOKEN, missing/
# placeholder repo_id) — both fail before donadataset.services.gbif ever
# calls huggingface_hub.upload_file, so nothing here touches the real
# HuggingFace Hub API.

def test_prepare_upload_to_huggingface_fails_without_token(
    tmp_path: Path, example_source_dataset: Path, monkeypatch,
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"

    result = _run_prepare(real_output, output_dir, [
        "--upload-to-huggingface", "--hf-repo-id", "someuser/somedataset",
    ])

    assert result.exit_code == 1
    assert "hf_token" in result.output.lower()
    # The archive itself is still generated locally before the upload attempt.
    assert (output_dir / "donadataset-camtrap-dp.zip").exists()


def test_prepare_upload_to_huggingface_fails_without_repo_id(
    tmp_path: Path, example_source_dataset: Path, monkeypatch,
):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    output_dir = tmp_path / "gbif_out"

    result = _run_prepare(real_output, output_dir, [
        "--upload-to-huggingface", "--hf-repo-id", "",
    ])

    assert result.exit_code == 1
    assert "repo_id" in result.output.lower()


# ── --link-media-to-huggingface ──────────────────────────────────────────────
#
# fetch_hfh_shard_urls is monkeypatched everywhere here so nothing touches the
# real HuggingFace Hub API — these tests only exercise how donadataset.services
# .gbif.build_camtrap_dp_resources consumes whatever mapping it's given.

def _fake_shard_urls_by_split(real_output: Path) -> dict[str, str]:
    """One fake .tar URL per split, assigned to every image_id in that split
    — mirrors what a real manifest.csv with a single shard per split would
    produce."""
    urls: dict[str, str] = {}
    for split in ("train", "val", "test"):
        shard_url = f"https://huggingface.co/datasets/x/resolve/main/data/{split}/{split}-00000.tar"
        for label_path in (real_output / "labels" / split).glob("*.txt"):
            urls[label_path.stem] = shard_url
    return urls


def test_prepare_link_media_to_huggingface_uses_shard_urls_per_split(
    tmp_path: Path, example_source_dataset: Path, monkeypatch,
):
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    monkeypatch.setattr(gbif_service, "fetch_hfh_shard_urls", lambda repo_id: _fake_shard_urls_by_split(real_output))

    output_dir = tmp_path / "gbif_out"
    result = _run_prepare(real_output, output_dir, ["--link-media-to-huggingface", "--hf-repo-id", "x"])
    assert result.exit_code == 0, result.output

    with (output_dir / "media.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        assert row["filePath"] == f"https://huggingface.co/datasets/x/resolve/main/data/{row['deploymentID']}/{row['deploymentID']}-00000.tar"
        assert "not an individually downloadable file" in row["mediaComments"]

    # Two images from the same split (train) share the same shard -> same filePath.
    train_paths = {row["filePath"] for row in rows if row["deploymentID"] == "train"}
    assert len(train_paths) == 1


def test_prepare_link_media_to_huggingface_fails_on_missing_manifest_entry(
    tmp_path: Path, example_source_dataset: Path, monkeypatch,
):
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    monkeypatch.setattr(gbif_service, "fetch_hfh_shard_urls", lambda repo_id: {})  # empty manifest

    output_dir = tmp_path / "gbif_out"
    result = _run_prepare(real_output, output_dir, ["--link-media-to-huggingface", "--hf-repo-id", "x"])

    assert result.exit_code == 1
    assert "no entry in the huggingface hub manifest" in result.output.lower()
