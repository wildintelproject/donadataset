"""Integration tests for the donadataset CLI wiring (typer app end-to-end)."""
from pathlib import Path

from typer.testing import CliRunner

from donadataset.main import app

runner = CliRunner()


def test_top_level_help_lists_generate_and_publish():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "publish" in result.output


def test_generate_help_lists_real_and_toy():
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "real" in result.output
    assert "toy" in result.output


def test_publish_help_lists_huggingface_and_zenodo():
    result = runner.invoke(app, ["publish", "--help"])
    assert result.exit_code == 0
    assert "huggingface" in result.output
    assert "zenodo" in result.output


def test_generate_real_rejects_invalid_split_choice():
    result = runner.invoke(app, ["generate", "real", "--split", "bogus"])
    assert result.exit_code != 0
    assert "bogus" in result.output


def test_generate_real_rejects_missing_source_dir(tmp_path: Path):
    missing_source = tmp_path / "does-not-exist"
    result = runner.invoke(app, [
        "generate", "real",
        "--source", str(missing_source),
        "--output", str(tmp_path / "out"),
    ])
    assert result.exit_code != 0


def test_generate_real_end_to_end(tmp_path: Path, example_source_dataset: Path):
    output = tmp_path / "out"

    result = runner.invoke(app, [
        "generate", "real",
        "--source", str(example_source_dataset),
        "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert (output / "donana_filtered.yaml").exists()


def test_generate_toy_end_to_end_after_real(tmp_path: Path, example_source_dataset: Path):
    real_output = tmp_path / "real"
    toy_output = tmp_path / "toy"

    real_result = runner.invoke(app, [
        "generate", "real",
        "--source", str(example_source_dataset),
        "--output", str(real_output),
    ])
    assert real_result.exit_code == 0, real_result.output

    toy_result = runner.invoke(app, [
        "generate", "toy",
        "--source", str(real_output),
        "--output", str(toy_output),
        "--samples-train", "1", "--samples-val", "1", "--samples-test", "1",
    ])

    assert toy_result.exit_code == 0, toy_result.output
    assert (toy_output / "donadatasetToy.yaml").exists()


def test_generate_real_verbose_by_default_shows_per_item_details(tmp_path: Path, example_source_dataset: Path):
    output = tmp_path / "out"

    result = runner.invoke(app, [
        "generate", "real",
        "--source", str(example_source_dataset),
        "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert "duplicadas" in result.output
    assert "missing label" in result.output


def test_generate_real_quiet_hides_per_item_details_but_keeps_summary(tmp_path: Path, example_source_dataset: Path):
    output = tmp_path / "out"

    result = runner.invoke(app, [
        "generate", "real",
        "--source", str(example_source_dataset),
        "--output", str(output),
        "--quiet",
    ])

    assert result.exit_code == 0, result.output
    assert "duplicadas" not in result.output
    assert "missing label" not in result.output
    assert "Dataset generado correctamente" in result.output
    assert (output / "donana_filtered.yaml").exists()


def test_generate_toy_quiet_hides_per_class_warnings_but_keeps_summary(
    tmp_path: Path, example_source_dataset: Path,
):
    real_output = tmp_path / "real"
    toy_output = tmp_path / "toy"
    runner.invoke(app, [
        "generate", "real",
        "--source", str(example_source_dataset),
        "--output", str(real_output),
    ])

    result = runner.invoke(app, [
        "generate", "toy",
        "--source", str(real_output),
        "--output", str(toy_output),
        # requesting far more than available triggers the per-class "Aviso" line
        "--samples-train", "100", "--samples-val", "100", "--samples-test", "100",
        "--quiet",
    ])

    assert result.exit_code == 0, result.output
    assert "Aviso" not in result.output
    assert "Dataset toy creado correctamente" in result.output


def test_generate_toy_verbose_by_default_shows_per_class_warnings(
    tmp_path: Path, example_source_dataset: Path,
):
    real_output = tmp_path / "real"
    toy_output = tmp_path / "toy"
    runner.invoke(app, [
        "generate", "real",
        "--source", str(example_source_dataset),
        "--output", str(real_output),
    ])

    result = runner.invoke(app, [
        "generate", "toy",
        "--source", str(real_output),
        "--output", str(toy_output),
        "--samples-train", "100", "--samples-val", "100", "--samples-test", "100",
    ])

    assert result.exit_code == 0, result.output
    assert "Aviso" in result.output


def test_publish_huggingface_help_lists_prepare_upload_download():
    result = runner.invoke(app, ["publish", "huggingface", "--help"])
    assert result.exit_code == 0
    assert "prepare" in result.output
    assert "upload" in result.output
    assert "download" in result.output


def test_publish_huggingface_prepare_missing_config_file_errors(tmp_path):
    missing_config = tmp_path / "does-not-exist.yaml"
    result = runner.invoke(app, [
        "publish", "huggingface", "prepare", "--config", str(missing_config),
    ])
    assert result.exit_code != 0


def test_publish_zenodo_help_lists_prepare_upload_download():
    result = runner.invoke(app, ["publish", "zenodo", "--help"])
    assert result.exit_code == 0
    assert "prepare" in result.output
    assert "upload" in result.output
    assert "download" in result.output
