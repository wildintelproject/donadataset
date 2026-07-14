"""Integration tests for 'publish huggingface prepare', chained after 'generate real'.

`prepare` always renders the bundled Jinja2 template
(templates/hfh/HuggingFaceHub.yaml.j2) using CLI flags (whose defaults come
from settings.toml's HUGGINGFACE section) — there is no --config flag and no
custom YAML file involved at all.

`prepare` expects an already-clean YOLO dataset (one label per image, no
duplicate stems) — exactly what 'generate real' produces. Feeding it
examples/source_dataset directly would fail validation (it has a duplicate
stem and a missing label on purpose, to exercise 'generate real''s cleanup).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

from donadataset.main import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _generate_real_dataset(tmp_path: Path, example_source_dataset: Path) -> Path:
    real_output = tmp_path / "real"
    result = runner.invoke(app, [
        "generate", "real",
        "--source", str(example_source_dataset),
        "--output", str(real_output),
    ])
    assert result.exit_code == 0, result.output
    return real_output


def test_prepare_creates_full_export_from_generated_real_dataset(
    tmp_path: Path, example_source_dataset: Path,
):
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    donana_yaml = yaml.safe_load((real_output / "donana_filtered.yaml").read_text())
    hf_output_dir = tmp_path / "HFH"

    result = runner.invoke(app, [
        "publish", "huggingface", "prepare",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(hf_output_dir),
    ])

    assert result.exit_code == 0, result.output
    assert (hf_output_dir / "manifest.csv").exists()
    assert (hf_output_dir / "manifest-files-sha256.csv").exists()
    assert (hf_output_dir / "metadata.csv").exists()
    assert (hf_output_dir / "dataset_info.json").exists()
    assert (hf_output_dir / "donana.yaml").exists()
    assert (hf_output_dir / "checksums-sha256.txt").exists()
    assert (hf_output_dir / "README.md").exists()
    assert (hf_output_dir / "LICENSE").exists()
    assert (hf_output_dir / "CITATION.cff").exists()
    assert (hf_output_dir / "validation_report.json").exists()

    report = json.loads((hf_output_dir / "verification_report_local.json").read_text())
    assert report["status"] == "passed"

    info = json.loads((hf_output_dir / "dataset_info.json").read_text())
    assert info["num_images"] == 3 + 3 + 3  # train + val + test kept images
    assert len(info["classes"]) == donana_yaml["nc"]
    # classes.names isn't a flag -> always auto-detected from --source-dataset-dir
    assert info["classes"] == {str(k): v for k, v in donana_yaml["names"].items()}

    for split in ("train", "val", "test"):
        shards = list((hf_output_dir / "data" / split).glob("*.tar"))
        assert len(shards) == 1


def test_prepare_rejects_source_with_duplicate_stems_and_missing_labels(
    tmp_path: Path, example_source_dataset: Path,
):
    """examples/source_dataset is deliberately 'dirty' (see its own README) —
    feeding it straight to 'prepare' (skipping 'generate real') must fail
    validation rather than silently produce a broken export."""
    result = runner.invoke(app, [
        "publish", "huggingface", "prepare",
        "--source-dataset-dir", str(example_source_dataset),
        "--output-dir", str(tmp_path / "HFH"),
    ])

    assert result.exit_code != 0
    assert not (tmp_path / "HFH").exists()


def test_prepare_overwrite_flag_allows_rerun(tmp_path: Path, example_source_dataset: Path):
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    hf_output_dir = tmp_path / "HFH"
    override_flags = ["--source-dataset-dir", str(real_output), "--output-dir", str(hf_output_dir)]

    first = runner.invoke(app, ["publish", "huggingface", "prepare", *override_flags])
    assert first.exit_code == 0, first.output

    # fail_if_output_dir_exists is fixed to true in the template -> a bare
    # second run must fail, and only --overwrite can make it succeed.
    second_without_overwrite = runner.invoke(app, ["publish", "huggingface", "prepare", *override_flags])
    assert second_without_overwrite.exit_code != 0

    second_with_overwrite = runner.invoke(
        app, ["publish", "huggingface", "prepare", *override_flags, "--overwrite"],
    )
    assert second_with_overwrite.exit_code == 0, second_with_overwrite.output


def test_prepare_fails_when_classes_missing_and_source_has_no_names_yaml(tmp_path: Path):
    source_dir = tmp_path / "source-without-yaml"
    for split in ("train", "val", "test"):
        (source_dir / "images" / split).mkdir(parents=True)
        (source_dir / "labels" / split).mkdir(parents=True)

    result = runner.invoke(app, [
        "publish", "huggingface", "prepare",
        "--source-dataset-dir", str(source_dir),
        "--output-dir", str(tmp_path / "HFH"),
    ])

    assert result.exit_code == 1
    assert "classes.names must be a non-empty dictionary" in result.output


# ── templates/hfh/HuggingFaceHub.yaml.j2 (Jinja2 template rendered by 'prepare') ─

def test_prepare_renders_jinja_template_using_cli_flags(tmp_path: Path, example_source_dataset: Path):
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    hf_output_dir = tmp_path / "HFH"

    result = runner.invoke(app, [
        "publish", "huggingface", "prepare",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(hf_output_dir),
        "--dataset-slug", "jinja-demo",
        "--dataset-name", "JinjaDemo",
        "--version", "0.1.0",
        "--description", "Jinja template smoke test",
        "--repo-id", "myuser/jinja-demo",
        "--license-id", "MIT",
    ])

    assert result.exit_code == 0, result.output
    dataset_info = json.loads((hf_output_dir / "dataset_info.json").read_text())
    assert dataset_info["dataset_slug"] == "jinja-demo"
    assert dataset_info["dataset_name"] == "JinjaDemo"
    assert dataset_info["version"] == "0.1.0"
    donana_yaml = yaml.safe_load((real_output / "donana_filtered.yaml").read_text())
    assert dataset_info["classes"] == {str(k): v for k, v in donana_yaml["names"].items()}

    readme = (hf_output_dir / "README.md").read_text()
    assert "myuser/jinja-demo" in readme
    assert "mit" in readme.lower()

    license_text = (hf_output_dir / "LICENSE").read_text()
    assert "MIT" in license_text

    citation = yaml.safe_load((hf_output_dir / "CITATION.cff").read_text())
    assert citation["repository-artifact"] == "https://huggingface.co/datasets/myuser/jinja-demo"
    # author flags weren't passed explicitly -> falls back to settings.toml's HUGGINGFACE defaults
    assert citation["authors"][0]["given-names"] == "WildINTEL Spain"
    assert citation["authors"][0]["family-names"] == "WildINTEL"
    assert citation["authors"][0]["affiliation"] == "University of Huelva"


def test_prepare_jinja_template_keeps_placeholder_for_unset_repo_id(tmp_path: Path, example_source_dataset: Path):
    """'prepare' doesn't require a real repo_id (that's enforced later, at
    upload/download/release time) — so it still succeeds, but the
    rendered YAML embedded in the export must keep the placeholder visible
    rather than silently invent a value."""
    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    hf_output_dir = tmp_path / "HFH"

    env = dict(os.environ, HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    unset_result = _run_donadataset(["config", "set", "HUGGINGFACE.repo_id="], env)
    assert unset_result.returncode == 0, unset_result.stdout + unset_result.stderr

    result = _run_donadataset([
        "publish", "huggingface", "prepare",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(hf_output_dir),
        "--dataset-slug", "jinja-demo",
        # --repo-id intentionally omitted, and settings.toml's own repo_id just unset above
    ], env)

    assert result.returncode == 0, result.stdout + result.stderr
    rendered_config = yaml.safe_load((hf_output_dir / "HuggingFaceHub.yaml").read_text())
    assert rendered_config["huggingface"]["repo_id"] == "REPLACE_WITH_HF_USER/REPLACE_WITH_DATASET_SLUG"


def _run_donadataset(args: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "donadataset.main", *args],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )


def test_prepare_fills_placeholders_from_settings_toml_defaults(tmp_path: Path, example_source_dataset: Path):
    """settings.toml's HUGGINGFACE.* (set via 'config set'/'config wizard')
    becomes the visible default of 'prepare's flags — fixed at process start,
    so this must run as a real subprocess (a CliRunner call in this same
    pytest process would still see the module's already-imported defaults)."""
    env = dict(os.environ, HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir()

    for assignment in [
        "HUGGINGFACE.repo_id=myuser/persisted-slug",
        "HUGGINGFACE.license_id=CC0-1.0",
        "HUGGINGFACE.license_name=CC0 1.0 Universal",
        "HUGGINGFACE.license_url=https://creativecommons.org/publicdomain/zero/1.0/",
        "HUGGINGFACE.author_given_names=Ada",
        "HUGGINGFACE.author_family_names=Lovelace",
        "HUGGINGFACE.author_affiliation=Analytical Engines Inc.",
    ]:
        set_result = _run_donadataset(["config", "set", assignment], env)
        assert set_result.returncode == 0, set_result.stdout + set_result.stderr

    real_output = _generate_real_dataset(tmp_path, example_source_dataset)
    hf_output_dir = tmp_path / "HFH"

    result = _run_donadataset([
        "publish", "huggingface", "prepare",
        "--source-dataset-dir", str(real_output),
        "--output-dir", str(hf_output_dir),
        "--dataset-slug", "jinja-demo",
        # --repo-id and --license-id intentionally omitted -> settings.toml fills them
    ], env)

    assert result.returncode == 0, result.stdout + result.stderr
    citation = yaml.safe_load((hf_output_dir / "CITATION.cff").read_text())
    assert citation["authors"][0]["given-names"] == "Ada"
    assert citation["authors"][0]["family-names"] == "Lovelace"
    assert citation["license"] == "CC0-1.0"
    assert citation["repository-artifact"] == "https://huggingface.co/datasets/myuser/persisted-slug"
    license_text = (hf_output_dir / "LICENSE").read_text()
    assert "CC0 1.0 Universal" in license_text


def test_prepare_help_shows_settings_toml_defaults(tmp_path: Path):
    """--help must show the actual stored value (not a placeholder), so the
    user can see what will be used without having to run 'config get' first."""
    env = dict(os.environ, HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir()

    set_result = _run_donadataset(["config", "set", "HUGGINGFACE.repo_id=myuser/help-demo"], env)
    assert set_result.returncode == 0, set_result.stdout + set_result.stderr

    result = _run_donadataset(["publish", "huggingface", "prepare", "--help"], env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "myuser/help-demo" in result.stdout
