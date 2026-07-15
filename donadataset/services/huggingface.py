"""Lógica de interacción con HuggingFace Hub (sin nada de CLI/Typer).

`run_export` crea una carpeta de export local (HFH/) lista para subir a
HuggingFace Hub a partir de un dataset YOLO (images/<split>/, labels/<split>/).

Espera un fichero de configuración YAML externo (puede tener cualquier
nombre). El dataset origen debe tener la estructura:

    source_dataset_dir/
      images/{train,val,test}/
      labels/{train,val,test}/

La carpeta de salida (normalmente HFH/) contiene:

    HFH/
      README.md
      LICENSE
      CITATION.cff
      HuggingFaceHub.yaml
      donana.yaml
      dataset_info.json
      metadata.csv
      manifest.csv
      manifest-files-sha256.csv
      checksums-sha256.txt
      validation_report.json
      verification_report_local.json
      data/
        train/*.tar
        val/*.tar
        test/*.tar

`run_upload` sube esa carpeta a un repositorio dataset de HuggingFace Hub, y
`run_download_and_verify` la descarga a un directorio temporal y verifica su
integridad (checksums globales + hashes internos de los tar) contra la misma
información generada por `run_export`.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import yaml
from huggingface_hub import HfApi, create_repo, snapshot_download, upload_folder, whoami
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from jinja2 import Template

from donadataset.config import REPO_ROOT
from donadataset.config import settings as global_settings
from donadataset.services.common import (
    as_bool,
    count_files,
    ensure_clean_dir,
    ensure_dict,
    fail,
    format_size,
    get_nested,
    load_yaml,
    read_checksums,
    remove_dir_if_exists,
    render_text_template,
    sha256_file,
    sha256_stream,
    total_size_bytes,
    utc_now_iso,
    write_json,
    write_yaml,
)

INTERNAL_CONFIG_FILENAME    = "HuggingFaceHub.yaml"
DEFAULT_OUTPUT_DIR_TEMPLATE = "HFH_Z_{dataset_slug}"
DEFAULT_SHARD_NAME_TEMPLATE = "{split}-{index:05d}.tar"

# Plantillas de contenido (prosa), separadas de la config .j2 (HuggingFaceHub.yaml.j2,
# que vive en la misma carpeta) — editables sin tocar este fichero.
TEMPLATES_DIR       = REPO_ROOT / "templates" / "hfh"
README_TEMPLATE_FILE = TEMPLATES_DIR / "README.md.j2"
LICENSE_TEMPLATE_FILE = TEMPLATES_DIR / "LICENSE.j2"
CITATION_TEMPLATE_FILE = TEMPLATES_DIR / "CITATION.cff.j2"


@dataclass
class DatasetItem:
    split: str
    image_abs_path: Path
    label_abs_path: Path
    image_rel_path: str
    label_rel_path: str
    image_id: str
    image_size_bytes: int
    label_size_bytes: int
    image_sha256: str
    label_sha256: str
    num_objects: int
    classes_present: List[int]
    shard_rel_path: str = ""


@dataclass
class ValidationIssue:
    level: str
    message: str
    path: str = ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_size_gb(value: Any) -> int:
    return int(float(value) * 1024 * 1024 * 1024)


def get_dataset_slug(config: Dict[str, Any]) -> str:
    return str(get_nested(config, ["project", "dataset_slug"], "donadataset"))


def get_dataset_name(config: Dict[str, Any]) -> str:
    return str(get_nested(config, ["project", "dataset_name"], "DonaDataset"))


def get_version(config: Dict[str, Any]) -> str:
    return str(get_nested(config, ["project", "version"], "1.0.0"))


def format_config_template(template: str, config: Dict[str, Any]) -> str:
    return template.format(
        dataset_slug=get_dataset_slug(config),
        dataset_name=get_dataset_name(config),
        version=get_version(config),
    )


def get_internal_config_filename(config: Dict[str, Any]) -> str:
    return str(
        get_nested(
            config,
            ["export", "internal_config_filename"],
            INTERNAL_CONFIG_FILENAME,
        )
    )


def get_output_dir(config: Dict[str, Any]) -> Path:
    """
    Return the export directory used by prepare/upload/download alike.

    `paths.output_dir`, if set, is an explicit override and wins outright.
    Otherwise the directory is derived from `export.output_dir_template`
    (dataset name/slug/version placeholders), falling back to
    DEFAULT_OUTPUT_DIR_TEMPLATE. Internal file names remain constant so
    downstream commands do not depend on the external config filename or on
    the dataset name.
    """
    configured_output_dir = get_nested(config, ["paths", "output_dir"], None)
    if configured_output_dir:
        return Path(str(configured_output_dir))

    output_dir_template = str(
        get_nested(
            config,
            ["export", "output_dir_template"],
            DEFAULT_OUTPUT_DIR_TEMPLATE,
        )
    )
    return Path(format_config_template(output_dir_template, config))


def get_shard_name(split: str, index: int, config: Dict[str, Any]) -> str:
    """
    Return a shard filename using the fixed internal naming policy.

    Shard filenames intentionally do not include the dataset name. The dataset
    identity is carried by the export directory, README, dataset_info.json, and
    Hugging Face repository metadata.
    """
    return DEFAULT_SHARD_NAME_TEMPLATE.format(split=split, index=index)


def get_split_names(config: Dict[str, Any]) -> List[str]:
    return [
        str(get_nested(config, ["splits", "train"], "train")),
        str(get_nested(config, ["splits", "validation"], "val")),
        str(get_nested(config, ["splits", "test"], "test")),
    ]


def get_source_dataset_dir(config: Dict[str, Any]) -> Path:
    return Path(get_nested(config, ["paths", "source_dataset_dir"], "./donadataset"))


def find_source_names_yaml(source_dataset_dir: Path) -> Optional[Path]:
    """Looks for a YOLO-style YAML (as written by 'generate real'/'generate toy',
    e.g. donana_filtered.yaml) directly inside source_dataset_dir with a
    non-empty top-level 'names' mapping."""
    if not source_dataset_dir.is_dir():
        return None

    candidates = sorted(source_dataset_dir.glob("*.yaml")) + sorted(source_dataset_dir.glob("*.yml"))
    for candidate in candidates:
        try:
            data = load_yaml(candidate)
        except Exception:
            continue
        if isinstance(data.get("names"), dict) and data["names"]:
            return candidate

    return None


def get_classes(config: Dict[str, Any]) -> Dict[int, str]:
    names = get_nested(config, ["classes", "names"], {})

    if not isinstance(names, dict) or not names:
        source_dataset_dir = get_source_dataset_dir(config)
        source_yaml = find_source_names_yaml(source_dataset_dir)
        if source_yaml is not None:
            logging.info("classes.names not set in config; using 'names' from %s", source_yaml)
            names = load_yaml(source_yaml)["names"]

    if not isinstance(names, dict) or not names:
        raise ValueError(
            "classes.names must be a non-empty dictionary. Set it explicitly in the config, "
            f"or make sure a YOLO YAML with a 'names' mapping exists directly inside "
            f"paths.source_dataset_dir ({get_source_dataset_dir(config)})."
        )

    result: Dict[int, str] = {}
    for key, value in names.items():
        result[int(key)] = str(value)
    return dict(sorted(result.items(), key=lambda x: x[0]))


def get_chunk_size_bytes(config: Dict[str, Any]) -> int:
    return int(get_nested(config, ["hashing", "chunk_size_mb"], 8)) * 1024 * 1024


def get_image_extensions(config: Dict[str, Any]) -> List[str]:
    extensions = get_nested(config, ["image_files", "allowed_extensions"], [".jpg", ".jpeg", ".png"])
    case_sensitive = as_bool(get_nested(config, ["image_files", "case_sensitive_extensions"], False))
    return [str(ext if case_sensitive else ext.lower()) for ext in extensions]


def is_image_file(path: Path, config: Dict[str, Any]) -> bool:
    case_sensitive = as_bool(get_nested(config, ["image_files", "case_sensitive_extensions"], False))
    suffix = path.suffix if case_sensitive else path.suffix.lower()
    return suffix in get_image_extensions(config)


def setup_logging(config: Dict[str, Any], output_dir: Optional[Path] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if output_dir and as_bool(get_nested(config, ["export", "create_logs"], True)):
        log_filename = str(get_nested(config, ["export", "log_filename"], "hfh_export.log"))
        handlers.append(logging.FileHandler(output_dir / log_filename, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def validate_source_structure(config: Dict[str, Any]) -> None:
    source_dataset_dir = get_source_dataset_dir(config)
    if not source_dataset_dir.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source_dataset_dir}")

    expected_image_dirs = get_nested(
        config,
        ["dataset_structure", "expected_image_dirs"],
        ["images/train", "images/val", "images/test"],
    )
    expected_label_dirs = get_nested(
        config,
        ["dataset_structure", "expected_label_dirs"],
        ["labels/train", "labels/val", "labels/test"],
    )

    for rel_dir in list(expected_image_dirs) + list(expected_label_dirs):
        path = source_dataset_dir / str(rel_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Expected dataset directory does not exist: {path}")


def find_images(images_dir: Path, config: Dict[str, Any]) -> List[Path]:
    recursive = as_bool(get_nested(config, ["image_files", "recursive_scan"], False))
    candidates = images_dir.rglob("*") if recursive else images_dir.iterdir()
    return sorted(
        [path for path in candidates if path.is_file() and is_image_file(path, config)],
        key=lambda p: p.as_posix().lower(),
    )


def find_duplicate_stems(paths: Iterable[Path]) -> List[str]:
    seen = set()
    duplicates = set()
    for path in paths:
        if path.stem in seen:
            duplicates.add(path.stem)
        seen.add(path.stem)
    return sorted(duplicates)


def parse_yolo_label(label_path: Path, classes: Dict[int, str], config: Dict[str, Any]) -> Tuple[int, List[int], List[ValidationIssue]]:
    issues: List[ValidationIssue] = []
    classes_present: List[int] = []

    allow_empty = as_bool(get_nested(config, ["label_files", "allow_empty_label_files"], True))
    validate_class_ids = as_bool(get_nested(config, ["validation", "validate_class_ids"], True))
    validate_coordinates = as_bool(get_nested(config, ["validation", "validate_normalized_coordinates"], True))
    coord_min = float(get_nested(config, ["validation", "coordinate_min_value"], 0.0))
    coord_max = float(get_nested(config, ["validation", "coordinate_max_value"], 1.0))

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        if allow_empty:
            return 0, [], issues
        return 0, [], [ValidationIssue("error", "Empty label file is not allowed.", str(label_path))]

    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            issues.append(ValidationIssue("error", f"Invalid YOLO row at line {line_number}: expected 5 values.", str(label_path)))
            continue

        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append(ValidationIssue("error", f"Invalid numeric value at line {line_number}.", str(label_path)))
            continue

        if validate_class_ids and class_id not in classes:
            issues.append(ValidationIssue("error", f"Class id {class_id} is not defined in classes.names.", str(label_path)))

        if validate_coordinates:
            for name, value in [
                ("x_center", x_center),
                ("y_center", y_center),
                ("width", width),
                ("height", height),
            ]:
                if value < coord_min or value > coord_max:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"Coordinate {name}={value} at line {line_number} is outside [{coord_min}, {coord_max}].",
                            str(label_path),
                        )
                    )
            if width <= 0 or height <= 0:
                issues.append(ValidationIssue("error", f"Width and height must be positive at line {line_number}.", str(label_path)))

        classes_present.append(class_id)

    return len(classes_present), sorted(set(classes_present)), issues


def scan_dataset(config: Dict[str, Any]) -> Tuple[List[DatasetItem], List[ValidationIssue]]:
    source_dataset_dir = get_source_dataset_dir(config)
    images_folder = str(get_nested(config, ["dataset_structure", "images_folder_name"], "images"))
    labels_folder = str(get_nested(config, ["dataset_structure", "labels_folder_name"], "labels"))
    label_extension = str(get_nested(config, ["label_files", "extension"], ".txt"))
    require_label = as_bool(get_nested(config, ["label_files", "require_label_for_each_image"], True))
    allow_orphan_label = as_bool(get_nested(config, ["label_files", "allow_label_without_image"], False))

    classes = get_classes(config)
    chunk_size_bytes = get_chunk_size_bytes(config)

    items: List[DatasetItem] = []
    issues: List[ValidationIssue] = []

    for split in get_split_names(config):
        image_dir = source_dataset_dir / images_folder / split
        label_dir = source_dataset_dir / labels_folder / split

        images = find_images(image_dir, config)
        image_stems = {image.stem for image in images}
        labels = sorted(label_dir.glob(f"*{label_extension}"), key=lambda p: p.as_posix().lower())
        label_stems = {label.stem for label in labels}

        for duplicate_stem in find_duplicate_stems(images):
            issues.append(ValidationIssue("error", f"Duplicate image stem in split '{split}': {duplicate_stem}", str(image_dir)))

        for stem in sorted(image_stems - label_stems):
            level = "error" if require_label else "warning"
            issues.append(ValidationIssue(level, f"Image without label in split '{split}': {stem}", str(image_dir / stem)))

        if not allow_orphan_label:
            for stem in sorted(label_stems - image_stems):
                issues.append(ValidationIssue("error", f"Label without image in split '{split}': {stem}", str(label_dir / f"{stem}{label_extension}")))

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}{label_extension}"
            if not label_path.exists():
                continue

            num_objects, classes_present, label_issues = parse_yolo_label(label_path, classes, config)
            issues.extend(label_issues)

            image_rel_path = f"{images_folder}/{split}/{image_path.name}"
            label_rel_path = f"{labels_folder}/{split}/{label_path.name}"

            items.append(
                DatasetItem(
                    split=split,
                    image_abs_path=image_path,
                    label_abs_path=label_path,
                    image_rel_path=image_rel_path,
                    label_rel_path=label_rel_path,
                    image_id=image_path.stem,
                    image_size_bytes=image_path.stat().st_size,
                    label_size_bytes=label_path.stat().st_size,
                    image_sha256=sha256_file(image_path, chunk_size_bytes),
                    label_sha256=sha256_file(label_path, chunk_size_bytes),
                    num_objects=num_objects,
                    classes_present=classes_present,
                )
            )

    return items, issues


def stop_if_validation_errors(issues: List[ValidationIssue]) -> None:
    errors = [issue for issue in issues if issue.level == "error"]
    if not errors:
        return
    for issue in errors[:50]:
        logging.error("%s | %s", issue.message, issue.path)
    if len(errors) > 50:
        logging.error("Additional validation errors not shown: %d", len(errors) - 50)
    raise RuntimeError(f"Dataset validation failed with {len(errors)} error(s).")


def prepare_output_dir(config: Dict[str, Any], config_path: Path) -> Path:
    output_dir = get_output_dir(config)
    overwrite = as_bool(get_nested(config, ["export", "overwrite_output_dir"], False))
    fail_if_exists = as_bool(get_nested(config, ["export", "fail_if_output_dir_exists"], True))

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif fail_if_exists:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Delete it yourself, or re-run with --overwrite to let 'prepare' delete and recreate it."
            )

    ensure_dir(output_dir)
    for split in get_split_names(config):
        ensure_dir(output_dir / "data" / split)

    # The external YAML can have any name and is not copied with that name.
    # The export contains one constant internal configuration filename so
    # downstream scripts can always find the same file.
    write_yaml(output_dir / get_internal_config_filename(config), config)

    return output_dir


def create_shards(items: List[DatasetItem], output_dir: Path, config: Dict[str, Any]) -> List[str]:
    archive_format = str(get_nested(config, ["sharding", "archive_format"], "tar"))
    compression = str(get_nested(config, ["sharding", "compression"], "none"))
    if archive_format != "tar" or compression != "none":
        raise ValueError("Only archive_format='tar' and compression='none' are currently supported.")

    max_shard_size_bytes = parse_size_gb(get_nested(config, ["sharding", "max_shard_size_gb"], 2.0))
    created_shards: List[str] = []

    for split in get_split_names(config):
        split_items = [item for item in items if item.split == split]
        if not split_items:
            logging.warning("No items found for split: %s", split)
            continue

        shard_index = 0
        current_size = 0
        current_tar: Optional[tarfile.TarFile] = None
        current_shard_rel = ""

        def open_new_shard() -> None:
            nonlocal shard_index, current_size, current_tar, current_shard_rel
            if current_tar is not None:
                current_tar.close()
            shard_name = get_shard_name(split, shard_index, config)
            shard_path = output_dir / "data" / split / shard_name
            current_shard_rel = shard_path.relative_to(output_dir).as_posix()
            current_tar = tarfile.open(shard_path, "w")
            current_size = 0
            created_shards.append(current_shard_rel)
            logging.info("Creating shard: %s", current_shard_rel)
            shard_index += 1

        open_new_shard()

        for item in split_items:
            pair_size = item.image_size_bytes + item.label_size_bytes
            if current_size > 0 and current_size + pair_size > max_shard_size_bytes:
                open_new_shard()
            assert current_tar is not None
            current_tar.add(item.image_abs_path, arcname=item.image_rel_path)
            current_tar.add(item.label_abs_path, arcname=item.label_rel_path)
            item.shard_rel_path = current_shard_rel
            current_size += pair_size

        if current_tar is not None:
            current_tar.close()

    return created_shards


def classes_to_text(class_ids: List[int], classes: Dict[int, str]) -> str:
    return ";".join(classes.get(class_id, str(class_id)) for class_id in class_ids)


def write_manifest_csv(items: List[DatasetItem], output_dir: Path, config: Dict[str, Any]) -> Path:
    classes = get_classes(config)
    path = output_dir / str(get_nested(config, ["manifest", "manifest_filename"], "manifest.csv"))
    fieldnames = [
        "image_id", "split", "image_path", "label_path", "shard",
        "image_size_bytes", "label_size_bytes", "image_sha256", "label_sha256",
        "num_objects", "class_ids_present", "classes_present",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({
                "image_id": item.image_id,
                "split": item.split,
                "image_path": item.image_rel_path,
                "label_path": item.label_rel_path,
                "shard": item.shard_rel_path,
                "image_size_bytes": item.image_size_bytes,
                "label_size_bytes": item.label_size_bytes,
                "image_sha256": item.image_sha256,
                "label_sha256": item.label_sha256,
                "num_objects": item.num_objects,
                "class_ids_present": ";".join(str(x) for x in item.classes_present),
                "classes_present": classes_to_text(item.classes_present, classes),
            })
    return path


def write_file_hash_manifest_csv(items: List[DatasetItem], output_dir: Path, config: Dict[str, Any]) -> Path:
    path = output_dir / str(get_nested(config, ["manifest", "file_hash_manifest_filename"], "manifest-files-sha256.csv"))
    fieldnames = ["split", "file_type", "relative_path", "sha256", "size_bytes", "shard"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({
                "split": item.split,
                "file_type": "image",
                "relative_path": item.image_rel_path,
                "sha256": item.image_sha256,
                "size_bytes": item.image_size_bytes,
                "shard": item.shard_rel_path,
            })
            writer.writerow({
                "split": item.split,
                "file_type": "label",
                "relative_path": item.label_rel_path,
                "sha256": item.label_sha256,
                "size_bytes": item.label_size_bytes,
                "shard": item.shard_rel_path,
            })
    return path


def write_metadata_csv(items: List[DatasetItem], output_dir: Path, config: Dict[str, Any]) -> Path:
    classes = get_classes(config)
    path = output_dir / str(get_nested(config, ["manifest", "metadata_filename"], "metadata.csv"))
    fieldnames = ["image_id", "split", "image_path", "label_path", "shard", "num_objects", "class_ids_present", "classes_present"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({
                "image_id": item.image_id,
                "split": item.split,
                "image_path": item.image_rel_path,
                "label_path": item.label_rel_path,
                "shard": item.shard_rel_path,
                "num_objects": item.num_objects,
                "class_ids_present": ";".join(str(x) for x in item.classes_present),
                "classes_present": classes_to_text(item.classes_present, classes),
            })
    return path


def write_yolo_yaml(output_dir: Path, config: Dict[str, Any]) -> Path:
    path = output_dir / str(get_nested(config, ["output_files", "yolo_yaml_filename"], "donana.yaml"))
    train_split, val_split, test_split = get_split_names(config)
    data = {
        "path": ".",
        "train": f"images/{train_split}",
        "val": f"images/{val_split}",
        "test": f"images/{test_split}",
        "nc": len(get_classes(config)),
        "names": get_classes(config),
    }
    write_yaml(path, data)
    return path


def write_dataset_info_json(items: List[DatasetItem], created_shards: List[str], output_dir: Path, config: Dict[str, Any]) -> Path:
    path = output_dir / str(get_nested(config, ["output_files", "dataset_info_filename"], "dataset_info.json"))
    splits: Dict[str, Dict[str, Any]] = {}
    for split in get_split_names(config):
        split_items = [item for item in items if item.split == split]
        split_shards = sorted({item.shard_rel_path for item in split_items})
        splits[split] = {
            "num_images": len(split_items),
            "num_labels": len(split_items),
            "num_shards": len(split_shards),
            "size_bytes_original_files": sum(item.image_size_bytes + item.label_size_bytes for item in split_items),
            "shards": split_shards,
        }
    data = {
        "dataset_name": get_dataset_name(config),
        "dataset_slug": get_dataset_slug(config),
        "version": get_version(config),
        "task": str(get_nested(config, ["project", "task"], "object-detection")),
        "annotation_format": str(get_nested(config, ["project", "annotation_format"], "YOLO")),
        "generated_at_utc": utc_now_iso(),
        "num_images": len(items),
        "num_labels": len(items),
        "num_shards": len(created_shards),
        "classes": {str(k): v for k, v in get_classes(config).items()},
        "splits": splits,
    }
    write_json(path, data)
    return path


def write_readme(output_dir: Path, config: Dict[str, Any], items: List[DatasetItem]) -> Path:
    path = output_dir / str(get_nested(config, ["output_files", "readme_filename"], "README.md"))
    dataset_name = get_dataset_name(config)
    version = get_version(config)
    description = str(get_nested(config, ["project", "description"], "YOLO object detection dataset."))
    repo_id = str(get_nested(config, ["huggingface", "repo_id"], "REPLACE_WITH_HF_USER/donadataset"))
    license_id = str(get_nested(config, ["license", "license_id"], "CC-BY-4.0"))
    repository_code = get_nested(config, ["citation", "repository_code"], None)
    if not repository_code or "REPLACE_WITH" in str(repository_code):
        repository_code = None
    classes = get_classes(config)

    splits = [
        {"name": split, "count": len([item for item in items if item.split == split])}
        for split in get_split_names(config)
    ]
    class_list = [{"id": class_id, "name": class_name} for class_id, class_name in classes.items()]

    text = render_text_template(
        README_TEMPLATE_FILE,
        dataset_name=dataset_name,
        version=version,
        description=description,
        repo_id=repo_id,
        license_id=license_id,
        repository_code=repository_code,
        splits=splits,
        classes=class_list,
    )
    path.write_text(text, encoding="utf-8")
    return path


def write_license(output_dir: Path, config: Dict[str, Any]) -> Path:
    path = output_dir / str(get_nested(config, ["output_files", "license_filename"], "LICENSE"))
    license_id = str(get_nested(config, ["license", "license_id"], "CC-BY-4.0"))
    license_name = str(get_nested(config, ["license", "license_name"], "Creative Commons Attribution 4.0 International"))
    license_url = str(get_nested(config, ["license", "license_url"], "https://creativecommons.org/licenses/by/4.0/"))
    text = render_text_template(
        LICENSE_TEMPLATE_FILE,
        license_id=license_id,
        license_name=license_name,
        license_url=license_url,
    )
    path.write_text(text, encoding="utf-8")
    return path


def write_citation(output_dir: Path, config: Dict[str, Any]) -> Path:
    path = output_dir / str(get_nested(config, ["output_files", "citation_filename"], "CITATION.cff"))
    authors_raw = get_nested(config, ["citation", "authors"], [])
    authors = []
    if isinstance(authors_raw, list):
        for author in authors_raw:
            if isinstance(author, dict):
                authors.append({
                    "given_names": str(author.get("given_names", "")),
                    "family_names": str(author.get("family_names", "")),
                    "affiliation": str(author.get("affiliation", "")),
                })

    def _optional(config_key: str) -> Optional[str]:
        value = get_nested(config, ["citation", config_key], None)
        return str(value) if value else None

    text = render_text_template(
        CITATION_TEMPLATE_FILE,
        cff_version=str(get_nested(config, ["citation", "cff_version"], "1.2.0")),
        title=str(get_nested(config, ["citation", "title"], get_dataset_name(config))),
        message=str(get_nested(config, ["citation", "message"], "If you use this dataset, please cite it as below.")),
        citation_type=str(get_nested(config, ["citation", "type"], "dataset")),
        authors=authors,
        version=get_version(config),
        date_released=str(get_nested(config, ["project", "creation_date"], datetime.now().date().isoformat())),
        license_id=str(get_nested(config, ["license", "license_id"], "CC-BY-4.0")),
        repository_code=_optional("repository_code"),
        repository_artifact=_optional("repository_artifact"),
        doi=_optional("doi"),
    )
    path.write_text(text, encoding="utf-8")
    return path


def collect_files_for_checksums(output_dir: Path, config: Dict[str, Any]) -> List[Path]:
    """
    Collect all files that must be included in checksums-sha256.txt.

    Files that can change while the script is still running are excluded.
    This includes log files and verification reports.
    """

    checksum_filename = str(
        get_nested(config, ["checksums", "checksum_filename"], "checksums-sha256.txt")
    )
    local_report_filename = str(
        get_nested(config, ["output_files", "local_verification_report_filename"], "verification_report_local.json")
    )
    downloaded_report_filename = str(
        get_nested(config, ["output_files", "downloaded_verification_report_filename"], "verification_report_downloaded.json")
    )
    log_filename = str(
        get_nested(config, ["export", "log_filename"], "hfh_export.log")
    )

    excluded_files = {
        checksum_filename,
        local_report_filename,
        downloaded_report_filename,
        log_filename,
    }

    files: List[Path] = []

    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(output_dir).as_posix()

        if rel in excluded_files:
            continue

        if rel.endswith(".log"):
            continue

        files.append(path)

    return files


def write_checksums(output_dir: Path, config: Dict[str, Any]) -> Path:
    path = output_dir / str(get_nested(config, ["checksums", "checksum_filename"], "checksums-sha256.txt"))
    chunk_size_bytes = get_chunk_size_bytes(config)
    files = collect_files_for_checksums(output_dir, config)
    with path.open("w", encoding="utf-8") as f:
        for file_path in files:
            rel = file_path.relative_to(output_dir).as_posix()
            f.write(f"{sha256_file(file_path, chunk_size_bytes)}  {rel}\n")
    return path


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """
    Read a CSV file and return its rows as dictionaries.

    This helper is used by the local export verification stage to inspect
    manifest.csv and manifest-files-sha256.csv.
    """
    rows: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    return rows


def read_file_hash_manifest(path: Path) -> List[Dict[str, str]]:
    """
    Read manifest-files-sha256.csv.

    This file must contain one row per internal file stored inside the tar
    shards. Each image and each label must have its own row.
    """
    return read_csv_rows(path)


def verify_global_checksums(
    output_dir: Path, config: Dict[str, Any], verify_data: bool = True,
) -> Tuple[int, List[str]]:
    """
    Verify checksums-sha256.txt.

    For every file listed in checksums-sha256.txt, this function recomputes
    the SHA256 hash and compares it with the expected value. checksums-sha256.txt
    also lists the data/<split>/*.tar shards themselves (not just metadata) —
    verify_data=False skips those entries too, since callers that pass it
    also skip downloading the shards in the first place (see
    download_repository).
    """
    errors: List[str] = []
    verified = 0

    checksum_filename = str(get_nested(config, ["checksums", "checksum_filename"], "checksums-sha256.txt"))
    checksum_path = output_dir / checksum_filename

    if not checksum_path.exists():
        return 0, [f"Missing checksum file: {checksum_filename}"]

    expected = read_checksums(checksum_path)
    chunk_size_bytes = get_chunk_size_bytes(config)

    if not expected:
        errors.append(f"Checksum file is empty: {checksum_filename}")

    for rel, expected_digest in expected.items():
        if not verify_data and (rel == "data" or rel.startswith("data/")):
            continue

        file_path = output_dir / rel

        if not file_path.exists():
            errors.append(f"Missing file listed in checksums: {rel}")
            continue

        actual_digest = sha256_file(file_path, chunk_size_bytes)

        if actual_digest != expected_digest:
            errors.append(f"Checksum mismatch for {rel}: expected {expected_digest}, got {actual_digest}")

        verified += 1

    return verified, errors


def verify_tar_internal_hashes(output_dir: Path, config: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    Verify internal tar members and internal file hashes.

    This function opens each tar shard referenced by manifest-files-sha256.csv,
    checks that every expected internal file exists, reads it as a stream, computes
    its SHA256 hash, and compares it with the expected hash.
    """
    errors: List[str] = []
    verified = 0

    manifest_filename = str(
        get_nested(config, ["manifest", "file_hash_manifest_filename"], "manifest-files-sha256.csv")
    )
    manifest_path = output_dir / manifest_filename

    if not manifest_path.exists():
        return 0, [f"Missing file hash manifest: {manifest_filename}"]

    rows = read_file_hash_manifest(manifest_path)
    chunk_size_bytes = get_chunk_size_bytes(config)

    rows_by_shard: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        shard = row.get("shard", "")
        if not shard:
            errors.append(f"Row without shard in {manifest_filename}: {row}")
            continue
        rows_by_shard.setdefault(shard, []).append(row)

    for shard_rel, shard_rows in rows_by_shard.items():
        shard_path = output_dir / shard_rel

        if not shard_path.exists():
            errors.append(f"Missing shard: {shard_rel}")
            continue

        try:
            with tarfile.open(shard_path, "r") as tar:
                members = {member.name: member for member in tar.getmembers()}

                for row in shard_rows:
                    rel_path = row.get("relative_path", "")
                    expected_digest = row.get("sha256", "")

                    if rel_path not in members:
                        errors.append(f"Missing tar member {rel_path} in {shard_rel}")
                        continue

                    member = members[rel_path]
                    extracted = tar.extractfile(member)

                    if extracted is None:
                        errors.append(f"Could not read tar member {rel_path} in {shard_rel}")
                        continue

                    actual_digest = sha256_stream(extracted, chunk_size_bytes)

                    if actual_digest != expected_digest:
                        errors.append(
                            f"Internal hash mismatch for {rel_path} in {shard_rel}: "
                            f"expected {expected_digest}, got {actual_digest}"
                        )

                    verified += 1

        except tarfile.TarError as exc:
            errors.append(f"Could not open tar shard {shard_rel}: {exc}")

    return verified, errors


def write_validation_report(output_dir: Path, issues: List[ValidationIssue]) -> Path:
    path = output_dir / "validation_report.json"
    write_json(path, {
        "generated_at_utc": utc_now_iso(),
        "num_issues": len(issues),
        "num_errors": len([issue for issue in issues if issue.level == "error"]),
        "num_warnings": len([issue for issue in issues if issue.level == "warning"]),
        "issues": [issue.__dict__ for issue in issues],
    })
    return path


def write_local_verification_report(output_dir: Path, config: Dict[str, Any]) -> Path:
    path = output_dir / str(get_nested(config, ["output_files", "local_verification_report_filename"], "verification_report_local.json"))
    global_count, global_errors = verify_global_checksums(output_dir, config)
    internal_count, internal_errors = verify_tar_internal_hashes(output_dir, config)
    errors = global_errors + internal_errors
    write_json(path, {
        "generated_at_utc": utc_now_iso(),
        "status": "passed" if not errors else "failed",
        "global_files_verified": global_count,
        "internal_tar_members_verified": internal_count,
        "num_errors": len(errors),
        "errors": errors,
    })
    if errors and as_bool(get_nested(config, ["local_verification", "stop_on_verification_error"], True)):
        for error in errors[:50]:
            logging.error(error)
        if len(errors) > 50:
            logging.error("Additional verification errors not shown: %d", len(errors) - 50)
        raise RuntimeError(f"Local verification failed with {len(errors)} error(s).")
    return path


def load_config_source(config_path: Path, **template_context: Any) -> Dict[str, Any]:
    """Loads a plain YAML config, or renders a Jinja2 template first if
    config_path ends in .j2 (e.g. templates/hfh/HuggingFaceHub.yaml.j2). The
    template_context values are the same overrides 'prepare' already accepts
    as CLI flags (dataset_slug, repo_id...), so a .j2 template can be used
    directly as --config without hand-editing a YAML copy of it first."""
    if config_path.suffix.lower() != ".j2":
        return load_yaml(config_path)

    rendered = Template(config_path.read_text(encoding="utf-8")).render(**template_context)
    data = yaml.safe_load(rendered)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML configuration rendered from template: {config_path}")
    return data


def run_export(
    config_path: Path,
    *,
    source_dataset_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    dataset_slug: Optional[str] = None,
    dataset_name: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None,
    repo_id: Optional[str] = None,
    license_id: Optional[str] = None,
    license_name: Optional[str] = None,
    license_url: Optional[str] = None,
    author_given_names: Optional[str] = None,
    author_family_names: Optional[str] = None,
    author_affiliation: Optional[str] = None,
    message: Optional[str] = None,
    repository_code: Optional[str] = None,
    overwrite_output_dir: Optional[bool] = None,
) -> None:
    """config_path is always templates/hfh/HuggingFaceHub.yaml.j2 (the only
    HFH config source — see commands.huggingface) — every value below is
    resolved entirely through Jinja rendering, there is no separate override
    step: what you pass here (or its settings.toml-derived default) is what
    ends up in the rendered YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config_source(
        config_path,
        source_dataset_dir=source_dataset_dir,
        output_dir=output_dir,
        dataset_slug=dataset_slug,
        dataset_name=dataset_name,
        version=version,
        description=description,
        repo_id=repo_id,
        license_id=license_id,
        license_name=license_name,
        license_url=license_url,
        author_given_names=author_given_names,
        author_family_names=author_family_names,
        author_affiliation=author_affiliation,
        message=message,
        repository_code=repository_code,
        overwrite_output_dir=overwrite_output_dir,
    )
    setup_logging(config)

    logging.info("Validating source dataset structure...")
    validate_source_structure(config)

    logging.info("Scanning source dataset, validating labels, and computing original file hashes...")
    items, issues = scan_dataset(config)
    stop_if_validation_errors(issues)
    logging.info("Valid dataset items: %d", len(items))

    logging.info("Preparing output directory...")
    output_dir = prepare_output_dir(config, config_path)
    setup_logging(config, output_dir)
    write_validation_report(output_dir, issues)

    logging.info("Creating tar shards...")
    created_shards = create_shards(items, output_dir, config)
    logging.info("Created shards: %d", len(created_shards))

    logging.info("Writing manifests and metadata...")
    write_manifest_csv(items, output_dir, config)
    write_file_hash_manifest_csv(items, output_dir, config)
    write_metadata_csv(items, output_dir, config)

    logging.info("Writing dataset documentation files...")
    write_yolo_yaml(output_dir, config)
    write_dataset_info_json(items, created_shards, output_dir, config)
    write_readme(output_dir, config, items)
    write_license(output_dir, config)
    write_citation(output_dir, config)

    logging.info("Writing global SHA256 checksums...")
    write_checksums(output_dir, config)

    logging.info("Running local export verification...")
    write_local_verification_report(output_dir, config)

    logging.info("HFH export completed successfully: %s", output_dir.resolve())


# ── Utilidades compartidas por upload y download ─────────────────────────────

def get_repo_id(config: Dict[str, Any]) -> str:
    repo_id = str(get_nested(config, ["huggingface", "repo_id"], "")).strip()

    if not repo_id or "REPLACE_WITH" in repo_id:
        fail(
            "Invalid huggingface.repo_id in the configuration YAML. "
            "Example: myuser/donadataset"
        )
    if "/" not in repo_id:
        fail(
            "huggingface.repo_id must have the form 'user_or_org/dataset_name'. "
            f"Current value: {repo_id}"
        )
    return repo_id


def get_repo_type(config: Dict[str, Any]) -> str:
    repo_type = str(get_nested(config, ["huggingface", "repo_type"], "dataset")).strip()
    if repo_type != "dataset":
        fail(
            "Only repo_type='dataset' is supported for Hugging Face Hub repositories. "
            f"Got repo_type='{repo_type}'."
        )
    return repo_type


def get_token_env_var(config: Dict[str, Any]) -> str:
    token_env_var = str(get_nested(config, ["huggingface", "token_env_var"], "HF_TOKEN")).strip()
    if not token_env_var:
        fail("huggingface.token_env_var cannot be empty.")
    return token_env_var


def get_token(config: Dict[str, Any]) -> str:
    """Resolves the HuggingFace Hub token: the environment variable always
    wins if set (lets CI/shared machines override without touching disk);
    otherwise falls back to huggingface.token in settings.toml (set via
    'donadataset publish huggingface config set token')."""
    token_env_var = get_token_env_var(config)
    token = os.environ.get(token_env_var)
    if token:
        return token
    if global_settings.HUGGINGFACE.token:
        return global_settings.HUGGINGFACE.token
    fail(
        f"No HuggingFace Hub token found. Set the environment variable {token_env_var} "
        f"(export {token_env_var}='hf_xxxxxxxxxxxxxxxxxxxxxxxxx'), or store it with "
        "'donadataset publish huggingface config set token'."
    )


def authenticate(token: str) -> Dict[str, Any]:
    try:
        info = whoami(token=token)
    except Exception as exc:
        fail(f"Could not authenticate with Hugging Face Hub: {exc}")
    if not isinstance(info, dict):
        fail("Unexpected Hugging Face Hub authentication response.")
    return info


def list_missing_required_files(
    directory: Path, config: Dict[str, Any], verify_data: bool = True,
) -> List[str]:
    """Files/dirs an HFH export must have. Shared by upload's pre-flight check
    and download's post-download verification — both look at the same layout.
    verify_data=False skips the data/<split>/ directory check too, since
    callers that pass it also skip downloading the shards in the first
    place (see download_repository)."""
    missing: List[str] = []

    required_files = [
        str(get_nested(config, ["output_files", "readme_filename"], "README.md")),
        str(get_nested(config, ["output_files", "license_filename"], "LICENSE")),
        str(get_nested(config, ["output_files", "citation_filename"], "CITATION.cff")),
        get_internal_config_filename(config),
        str(get_nested(config, ["output_files", "yolo_yaml_filename"], "donana.yaml")),
        str(get_nested(config, ["output_files", "dataset_info_filename"], "dataset_info.json")),
        str(get_nested(config, ["manifest", "metadata_filename"], "metadata.csv")),
        str(get_nested(config, ["manifest", "manifest_filename"], "manifest.csv")),
        str(get_nested(config, ["manifest", "file_hash_manifest_filename"], "manifest-files-sha256.csv")),
        str(get_nested(config, ["checksums", "checksum_filename"], "checksums-sha256.txt")),
        str(get_nested(config, ["output_files", "local_verification_report_filename"], "verification_report_local.json")),
    ]
    for rel in required_files:
        if not (directory / rel).is_file():
            missing.append(rel)

    if verify_data:
        for split in get_split_names(config):
            rel_dir = f"data/{split}"
            if not (directory / rel_dir).is_dir():
                missing.append(rel_dir)

    return missing


# ── Upload ────────────────────────────────────────────────────────────────────

def get_create_repo_if_missing(config: Dict[str, Any]) -> bool:
    return as_bool(get_nested(config, ["huggingface", "create_repo_if_missing"], True), True)


def get_commit_message(config: Dict[str, Any]) -> str:
    return str(get_nested(config, ["huggingface", "commit_message"], "Upload HFH dataset export"))


def get_ignore_patterns(config: Dict[str, Any]) -> Optional[List[str]]:
    patterns = get_nested(config, ["huggingface", "upload_ignore_patterns"], None)
    if patterns is None:
        return None
    if not isinstance(patterns, list):
        fail("huggingface.upload_ignore_patterns must be a list.")
    return [str(pattern) for pattern in patterns]


def get_delete_patterns(config: Dict[str, Any]) -> Optional[List[str]]:
    patterns = get_nested(config, ["huggingface", "upload_delete_patterns"], None)
    if patterns is None:
        return None
    if not isinstance(patterns, list):
        fail("huggingface.upload_delete_patterns must be a list.")
    return [str(pattern) for pattern in patterns]


def get_retry_config(config: Dict[str, Any]) -> Dict[str, Any]:
    retry_config = get_nested(config, ["huggingface", "retry"], {}) or {}
    if not isinstance(retry_config, dict):
        fail("huggingface.retry must be a dictionary.")

    max_attempts = int(retry_config.get("max_attempts", 5))
    initial_wait_minutes = float(retry_config.get("initial_wait_minutes", 2))
    retry_on_status_codes = retry_config.get("retry_on_status_codes", [429, 500, 502, 503, 504])
    if not isinstance(retry_on_status_codes, list):
        fail("huggingface.retry.retry_on_status_codes must be a list.")
    if max_attempts < 1:
        fail("huggingface.retry.max_attempts must be >= 1.")
    if initial_wait_minutes < 0:
        fail("huggingface.retry.initial_wait_minutes must be >= 0.")

    return {
        "enabled": bool(retry_config.get("enabled", True)),
        "max_attempts": max_attempts,
        "initial_wait_minutes": initial_wait_minutes,
        "incremental_wait": bool(retry_config.get("incremental_wait", True)),
        "retry_on_status_codes": [int(code) for code in retry_on_status_codes],
    }


def is_retryable_hfh_error(exc: Exception, retry_on_status_codes: List[int]) -> bool:
    message = str(exc)
    for status_code in retry_on_status_codes:
        if str(status_code) in message:
            return True

    retryable_fragments = [
        "Gateway Time-out", "Gateway Timeout", "Read timed out",
        "Connection timed out", "Connection aborted", "Connection reset",
        "temporarily unavailable", "Temporary failure", "Max retries exceeded",
        "Remote end closed connection",
    ]
    message_lower = message.lower()
    return any(fragment.lower() in message_lower for fragment in retryable_fragments)


def run_with_hfh_retries(operation_name: str, operation, retry_config: Dict[str, Any]):
    if not retry_config["enabled"]:
        logging.info("%s | retries disabled", operation_name)
        return operation()

    max_attempts = retry_config["max_attempts"]
    initial_wait_minutes = retry_config["initial_wait_minutes"]
    incremental_wait = retry_config["incremental_wait"]
    retry_on_status_codes = retry_config["retry_on_status_codes"]

    last_exception: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            logging.info("%s | attempt %d/%d", operation_name, attempt, max_attempts)
            return operation()
        except Exception as exc:
            last_exception = exc
            if not is_retryable_hfh_error(exc, retry_on_status_codes):
                logging.error("%s failed with a non-retryable error.", operation_name)
                raise
            if attempt >= max_attempts:
                logging.error("%s failed after %d attempts.", operation_name, max_attempts)
                raise

            wait_minutes = initial_wait_minutes * attempt if incremental_wait else initial_wait_minutes
            wait_seconds = int(wait_minutes * 60)
            logging.warning("%s failed with a retryable error: %s", operation_name, exc)
            logging.warning("Waiting %.2f minutes before retrying...", wait_minutes)
            if wait_seconds > 0:
                time.sleep(wait_seconds)

    raise last_exception


def validate_hfh_folder(output_dir: Path, config: Dict[str, Any]) -> None:
    if not output_dir.exists():
        fail(f"HFH export folder does not exist: {output_dir}")
    if not output_dir.is_dir():
        fail(f"HFH export path is not a directory: {output_dir}")

    missing = list_missing_required_files(output_dir, config)
    if missing:
        fail(
            "The HFH export folder is incomplete. Missing required files/directories:\n"
            + "\n".join(f"  - {item}" for item in missing)
        )


def validate_local_verification_report(output_dir: Path, config: Dict[str, Any]) -> None:
    report_path = output_dir / str(
        get_nested(config, ["output_files", "local_verification_report_filename"], "verification_report_local.json")
    )
    if not report_path.exists():
        fail(f"Missing local verification report: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = report.get("status")
    if status != "passed":
        errors = report.get("errors", [])
        preview = ""
        if isinstance(errors, list) and errors:
            preview = "\nFirst errors:\n" + "\n".join(f"  - {err}" for err in errors[:10])
        fail(
            "Local verification did not pass. "
            f"status={status}, num_errors={report.get('num_errors', 'unknown')}.{preview}"
        )


def repo_exists(api: HfApi, repo_id: str, repo_type: str, token: str) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        return True
    except RepositoryNotFoundError:
        return False
    except HfHubHTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code == 404:
            return False
        raise


def create_dataset_repo_if_needed(
    api: HfApi, repo_id: str, repo_type: str, private: bool, token: str, create_if_missing: bool,
) -> None:
    if repo_exists(api, repo_id, repo_type, token):
        logging.info("Hugging Face Hub repository already exists: %s", repo_id)
        return

    if not create_if_missing:
        fail(
            f"Repository does not exist: {repo_id}. "
            "Set huggingface.create_repo_if_missing: true to create it automatically."
        )

    logging.info("Creating Hugging Face Hub dataset repository: %s", repo_id)
    create_repo(repo_id=repo_id, repo_type=repo_type, private=private, token=token, exist_ok=True)


def upload_hfh_folder(
    output_dir: Path,
    repo_id: str,
    repo_type: str,
    token: str,
    commit_message: str,
    ignore_patterns: Optional[List[str]],
    delete_patterns: Optional[List[str]],
) -> str:
    logging.info("Uploading folder to Hugging Face Hub...")
    logging.info("Local folder: %s", output_dir)
    logging.info("Repository: %s", repo_id)
    if ignore_patterns:
        logging.info("Upload ignore patterns: %s", ignore_patterns)
    if delete_patterns:
        logging.info("Remote delete patterns: %s", delete_patterns)

    result = upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
        delete_patterns=delete_patterns,
    )
    return str(result)


def build_dataset_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def get_private(config: Dict[str, Any]) -> bool:
    return as_bool(get_nested(config, ["huggingface", "private"], True), True)


def run_upload(config_path: Path, dry_run: bool = False) -> None:
    if not config_path.exists():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_yaml(config_path)

    repo_id = get_repo_id(config)
    repo_type = get_repo_type(config)
    private = get_private(config)
    token = get_token(config)
    create_if_missing = get_create_repo_if_missing(config)
    commit_message = get_commit_message(config)
    ignore_patterns = get_ignore_patterns(config)
    delete_patterns = get_delete_patterns(config)
    output_dir = get_output_dir(config)
    retry_config = get_retry_config(config)

    logging.info("Validating local HFH export folder...")
    validate_hfh_folder(output_dir, config)

    logging.info("Validating local verification report...")
    validate_local_verification_report(output_dir, config)

    file_count = count_files(output_dir)
    total_size = total_size_bytes(output_dir)

    logging.info("Files to upload: %d", file_count)
    logging.info("Total upload size: %s", format_size(total_size))
    logging.info("Upload ignore patterns: %s", ignore_patterns)
    logging.info("Remote delete patterns: %s", delete_patterns)
    logging.info("Retry enabled: %s", retry_config["enabled"])
    logging.info("Retry max attempts: %s", retry_config["max_attempts"])
    logging.info("Retry initial wait minutes: %s", retry_config["initial_wait_minutes"])
    logging.info("Retry incremental wait: %s", retry_config["incremental_wait"])
    logging.info("Retry status codes: %s", retry_config["retry_on_status_codes"])

    logging.info("Target repo_id: %s", repo_id)
    logging.info("Target repo_type: %s", repo_type)
    logging.info("Private repository: %s", private)
    logging.info("Token environment variable: %s", get_token_env_var(config))

    if dry_run:
        logging.info("Dry run enabled. No repository will be created and no files will be uploaded.")
        return

    logging.info("Authenticating with Hugging Face Hub...")
    user_info = authenticate(token)
    logging.info("Authenticated as: %s", user_info.get("name", "unknown"))

    api = HfApi(token=token)

    run_with_hfh_retries(
        operation_name="Create or check Hugging Face Hub repository",
        operation=lambda: create_dataset_repo_if_needed(
            api=api, repo_id=repo_id, repo_type=repo_type, private=private,
            token=token, create_if_missing=create_if_missing,
        ),
        retry_config=retry_config,
    )

    commit_url = run_with_hfh_retries(
        operation_name="Upload HFH folder to Hugging Face Hub",
        operation=lambda: upload_hfh_folder(
            output_dir=output_dir, repo_id=repo_id, repo_type=repo_type, token=token,
            commit_message=commit_message, ignore_patterns=ignore_patterns, delete_patterns=delete_patterns,
        ),
        retry_config=retry_config,
    )

    dataset_url = build_dataset_url(repo_id)
    logging.info("Upload completed successfully.")
    logging.info("Dataset URL: %s", dataset_url)
    logging.info("Commit URL: %s", commit_url)


# ── Download + verificación ──────────────────────────────────────────────────

def get_download_dir(config: Dict[str, Any]) -> Path:
    return Path(get_nested(config, ["downloaded_verification", "download_dir"], "./HFH_download_tmp"))


def get_delete_download_dir_after_success(config: Dict[str, Any]) -> bool:
    return as_bool(
        get_nested(config, ["downloaded_verification", "delete_download_dir_after_verification"], True), True,
    )


def get_downloaded_report_path(config: Dict[str, Any]) -> Path:
    return Path(str(
        get_nested(
            config, ["output_files", "downloaded_verification_report_filename"],
            "verification_report_downloaded.json",
        )
    ))


def download_repository(
    repo_id: str, repo_type: str, token: str, download_dir: Path, verify_data: bool = True,
) -> Path:
    """verify_data=False skips the heavy data/<split>/*.tar shards entirely
    (ignore_patterns on the snapshot download) — only the small metadata/
    evidence files are fetched. Used by callers that only need to prove the
    published *metadata* is internally consistent, not re-download and
    re-hash the whole dataset every run (see zenodo prepare's --verify-data)."""
    logging.info("Downloading Hugging Face Hub repository...")
    logging.info("Repository: %s", repo_id)
    logging.info("Download directory: %s", download_dir)
    logging.info("Including data/ shards: %s", verify_data)

    ensure_clean_dir(download_dir)
    snapshot_path = snapshot_download(
        repo_id=repo_id, repo_type=repo_type, token=token, local_dir=str(download_dir),
        ignore_patterns=None if verify_data else ["data/**", "data/*"],
    )
    return Path(snapshot_path)


def create_downloaded_verification_report(
    report_path: Path,
    repo_id: str,
    repo_type: str,
    downloaded_dir: Path,
    structural_errors: List[str],
    checksum_verified_count: int,
    checksum_errors: List[str],
    internal_verified_count: int,
    internal_errors: List[str],
    deleted_download_dir: bool,
    data_verified: bool = True,
) -> Dict[str, Any]:
    all_errors = structural_errors + checksum_errors + internal_errors

    report = {
        "generated_at_utc": utc_now_iso(),
        "status": "passed" if not all_errors else "failed",
        "repo_id": repo_id,
        "repo_type": repo_type,
        "downloaded_dir": str(downloaded_dir),
        "downloaded_dir_deleted": deleted_download_dir,
        "downloaded_files_count": count_files(downloaded_dir) if downloaded_dir.exists() else 0,
        "downloaded_total_size_bytes": total_size_bytes(downloaded_dir) if downloaded_dir.exists() else 0,
        "data_verified": data_verified,
        "global_files_verified": checksum_verified_count,
        "internal_tar_members_verified": internal_verified_count,
        "num_errors": len(all_errors),
        "structural_errors": structural_errors,
        "checksum_errors": checksum_errors,
        "internal_errors": internal_errors,
        "errors": all_errors,
    }
    write_json(report_path, report)
    return report


def download_and_verify_hfh(
    config: Dict[str, Any], token: str, download_dir: Path, report_path: Path, delete_after_success: bool,
    verify_data: bool = True,
) -> Dict[str, Any]:
    """Downloads the HFH repo fresh and verifies it against the same
    checksums/manifest 'prepare' wrote, writing (and returning) a report.
    Fails loudly if verification doesn't pass. Shared by 'huggingface
    download' (report kept next to the export) and 'zenodo prepare' (report
    generated live, inside Zenodo's own directory, rather than trusting a
    possibly-stale report from an earlier, separate download).

    verify_data=False skips the data/<split>/*.tar shards entirely — not
    downloaded, not checked against checksums-sha256.txt, and no internal
    tar member hashing — leaving only the small metadata/evidence files
    verified. The report's own "data_verified" field records which mode
    was used, so nobody mistakes a metadata-only pass for a full one."""
    repo_id = get_repo_id(config)
    repo_type = get_repo_type(config)

    logging.info("Authenticating with Hugging Face Hub...")
    user_info = authenticate(token)
    logging.info("Authenticated as: %s", user_info.get("name", "unknown"))

    downloaded_path = download_repository(repo_id, repo_type, token, download_dir, verify_data=verify_data)

    logging.info("Download completed.")
    logging.info("Downloaded path: %s", downloaded_path)
    logging.info("Downloaded files: %d", count_files(downloaded_path))
    logging.info("Downloaded size: %s", format_size(total_size_bytes(downloaded_path)))

    logging.info("Validating downloaded folder structure...")
    structural_errors = list_missing_required_files(downloaded_path, config, verify_data=verify_data)

    logging.info("Verifying global checksums...")
    checksum_verified_count, checksum_errors = verify_global_checksums(downloaded_path, config, verify_data=verify_data)

    if verify_data:
        logging.info("Verifying internal tar file hashes...")
        internal_verified_count, internal_errors = verify_tar_internal_hashes(downloaded_path, config)
    else:
        logging.info("Skipping internal tar file hashes (verify_data=False, shards were not downloaded).")
        internal_verified_count, internal_errors = 0, []

    all_errors = structural_errors + checksum_errors + internal_errors
    deleted_download_dir = False

    if not all_errors and delete_after_success:
        logging.info("Verification passed. Deleting temporary download directory...")
        remove_dir_if_exists(download_dir)
        deleted_download_dir = True
    elif all_errors:
        logging.warning("Verification failed. Temporary download directory will be kept for inspection.")
        logging.warning("Temporary download directory: %s", download_dir)

    report = create_downloaded_verification_report(
        report_path=report_path,
        repo_id=repo_id,
        repo_type=repo_type,
        downloaded_dir=downloaded_path,
        structural_errors=structural_errors,
        checksum_verified_count=checksum_verified_count,
        checksum_errors=checksum_errors,
        internal_verified_count=internal_verified_count,
        internal_errors=internal_errors,
        deleted_download_dir=deleted_download_dir,
        data_verified=verify_data,
    )

    if report["status"] == "passed":
        logging.info("Downloaded verification completed successfully.")
        logging.info("Report: %s", report_path.resolve())
        return report

    for error in all_errors[:50]:
        logging.error(error)
    if len(all_errors) > 50:
        logging.error("Additional errors not shown: %d", len(all_errors) - 50)
    fail(f"Downloaded verification failed with {len(all_errors)} error(s).")


def run_download_and_verify(config_path: Path, dry_run: bool = False) -> None:
    if not config_path.exists():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_yaml(config_path)

    repo_id = get_repo_id(config)
    repo_type = get_repo_type(config)
    token = get_token(config)
    token_env_var = get_token_env_var(config)
    download_dir = get_download_dir(config)
    delete_after_success = get_delete_download_dir_after_success(config)
    output_dir = get_output_dir(config)
    report_path = output_dir / get_downloaded_report_path(config)

    logging.info("Target repo_id: %s", repo_id)
    logging.info("Target repo_type: %s", repo_type)
    logging.info("Token environment variable: %s", token_env_var)
    logging.info("Temporary download directory: %s", download_dir)
    logging.info("Delete download directory after successful verification: %s", delete_after_success)
    logging.info("Downloaded verification report: %s", report_path)

    if dry_run:
        logging.info("Dry run enabled. No files will be downloaded.")
        return

    download_and_verify_hfh(config, token, download_dir, report_path, delete_after_success)


# ═══════════════════════════════════════════════════════════════════════════
# Hacer público el repositorio ("huggingface release")
# ═══════════════════════════════════════════════════════════════════════════
#
# Cambia la visibilidad del repo de private a public, y verifica de verdad que
# es accesible sin token (no basta con que la API diga "public"). Pensado para
# ejecutarse al final, una vez el DOI de Zenodo ya está en la metadata local y
# se ha vuelto a subir con 'huggingface upload'.

def get_tree_url(repo_id: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo_id}/tree/{revision}"


def get_public_visibility_report_path(config: Dict[str, Any]) -> Path:
    return Path(str(get_nested(config, ["huggingface", "public_visibility_report_filename"], "hfh_publication_report.json")))


def should_update_config_private_flag(config: Dict[str, Any]) -> bool:
    return bool(get_nested(config, ["huggingface", "update_config_private_flag"], True))


def get_public_check_timeout(config: Dict[str, Any]) -> int:
    return int(get_nested(config, ["huggingface", "public_check_timeout_seconds"], 20))


def get_dataset_visibility(api: HfApi, repo_id: str, token: str) -> Dict[str, Any]:
    try:
        info = api.dataset_info(repo_id=repo_id, token=token)
    except Exception as exc:
        fail(f"Could not read Hugging Face dataset info for {repo_id}: {exc}")

    private = getattr(info, "private", None)
    if private is None:
        fail("Could not determine repository visibility. The dataset_info response has no 'private' attribute.")

    return {"repo_id": repo_id, "private": bool(private), "public": not bool(private)}


def make_dataset_public(api: HfApi, repo_id: str, repo_type: str, token: str) -> None:
    try:
        # Stable parameter across older huggingface_hub clients.
        api.update_repo_settings(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
    except TypeError:
        # Newer clients prefer visibility="public".
        api.update_repo_settings(repo_id=repo_id, repo_type=repo_type, visibility="public", token=token)
    except Exception as exc:
        fail(f"Could not update repository visibility to public: {exc}")


def check_public_url(url: str, timeout_seconds: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {"url": url, "status": "unknown", "http_status": None, "reason": None}

    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout_seconds)
        if response.status_code in {403, 405}:
            response = requests.get(url, allow_redirects=True, timeout=timeout_seconds, stream=True)

        result["http_status"] = response.status_code
        result["reason"] = response.reason
        result["status"] = "passed" if 200 <= response.status_code < 400 else "failed"
    except requests.RequestException as exc:
        result["status"] = "failed"
        result["reason"] = str(exc)

    return result


def update_local_config_private_flag(config_path: Path, config: Dict[str, Any]) -> None:
    huggingface = ensure_dict(config, "huggingface")
    huggingface["private"] = False
    huggingface["last_visibility_update_utc"] = utc_now_iso()
    huggingface["last_visibility_status"] = "public"
    write_yaml(config_path, config)


def build_public_visibility_report(
    repo_id: str, repo_type: str, user_info: Dict[str, Any],
    before_visibility: Dict[str, Any], after_visibility: Dict[str, Any],
    dataset_url_check: Dict[str, Any], tree_url_check: Dict[str, Any],
    dry_run: bool, verify_only: bool, changed_visibility: bool, config_updated: bool,
) -> Dict[str, Any]:
    errors: List[str] = []
    dry_run_expected_errors: List[str] = []

    if not after_visibility.get("public"):
        message = f"Repository is not public according to Hugging Face API: {repo_id}"
        (dry_run_expected_errors if dry_run else errors).append(message)

    for label, check in [("dataset_url", dataset_url_check), ("tree_url", tree_url_check)]:
        if check.get("status") != "passed":
            message = (
                f"Public URL check failed for {label}: "
                f"{check.get('url')} ({check.get('http_status')}, {check.get('reason')})"
            )
            (dry_run_expected_errors if dry_run else errors).append(message)

    status = "dry_run" if dry_run else ("passed" if not errors else "failed")

    return {
        "generated_at_utc": utc_now_iso(),
        "status": status,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "authenticated_as": user_info.get("name") or user_info.get("fullname") or "unknown",
        "dry_run": dry_run,
        "verify_only": verify_only,
        "changed_visibility": changed_visibility,
        "config_updated": config_updated,
        "before_visibility": before_visibility,
        "after_visibility": after_visibility,
        "public_urls": {"dataset_url": dataset_url_check, "tree_url": tree_url_check},
        "num_errors": len(errors),
        "errors": errors,
        "dry_run_expected_errors": dry_run_expected_errors,
        "next_recommended_steps": [
            "Open the Hugging Face Hub dataset URL in a private/incognito browser window.",
            "Review the Zenodo draft manually.",
            "Publish the Zenodo draft only after the public HFH links work for unauthenticated users.",
        ],
    }


def run_release(config_path: Path, dry_run: bool = False, verify_only: bool = False) -> None:
    if not config_path.exists():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_yaml(config_path)

    repo_id = get_repo_id(config)
    repo_type = get_repo_type(config)
    token_env_var = get_token_env_var(config)
    token = get_token(config)
    output_dir = get_output_dir(config)
    report_path = output_dir / get_public_visibility_report_path(config)
    timeout_seconds = get_public_check_timeout(config)

    logging.info("Target repo_id: %s", repo_id)
    logging.info("Target repo_type: %s", repo_type)
    logging.info("Token environment variable: %s", token_env_var)
    logging.info("Public visibility report: %s", report_path)

    logging.info("Authenticating with Hugging Face Hub...")
    user_info = authenticate(token)
    logging.info("Authenticated as: %s", user_info.get("name", "unknown"))

    api = HfApi(token=token)

    logging.info("Reading current Hugging Face Hub visibility...")
    before_visibility = get_dataset_visibility(api, repo_id, token)
    logging.info("Current visibility: %s", "private" if before_visibility["private"] else "public")

    changed_visibility = False
    config_updated = False

    if verify_only:
        logging.info("Verify-only mode enabled. Repository visibility will not be changed.")
    elif dry_run:
        logging.info("Dry run enabled. Repository visibility will not be changed.")
        if before_visibility["private"]:
            logging.info("Would change repository visibility from private to public.")
        else:
            logging.info("Repository is already public. No visibility change would be needed.")
    else:
        if before_visibility["private"]:
            logging.info("Changing repository visibility to public...")
            make_dataset_public(api, repo_id, repo_type, token)
            changed_visibility = True
        else:
            logging.info("Repository is already public. No visibility change needed.")

    logging.info("Reading Hugging Face Hub visibility after requested operation...")
    after_visibility = get_dataset_visibility(api, repo_id, token)
    logging.info("Visibility after operation: %s", "private" if after_visibility["private"] else "public")

    dataset_url = build_dataset_url(repo_id)
    tree_url = get_tree_url(repo_id)

    logging.info("Checking public dataset URL without token: %s", dataset_url)
    dataset_url_check = check_public_url(dataset_url, timeout_seconds)
    logging.info("Dataset URL check: %s (%s)", dataset_url_check["status"], dataset_url_check.get("http_status"))

    logging.info("Checking public tree URL without token: %s", tree_url)
    tree_url_check = check_public_url(tree_url, timeout_seconds)
    logging.info("Tree URL check: %s (%s)", tree_url_check["status"], tree_url_check.get("http_status"))

    if (
        not dry_run and not verify_only
        and after_visibility.get("public")
        and dataset_url_check.get("status") == "passed"
        and should_update_config_private_flag(config)
    ):
        logging.info("Updating local configuration file: huggingface.private = false")
        update_local_config_private_flag(config_path, config)
        config_updated = True

    report = build_public_visibility_report(
        repo_id=repo_id, repo_type=repo_type, user_info=user_info,
        before_visibility=before_visibility, after_visibility=after_visibility,
        dataset_url_check=dataset_url_check, tree_url_check=tree_url_check,
        dry_run=dry_run, verify_only=verify_only,
        changed_visibility=changed_visibility, config_updated=config_updated,
    )

    write_json(report_path, report)
    logging.info("Report written: %s", report_path)

    if dry_run:
        if report.get("dry_run_expected_errors"):
            logging.info("Dry run completed. The following checks are expected to fail because the repository was not modified:")
            for error in report["dry_run_expected_errors"]:
                logging.info("  - %s", error)
        logging.info("Dry run completed successfully. No repository settings were changed.")
        return

    if report["status"] != "passed":
        for error in report["errors"]:
            logging.error(error)
        fail(f"HFH public visibility verification failed with {len(report['errors'])} error(s).")

    logging.info("HFH repository is public and public URLs are accessible.")
    logging.info("Dataset URL: %s", dataset_url)
    logging.info("Tree URL: %s", tree_url)


# ═══════════════════════════════════════════════════════════════════════════
# Sincronizar el DOI nativo de HuggingFace Hub ("huggingface sync-doi")
# ═══════════════════════════════════════════════════════════════════════════
#
# HuggingFace Hub no permite generar un DOI por API — solo mediante el botón
# "Generate DOI" en la web del repo (Settings). Una vez generado, aparece
# como un tag "doi:10.xxxx/hf/xxxx" en la información pública del repo. Este
# comando lee ese tag y lo refleja en el CITATION.cff local (y recalcula
# checksums-sha256.txt), para que quede reflejado la próxima vez que hagas
# 'huggingface upload'. Si ya existe un DOI de Zenodo en citation.doi (el que
# cita el registro de metadatos enlazado), el de HuggingFace Hub se añade a
# citation.identifiers en vez de sobreescribirlo — son identificadores de
# alcance distinto (el repo de datos en sí, frente al registro de Zenodo).

def get_doi_sync_report_path(config: Dict[str, Any]) -> Path:
    return Path(str(get_nested(config, ["huggingface", "doi_sync_report_filename"], "hfh_doi_sync_report.json")))


def extract_doi_tag(tags: Optional[List[str]]) -> Optional[str]:
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith("doi:"):
            return tag[len("doi:"):]
    return None


def get_repo_doi(api: HfApi, repo_id: str, repo_type: str, token: str) -> Optional[str]:
    try:
        info = api.dataset_info(repo_id=repo_id, token=token) if repo_type == "dataset" else None
    except Exception as exc:
        fail(f"Could not read Hugging Face Hub repo info for {repo_id}: {exc}")
    return extract_doi_tag(getattr(info, "tags", None))


def update_citation_cff_with_hfh_doi(citation_path: Path, doi: str, repo_id: str) -> bool:
    """Writes the Hugging Face Hub DOI into CITATION.cff. Returns False (no
    change) if it was already in sync."""
    citation = load_yaml(citation_path)

    if citation.get("doi") == doi:
        return False

    if not citation.get("doi"):
        citation["doi"] = doi
        citation["url"] = build_dataset_url(repo_id)
    else:
        identifiers = citation.get("identifiers")
        if not isinstance(identifiers, list):
            identifiers = []
        identifiers = [
            item for item in identifiers
            if not (isinstance(item, dict) and item.get("description") == "Hugging Face Hub dataset DOI")
        ]
        identifiers.append({
            "type": "doi", "value": doi, "description": "Hugging Face Hub dataset DOI",
        })
        citation["identifiers"] = identifiers

    write_yaml(citation_path, citation)
    return True


def run_sync_hfh_doi(config_path: Path, dry_run: bool = False) -> None:
    if not config_path.exists():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_yaml(config_path)

    repo_id = get_repo_id(config)
    repo_type = get_repo_type(config)
    output_dir = get_output_dir(config)
    citation_path = output_dir / str(get_nested(config, ["output_files", "citation_filename"], "CITATION.cff"))
    report_path = output_dir / get_doi_sync_report_path(config)

    if not citation_path.is_file():
        fail(f"CITATION.cff not found: {citation_path}. Run 'huggingface prepare' first.")

    token = get_token(config)
    api = HfApi()

    logging.info("Checking Hugging Face Hub for a generated DOI on %s...", repo_id)
    doi = get_repo_doi(api, repo_id, repo_type, token)

    if not doi:
        message = (
            f"No DOI tag found yet on {repo_id}. Generate one manually first: "
            f"https://huggingface.co/datasets/{repo_id}/settings, "
            "'Digital Object Identifier (DOI)' section, 'Generate DOI' button."
        )
        write_json(report_path, {
            "generated_at_utc": utc_now_iso(),
            "status": "no_doi_found",
            "repo_id": repo_id,
            "message": message,
        })
        logging.warning(message)
        return

    logging.info("Found Hugging Face Hub DOI: %s", doi)

    if dry_run:
        logging.info("Dry run enabled. CITATION.cff and checksums will not be modified.")
        return

    changed = update_citation_cff_with_hfh_doi(citation_path, doi, repo_id)

    if changed:
        logging.info("CITATION.cff changed — recomputing checksums...")
        write_checksums(output_dir, config)
    else:
        logging.info("CITATION.cff already had this DOI. Nothing to update.")

    write_json(report_path, {
        "generated_at_utc": utc_now_iso(),
        "status": "passed",
        "repo_id": repo_id,
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}",
        "citation_updated": changed,
    })

    if changed:
        logging.info("Remember to run 'huggingface upload' again so the updated CITATION.cff is published.")
