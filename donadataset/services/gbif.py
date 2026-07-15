"""Lógica de interacción con GBIF (sin nada de CLI/Typer).

GBIF solo ingiere datos ya convertidos a un formato reconocido — este
proyecto usa Camtrap DP (https://camtrap-dp.tdwg.org/), el estándar
recomendado por GBIF/TDWG para datos de cámaras trampa (IPT v3+ lo soporta
nativamente). Camtrap DP es un Frictionless Data Package: un
`datapackage.json` describiendo tres tablas (`deployments.csv`,
`media.csv`, `observations.csv`).

Tres operaciones:

- `run_prepare`: escanea el dataset YOLO ya limpio (el mismo que usa
  `huggingface prepare`) y genera el paquete Camtrap DP completo, sin pedir
  nada relleno a mano. Puramente local — no sube nada a ningún sitio. Como
  este pipeline no rastrea GPS ni fecha de despliegue por cámara en ningún
  punto, se asume **un deployment por split** (train/val/test) con
  coordenadas fijas ilustrativas dentro de Doñana; la fecha de cada imagen
  se lee de su EXIF si existe, y si no, se reparte dentro del rango que sí
  tenga EXIF en ese split (o, si ninguna imagen del split tiene EXIF,
  dentro de un año-placeholder fijo). Todo esto queda anotado en
  `deploymentComments`/`mediaComments` para que no se confunda con datos
  reales.
- `run_upload`: sube el `.zip` que ya generó `run_prepare` al repo de
  HuggingFace Hub ya publicado (mismo repo_id que usan `huggingface
  prepare`/`upload`) — lo copia al export local de HuggingFace Hub y
  regenera su checksums-sha256.txt, mismo patrón de "sincroniza local,
  vuelve a subir solo lo cambiado" que usa `zenodo sync-doi`. Da una URL
  persistente lista para `gbif register --archive-url` sin depender de
  alojarlo en otro sitio tú mismo.
- `run_register`: usa esa URL (la de `run_upload`, o una que hayas alojado
  tú mismo en otro sitio) para registrar/actualizar el dataset en el
  Registry de GBIF (endpoint tipo CAMTRAP_DP) — mismo patrón de "registro
  enlazado" que donadataset.services.zenodo/b2share con HuggingFace Hub.
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from donadataset.config import get_gbif_output_dir, get_hfh_output_dir
from donadataset.config import settings as global_settings
from donadataset.services.common import (
    fail,
    find_tar_files,
    read_checksums,
    read_json,
    safe_extract_tar,
    sha256_file,
    utc_now_iso,
    write_json,
)
from donadataset.services.huggingface import verify_global_checksums
from donadataset.services.huggingface import (
    download_repository,
    find_source_names_yaml,
    get_classes,
    scan_dataset,
    stop_if_validation_errors,
)

# ── Camtrap DP generation ───────────────────────────────────────────────────

# No hay GPS real en ningún punto del pipeline (ni EXIF, ni nombre de
# fichero, ni manifest) — un punto ilustrativo por split, dentro del
# rectángulo aproximado del Parque Nacional de Doñana (~36.83–37.15 N,
# -6.55– -6.20 W). No confundir con coordenadas reales de cámara.
SPLIT_DEPLOYMENT_COORDINATES: Dict[str, Tuple[float, float]] = {
    "train": (37.0160, -6.4400),
    "val":   (36.9550, -6.3300),
    "test":  (37.0850, -6.2600),
}
DEFAULT_DEPLOYMENT_COORDINATES = (37.0, -6.35)  # fallback si algún split no está en el dict de arriba

# Solo se usa si NINGUNA imagen de un split tiene fecha EXIF legible — un
# año de calendario completo y obviamente sintético.
PLACEHOLDER_DEPLOYMENT_START = datetime(2023, 1, 1, 0, 0, 0)
PLACEHOLDER_DEPLOYMENT_END   = datetime(2023, 12, 31, 23, 59, 59)

EXIF_IFD_POINTER            = 0x8769  # puntero al Exif SubIFD
EXIF_DATETIME_TAG           = 0x0132  # DateTime, en IFD0
EXIF_DATETIME_ORIGINAL_TAG  = 0x9003  # DateTimeOriginal, en el Exif SubIFD

MEDIA_TYPES_BY_EXTENSION = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff",
    ".webp": "image/webp",
}

# Clases que no representan un animal detectado (fondo/ninguna detección) —
# esas imágenes generan una observación 'blank', no una fila por especie.
NON_OBSERVATION_CLASS_NAMES = {"empty"}

CAMTRAP_DP_PROFILE_URL = "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0/camtrap-dp-profile.json"

DEPLOYMENTS_FILENAME  = "deployments.csv"
MEDIA_FILENAME        = "media.csv"
OBSERVATIONS_FILENAME = "observations.csv"
DATAPACKAGE_FILENAME  = "datapackage.json"

DEPLOYMENT_FIELDNAMES = [
    "deploymentID", "locationID", "locationName", "latitude", "longitude",
    "deploymentStart", "deploymentEnd", "deploymentComments",
]
MEDIA_FIELDNAMES = [
    "mediaID", "deploymentID", "captureMethod", "timestamp",
    "filePath", "fileName", "fileMediatype", "mediaComments",
]
OBSERVATION_FIELDNAMES = [
    "observationID", "deploymentID", "mediaID", "eventID", "eventStart", "eventEnd",
    "observationLevel", "observationType", "scientificName", "count",
    "classificationMethod", "classifiedBy",
]

# Not configurable — every label in this dataset was produced by human
# review, so this is a fact about the dataset, not a per-run setting.
CLASSIFICATION_METHOD = "human"
CLASSIFIED_BY = "WildINTEL experts"


def _to_iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def read_exif_datetime(image_path: Path) -> Optional[datetime]:
    """Reads DateTimeOriginal (preferred) or DateTime from an image's EXIF,
    if present. Returns None for images without EXIF (e.g. this project's
    PNGs, or JPEGs stripped of metadata) or with an unparseable value —
    callers fall back to a placeholder in that case, never raise."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return None
            raw = exif.get(EXIF_DATETIME_TAG)
            if not raw:
                try:
                    sub_ifd = exif.get_ifd(EXIF_IFD_POINTER)
                except Exception:
                    sub_ifd = {}
                raw = sub_ifd.get(EXIF_DATETIME_ORIGINAL_TAG)
    except Exception:
        return None

    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def resolve_deployment_timestamps(items: List[Any]) -> Dict[str, Tuple[datetime, bool]]:
    """Returns {image_id: (timestamp, is_from_exif)} for one deployment's
    (one split's) images.

    Real EXIF dates are used as-is. Images without a readable EXIF date are
    spread evenly across the range established by that same deployment's
    own EXIF dates, so the deployment's date range never gets distorted by
    a placeholder — unless *no* image in the deployment has EXIF at all, in
    which case every image falls back to a fixed synthetic placeholder year."""
    exif_dates: Dict[str, datetime] = {}
    for item in items:
        found = read_exif_datetime(item.image_abs_path)
        if found is not None:
            exif_dates[item.image_id] = found

    if exif_dates:
        anchor_start, anchor_end = min(exif_dates.values()), max(exif_dates.values())
    else:
        anchor_start, anchor_end = PLACEHOLDER_DEPLOYMENT_START, PLACEHOLDER_DEPLOYMENT_END

    resolved: Dict[str, Tuple[datetime, bool]] = {
        image_id: (value, True) for image_id, value in exif_dates.items()
    }

    missing = sorted((item for item in items if item.image_id not in exif_dates), key=lambda i: i.image_id)
    span = anchor_end - anchor_start
    for index, item in enumerate(missing):
        fraction = index / len(missing) if len(missing) > 1 else 0.0
        resolved[item.image_id] = (anchor_start + span * fraction, False)

    return resolved


def count_classes_in_label(label_path: Path) -> Dict[int, int]:
    """Counts how many boxes of each class a YOLO label file has.

    scan_dataset()+stop_if_validation_errors() already validated the label
    format before this runs, so no re-validation is needed here."""
    counts: Dict[int, int] = {}
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return counts
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        class_id = int(float(line.split()[0]))
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def build_camtrap_dp_resources(
    items: List[Any],
    classes: Dict[int, str],
    *,
    shard_url_by_image_id: Dict[str, str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Builds the three Camtrap DP tables from a scanned dataset.

    One deployment per split present in the data (skips a split entirely if
    it has no images). One media row per image. One observation row per
    image+species with at least one box (individualCount-equivalent via
    'count'), or a single 'blank' observation for images whose only class is
    'Empty' (or that have no boxes at all).

    shard_url_by_image_id (see read_local_shard_urls) maps every item's
    image_id to the persistent URL of the HuggingFace Hub .tar shard it was
    actually packed into by 'huggingface prepare' — used as media.filePath.
    It must cover every item; a missing image_id fails loudly rather than
    silently falling back to a local path, since that means the source
    dataset used here doesn't match what's actually published."""
    deployment_rows: List[Dict[str, str]] = []
    media_rows: List[Dict[str, str]] = []
    observation_rows: List[Dict[str, str]] = []

    for split in ("train", "val", "test"):
        split_items = sorted((item for item in items if item.split == split), key=lambda i: i.image_id)
        if not split_items:
            continue

        deployment_id = split
        latitude, longitude = SPLIT_DEPLOYMENT_COORDINATES.get(split, DEFAULT_DEPLOYMENT_COORDINATES)
        timestamps = resolve_deployment_timestamps(split_items)
        deployment_start = min(value for value, _ in timestamps.values())
        deployment_end = max(value for value, _ in timestamps.values())
        any_placeholder = any(not is_real for _, is_real in timestamps.values())

        comments = "Synthetic deployment: one per dataset split, not a real camera placement."
        if any_placeholder:
            comments += " Some/all media timestamps are estimated (no EXIF datetime found)."

        deployment_rows.append({
            "deploymentID": deployment_id,
            "locationID": deployment_id,
            "locationName": f"Doñana National Park — {split} split (placeholder location)",
            "latitude": f"{latitude}",
            "longitude": f"{longitude}",
            "deploymentStart": _to_iso(deployment_start),
            "deploymentEnd": _to_iso(deployment_end),
            "deploymentComments": comments,
        })

        for item in split_items:
            timestamp, is_real = timestamps[item.image_id]
            extension = item.image_abs_path.suffix.lower()

            media_comments = [] if is_real else ["timestamp estimated: no EXIF datetime found in the source image"]

            if item.image_id not in shard_url_by_image_id:
                fail(
                    f"Image {item.image_id!r} has no entry in the HuggingFace Hub manifest.csv — "
                    "the local source dataset doesn't match what's published. Re-run 'huggingface "
                    "prepare'/'upload' with the same source dataset before 'gbif prepare'."
                )
            file_path = shard_url_by_image_id[item.image_id]
            media_comments.append(
                f"filePath points to the .tar shard containing {item.image_abs_path.name} on "
                "HuggingFace Hub, not an individually downloadable file."
            )

            media_rows.append({
                "mediaID": item.image_id,
                "deploymentID": deployment_id,
                "captureMethod": "activityDetection",
                "timestamp": _to_iso(timestamp),
                "filePath": file_path,
                "fileName": item.image_rel_path,
                "fileMediatype": MEDIA_TYPES_BY_EXTENSION.get(extension, "application/octet-stream"),
                "mediaComments": " ".join(media_comments),
            })

            class_counts = count_classes_in_label(item.label_abs_path)
            observed_species = {
                class_id: count for class_id, count in class_counts.items()
                if classes.get(class_id, "").strip().lower() not in NON_OBSERVATION_CLASS_NAMES
            }

            if not observed_species:
                observation_rows.append({
                    "observationID": f"{item.image_id}:blank",
                    "deploymentID": deployment_id,
                    "mediaID": item.image_id,
                    "eventID": item.image_id,
                    "eventStart": _to_iso(timestamp),
                    "eventEnd": _to_iso(timestamp),
                    "observationLevel": "media",
                    "observationType": "blank",
                    "scientificName": "",
                    "count": "",
                    "classificationMethod": CLASSIFICATION_METHOD,
                    "classifiedBy": CLASSIFIED_BY,
                })
                continue

            for class_id in sorted(observed_species):
                observation_rows.append({
                    "observationID": f"{item.image_id}:{class_id}",
                    "deploymentID": deployment_id,
                    "mediaID": item.image_id,
                    "eventID": item.image_id,
                    "eventStart": _to_iso(timestamp),
                    "eventEnd": _to_iso(timestamp),
                    "observationLevel": "media",
                    "observationType": "animal",
                    "scientificName": classes.get(class_id, ""),
                    "count": str(observed_species[class_id]),
                    "classificationMethod": CLASSIFICATION_METHOD,
                    "classifiedBy": CLASSIFIED_BY,
                })

    return deployment_rows, media_rows, observation_rows


def _resource_descriptor(name: str, path: str, fieldnames: List[str]) -> Dict[str, Any]:
    """Minimal Frictionless tabular-data-resource descriptor (every field
    typed as 'string') — not the full constrained official Camtrap DP table
    schema (which also declares enums/required/foreign keys), just enough
    for the package to be structurally self-describing."""
    return {
        "name": name,
        "path": path,
        "profile": "tabular-data-resource",
        "format": "csv",
        "mediatype": "text/csv",
        "encoding": "utf-8",
        "schema": {"fields": [{"name": field, "type": "string"} for field in fieldnames]},
    }


def build_datapackage(
    deployment_rows: List[Dict[str, str]],
    observation_rows: List[Dict[str, str]],
    *,
    dataset_slug: str,
    dataset_name: str,
    description: str,
    license_id: str,
    license_name: str,
    license_url: str,
    rights_holder: str,
    contact_name: str,
    contact_email: Optional[str],
) -> Dict[str, Any]:
    species_names = sorted({row["scientificName"] for row in observation_rows if row["scientificName"]})
    latitudes = [float(row["latitude"]) for row in deployment_rows]
    longitudes = [float(row["longitude"]) for row in deployment_rows]
    starts = sorted(row["deploymentStart"] for row in deployment_rows)
    ends = sorted(row["deploymentEnd"] for row in deployment_rows)

    contributor: Dict[str, Any] = {"title": contact_name, "role": "contact", "organization": rights_holder}
    if contact_email:
        contributor["email"] = contact_email

    return {
        "profile": CAMTRAP_DP_PROFILE_URL,
        "name": dataset_slug,
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://donadataset.local/{dataset_slug}")),
        "created": utc_now_iso(),
        "title": dataset_name,
        "contributors": [contributor],
        "description": description,
        "version": "1.0",
        "keywords": ["camera traps", "Doñana", "mammals", "Camtrap DP"],
        "licenses": [
            {"name": license_id, "title": license_name, "path": license_url, "scope": "data"},
            {"name": license_id, "title": license_name, "path": license_url, "scope": "media"},
        ],
        "project": {
            "id": dataset_slug,
            "title": dataset_name,
            "description": description,
            "samplingDesign": "opportunistic",
            "captureMethod": ["activityDetection"],
            "individualAnimals": True,
            "observationLevel": ["media"],
        },
        "spatial": {
            "type": "MultiPoint",
            "coordinates": [[longitude, latitude] for longitude, latitude in zip(longitudes, latitudes)],
        },
        "temporal": {"start": starts[0][:10], "end": ends[-1][:10]},
        "taxonomic": [{"scientificName": name} for name in species_names],
        "resources": [
            _resource_descriptor("deployments", DEPLOYMENTS_FILENAME, DEPLOYMENT_FIELDNAMES),
            _resource_descriptor("media", MEDIA_FILENAME, MEDIA_FIELDNAMES),
            _resource_descriptor("observations", OBSERVATIONS_FILENAME, OBSERVATION_FIELDNAMES),
        ],
    }


def _write_csv(rows: List[Dict[str, str]], fieldnames: List[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


HUGGINGFACE_TOKEN_ENV_VAR = "HF_TOKEN"
HFH_CHECKSUMS_FILENAME = "checksums-sha256.txt"  # matches write_checksums()'s own default


def run_upload(
    output_dir: Path, repo_id: Optional[str], hfh_output_dir: Optional[Path] = None, *, dry_run: bool = False,
) -> Tuple[str, str, Optional[Path]]:
    """Stages the already-built Camtrap DP .zip (see run_prepare) into the
    local HuggingFace Hub export and regenerates that export's
    checksums-sha256.txt — the same "sync local metadata, then re-upload
    just the changed files" pattern 'zenodo sync-doi' uses, so the actual
    push (a scoped 'huggingface upload') stays in commands/gbif.py, right
    next to the rest of the HuggingFace Hub wiring. hfh_output_dir, if
    given, always wins; otherwise it's resolved the same "local first,
    download-and-cache otherwise" way as run_prepare's own source dataset
    (see resolve_gbif_hfh_output_dir). Returns (archive_filename,
    checksums_filename, resolved_hfh_output_dir) — the caller needs the
    first two for allow_patterns and the persistent download URL, and the
    third to build the HuggingFace Hub config path for the actual push.
    resolved_hfh_output_dir is None only when dry_run=True and nothing is
    local yet (nothing to stage, so there's nothing to push either)."""
    if not repo_id or "REPLACE_WITH" in repo_id:
        fail(
            "A valid HuggingFace repo_id is required to upload the Camtrap DP archive. "
            "Pass --hf-repo-id, or set one with 'donadataset publish huggingface config "
            "set repo_id=user_or_org/dataset'."
        )

    archive_candidates = sorted(output_dir.glob("*-camtrap-dp.zip"))
    if not archive_candidates:
        fail(f"No Camtrap DP package found in {output_dir}. Run 'gbif prepare' first.")
    if len(archive_candidates) > 1:
        fail(
            f"Several Camtrap DP packages found in {output_dir}: "
            f"{[p.name for p in archive_candidates]}. Re-run 'gbif prepare --overwrite' so "
            "only one remains before uploading."
        )
    archive_path = archive_candidates[0]

    resolved_hfh_output_dir = resolve_gbif_hfh_output_dir(repo_id, hfh_output_dir, dry_run=dry_run)
    if resolved_hfh_output_dir is None:
        return archive_path.name, HFH_CHECKSUMS_FILENAME, None

    if not resolved_hfh_output_dir.is_dir():
        fail(
            f"HuggingFace Hub export folder not found: {resolved_hfh_output_dir}. Run "
            "'huggingface prepare' for this dataset first — 'gbif upload' adds the archive "
            "to that local export and refreshes its checksums before pushing."
        )

    if dry_run:
        logging.info(
            "Dry run enabled. Would copy %s into %s and regenerate %s.",
            archive_path.name, resolved_hfh_output_dir, HFH_CHECKSUMS_FILENAME,
        )
        return archive_path.name, HFH_CHECKSUMS_FILENAME, resolved_hfh_output_dir

    logging.info("Copying %s into the HuggingFace Hub export...", archive_path.name)
    shutil.copy2(archive_path, resolved_hfh_output_dir / archive_path.name)

    logging.info("Regenerating %s...", HFH_CHECKSUMS_FILENAME)
    regenerate_hfh_checksums(resolved_hfh_output_dir, [archive_path.name])

    logging.info("Verifying regenerated checksums...")
    checksum_verified_count, checksum_errors = verify_global_checksums(resolved_hfh_output_dir, {})
    if checksum_errors:
        for error in checksum_errors:
            logging.error(error)
        fail("The Camtrap DP archive was copied, but checksum verification failed.")
    logging.info("Checksums verified: %d", checksum_verified_count)

    return archive_path.name, HFH_CHECKSUMS_FILENAME, resolved_hfh_output_dir


HFH_MANIFEST_FILENAME = "manifest.csv"  # matches huggingface.py's write_manifest_csv() default


def read_local_shard_urls(source_dir: Path, repo_id: str) -> Dict[str, str]:
    """media.filePath always links to HuggingFace Hub — no opt-in flag.
    Reads manifest.csv straight from source_dir (the same local HFH copy
    'gbif prepare' already resolved images/labels from — see
    resolve_gbif_source_dataset_dir) instead of fetching it over the
    network: this is the exact file that produced the images being
    scanned, so it can never describe a different snapshot, and it works
    even before the repo has ever been published (an explicit
    --source-dataset-dir must still point at a real 'huggingface prepare'
    output for this to exist). Returns {image_id: persistent shard URL} —
    the .tar each image was actually packed into, the closest thing to a
    real per-image URL this pipeline can offer (see
    build_camtrap_dp_resources for why it's the shard, not the image
    itself)."""
    manifest_path = source_dir / HFH_MANIFEST_FILENAME
    if not manifest_path.is_file():
        fail(
            f"{HFH_MANIFEST_FILENAME} not found in {source_dir}. 'gbif prepare' always links "
            "media.filePath to the HuggingFace Hub .tar shard, which needs this file — run "
            "'huggingface prepare' for this dataset first."
        )

    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    urls: Dict[str, str] = {}
    with open(manifest_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            shard = row.get("shard", "").strip()
            if row.get("image_id") and shard:
                urls[row["image_id"]] = f"{base_url}/{shard}"
    return urls


# ── Resolución del dataset fuente desde HuggingFace Hub ──────────────────────
#
# 'gbif prepare' necesita leer cada imagen (EXIF) y cada label (conteo de
# cajas) — a diferencia de Zenodo, que solo copia ficheros de evidencia ya
# existentes, esto obliga a tener el dataset YOLO completo en disco. Por eso
# NO se descarga en vivo cada vez (como sí hace Zenodo): primero se reutiliza
# lo que 'huggingface prepare' ya dejó en local, y solo se descarga de la red
# si no está.

def _looks_like_hfh_export(directory: Path) -> bool:
    """True if `directory` looks like a real 'huggingface prepare' output:
    a YOLO-style YAML with a 'names' mapping directly inside it, plus at
    least one packed data/<split>/*.tar shard."""
    if not directory.is_dir():
        return False
    if find_source_names_yaml(directory) is None:
        return False
    return any(directory.glob("data/*/*.tar"))


def _looks_like_extracted_yolo_dataset(directory: Path) -> bool:
    """True if `directory` already has an extracted images/<split>/+labels/<split>/
    tree, a names YAML, and manifest.csv — i.e. a previous 'gbif prepare' run
    already did the tar-extraction work AND media.filePath's shard-URL lookup
    has what it needs, so it can be reused as-is. manifest.csv is checked too
    (not just the images/labels tree) so a cache extracted before
    manifest.csv started traveling with it gets re-synced instead of being
    silently reused half-broken."""
    if not directory.is_dir():
        return False
    if find_source_names_yaml(directory) is None:
        return False
    if not (directory / HFH_MANIFEST_FILENAME).is_file():
        return False
    return any(directory.glob("images/*/*"))


def get_gbif_hfh_download_dir(repo_id: str) -> Path:
    """Lives INSIDE get_gbif_output_dir() (--output-dir) on purpose, named
    after what it holds — a HuggingFace Hub download, extracted in place.
    'run_prepare's own --overwrite handling only ever touches the Camtrap DP
    deliverables (deployments/media/observations.csv, datapackage.json, the
    .zip), never this directory, so it survives across runs regardless of
    --overwrite."""
    return get_gbif_output_dir(repo_id) / "hfh_download"


def extract_hfh_export_to_yolo_source(source_dir: Path, extracted_dir: Path) -> Path:
    """source_dir and extracted_dir may be the SAME directory (the
    freshly-downloaded case, see resolve_gbif_source_dataset_dir) — nothing
    here force-deletes extracted_dir first, so extracting in place is safe;
    tarfile.extractall() onto files that already match is a no-op."""
    if _looks_like_extracted_yolo_dataset(extracted_dir):
        logging.info("Reusing already-extracted YOLO dataset: %s", extracted_dir)
        return extracted_dir

    logging.info("Extracting HuggingFace Hub shards from %s into %s...", source_dir, extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    for tar_path in find_tar_files(source_dir):
        safe_extract_tar(tar_path, extracted_dir)

    names_yaml = find_source_names_yaml(source_dir)
    if names_yaml is None:
        fail(f"Could not find a YOLO-style YAML with a 'names' mapping inside {source_dir}.")
    if names_yaml.parent != extracted_dir:
        shutil.copy2(names_yaml, extracted_dir / names_yaml.name)

    # media.filePath always links to HuggingFace Hub (see read_local_shard_urls),
    # which needs manifest.csv sitting right next to the extracted images —
    # not just the tar shards themselves.
    manifest_path = source_dir / HFH_MANIFEST_FILENAME
    if manifest_path.is_file() and manifest_path.parent != extracted_dir:
        shutil.copy2(manifest_path, extracted_dir / HFH_MANIFEST_FILENAME)

    return extracted_dir


def resolve_gbif_source_dataset_dir(
    repo_id: Optional[str], explicit_source_dataset_dir: Optional[Path],
) -> Path:
    """--source-dataset-dir, if given explicitly, always wins (skips
    HuggingFace Hub entirely — useful to try 'gbif prepare' before anything
    is published). Otherwise resolves the dataset from --repo-id: reuses
    hfh_download/ as-is if a previous run already extracted it there;
    failing that, reuses the local 'huggingface prepare' export at
    <Documents>/donadataset/HFH/<repo_id> if it's already there, downloading
    it fresh into hfh_download/ only when neither is available. Either way,
    the .tar shards get extracted once into that same hfh_download/ and
    reused on later runs instead of re-extracting every time."""
    if explicit_source_dataset_dir is not None:
        return explicit_source_dataset_dir

    if not repo_id or "REPLACE_WITH" in repo_id:
        fail(
            "Either --source-dataset-dir or a valid --repo-id (huggingface.repo_id) is "
            "required so 'gbif prepare' knows where to read images/labels from."
        )

    download_dir = get_gbif_hfh_download_dir(repo_id)
    if _looks_like_extracted_yolo_dataset(download_dir):
        logging.info("Reusing already-extracted YOLO dataset: %s", download_dir)
        return download_dir

    hfh_output_dir = get_hfh_output_dir(repo_id)
    if _looks_like_hfh_export(hfh_output_dir):
        logging.info("Found a local HuggingFace Hub export already on disk: %s", hfh_output_dir)
        source_dir = hfh_output_dir
    else:
        logging.info(
            "No local HuggingFace Hub export found at %s — downloading the published repo "
            "%s into %s instead...", hfh_output_dir, repo_id, download_dir,
        )
        token = os.environ.get(HUGGINGFACE_TOKEN_ENV_VAR) or None
        download_repository(
            repo_id=repo_id, repo_type="dataset", token=token, download_dir=download_dir, verify_data=True,
        )
        source_dir = download_dir

    return extract_hfh_export_to_yolo_source(source_dir, download_dir)


def resolve_gbif_hfh_output_dir(
    repo_id: Optional[str], explicit_hfh_output_dir: Optional[Path], *, dry_run: bool = False,
) -> Optional[Path]:
    """Same "local first, download-and-cache otherwise" resolution as
    resolve_gbif_source_dataset_dir, but for the RAW HuggingFace Hub export
    'gbif upload' needs (tar shards + README/CITATION.cff/manifest.csv/
    checksums-sha256.txt as actually published — not the images/labels/
    tree resolve_gbif_source_dataset_dir extracts for scanning). Reuses the
    very same hfh_download/ cache 'gbif prepare' already populates; that
    directory may ALSO carry prepare's own extracted images/labels/ from an
    earlier run, but run_upload's checksum regeneration only ever re-hashes
    files already listed in checksums-sha256.txt plus the new archive,
    never a fresh directory scan (see regenerate_hfh_checksums), so that
    clutter can never leak into what gets re-published. dry_run never
    downloads — if neither location has the export yet, it logs what would
    happen and returns None."""
    if explicit_hfh_output_dir is not None:
        return explicit_hfh_output_dir

    hfh_output_dir = get_hfh_output_dir(repo_id)
    if _looks_like_hfh_export(hfh_output_dir):
        return hfh_output_dir

    download_dir = get_gbif_hfh_download_dir(repo_id)
    if _looks_like_hfh_export(download_dir):
        return download_dir

    if dry_run:
        logging.info(
            "Dry run enabled. Neither %s nor %s has a local HuggingFace Hub export yet — "
            "would download the published repo %s into %s.",
            hfh_output_dir, download_dir, repo_id, download_dir,
        )
        return None

    logging.info(
        "No local HuggingFace Hub export found at %s or %s — downloading the published repo "
        "%s into %s instead...", hfh_output_dir, download_dir, repo_id, download_dir,
    )
    token = os.environ.get(HUGGINGFACE_TOKEN_ENV_VAR) or None
    download_repository(
        repo_id=repo_id, repo_type="dataset", token=token, download_dir=download_dir, verify_data=True,
    )
    return download_dir


def regenerate_hfh_checksums(hfh_output_dir: Path, extra_relative_paths: List[str]) -> Path:
    """Recomputes checksums-sha256.txt from exactly the files it ALREADY
    lists (re-hashed from disk, not trusted as-is) plus extra_relative_paths
    — deliberately never a fresh directory scan (unlike huggingface.py's own
    write_checksums), so local-only clutter that was never part of the
    published HuggingFace Hub repo (e.g. 'gbif prepare's own extracted
    images/labels/ sitting inside a shared hfh_download/ cache — see
    resolve_gbif_hfh_output_dir) can never leak into what gets
    re-published."""
    checksum_path = hfh_output_dir / HFH_CHECKSUMS_FILENAME
    if not checksum_path.is_file():
        fail(
            f"{HFH_CHECKSUMS_FILENAME} not found in {hfh_output_dir} — can't safely refresh it "
            "without the existing file list. Run 'huggingface prepare' for this dataset first."
        )
    existing = read_checksums(checksum_path)
    relative_paths = sorted(set(existing) | set(extra_relative_paths))

    with checksum_path.open("w", encoding="utf-8") as f:
        for rel in relative_paths:
            f.write(f"{sha256_file(hfh_output_dir / rel)}  {rel}\n")

    return checksum_path


def run_prepare(
    source_dataset_dir: Optional[Path],
    output_dir: Path,
    *,
    repo_id: Optional[str] = None,
    dataset_slug: str = "donadataset",
    dataset_name: str = "DonaDataset",
    description: str = "",
    license_id: str = "CC-BY-4.0",
    license_name: str = "Creative Commons Attribution 4.0 International",
    license_url: str = "https://creativecommons.org/licenses/by/4.0/",
    rights_holder: str = "",
    institution_code: str = "",
    contact_name: str = "",
    contact_email: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Purely local — never uploads anything (see run_upload for that).
    media.filePath always links to the HuggingFace Hub .tar shard each
    image was packed into — there is no local-relative-path mode — so
    repo_id is required even when --source-dataset-dir is used explicitly."""
    if not repo_id or "REPLACE_WITH" in repo_id:
        fail(
            "A valid HuggingFace repo_id is required — media.filePath always links to the "
            "HuggingFace Hub .tar shard each image was packed into. Pass --hf-repo-id, or "
            "set one with 'donadataset publish huggingface config set repo_id=user_or_org/dataset'."
        )

    source_dataset_dir = resolve_gbif_source_dataset_dir(repo_id, source_dataset_dir)
    if not source_dataset_dir.is_dir():
        fail(f"Source dataset directory not found: {source_dataset_dir}")

    config: Dict[str, Any] = {"paths": {"source_dataset_dir": str(source_dataset_dir)}}

    logging.info("Scanning source dataset: %s", source_dataset_dir)
    items, issues = scan_dataset(config)
    stop_if_validation_errors(issues)
    if not items:
        fail(f"No valid images found under {source_dataset_dir}.")

    classes = get_classes(config)
    shard_url_by_image_id = read_local_shard_urls(source_dataset_dir, repo_id)

    logging.info("Reading EXIF capture dates and building Camtrap DP deployments/media/observations...")
    deployment_rows, media_rows, observation_rows = build_camtrap_dp_resources(
        items, classes, shard_url_by_image_id=shard_url_by_image_id,
    )
    if not deployment_rows:
        fail(f"No deployments were generated — is {source_dataset_dir} empty for every split?")

    logging.info(
        "Deployments: %d, media: %d, observations: %d",
        len(deployment_rows), len(media_rows), len(observation_rows),
    )

    # Only the Camtrap DP deliverables are checked/cleared here — never the
    # whole directory. --output-dir also hosts hfh_download/ (see
    # get_gbif_hfh_download_dir), which must survive across runs regardless
    # of --overwrite, and whose mere presence would otherwise make every
    # first run look like "output already exists".
    existing_package_files = [
        path for path in [
            output_dir / DEPLOYMENTS_FILENAME,
            output_dir / MEDIA_FILENAME,
            output_dir / OBSERVATIONS_FILENAME,
            output_dir / DATAPACKAGE_FILENAME,
            *output_dir.glob("*-camtrap-dp.zip"),
        ]
        if path.exists()
    ]

    if existing_package_files:
        if not overwrite:
            fail(
                f"A Camtrap DP package already exists in {output_dir}. "
                "Delete it yourself, or re-run with --overwrite to let 'prepare' replace it."
            )
        logging.info("Removing existing Camtrap DP package files before regenerating...")
        for path in existing_package_files:
            path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(deployment_rows, DEPLOYMENT_FIELDNAMES, output_dir / DEPLOYMENTS_FILENAME)
    _write_csv(media_rows, MEDIA_FIELDNAMES, output_dir / MEDIA_FILENAME)
    _write_csv(observation_rows, OBSERVATION_FIELDNAMES, output_dir / OBSERVATIONS_FILENAME)

    datapackage = build_datapackage(
        deployment_rows, observation_rows,
        dataset_slug=dataset_slug, dataset_name=dataset_name, description=description,
        license_id=license_id, license_name=license_name, license_url=license_url,
        rights_holder=rights_holder, contact_name=contact_name, contact_email=contact_email,
    )
    write_json(output_dir / DATAPACKAGE_FILENAME, datapackage)

    archive_path = output_dir / f"{dataset_slug}-camtrap-dp.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in (DATAPACKAGE_FILENAME, DEPLOYMENTS_FILENAME, MEDIA_FILENAME, OBSERVATIONS_FILENAME):
            zf.write(output_dir / filename, arcname=filename)

    logging.info("Camtrap DP package ready: %s", archive_path)
    logging.info(
        "Either upload it to your GBIF IPT (v3+, supports Camtrap DP natively) by hand, or "
        "run 'donadataset publish gbif upload' to host it yourself on the already-published "
        "HuggingFace Hub repo, then run 'donadataset publish gbif register --archive-url "
        "<that URL>' to register it via the Registry API."
    )


# ── register ──────────────────────────────────────────────────────────────

GBIF_USERNAME_ENV_VAR = "GBIF_USERNAME"
GBIF_PASSWORD_ENV_VAR = "GBIF_PASSWORD"
GBIF_ENDPOINT_TYPE    = "CAMTRAP_DP"

GBIF_REGISTRY_BASE_URLS = {
    "sandbox": "https://api.gbif-test.org",
    "production": "https://api.gbif.org",
}
GBIF_DATASET_PAGE_URL_TEMPLATES = {
    "sandbox": "https://registry.gbif-test.org/dataset/{key}",
    "production": "https://www.gbif.org/dataset/{key}",
}
GBIF_LINKED_RECORD_FILENAME = "gbif_linked_dataset_record.json"


def get_gbif_credentials() -> Tuple[str, str]:
    """Resolves GBIF Registry API credentials: each environment variable
    always wins if set; otherwise falls back to gbif.username/gbif.password
    in settings.toml (set via 'donadataset publish gbif config set
    username'/'config set password')."""
    username = os.environ.get(GBIF_USERNAME_ENV_VAR) or global_settings.GBIF.username
    password = os.environ.get(GBIF_PASSWORD_ENV_VAR) or global_settings.GBIF.password
    if not username or not password:
        fail(
            f"No GBIF Registry API credentials found (your gbif.org account — ideally an "
            f"institutional account, not a personal one). Set the environment variables "
            f"{GBIF_USERNAME_ENV_VAR}/{GBIF_PASSWORD_ENV_VAR}, or store them with "
            "'donadataset publish gbif config set username' / 'config set password'."
        )
    return username, password


def run_register(
    archive_url: str,
    output_dir: Path,
    *,
    environment: str,
    publishing_organization_key: Optional[str],
    installation_key: Optional[str],
    dataset_name: str,
    description: str,
    license_url: str,
    registry_language: str,
    dry_run: bool = False,
) -> None:
    """Registers (first run) or updates (later runs) the dataset in the GBIF
    Registry directly via API, bypassing the IPT entirely.

    GBIF still has to be able to fetch the archive over HTTP(S) — this
    function does not upload anything anywhere, it just tells the Registry
    where to find a copy you already host (--archive-url). The dataset type
    is fixed to 'OCCURRENCE' (GBIF ingests Camtrap DP by converting it to
    Darwin Core Occurrence records internally) and the endpoint type to
    'CAMTRAP_DP', matching what run_prepare() produces."""
    if not (archive_url.startswith("http://") or archive_url.startswith("https://")):
        fail(f"--archive-url must be a public http(s) URL, got: {archive_url}")
    if environment not in GBIF_REGISTRY_BASE_URLS:
        fail(f"gbif.environment must be 'sandbox' or 'production', got: {environment}")
    if not publishing_organization_key or not installation_key:
        fail(
            "gbif.publishing_organization_key and gbif.installation_key must both be set — they "
            "can't be guessed. Register an organisation/installation once on gbif.org (or its "
            "sandbox at gbif-test.org) and set them with 'donadataset publish gbif config set "
            "publishing_organization_key=...' and 'installation_key=...'."
        )

    username, password = get_gbif_credentials()
    base_url = GBIF_REGISTRY_BASE_URLS[environment]
    auth = (username, password)

    record_path = output_dir / GBIF_LINKED_RECORD_FILENAME
    existing_record = read_json(record_path) if record_path.exists() else {}
    dataset_key = existing_record.get("dataset_key")

    dataset_payload = {
        "publishingOrganizationKey": publishing_organization_key,
        "installationKey": installation_key,
        "type": "OCCURRENCE",
        "title": dataset_name,
        "description": description,
        "language": registry_language,
        "license": license_url,
    }

    if dry_run:
        logging.info(
            "Dry run enabled. Would %s the GBIF dataset at %s with: %s",
            "update" if dataset_key else "create", base_url, dataset_payload,
        )
        logging.info("Would then point its %s endpoint at: %s", GBIF_ENDPOINT_TYPE, archive_url)
        return

    if dataset_key:
        logging.info("Updating existing GBIF dataset: %s", dataset_key)
        response = requests.put(
            f"{base_url}/v1/dataset/{dataset_key}", json={**dataset_payload, "key": dataset_key}, auth=auth,
        )
        response.raise_for_status()
    else:
        logging.info("Registering new GBIF dataset (%s)...", environment)
        response = requests.post(f"{base_url}/v1/dataset", json=dataset_payload, auth=auth)
        response.raise_for_status()
        dataset_key = response.json()
        logging.info("Created GBIF dataset: %s", dataset_key)

    # Drop any endpoint of our type left over from a previous 'register' run
    # before adding the current one, so re-registering never leaves stale
    # duplicate endpoints pointing at old/renamed archive URLs.
    endpoints_response = requests.get(f"{base_url}/v1/dataset/{dataset_key}/endpoint", auth=auth)
    endpoints_response.raise_for_status()
    for endpoint in endpoints_response.json():
        if endpoint.get("type") == GBIF_ENDPOINT_TYPE:
            delete_response = requests.delete(
                f"{base_url}/v1/dataset/{dataset_key}/endpoint/{endpoint['key']}", auth=auth,
            )
            delete_response.raise_for_status()

    logging.info("Adding %s endpoint: %s", GBIF_ENDPOINT_TYPE, archive_url)
    endpoint_response = requests.post(
        f"{base_url}/v1/dataset/{dataset_key}/endpoint",
        json={"type": GBIF_ENDPOINT_TYPE, "url": archive_url},
        auth=auth,
    )
    endpoint_response.raise_for_status()

    dataset_page_url = GBIF_DATASET_PAGE_URL_TEMPLATES[environment].format(key=dataset_key)
    write_json(record_path, {
        "dataset_key": dataset_key,
        "environment": environment,
        "archive_url": archive_url,
        "dataset_page_url": dataset_page_url,
        "registered_at_utc": utc_now_iso(),
    })

    logging.info("GBIF dataset registered: %s", dataset_page_url)
    logging.info("GBIF crawls new/updated endpoints within a few hours — check back at the link above.")
