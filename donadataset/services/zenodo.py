"""Lógica de interacción con Zenodo (sin nada de CLI/Typer).

Implementa un flujo de "registro Zenodo enlazado" (linked dataset record):

- Los shards pesados del dataset se quedan en HuggingFace Hub.
- Zenodo almacena metadata, manifests, checksums, reports de verificación y
  enlaces (no es un archivo físico completo de todas las imágenes).
- Zenodo asigna un DOI al registro enlazado y verificado.

Tres operaciones:

- `run_zenodo_linked_dataset_creation` / `run_zenodo_existing_draft_sync`:
  crea (o sincroniza) el depósito Zenodo y sube los ficheros de evidencia
  (manifests, checksums, reports...) — no los shards, que ya están en HFH.
- `run_update_local_metadata_with_doi`: una vez asignado el DOI, lo inserta en
  los metadatos locales (HuggingFaceHub.yaml, dataset_info.json, CITATION.cff,
  README.md) y regenera checksums-sha256.txt.
- `run_download_and_deploy`: descarga un registro Zenodo (completo o
  enlazado a HuggingFace Hub) y despliega el dataset en formato YOLO.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from huggingface_hub import HfApi

from donadataset.config import settings as global_settings
from donadataset.services.common import (
    as_bool,
    ensure_dict,
    fail,
    format_size,
    get_nested,
    load_yaml,
    md5_file,
    read_checksums,
    read_json,
    setup_logging,
    sha256_file,
    utc_now_iso,
    write_json,
    write_yaml,
)
from donadataset.services.huggingface import (
    authenticate,
    build_dataset_url,
    download_and_verify_hfh,
    ensure_dir,
    get_chunk_size_bytes,
    get_dataset_slug,
    get_dataset_visibility,
    get_internal_config_filename,
    get_output_dir,
    get_public_visibility_report_path,
    get_repo_id,
    get_repo_type,
    get_token,
    get_token_env_var,
    get_tree_url,
    load_config_source,
    check_public_url,
    verify_global_checksums,
)

# ── Naming policy ─────────────────────────────────────────────────────────────

INTERNAL_CONFIG_FILENAME = "HuggingFaceHub.yaml"

# Evidence files stored at the root of the HFH_Z_<dataset_slug> export folder.
# verification_report_downloaded.json is intentionally excluded: Zenodo always
# generates its own fresh copy (get_zenodo_downloaded_report_path, written
# into Zenodo's own output dir) instead of trusting whatever a separate,
# possibly-stale 'huggingface download' run last produced.
DEFAULT_ZENODO_EVIDENCE_FILENAMES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    INTERNAL_CONFIG_FILENAME,
    "donana.yaml",
    "dataset_info.json",
    "metadata.csv",
    "manifest.csv",
    "manifest-files-sha256.csv",
    "checksums-sha256.txt",
    "validation_report.json",
    "verification_report_local.json",
]

# Historical naming this project no longer uses, kept only so an old external
# YAML with stale zenodo.files_to_upload entries still resolves correctly.
# HuggingFaceHub_Zenodo.yaml was itself the *current* name before this file
# went back to plain HuggingFaceHub.yaml (see donadataset/services/huggingface.py) —
# 'huggingface prepare' never generates anything Zenodo-specific, so baking a
# downstream repository's name into it was a naming leak, not a hard
# requirement of the format.
OLD_INTERNAL_CONFIG_FILENAMES = {
    "HuggingFaceHub_Zenodo.yaml",
    "HuggingFaceHub_Toy.yaml",
    "HuggingFaceHub_Full.yaml",
}
OLD_HFH_EXPORT_DIR_NAMES = {"HFH", "HFH_Toy"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTENSIONS = {".txt"}
HFH_DATASET_URL_RE = re.compile(
    r"https?://huggingface\.co/datasets/([^/\s]+/[^/\s?#]+)(?:/tree/([^\s?#]+))?"
)


# ── Configuración ─────────────────────────────────────────────────────────────

def get_zenodo_environment(config: Dict[str, Any]) -> str:
    env = str(get_nested(config, ["zenodo", "environment"], "sandbox")).strip().lower()
    if env not in {"sandbox", "production"}:
        fail("zenodo.environment must be either 'sandbox' or 'production'.")
    return env


def get_zenodo_base_url(config: Dict[str, Any]) -> str:
    return "https://sandbox.zenodo.org" if get_zenodo_environment(config) == "sandbox" else "https://zenodo.org"


def get_zenodo_api_base_url(config: Dict[str, Any]) -> str:
    return f"{get_zenodo_base_url(config)}/api"


def get_zenodo_token_env_var(config: Dict[str, Any]) -> str:
    token_env_var = str(get_nested(config, ["zenodo", "token_env_var"], "")).strip()
    if not token_env_var:
        env = get_zenodo_environment(config)
        token_env_var = "ZENODO_SANDBOX_TOKEN" if env == "sandbox" else "ZENODO_TOKEN"
    return token_env_var


def get_zenodo_token(config: Dict[str, Any]) -> str:
    """Resolves the Zenodo token: the environment variable always wins if
    set; otherwise falls back to zenodo.token in settings.toml (set via
    'donadataset publish zenodo config set token')."""
    token_env_var = get_zenodo_token_env_var(config)
    token = os.environ.get(token_env_var)
    if token:
        return token
    if global_settings.ZENODO.token:
        return global_settings.ZENODO.token
    base_url = get_zenodo_base_url(config)
    fail(
        f"No Zenodo token found. Get one at "
        f"{base_url}/account/settings/applications/tokens/new/ "
        f"(check the 'deposit:write' scope, and also 'deposit:actions' if you'll "
        f"use 'zenodo release'), then either set it with: "
        f"export {token_env_var}='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', or store it "
        "with 'donadataset publish zenodo config set token'."
    )


def get_publish_flag(config: Dict[str, Any]) -> bool:
    return as_bool(get_nested(config, ["zenodo", "publish"], False), False)


def get_hfh_export_dir(config: Dict[str, Any]) -> Path:
    """Same export directory 'huggingface prepare' created (paths.output_dir /
    export.output_dir_template) — Zenodo evidence files live inside it."""
    return get_output_dir(config)


def build_zenodo_template_context(
    *,
    hfh_output_dir: Optional[str] = None,
    zenodo_output_dir: Optional[str] = None,
    repo_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    description: Optional[str] = None,
    license_id: Optional[str] = None,
    author_given_names: Optional[str] = None,
    author_family_names: Optional[str] = None,
    author_affiliation: Optional[str] = None,
    environment: Optional[str] = None,
) -> Dict[str, Any]:
    """Builds the Jinja context for templates/Zenodo.yaml.j2. All fallbacks
    are resolved here (never inside the template) because the template
    concatenates some of these into longer strings, which would raise on a
    bare None. huggingface_dataset_url/tree_url are derived from repo_id —
    never written by hand.

    Two directories are genuinely different things: hfh_output_dir is where
    'huggingface prepare' already put the export (an INPUT to read evidence
    files from); zenodo_output_dir is where 'zenodo prepare' downloads
    HuggingFace Hub fresh and stages the package it uploads to Zenodo (its
    own OUTPUT)."""
    if repo_id:
        huggingface_dataset_url = build_dataset_url(repo_id)
        huggingface_tree_url = get_tree_url(repo_id)
    else:
        huggingface_dataset_url = "https://huggingface.co/datasets/REPLACE_WITH_HF_USER/REPLACE_WITH_DATASET_SLUG"
        huggingface_tree_url = huggingface_dataset_url + "/tree/main"

    return {
        "output_dir": hfh_output_dir,
        "zenodo_output_dir": zenodo_output_dir or "REPLACE_WITH_ZENODO_OUTPUT_DIR",
        "repo_id": repo_id,
        "dataset_name": dataset_name or "REPLACE_WITH_DATASET_NAME",
        "description": description or "REPLACE_WITH_DESCRIPTION.",
        "license_id": license_id or "REPLACE_WITH_LICENSE_ID",
        "author_given_names": author_given_names or "",
        "author_family_names": author_family_names or "REPLACE_WITH_FAMILY_NAMES",
        "author_affiliation": author_affiliation,
        "huggingface_dataset_url": huggingface_dataset_url,
        "huggingface_tree_url": huggingface_tree_url,
        "environment": environment or "sandbox",
    }


def get_zenodo_output_dir(config: Dict[str, Any]) -> Path:
    """Directory where everything Zenodo-related is staged: copies of the
    evidence files actually uploaded, plus the JSON reports 'prepare'/
    'upload'/'check-readiness'/'release' write. Deliberately separate from
    the HuggingFace export dir (zenodo.output_dir override wins; otherwise a
    sibling of the HFH export dir named Zenodo_{dataset_slug})."""
    configured = get_nested(config, ["zenodo", "output_dir"], None)
    if configured:
        return Path(str(configured))

    hfh_export_dir = get_hfh_export_dir(config)
    return hfh_export_dir.parent / f"Zenodo_{get_dataset_slug(config)}"


def stage_files_for_zenodo(files: List[Path], output_dir: Path) -> List[Path]:
    """Copy each file into output_dir (flat, by filename) so the directory
    ends up containing exactly what was uploaded to Zenodo — files are
    validated at their original location first, so this only runs once
    that's already confirmed to succeed."""
    ensure_dir(output_dir)
    staged: List[Path] = []
    for path in files:
        destination = output_dir / path.name
        if destination.resolve() != path.resolve():
            shutil.copy2(path, destination)
        staged.append(destination)
    return staged


def get_zenodo_hfh_download_dir(config: Dict[str, Any]) -> Path:
    return get_zenodo_output_dir(config) / "hfh_download"


def get_zenodo_downloaded_report_path(config: Dict[str, Any]) -> Path:
    """Where 'zenodo prepare' writes its OWN, freshly generated download
    verification report — deliberately separate from get_downloaded_report_path
    (huggingface.py), which is 'huggingface download's own report and may be
    stale or configured to live somewhere else entirely."""
    return get_zenodo_output_dir(config) / "verification_report_downloaded.json"


def ensure_fresh_hfh_download_report(config: Dict[str, Any], verify_data: bool = False) -> Dict[str, Any]:
    """Downloads the HuggingFace Hub repo right now and verifies it against
    the local manifest/checksums, so 'zenodo prepare' never has to trust a
    possibly-stale report from an earlier, separate 'huggingface download'
    run — what's staged in Zenodo's directory is guaranteed to match what's
    live on HuggingFace Hub at the moment 'zenodo prepare' actually runs.

    verify_data=False (the default) skips the heavy data/<split>/*.tar
    shards entirely — Zenodo never uploads them anyway (see module
    docstring), so by default this only downloads and verifies the small
    evidence files. Pass verify_data=True for the older, slower behaviour:
    download every shard too and re-hash its contents against
    manifest-files-sha256.csv, for an extra guarantee that the published
    images/labels themselves — not just the metadata describing them —
    still match what 'prepare' originally wrote."""
    download_dir = get_zenodo_hfh_download_dir(config)
    report_path = get_zenodo_downloaded_report_path(config)
    token = get_token(config)

    logging.info("Downloading HuggingFace Hub repository to verify it matches the local export...")
    logging.info("Download directory: %s", download_dir)
    logging.info("Verifying data/ shards too: %s", verify_data)
    return download_and_verify_hfh(
        config, token, download_dir, report_path, delete_after_success=False, verify_data=verify_data,
    )


def get_default_files_to_upload(config: Dict[str, Any]) -> List[Path]:
    """Evidence files come from the live HuggingFace Hub download (staged at
    get_zenodo_hfh_download_dir), not from a separate local pre-upload
    directory — whatever was pushed there with 'huggingface upload' already
    contains README/LICENSE/CITATION.cff/manifests/checksums, so there is no
    need to also know where the original, pre-upload export sits."""
    hfh_download_dir = get_zenodo_hfh_download_dir(config)
    files = [hfh_download_dir / filename for filename in DEFAULT_ZENODO_EVIDENCE_FILENAMES]
    files.append(get_zenodo_downloaded_report_path(config))
    return files


def normalize_zenodo_upload_path(path: Path, config: Dict[str, Any]) -> Path:
    """Map historical export-folder references to the current naming policy."""
    hfh_export_dir = get_hfh_export_dir(config)

    if path.name in OLD_INTERNAL_CONFIG_FILENAMES:
        return hfh_export_dir / get_internal_config_filename(config)

    if path.parts and (path.parts[0] in OLD_HFH_EXPORT_DIR_NAMES or path.parts[0].startswith("HFH_Z_")):
        if path.name in OLD_INTERNAL_CONFIG_FILENAMES:
            return hfh_export_dir / get_internal_config_filename(config)
        return hfh_export_dir.joinpath(*path.parts[1:])

    return path


def unique_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    result: List[Path] = []
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def get_required_zenodo_evidence_files(config: Dict[str, Any]) -> List[Path]:
    return get_default_files_to_upload(config)


def get_files_to_upload(config: Dict[str, Any]) -> List[Path]:
    files = get_nested(config, ["zenodo", "files_to_upload"], None)

    if files is None or files == []:
        return unique_paths(get_default_files_to_upload(config))

    if not isinstance(files, list):
        fail("zenodo.files_to_upload must be a list when provided.")

    configured_files = [normalize_zenodo_upload_path(Path(str(p)), config) for p in files]
    return unique_paths(configured_files + get_required_zenodo_evidence_files(config))


def get_output_filename(config: Dict[str, Any], key: str, default: str) -> Path:
    configured = get_nested(config, ["zenodo", "output", key], None)
    if configured:
        return Path(str(configured))
    return get_zenodo_output_dir(config) / default


def get_related_links(config: Dict[str, Any]) -> Dict[str, str]:
    related = get_nested(config, ["zenodo", "related_identifiers"], {})
    links: Dict[str, str] = {}
    if isinstance(related, dict):
        for key, value in related.items():
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str and "REPLACE_WITH" not in value_str:
                links[str(key)] = value_str
    return links


def get_link_check_timeout(config: Dict[str, Any]) -> int:
    return int(get_nested(config, ["zenodo", "link_checking", "timeout_seconds"], 20))


def get_allowed_private_hfh_status_codes(config: Dict[str, Any]) -> List[int]:
    values = get_nested(config, ["zenodo", "link_checking", "allow_private_hfh_status_codes"], [401, 403, 404])
    if not isinstance(values, list):
        return [401, 403, 404]
    result = []
    for value in values:
        try:
            result.append(int(value))
        except ValueError:
            continue
    return result


# ── Precondiciones ────────────────────────────────────────────────────────────


def validate_files_to_upload(files: List[Path]) -> None:
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        fail("Some Zenodo evidence files do not exist:\n" + "\n".join(f"  - {path}" for path in missing))


# ── Metadata Zenodo ───────────────────────────────────────────────────────────

def build_zenodo_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    creators_raw = get_nested(config, ["zenodo", "creators"], [])
    if not isinstance(creators_raw, list) or not creators_raw:
        fail("zenodo.creators must be a non-empty list.")

    creators = []
    for creator in creators_raw:
        if not isinstance(creator, dict):
            continue
        name = str(creator.get("name", "")).strip()
        if not name:
            continue
        item: Dict[str, Any] = {"name": name}
        affiliation = creator.get("affiliation")
        orcid = creator.get("orcid")
        if affiliation and "REPLACE_WITH" not in str(affiliation):
            item["affiliation"] = str(affiliation)
        if orcid:
            item["orcid"] = str(orcid)
        creators.append(item)

    if not creators:
        fail("zenodo.creators does not contain any valid creator.")

    keywords = get_nested(config, ["zenodo", "keywords"], [])
    if not isinstance(keywords, list):
        keywords = []

    related_identifiers = [
        {"identifier": url, "relation": "isSupplementTo", "resource_type": "dataset", "scheme": "url"}
        for url in get_related_links(config).values()
    ]

    return {
        "title": str(get_nested(config, ["zenodo", "title"], "Linked dataset record")),
        "upload_type": str(get_nested(config, ["zenodo", "upload_type"], "dataset")),
        "description": str(get_nested(config, ["zenodo", "description"], "")),
        "creators": creators,
        "access_right": str(get_nested(config, ["zenodo", "access_right"], "open")),
        "license": str(get_nested(config, ["zenodo", "license"], "cc-by-4.0")),
        "keywords": [str(keyword) for keyword in keywords],
        "notes": str(get_nested(config, ["zenodo", "notes"], "")),
        "related_identifiers": related_identifiers,
        "prereserve_doi": True,
    }


# ── API de Zenodo ─────────────────────────────────────────────────────────────

def zenodo_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def zenodo_binary_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"}


def check_response(response: requests.Response, expected: Tuple[int, ...], context: str) -> None:
    if response.status_code in expected:
        return
    try:
        body = response.json()
    except Exception:
        body = response.text
    fail(f"{context} failed. HTTP status={response.status_code}. Response={body}")


def create_deposition(api_base_url: str, token: str) -> Dict[str, Any]:
    response = requests.post(f"{api_base_url}/deposit/depositions", headers=zenodo_headers(token), json={}, timeout=60)
    check_response(response, (200, 201), "Create Zenodo deposition")
    return response.json()


def update_deposition_metadata(api_base_url: str, token: str, deposition_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{api_base_url}/deposit/depositions/{deposition_id}"
    response = requests.put(url, headers=zenodo_headers(token), json={"metadata": metadata}, timeout=60)
    check_response(response, (200, 201), "Update Zenodo deposition metadata")
    return response.json()


def get_deposition(api_base_url: str, token: str, deposition_id: int) -> Dict[str, Any]:
    url = f"{api_base_url}/deposit/depositions/{deposition_id}"
    response = requests.get(url, headers=zenodo_headers(token), timeout=60)
    check_response(response, (200,), "Get Zenodo deposition")
    return response.json()


def upload_file_to_bucket(bucket_url: str, token: str, file_path: Path, remote_filename: str) -> Dict[str, Any]:
    url = f"{bucket_url}/{quote(remote_filename)}"
    with file_path.open("rb") as f:
        response = requests.put(url, headers=zenodo_binary_headers(token), data=f, timeout=300)
    check_response(response, (200, 201), f"Upload file to Zenodo bucket: {remote_filename}")
    return response.json()


def publish_deposition(api_base_url: str, token: str, deposition_id: int) -> Dict[str, Any]:
    url = f"{api_base_url}/deposit/depositions/{deposition_id}/actions/publish"
    response = requests.post(url, headers=zenodo_headers(token), timeout=60)
    check_response(response, (200, 201, 202), "Publish Zenodo deposition")
    return response.json()


# ── DOI y registro enlazado ───────────────────────────────────────────────────

def extract_reserved_doi(deposition: Dict[str, Any]) -> Optional[str]:
    metadata = deposition.get("metadata", {})
    if isinstance(metadata, dict):
        prereserve = metadata.get("prereserve_doi")
        if isinstance(prereserve, dict) and prereserve.get("doi"):
            return str(prereserve["doi"])
        if metadata.get("doi"):
            return str(metadata["doi"])
    if deposition.get("doi"):
        return str(deposition["doi"])
    return None


def build_doi_url(doi: Optional[str]) -> Optional[str]:
    return f"https://doi.org/{doi}" if doi else None


def build_record_url(base_url: str, deposition: Dict[str, Any]) -> Optional[str]:
    # Prefer the URL the API itself returns (more reliable once published)
    # before falling back to constructing it from record_id/id.
    links = deposition.get("links", {})
    if isinstance(links, dict):
        record_html = links.get("record_html") or links.get("html")
        if record_html and "/records/" in str(record_html):
            return str(record_html)

    record_id = deposition.get("record_id") or deposition.get("id")
    return f"{base_url}/records/{record_id}" if record_id else None


def create_linked_dataset_record(
    config: Dict[str, Any], deposition: Dict[str, Any], files: List[Path], downloaded_report: Dict[str, Any],
) -> Dict[str, Any]:
    base_url = get_zenodo_base_url(config)
    doi = extract_reserved_doi(deposition)

    file_entries = [
        {
            "path": str(path), "name": path.name, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "md5": md5_file(path),
        }
        for path in files
    ]

    return {
        "generated_at_utc": utc_now_iso(),
        "record_type": "zenodo_linked_dataset_record",
        "record_scope": (
            "This record stores metadata, manifests, checksums, verification reports, "
            "and links. Heavy dataset shards are hosted on Hugging Face Hub."
        ),
        "zenodo_environment": get_zenodo_environment(config),
        "zenodo_base_url": base_url,
        "deposition_id": deposition.get("id"),
        "record_id": deposition.get("record_id"),
        "reserved_doi": doi,
        "doi_url": build_doi_url(doi),
        "record_url": build_record_url(base_url, deposition),
        "huggingface_verification": {
            "status": downloaded_report.get("status"),
            "repo_id": downloaded_report.get("repo_id"),
            "repo_type": downloaded_report.get("repo_type"),
            "global_files_verified": downloaded_report.get("global_files_verified"),
            "internal_tar_members_verified": downloaded_report.get("internal_tar_members_verified"),
            "num_errors": downloaded_report.get("num_errors"),
        },
        "related_links": get_related_links(config),
        "files_to_upload": file_entries,
    }


# ── Verificación de ficheros remotos ─────────────────────────────────────────

def normalize_remote_checksum(value: Any) -> Optional[Tuple[str, str]]:
    if not value:
        return None
    raw = str(value)
    if ":" in raw:
        algorithm, digest = raw.split(":", 1)
        return algorithm.lower(), digest.lower()
    return "md5", raw.lower()


def get_remote_files_from_deposition(deposition: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    files = deposition.get("files", [])
    if not isinstance(files, list):
        return result
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        filename = file_info.get("filename") or file_info.get("key") or file_info.get("name")
        if filename:
            result[str(filename)] = file_info
    return result


def verify_uploaded_files(local_files: List[Path], deposition: Dict[str, Any]) -> Dict[str, Any]:
    remote_files = get_remote_files_from_deposition(deposition)
    errors: List[str] = []
    verified_files = []

    for local_path in local_files:
        remote_name = local_path.name
        remote = remote_files.get(remote_name)

        if remote is None:
            errors.append(f"Missing remote file in Zenodo deposition: {remote_name}")
            continue

        local_size = local_path.stat().st_size
        remote_size = remote.get("filesize") or remote.get("size")
        size_ok = None
        if remote_size is not None:
            try:
                size_ok = int(remote_size) == int(local_size)
            except Exception:
                size_ok = False
            if not size_ok:
                errors.append(f"Size mismatch for {remote_name}: local={local_size}, remote={remote_size}")

        checksum_ok = None
        checksum_info = normalize_remote_checksum(remote.get("checksum"))
        if checksum_info is not None:
            algorithm, digest = checksum_info
            local_digest = None
            if algorithm == "md5":
                local_digest = md5_file(local_path)
                checksum_ok = local_digest.lower() == digest.lower()
            elif algorithm == "sha256":
                local_digest = sha256_file(local_path)
                checksum_ok = local_digest.lower() == digest.lower()
            if checksum_ok is False:
                errors.append(
                    f"Checksum mismatch for {remote_name}: algorithm={algorithm}, "
                    f"expected_remote={digest}, local={local_digest}"
                )

        verified_files.append({
            "filename": remote_name, "local_path": str(local_path),
            "local_size_bytes": local_size, "remote_size_bytes": remote_size, "size_ok": size_ok,
            "remote_checksum": remote.get("checksum"), "checksum_ok": checksum_ok,
        })

    return {
        "generated_at_utc": utc_now_iso(),
        "status": "passed" if not errors else "failed",
        "num_local_files": len(local_files),
        "num_remote_files": len(remote_files),
        "num_errors": len(errors),
        "errors": errors,
        "verified_files": verified_files,
    }


# ── Verificación de enlaces ───────────────────────────────────────────────────

def check_single_link(
    label: str, url: str, timeout_seconds: int, allowed_private_hfh_status_codes: List[int],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"label": label, "url": url, "status": "unknown", "http_status": None, "reason": None}

    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout_seconds)
        if response.status_code in {405, 403}:
            response = requests.get(url, allow_redirects=True, timeout=timeout_seconds, stream=True)

        result["http_status"] = response.status_code
        result["reason"] = response.reason
        is_hfh = "huggingface.co" in url.lower()

        if 200 <= response.status_code < 400:
            result["status"] = "passed"
        elif is_hfh and response.status_code in allowed_private_hfh_status_codes:
            result["status"] = "private_or_restricted"
        else:
            result["status"] = "failed"
    except requests.RequestException as exc:
        result["status"] = "failed"
        result["reason"] = str(exc)

    return result


def verify_links(config: Dict[str, Any], linked_record: Dict[str, Any]) -> Dict[str, Any]:
    """Verify zenodo.related_identifiers plus (when present) the DOI/record URLs.

    Zenodo Sandbox DOIs may not resolve through doi.org (or only after a
    delay), so in Sandbox a 404 on the DOI URL is a warning, not a hard error.
    """
    if not as_bool(get_nested(config, ["zenodo", "link_checking", "enabled"], True), True):
        return {"generated_at_utc": utc_now_iso(), "status": "skipped", "links": [], "num_errors": 0, "errors": []}

    links = dict(get_related_links(config))
    if linked_record.get("doi_url"):
        links["doi_url"] = str(linked_record["doi_url"])
    if linked_record.get("record_url"):
        links["zenodo_record_url"] = str(linked_record["record_url"])

    timeout_seconds = get_link_check_timeout(config)
    allowed_private_hfh_status_codes = get_allowed_private_hfh_status_codes(config)
    environment = get_zenodo_environment(config)

    checked = []
    errors = []
    for label, url in links.items():
        result = check_single_link(label, url, timeout_seconds, allowed_private_hfh_status_codes)

        if (
            environment == "sandbox" and label == "doi_url"
            and result["status"] == "failed" and result.get("http_status") == 404
        ):
            result["status"] = "sandbox_doi_not_resolved"
            result["reason"] = (
                "DOI URL returned 404 in Zenodo Sandbox. Accepted as non-critical for Sandbox testing."
            )

        checked.append(result)
        if result["status"] == "failed":
            errors.append(f"Link check failed for {label}: {url} ({result.get('http_status')})")

    return {
        "generated_at_utc": utc_now_iso(),
        "status": "passed" if not errors else "failed",
        "links": checked,
        "num_errors": len(errors),
        "errors": errors,
    }


# ── Sincronización de un draft existente ─────────────────────────────────────

def get_linked_record_path(config: Dict[str, Any]) -> Path:
    return get_output_filename(config, "linked_record_filename", "zenodo_linked_dataset_record.json")


def get_existing_deposition_id_from_linked_record(config: Dict[str, Any]) -> int:
    linked_record_path = get_linked_record_path(config)
    if not linked_record_path.is_file():
        fail(
            f"Zenodo linked dataset record not found: {linked_record_path}. "
            "Run 'zenodo prepare' first, or provide an existing draft."
        )
    record = read_json(linked_record_path)
    deposition_id = record.get("deposition_id")
    if deposition_id is None:
        fail(f"{linked_record_path} does not contain deposition_id.")
    return int(deposition_id)


def upload_evidence_files_to_deposition(
    config: Dict[str, Any], token: str, deposition: Dict[str, Any],
    files_to_upload: List[Path], downloaded_report: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    api_base_url = get_zenodo_api_base_url(config)
    bucket_url = get_nested(deposition, ["links", "bucket"], None)
    if not bucket_url:
        fail("Zenodo deposition response does not contain links.bucket for file upload.")

    deposition_id = deposition.get("id")
    if deposition_id is None:
        fail("Zenodo deposition response does not contain id.")

    zenodo_output_dir = get_zenodo_output_dir(config)
    logging.info("Staging evidence files in: %s", zenodo_output_dir)
    files_to_upload = stage_files_for_zenodo(files_to_upload, zenodo_output_dir)

    linked_record_path = get_linked_record_path(config)
    linked_record = create_linked_dataset_record(config, deposition, files_to_upload, downloaded_report)
    write_json(linked_record_path, linked_record)
    logging.info("Linked dataset record written: %s", linked_record_path)

    logging.info("Uploading configured evidence files to Zenodo...")
    for file_path in files_to_upload:
        logging.info("Uploading: %s", file_path)
        upload_file_to_bucket(str(bucket_url), token, file_path, file_path.name)

    logging.info("Uploading linked dataset record JSON to Zenodo...")
    upload_file_to_bucket(str(bucket_url), token, linked_record_path, linked_record_path.name)

    all_uploaded_local_files = unique_paths(files_to_upload + [linked_record_path])

    logging.info("Refreshing deposition information after upload...")
    deposition = get_deposition(api_base_url, token, int(deposition_id))

    deposition_response_path = get_output_filename(config, "deposition_response_filename", "zenodo_deposition_response.json")
    write_json(deposition_response_path, deposition)

    logging.info("Verifying uploaded Zenodo files...")
    file_verification = verify_uploaded_files(all_uploaded_local_files, deposition)

    file_verification_report_path = get_output_filename(
        config, "file_verification_report_filename", "zenodo_file_verification_report.json",
    )
    write_json(file_verification_report_path, file_verification)

    if file_verification["status"] != "passed":
        for error in file_verification["errors"]:
            logging.error(error)
        fail("Zenodo file verification failed.")

    return deposition, linked_record, file_verification


def run_zenodo_existing_draft_sync(
    config_path: Path, dry_run: bool = False, template_context: Optional[Dict[str, Any]] = None,
    verify_data: bool = False,
) -> None:
    if not config_path.exists():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_config_source(config_path, **(template_context or {}))

    if not as_bool(get_nested(config, ["zenodo", "enabled"], False), False):
        fail("zenodo.enabled is false. Set zenodo.enabled: true to use this command.")

    environment = get_zenodo_environment(config)
    api_base_url = get_zenodo_api_base_url(config)
    base_url = get_zenodo_base_url(config)
    token_env_var = get_zenodo_token_env_var(config)
    # All evidence files (and the download-verification report itself) live
    # inside the live HuggingFace Hub download below — nothing to validate
    # locally before that happens.
    files_to_upload = get_files_to_upload(config)
    deposition_id = get_existing_deposition_id_from_linked_record(config)

    logging.info("Zenodo environment: %s", environment)
    logging.info("Zenodo base URL: %s", base_url)
    logging.info("Zenodo API base URL: %s", api_base_url)
    logging.info("Zenodo token environment variable: %s", token_env_var)
    logging.info("Existing deposition id: %s", deposition_id)
    logging.info("Evidence files that will be synced (fetched from a live HuggingFace Hub download): %d", len(files_to_upload))

    if dry_run:
        logging.info("Dry run enabled. HuggingFace Hub will not be downloaded; existing Zenodo draft will not be modified.")
        for path in files_to_upload:
            logging.info("  - %s", path)
        return

    downloaded_report = ensure_fresh_hfh_download_report(config, verify_data=verify_data)

    logging.info("Validating downloaded evidence files...")
    validate_files_to_upload(files_to_upload)

    total_upload_size = sum(path.stat().st_size for path in files_to_upload)
    logging.info("Total Zenodo evidence upload size: %s", format_size(total_upload_size))

    token = get_zenodo_token(config)

    logging.info("Reading existing Zenodo draft deposition...")
    deposition = get_deposition(api_base_url, token, deposition_id)

    deposition_state = str(deposition.get("state", "")).lower()
    if deposition_state and deposition_state not in {"unsubmitted", "inprogress", "draft"}:
        logging.warning(
            "Zenodo deposition state is %r. Uploading files is expected to work only for editable drafts.",
            deposition_state,
        )

    deposition, linked_record, file_verification = upload_evidence_files_to_deposition(
        config, token, deposition, files_to_upload, downloaded_report,
    )

    logging.info("Verifying configured external links before publication...")
    prepublish_link_verification = verify_links(config, {"doi_url": None, "record_url": None})

    link_verification_report_path = get_output_filename(
        config, "link_verification_report_filename", "zenodo_link_verification_report.json",
    )
    write_json(link_verification_report_path, prepublish_link_verification)

    if prepublish_link_verification["status"] != "passed":
        for error in prepublish_link_verification["errors"]:
            logging.error(error)
        fail("Zenodo external link verification failed.")

    logging.info("Existing Zenodo draft synchronization completed successfully.")
    logging.info("Deposition id: %s", deposition.get("id"))
    logging.info("Reserved DOI: %s", linked_record.get("reserved_doi"))
    logging.info("DOI URL: %s", linked_record.get("doi_url"))
    logging.info("Record URL: %s", linked_record.get("record_url"))
    logging.info("Files verified in Zenodo: %s", file_verification.get("num_local_files"))
    logging.info("Link verification report: %s", link_verification_report_path)


# ── Creación del registro enlazado (flujo principal) ─────────────────────────

def run_zenodo_linked_dataset_creation(
    config_path: Path, dry_run: bool = False, template_context: Optional[Dict[str, Any]] = None,
    verify_data: bool = False,
) -> None:
    if not config_path.exists():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_config_source(config_path, **(template_context or {}))

    if not as_bool(get_nested(config, ["zenodo", "enabled"], False), False):
        fail("zenodo.enabled is false. Set zenodo.enabled: true to use this command.")

    environment = get_zenodo_environment(config)
    api_base_url = get_zenodo_api_base_url(config)
    base_url = get_zenodo_base_url(config)
    token_env_var = get_zenodo_token_env_var(config)
    publish = get_publish_flag(config)
    # All evidence files (and the download-verification report itself) live
    # inside the live HuggingFace Hub download below — nothing to validate
    # locally before that happens.
    files_to_upload = get_files_to_upload(config)

    logging.info("Zenodo environment: %s", environment)
    logging.info("Zenodo base URL: %s", base_url)
    logging.info("Zenodo API base URL: %s", api_base_url)
    logging.info("Zenodo token environment variable: %s", token_env_var)
    logging.info("Publish after upload: %s", publish)
    logging.info("Files that will be uploaded (fetched from a live HuggingFace Hub download): %d", len(files_to_upload))

    if dry_run:
        logging.info("Dry run enabled. HuggingFace Hub will not be downloaded; no Zenodo deposition will be created.")
        return

    downloaded_report = ensure_fresh_hfh_download_report(config, verify_data=verify_data)

    logging.info("Validating downloaded files configured for Zenodo upload...")
    validate_files_to_upload(files_to_upload)

    total_upload_size = sum(path.stat().st_size for path in files_to_upload)
    logging.info("Total Zenodo evidence upload size: %s", format_size(total_upload_size))

    token = get_zenodo_token(config)

    logging.info("Creating Zenodo deposition...")
    deposition = create_deposition(api_base_url, token)
    deposition_id = deposition.get("id")
    if deposition_id is None:
        fail("Zenodo deposition response does not contain an id.")
    logging.info("Created deposition id: %s", deposition_id)

    metadata = build_zenodo_metadata(config)

    logging.info("Updating deposition metadata and reserving DOI...")
    deposition = update_deposition_metadata(api_base_url, token, int(deposition_id), metadata)

    doi = extract_reserved_doi(deposition)
    logging.info("Reserved DOI: %s", doi or "not returned yet")

    deposition, linked_record, file_verification = upload_evidence_files_to_deposition(
        config, token, deposition, files_to_upload, downloaded_report,
    )
    deposition_response_path = get_deposition_response_path(config)
    linked_record_path = get_linked_record_path(config)
    file_verification_report_path = get_output_filename(
        config, "file_verification_report_filename", "zenodo_file_verification_report.json",
    )

    if file_verification["status"] != "passed":
        for error in file_verification["errors"]:
            logging.error(error)
        fail("Zenodo file verification failed.")

    # Only the links configured in the YAML are checked before publication —
    # the DOI/public record URL may legitimately still 404 while it's a draft.
    logging.info("Verifying configured external links before publication...")
    prepublish_link_verification = verify_links(config, {"doi_url": None, "record_url": None})

    link_verification_report_path = get_output_filename(
        config, "link_verification_report_filename", "zenodo_link_verification_report.json",
    )
    write_json(link_verification_report_path, prepublish_link_verification)

    if prepublish_link_verification["status"] != "passed":
        for error in prepublish_link_verification["errors"]:
            logging.error(error)
        fail("Zenodo external link verification failed.")

    if publish:
        logging.info("Publishing Zenodo deposition...")
        publish_response = publish_deposition(api_base_url, token, int(deposition_id))

        publish_response_path = get_output_filename(config, "publish_response_filename", "zenodo_publish_response.json")
        write_json(publish_response_path, publish_response)
        logging.info("Published Zenodo record.")
        logging.info("Publish response written: %s", publish_response_path)

        logging.info("Refreshing deposition information after publication...")
        deposition = get_deposition(api_base_url, token, int(deposition_id))
        write_json(deposition_response_path, deposition)

        linked_record = create_linked_dataset_record(config, deposition, files_to_upload, downloaded_report)
        write_json(linked_record_path, linked_record)

        logging.info("Verifying DOI URL and public Zenodo record URL after publication...")
        postpublish_link_verification = verify_links(config, linked_record)
        write_json(link_verification_report_path, postpublish_link_verification)

        if postpublish_link_verification["status"] != "passed":
            for error in postpublish_link_verification["errors"]:
                logging.error(error)
            fail("Zenodo post-publication link verification failed.")
    else:
        logging.info("zenodo.publish is false. Deposition remains as draft.")
        linked_record = create_linked_dataset_record(config, deposition, files_to_upload, downloaded_report)
        write_json(linked_record_path, linked_record)

    logging.info("Zenodo linked dataset workflow completed successfully.")
    logging.info("Deposition id: %s", deposition_id)
    logging.info("Reserved DOI: %s", linked_record.get("reserved_doi"))
    logging.info("DOI URL: %s", linked_record.get("doi_url"))
    logging.info("Record URL: %s", linked_record.get("record_url"))
    logging.info("File verification report: %s", file_verification_report_path)
    logging.info("Link verification report: %s", link_verification_report_path)


# ═══════════════════════════════════════════════════════════════════════════
# Actualizar la metadata local con el DOI de Zenodo ("zenodo upload")
# ═══════════════════════════════════════════════════════════════════════════
#
# No sube nada a Zenodo — inserta el DOI ya reservado por 'zenodo prepare' en
# los ficheros locales (HuggingFaceHub.yaml, dataset_info.json, CITATION.cff,
# README.md) y regenera checksums-sha256.txt, ya que esos ficheros cambiaron.
# El siguiente paso recomendado es volver a subir con 'huggingface upload'.

def backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    return backup_path


def path_as_posix(path: Path) -> str:
    return path.as_posix()


def get_metadata_update_report_path(config: Dict[str, Any]) -> Path:
    return get_output_filename(config, "metadata_update_report_filename", "metadata_update_report.json")


def get_checksums_filename(config: Dict[str, Any]) -> str:
    return str(get_nested(config, ["checksums", "checksum_filename"], "checksums-sha256.txt"))


def validate_and_extract_zenodo_info(record: Dict[str, Any]) -> Dict[str, Any]:
    environment = record.get("zenodo_environment")
    deposition_id = record.get("deposition_id")
    record_id = record.get("record_id")
    doi = record.get("reserved_doi")
    doi_url = record.get("doi_url")
    record_url = record.get("record_url")

    if environment not in {"sandbox", "production"}:
        fail("zenodo_linked_dataset_record.json must contain 'zenodo_environment' with value 'sandbox' or 'production'.")
    if not deposition_id:
        fail("zenodo_linked_dataset_record.json does not contain deposition_id.")
    if not doi:
        fail("zenodo_linked_dataset_record.json does not contain reserved_doi.")
    if not doi_url:
        doi_url = f"https://doi.org/{doi}"
    if not record_url:
        fail("zenodo_linked_dataset_record.json does not contain record_url.")

    return {
        "environment": str(environment),
        "deposition_id": deposition_id,
        "record_id": record_id,
        "doi": str(doi),
        "doi_url": str(doi_url),
        "record_url": str(record_url),
        "record_type": str(record.get("record_type", "zenodo_linked_dataset_record")),
        "record_scope": str(record.get("record_scope", "")),
        "generated_at_utc": record.get("generated_at_utc"),
    }


def is_sandbox_zenodo(zenodo_info: Dict[str, Any]) -> bool:
    return zenodo_info["environment"] == "sandbox"


def update_huggingfacehub_yaml(config_path: Path, config: Dict[str, Any], zenodo_info: Dict[str, Any]) -> None:
    zenodo = ensure_dict(config, "zenodo")
    zenodo["last_metadata_update_utc"] = utc_now_iso()
    zenodo["last_linked_record_type"] = zenodo_info["record_type"]
    zenodo["last_deposition_id"] = zenodo_info["deposition_id"]
    zenodo["last_record_id"] = zenodo_info["record_id"]

    if is_sandbox_zenodo(zenodo_info):
        zenodo["sandbox_doi"] = zenodo_info["doi"]
        zenodo["sandbox_doi_url"] = zenodo_info["doi_url"]
        zenodo["sandbox_record_url"] = zenodo_info["record_url"]
        zenodo["sandbox_note"] = "Sandbox DOI for workflow testing only. Not intended for formal citation."
    else:
        zenodo["doi"] = zenodo_info["doi"]
        zenodo["doi_url"] = zenodo_info["doi_url"]
        zenodo["record_url"] = zenodo_info["record_url"]
        zenodo["doi_note"] = "Production Zenodo DOI for this linked dataset record."

    write_yaml(config_path, config)


def update_internal_config_yaml(internal_config_path: Path, config: Dict[str, Any]) -> None:
    write_yaml(internal_config_path, config)


def update_downloaded_verification_report(report_path: Path, zenodo_info: Dict[str, Any]) -> None:
    report = read_json(report_path)
    report["zenodo"] = {
        "environment": zenodo_info["environment"],
        "deposition_id": zenodo_info["deposition_id"],
        "record_id": zenodo_info["record_id"],
        "doi": zenodo_info["doi"],
        "doi_url": zenodo_info["doi_url"],
        "record_url": zenodo_info["record_url"],
        "record_type": zenodo_info["record_type"],
        "record_scope": zenodo_info["record_scope"],
        "is_sandbox": is_sandbox_zenodo(zenodo_info),
        "note": (
            "Sandbox DOI for workflow testing only. Not intended for formal citation."
            if is_sandbox_zenodo(zenodo_info)
            else "Production DOI for formal citation of this linked dataset record."
        ),
        "updated_at_utc": utc_now_iso(),
    }
    write_json(report_path, report)


def update_dataset_info_json(dataset_info_path: Path, zenodo_info: Dict[str, Any]) -> None:
    data = read_json(dataset_info_path)
    data["zenodo"] = {
        "environment": zenodo_info["environment"],
        "deposition_id": zenodo_info["deposition_id"],
        "record_id": zenodo_info["record_id"],
        "doi": zenodo_info["doi"],
        "doi_url": zenodo_info["doi_url"],
        "record_url": zenodo_info["record_url"],
        "record_type": zenodo_info["record_type"],
        "record_scope": zenodo_info["record_scope"],
        "is_sandbox": is_sandbox_zenodo(zenodo_info),
        "note": (
            "Sandbox DOI for workflow testing only. Not intended for formal citation."
            if is_sandbox_zenodo(zenodo_info)
            else "Production DOI for formal citation of this linked dataset record."
        ),
        "updated_at_utc": utc_now_iso(),
    }
    write_json(dataset_info_path, data)


def update_citation_cff(citation_path: Path, zenodo_info: Dict[str, Any]) -> None:
    citation = load_yaml(citation_path)

    if is_sandbox_zenodo(zenodo_info):
        identifiers = citation.get("identifiers")
        if not isinstance(identifiers, list):
            identifiers = []
        identifiers = [
            item for item in identifiers
            if not (isinstance(item, dict) and item.get("description") == "Zenodo Sandbox DOI for workflow testing only")
        ]
        identifiers.append({
            "type": "doi",
            "value": zenodo_info["doi"],
            "description": "Zenodo Sandbox DOI for workflow testing only",
        })
        citation["identifiers"] = identifiers
        citation["notes"] = (
            "This CITATION.cff contains a Zenodo Sandbox DOI for workflow testing only. "
            "Do not use the Sandbox DOI for formal citation."
        )
    else:
        citation["doi"] = zenodo_info["doi"]
        citation["url"] = zenodo_info["record_url"]

    write_yaml(citation_path, citation)


def build_zenodo_readme_section(zenodo_info: Dict[str, Any]) -> str:
    if is_sandbox_zenodo(zenodo_info):
        note = "Note: this is a Zenodo Sandbox DOI for workflow testing only. It must not be used for formal citation."
        heading = "## Zenodo Sandbox DOI"
    else:
        note = "This DOI can be used to cite this dataset version."
        heading = "## Zenodo DOI"

    return f"""{heading}

This dataset version has an associated Zenodo linked dataset record.

- DOI: `{zenodo_info["doi"]}`
- DOI URL: {zenodo_info["doi_url"]}
- Zenodo record: {zenodo_info["record_url"]}
- Record type: `{zenodo_info["record_type"]}`

{note}
"""


def replace_or_append_readme_section(readme_text: str, zenodo_info: Dict[str, Any]) -> str:
    section = build_zenodo_readme_section(zenodo_info).rstrip() + "\n"
    possible_headings = ["## Zenodo DOI", "## Zenodo Sandbox DOI"]

    lines = readme_text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() in possible_headings:
            start_idx = idx
            break

    if start_idx is None:
        if readme_text.endswith("\n"):
            return readme_text + "\n" + section
        return readme_text + "\n\n" + section

    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].startswith("## "):
            end_idx = idx
            break

    new_lines = lines[:start_idx] + section.splitlines() + lines[end_idx:]
    return "\n".join(new_lines).rstrip() + "\n"


def update_readme(readme_path: Path, zenodo_info: Dict[str, Any]) -> None:
    text = readme_path.read_text(encoding="utf-8")
    updated = replace_or_append_readme_section(text, zenodo_info)
    readme_path.write_text(updated, encoding="utf-8")


def collect_files_for_global_checksums(output_dir: Path, checksum_filename: str) -> List[Path]:
    files: List[Path] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        if rel == checksum_filename or rel.endswith(".bak"):
            continue
        files.append(path)
    return files


def write_global_checksums(output_dir: Path, config: Dict[str, Any]) -> Path:
    checksum_filename = get_checksums_filename(config)
    checksum_path = output_dir / checksum_filename
    chunk_size_bytes = get_chunk_size_bytes(config)
    files = collect_files_for_global_checksums(output_dir, checksum_filename)

    with checksum_path.open("w", encoding="utf-8") as f:
        for file_path in files:
            rel = file_path.relative_to(output_dir).as_posix()
            f.write(f"{sha256_file(file_path, chunk_size_bytes)}  {rel}\n")

    return checksum_path


def validate_required_metadata_files(
    config_path: Path, config: Dict[str, Any], zenodo_record_path: Path, output_dir: Path, downloaded_report_path: Path,
) -> Dict[str, Path]:
    if not config_path.is_file():
        fail(f"Configuration file not found: {config_path}")
    if not zenodo_record_path.is_file():
        fail(f"Zenodo linked dataset record not found: {zenodo_record_path}. Run 'zenodo prepare' first.")
    if not downloaded_report_path.is_file():
        fail(f"Downloaded verification report not found: {downloaded_report_path}. Run 'zenodo prepare' first.")
    if not output_dir.is_dir():
        fail(f"HFH output directory not found: {output_dir}")

    required = {
        "readme": output_dir / "README.md",
        "citation": output_dir / "CITATION.cff",
        "internal_config": output_dir / get_internal_config_filename(config),
        "dataset_info": output_dir / "dataset_info.json",
        "checksums": output_dir / get_checksums_filename(config),
    }

    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        fail("Missing required HFH metadata files:\n" + "\n".join(f"  - {path}" for path in missing))

    return required


def create_metadata_update_report(
    report_path: Path, config_path: Path, zenodo_info: Dict[str, Any],
    updated_files: List[Path], backup_files: List[Path],
    checksum_verified_count: int, checksum_errors: List[str],
) -> Dict[str, Any]:
    report = {
        "generated_at_utc": utc_now_iso(),
        "status": "passed" if not checksum_errors else "failed",
        "zenodo": {
            "environment": zenodo_info["environment"],
            "deposition_id": zenodo_info["deposition_id"],
            "record_id": zenodo_info["record_id"],
            "doi": zenodo_info["doi"],
            "doi_url": zenodo_info["doi_url"],
            "record_url": zenodo_info["record_url"],
            "record_type": zenodo_info["record_type"],
            "is_sandbox": is_sandbox_zenodo(zenodo_info),
            "note": (
                "Sandbox DOI for workflow testing only. Not intended for formal citation."
                if is_sandbox_zenodo(zenodo_info)
                else "Production DOI for formal citation of this linked dataset record."
            ),
        },
        "updated_files": [path_as_posix(path) for path in updated_files],
        "backup_files": [path_as_posix(path) for path in backup_files],
        "checksum_files_verified": checksum_verified_count,
        "checksum_errors": checksum_errors,
        "next_recommended_steps": [
            "donadataset publish huggingface upload",
            "donadataset publish huggingface download",
        ],
    }
    write_json(report_path, report)
    return report


def run_update_local_metadata_with_doi(
    config_path: Path, dry_run: bool = False, no_backup: bool = False,
    template_context: Optional[Dict[str, Any]] = None,
) -> None:
    logging.info("Reading configuration: %s", config_path)
    config = load_config_source(config_path, **(template_context or {}))

    output_dir = get_output_dir(config)
    zenodo_record_path = get_linked_record_path(config)
    downloaded_report_path = get_zenodo_downloaded_report_path(config)
    report_path = get_metadata_update_report_path(config)

    required_paths = validate_required_metadata_files(
        config_path, config, zenodo_record_path, output_dir, downloaded_report_path,
    )

    logging.info("Reading Zenodo linked dataset record: %s", zenodo_record_path)
    zenodo_record = read_json(zenodo_record_path)
    zenodo_info = validate_and_extract_zenodo_info(zenodo_record)

    logging.info("Zenodo environment: %s", zenodo_info["environment"])
    logging.info("Zenodo DOI: %s", zenodo_info["doi"])
    logging.info("Zenodo DOI URL: %s", zenodo_info["doi_url"])
    logging.info("Zenodo record URL: %s", zenodo_info["record_url"])

    if is_sandbox_zenodo(zenodo_info):
        logging.warning("This is a Sandbox DOI. It will be marked as testing-only.")

    files_to_update = [
        config_path, downloaded_report_path, required_paths["internal_config"],
        required_paths["dataset_info"], required_paths["citation"], required_paths["readme"],
        required_paths["checksums"],
    ]

    if dry_run:
        logging.info("Dry run enabled. No files will be modified.")
        logging.info("Files that would be updated:")
        for path in files_to_update:
            logging.info("  - %s", path)
        return

    backup_files: List[Path] = []
    if not no_backup:
        logging.info("Creating backups...")
        for path in files_to_update:
            backup = backup_file(path)
            if backup is not None:
                backup_files.append(backup)

    updated_files: List[Path] = []

    if config_path.suffix.lower() == ".j2":
        # Never overwrite a Jinja template with resolved YAML — write the
        # DOI-updated config to a plain sibling copy instead.
        external_config_path = get_zenodo_output_dir(config) / "Zenodo_resolved.yaml"
        ensure_dir(external_config_path.parent)
        logging.info(
            "Config is a Jinja template (%s); writing updated metadata to %s instead.",
            config_path, external_config_path,
        )
    else:
        external_config_path = config_path

    logging.info("Updating external configuration file: %s", external_config_path)
    update_huggingfacehub_yaml(external_config_path, config, zenodo_info)
    updated_files.append(external_config_path)

    logging.info("Updating internal HFH configuration file: %s", required_paths["internal_config"])
    update_internal_config_yaml(required_paths["internal_config"], config)
    updated_files.append(required_paths["internal_config"])

    logging.info("Updating verification_report_downloaded.json...")
    update_downloaded_verification_report(downloaded_report_path, zenodo_info)
    updated_files.append(downloaded_report_path)

    logging.info("Updating %s...", required_paths["dataset_info"])
    update_dataset_info_json(required_paths["dataset_info"], zenodo_info)
    updated_files.append(required_paths["dataset_info"])

    logging.info("Updating %s...", required_paths["citation"])
    update_citation_cff(required_paths["citation"], zenodo_info)
    updated_files.append(required_paths["citation"])

    logging.info("Updating %s...", required_paths["readme"])
    update_readme(required_paths["readme"], zenodo_info)
    updated_files.append(required_paths["readme"])

    logging.info("Regenerating %s...", output_dir / get_checksums_filename(config))
    checksum_path = write_global_checksums(output_dir, config)
    updated_files.append(checksum_path)

    logging.info("Verifying regenerated checksums...")
    checksum_verified_count, checksum_errors = verify_global_checksums(output_dir, config)

    logging.info("Writing metadata update report...")
    report = create_metadata_update_report(
        report_path, config_path, zenodo_info, updated_files, backup_files,
        checksum_verified_count, checksum_errors,
    )

    if report["status"] != "passed":
        for error in checksum_errors:
            logging.error(error)
        fail("Metadata update completed, but checksum verification failed.")

    logging.info("Metadata update completed successfully.")
    logging.info("Report: %s", report_path)
    logging.info("Next recommended steps:")
    logging.info("  donadataset publish huggingface upload")
    logging.info("  donadataset publish huggingface download")


# ═══════════════════════════════════════════════════════════════════════════
# Descarga y despliegue local ("zenodo download")
# ═══════════════════════════════════════════════════════════════════════════
#
# Descarga un registro Zenodo (completo, o enlazado a HuggingFace Hub) y lo
# despliega en formato YOLO. No depende del esquema HuggingFaceHub.yaml: solo
# necesita el record id/DOI de Zenodo y un directorio de destino.

def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 0:
        seconds = 0
    seconds_int = int(round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def get_hfh_file_sizes(
    repo_id: str, revision: Optional[str], token: Optional[str], filenames: List[str],
) -> Dict[str, Optional[int]]:
    """Return file sizes from the linked Hugging Face Hub repository when available."""
    wanted = set(filenames)
    sizes: Dict[str, Optional[int]] = {filename: None for filename in filenames}

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token) if token else HfApi()
        for item in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", revision=revision, recursive=True):
            path = getattr(item, "path", None)
            if path in wanted:
                sizes[path] = getattr(item, "size", None)
    except Exception as exc:
        logging.warning(
            "Could not read file sizes from Hugging Face Hub. "
            "Download progress will still be shown per file by huggingface_hub. Details: %s",
            exc,
        )

    return sizes


@dataclass
class DeployConfig:
    deploy_dir: Path

    zenodo_record: Optional[str] = None
    doi: Optional[str] = None

    download_dir: Optional[Path] = None
    keep_download: bool = False
    clean_deploy_dir: bool = False
    verify_checksums: bool = True
    copy_metadata: bool = True

    allow_linked_hfh: bool = True
    hf_token_env: Optional[str] = None


def resolve_zenodo_record_id(zenodo_record: Optional[str], doi: Optional[str]) -> str:
    """Resolve a Zenodo record id from a record id, record URL, DOI, or DOI URL."""
    if zenodo_record and doi:
        fail("Provide only one of zenodo_record or doi, not both.")
    if not zenodo_record and not doi:
        fail("You must provide either a Zenodo record or a DOI.")

    value = (zenodo_record if zenodo_record else doi).strip().rstrip("/")

    if value.isdigit():
        return value
    if "/records/" in value:
        record_id = value.split("/records/", 1)[1].split("?", 1)[0].split("#", 1)[0]
        if record_id.isdigit():
            return record_id
    if "zenodo." in value:
        record_id = value.split("zenodo.", 1)[1].split("?", 1)[0].split("#", 1)[0]
        if record_id.isdigit():
            return record_id

    fail(f"Could not resolve a Zenodo record id from: {value}")


def get_zenodo_record_metadata(record_id: str) -> Dict[str, Any]:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    logging.info("Querying Zenodo API: %s", api_url)
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    return response.json()


def download_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading: %s", output_path.name)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_from_zenodo(record_id: str, download_dir: Path) -> Dict[str, Any]:
    record = get_zenodo_record_metadata(record_id)

    metadata_output = download_dir / "zenodo_record_metadata.json"
    write_json(metadata_output, record)
    logging.info("Saved Zenodo metadata: %s", metadata_output)

    files = record.get("files", [])
    if not files:
        fail(f"The Zenodo record {record_id} does not contain files.")

    for item in files:
        filename = item.get("key")
        file_url = item.get("links", {}).get("self")

        if not filename or not file_url:
            logging.warning("Skipping Zenodo file entry without key or URL: %s", item)
            continue

        output_path = download_dir / filename
        if output_path.exists():
            logging.info("File already exists, skipping download: %s", output_path.name)
        else:
            download_url(file_url, output_path)

        checksum = item.get("checksum")
        if checksum and checksum.startswith("md5:"):
            expected_md5 = checksum.split("md5:", 1)[1]
            actual_md5 = md5_file(output_path)
            if actual_md5.lower() != expected_md5.lower():
                fail(f"MD5 mismatch for {output_path.name}: expected {expected_md5}, got {actual_md5}")
            logging.info("MD5 verified: %s", output_path.name)

    return record


def find_checksum_file(download_dir: Path) -> Optional[Path]:
    candidates = sorted(download_dir.rglob("checksums-sha256.txt"))
    if not candidates:
        return None
    if len(candidates) > 1:
        logging.warning("Several checksums-sha256.txt files were found. Using: %s", candidates[0])
    return candidates[0]


def missing_files_from_sha256(download_dir: Path) -> List[Path]:
    checksum_file = find_checksum_file(download_dir)
    if checksum_file is None:
        return []

    base_dir = checksum_file.parent
    expected = read_checksums(checksum_file)
    return [Path(rel) for rel in expected if not (base_dir / rel).exists()]


def extract_linked_hfh_repo(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (repo_id, revision) from Zenodo related identifiers, when present."""
    metadata = record.get("metadata", {})
    for item in metadata.get("related_identifiers", []):
        match = HFH_DATASET_URL_RE.search(str(item.get("identifier", "")))
        if match:
            return match.group(1), match.group(2)
    return None, None


def download_missing_files_from_linked_hfh(
    record: Dict[str, Any], download_dir: Path, config: "DeployConfig",
) -> List[str]:
    """checksums-sha256.txt can reference files stored on Hugging Face Hub for
    linked records. Download those missing files, keeping their relative paths."""
    if not config.allow_linked_hfh:
        logging.info("Linked Hugging Face Hub downloads are disabled.")
        return []

    missing = missing_files_from_sha256(download_dir)
    if not missing:
        return []

    repo_id, revision = extract_linked_hfh_repo(record)
    if not repo_id:
        logging.warning(
            "checksums-sha256.txt references missing files, but no linked "
            "Hugging Face Hub dataset repository was found in Zenodo metadata."
        )
        return []

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        fail(
            "The Zenodo record is linked to Hugging Face Hub and some files are "
            "missing locally, but huggingface_hub is not installed."
        )
        raise exc  # unreachable, fail() always raises — keeps type-checkers happy

    token = None
    if config.hf_token_env:
        token = os.environ.get(config.hf_token_env)
        if not token:
            logging.warning(
                "The environment variable %s is not defined. If the linked HFH "
                "repository is private, the download will fail.",
                config.hf_token_env,
            )

    missing_filenames = [relative_path.as_posix() for relative_path in missing]
    missing_tar_filenames = [f for f in missing_filenames if f.lower().endswith(".tar")]
    file_sizes = get_hfh_file_sizes(repo_id, revision, token, missing_filenames)
    total_tar_bytes = sum(file_sizes.get(f) or 0 for f in missing_tar_filenames)

    logging.info("Zenodo record references files stored in Hugging Face Hub: %s", repo_id)
    logging.info("Missing files to download from linked HFH repository: %d", len(missing))
    logging.info(
        "Missing linked HFH .tar files: %d%s",
        len(missing_tar_filenames),
        f" | total known size: {format_size(total_tar_bytes)}" if total_tar_bytes else "",
    )
    for index, filename in enumerate(missing_tar_filenames, start=1):
        logging.info("  TAR %d/%d: %s (%s)", index, len(missing_tar_filenames), filename, format_size(file_sizes.get(filename) or 0))

    downloaded: List[str] = []
    failures: List[str] = []
    tar_download_start = time.monotonic()
    completed_tar_bytes = 0
    completed_tar_count = 0

    for relative_path in missing:
        filename = relative_path.as_posix()
        target_path = download_dir / relative_path
        is_tar = filename.lower().endswith(".tar")

        if target_path.exists():
            if is_tar:
                completed_tar_count += 1
                completed_tar_bytes += target_path.stat().st_size
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if is_tar:
            tar_index = completed_tar_count + 1
            elapsed = time.monotonic() - tar_download_start
            average_rate = completed_tar_bytes / elapsed if elapsed > 0 and completed_tar_bytes > 0 else None
            remaining_after_current = max(total_tar_bytes - completed_tar_bytes, 0) if total_tar_bytes else None
            estimated_remaining = (
                remaining_after_current / average_rate if average_rate and remaining_after_current is not None else None
            )
            logging.info(
                "Downloading linked HFH TAR %d/%d: %s (%s). Estimated remaining TAR download time: %s",
                tar_index, len(missing_tar_filenames), filename,
                format_size(file_sizes.get(filename) or 0), format_duration(estimated_remaining),
            )
        else:
            logging.info("Downloading linked HFH file: %s", filename)

        file_start = time.monotonic()
        try:
            hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename=filename,
                revision=revision, token=token, local_dir=str(download_dir),
            )
            downloaded.append(filename)

            if is_tar:
                elapsed_file = time.monotonic() - file_start
                completed_tar_count += 1
                actual_size = target_path.stat().st_size if target_path.exists() else (file_sizes.get(filename) or 0)
                completed_tar_bytes += actual_size

                total_elapsed = time.monotonic() - tar_download_start
                average_rate = completed_tar_bytes / total_elapsed if total_elapsed > 0 else None
                remaining_bytes = max(total_tar_bytes - completed_tar_bytes, 0) if total_tar_bytes else None
                estimated_remaining = (
                    remaining_bytes / average_rate if average_rate and remaining_bytes is not None else None
                )
                logging.info(
                    "Completed TAR %d/%d: %s in %s. Estimated remaining TAR download time: %s",
                    completed_tar_count, len(missing_tar_filenames), filename,
                    format_duration(elapsed_file), format_duration(estimated_remaining),
                )
        except Exception as exc:
            failures.append(f"{filename}: {exc}")

    if failures:
        message = "\n".join(failures[:20])
        if len(failures) > 20:
            message += f"\n... and {len(failures) - 20} more failures."
        fail(f"Could not download all linked HFH files:\n{message}")

    logging.info("Downloaded %d linked HFH files.", len(downloaded))
    return downloaded


def verify_sha256_checksums(download_dir: Path) -> None:
    checksum_file = find_checksum_file(download_dir)
    if checksum_file is None:
        logging.warning("checksums-sha256.txt was not found. Internal SHA256 verification will be skipped.")
        return

    logging.info("Verifying SHA256 checksums from: %s", checksum_file)
    base_dir = checksum_file.parent
    expected = read_checksums(checksum_file)

    if not expected:
        fail(f"The checksum file is empty: {checksum_file}")

    errors: List[str] = []
    for rel_path, expected_hash in expected.items():
        file_path = base_dir / rel_path
        if not file_path.exists():
            errors.append(f"Missing file: {rel_path}")
            continue
        actual_hash = sha256_file(file_path)
        if actual_hash.lower() != expected_hash.lower():
            errors.append(f"SHA256 mismatch: {rel_path} | expected {expected_hash} | got {actual_hash}")

    if errors:
        message = "\n".join(errors[:20])
        if len(errors) > 20:
            message += f"\n... and {len(errors) - 20} more errors."
        fail(f"SHA256 verification failed:\n{message}")

    logging.info("SHA256 verification passed for %d files.", len(expected))


def find_tar_files(download_dir: Path) -> List[Path]:
    tar_files = sorted(p for p in download_dir.rglob("*.tar") if ".git" not in p.parts)
    if not tar_files:
        fail(f"No .tar files were found in: {download_dir}")

    logging.info("Found %d .tar files.", len(tar_files))
    for tar_path in tar_files:
        logging.info("  - %s", tar_path.relative_to(download_dir))

    return tar_files


def is_safe_tar_member(destination: Path, member_name: str) -> bool:
    """Prevent path traversal attacks when extracting tar files."""
    destination = destination.resolve()
    target = (destination / member_name).resolve()
    try:
        target.relative_to(destination)
        return True
    except ValueError:
        return False


def safe_extract_tar(tar_path: Path, destination: Path) -> int:
    logging.info("Extracting: %s", tar_path.name)
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in members:
            if not is_safe_tar_member(destination, member.name):
                fail(f"Unsafe path inside {tar_path.name}: {member.name}")
        tar.extractall(destination)
    return len(members)


def prepare_deploy_dir(config: "DeployConfig") -> None:
    if config.deploy_dir.exists() and config.clean_deploy_dir:
        logging.warning("Removing existing deployment directory: %s", config.deploy_dir)
        shutil.rmtree(config.deploy_dir)
    config.deploy_dir.mkdir(parents=True, exist_ok=True)


def should_copy_metadata(path: Path) -> bool:
    if path.is_dir() or path.suffix.lower() == ".tar" or ".git" in path.parts:
        return False
    return True


def copy_metadata_files(download_dir: Path, deploy_dir: Path) -> List[str]:
    copied: List[str] = []
    for path in sorted(download_dir.rglob("*")):
        if not should_copy_metadata(path):
            continue
        relative_path = path.relative_to(download_dir)
        if any(part.startswith(".cache") for part in relative_path.parts):
            continue
        target_path = deploy_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_path)
        copied.append(str(relative_path))

    logging.info("Copied %d metadata/support files.", len(copied))
    return copied


def count_dataset_files(deploy_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"splits": {}, "total_images": 0, "total_labels": 0}

    for split in ["train", "val", "test"]:
        image_dir = deploy_dir / "images" / split
        label_dir = deploy_dir / "labels" / split

        images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS] if image_dir.exists() else []
        labels = [p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in LABEL_EXTENSIONS] if label_dir.exists() else []

        result["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_image_dir": not image_dir.exists(),
            "missing_label_dir": not label_dir.exists(),
        }
        result["total_images"] += len(images)
        result["total_labels"] += len(labels)

    return result


def check_yolo_structure(deploy_dir: Path) -> List[str]:
    warnings: List[str] = []
    for split in ["train", "val", "test"]:
        for kind in ("images", "labels"):
            split_dir = deploy_dir / kind / split
            if not split_dir.exists():
                warnings.append(f"Missing directory: {split_dir}")
    return warnings


def extract_basic_zenodo_info(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata", {})
    creator_names = [c["name"] for c in metadata.get("creators", []) if c.get("name")]
    repo_id, revision = extract_linked_hfh_repo(record)

    return {
        "record_id": record.get("id"),
        "conceptrecid": record.get("conceptrecid"),
        "doi": metadata.get("doi"),
        "title": metadata.get("title"),
        "publication_date": metadata.get("publication_date"),
        "version": metadata.get("version"),
        "license": metadata.get("license"),
        "creators": creator_names,
        "zenodo_url": record.get("links", {}).get("html") or record.get("links", {}).get("self_html"),
        "linked_huggingface_hub_repo": repo_id,
        "linked_huggingface_hub_revision": revision,
    }


def write_deployment_report(
    deploy_dir: Path, record: Dict[str, Any], tar_files: List[Path], copied_metadata: List[str],
    counts: Dict[str, Any], structure_warnings: List[str], linked_hfh_downloads: List[str],
) -> None:
    report = {
        "zenodo": extract_basic_zenodo_info(record),
        "deployment": {
            "deploy_dir": str(deploy_dir),
            "tar_files_extracted": [str(p) for p in tar_files],
            "metadata_files_copied": copied_metadata,
            "linked_huggingface_hub_downloads": linked_hfh_downloads,
            "counts": counts,
            "structure_warnings": structure_warnings,
        },
    }
    write_json(deploy_dir / "deployment_report.json", report)
    logging.info("Deployment report written to: %s", deploy_dir / "deployment_report.json")


def deploy_dataset(
    config: "DeployConfig", download_dir: Path, record: Dict[str, Any], linked_hfh_downloads: List[str],
) -> None:
    prepare_deploy_dir(config)
    tar_files = find_tar_files(download_dir)

    total_members = sum(safe_extract_tar(tar_path, config.deploy_dir) for tar_path in tar_files)
    logging.info("Extracted %d members from .tar files.", total_members)

    copied_metadata = copy_metadata_files(download_dir, config.deploy_dir) if config.copy_metadata else []

    counts = count_dataset_files(config.deploy_dir)
    structure_warnings = check_yolo_structure(config.deploy_dir)
    for warning in structure_warnings:
        logging.warning(warning)

    logging.info("Deployment summary:")
    logging.info("  Total images: %d", counts["total_images"])
    logging.info("  Total labels: %d", counts["total_labels"])
    for split, values in counts["splits"].items():
        logging.info("  %s | images: %d | labels: %d", split, values["images"], values["labels"])

    write_deployment_report(
        config.deploy_dir, record, tar_files, copied_metadata, counts, structure_warnings, linked_hfh_downloads,
    )


def run_download_and_deploy(
    deploy_dir: Path,
    zenodo_record: Optional[str] = None,
    doi: Optional[str] = None,
    download_dir: Optional[Path] = None,
    keep_download: bool = False,
    clean_deploy_dir: bool = False,
    verify_checksums: bool = True,
    copy_metadata: bool = True,
    allow_linked_hfh: bool = True,
    hf_token_env: Optional[str] = None,
) -> None:
    config = DeployConfig(
        deploy_dir=deploy_dir,
        zenodo_record=zenodo_record,
        doi=doi,
        download_dir=download_dir,
        keep_download=keep_download,
        clean_deploy_dir=clean_deploy_dir,
        verify_checksums=verify_checksums,
        copy_metadata=copy_metadata,
        allow_linked_hfh=allow_linked_hfh,
        hf_token_env=hf_token_env,
    )

    record_id = resolve_zenodo_record_id(config.zenodo_record, config.doi)
    logging.info("Resolved Zenodo record id: %s", record_id)
    logging.info("Deployment directory: %s", config.deploy_dir)

    if config.download_dir:
        download_dir_path = config.download_dir
        download_dir_path.mkdir(parents=True, exist_ok=True)
        remove_download_dir_when_done = False
    else:
        # Deliberately not tempfile.TemporaryDirectory(): its finalizer removes
        # the directory as soon as the object is garbage-collected, regardless
        # of whether .cleanup() is ever called — which would silently defeat
        # --keep-download. mkdtemp() has no such finalizer; cleanup here is
        # always explicit, controlled by remove_download_dir_when_done below.
        download_dir_path = Path(tempfile.mkdtemp(prefix="donadataset_zenodo_download_"))
        remove_download_dir_when_done = not config.keep_download

    logging.info("Download directory: %s", download_dir_path)

    record = download_from_zenodo(record_id, download_dir_path)
    linked_hfh_downloads = download_missing_files_from_linked_hfh(record, download_dir_path, config)

    if config.verify_checksums:
        verify_sha256_checksums(download_dir_path)
    else:
        logging.warning("Internal SHA256 verification disabled.")

    deploy_dataset(config, download_dir_path, record, linked_hfh_downloads)

    if remove_download_dir_when_done:
        shutil.rmtree(download_dir_path, ignore_errors=True)
    else:
        logging.info("Download directory kept at: %s", download_dir_path)

    logging.info("Download and deployment completed successfully.")


# ═══════════════════════════════════════════════════════════════════════════
# Chequeo final antes de publicar ("zenodo check-readiness")
# ═══════════════════════════════════════════════════════════════════════════
#
# Solo lectura — no publica nada, no cambia visibilidad, no sube ficheros.
# Junta en un único veredicto (ready_to_publish_zenodo) los tres chequeos que
# hasta ahora estaban dispersos: HFH público de verdad (URL + API), el draft
# de Zenodo bien formado (metadata + ficheros de evidencia realmente subidos),
# y que los reports locales previos digan "passed".

DEFAULT_READINESS_REPORT_FILENAME = "public_release_readiness_report.json"

# Mismos ficheros que 'huggingface prepare' escribe dentro del export — si
# alguno falta en el depósito de Zenodo, algo se subió incompleto.
EXPECTED_ZENODO_DEPOSITION_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    INTERNAL_CONFIG_FILENAME,
    "donana.yaml",
    "dataset_info.json",
    "metadata.csv",
    "manifest.csv",
    "manifest-files-sha256.csv",
    "checksums-sha256.txt",
    "verification_report_local.json",
    "verification_report_downloaded.json",
    "zenodo_linked_dataset_record.json",
]


def normalize_url(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or "REPLACE_WITH" in text:
        return None
    return text


def get_optional_hfh_token(config: Dict[str, Any]) -> Optional[str]:
    """Like get_token() in services.huggingface, but returns None instead of
    failing — this check must still report something useful without a token."""
    return os.environ.get(get_token_env_var(config))


def get_readiness_report_path(config: Dict[str, Any]) -> Path:
    return get_output_filename(
        config, "public_release_readiness_report_filename", DEFAULT_READINESS_REPORT_FILENAME,
    )


def get_public_hfh_urls(config: Dict[str, Any], linked_record: Dict[str, Any]) -> Dict[str, str]:
    repo_id = get_repo_id(config)
    urls: Dict[str, str] = {
        "huggingface_dataset_url": build_dataset_url(repo_id),
        "huggingface_tree_url": get_tree_url(repo_id),
    }

    for source in (get_related_links(config), linked_record.get("related_links", {})):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            url = normalize_url(value)
            if url and "huggingface.co" in url.lower():
                urls[str(key)] = url

    return urls


def verify_hfh_public_urls(config: Dict[str, Any], linked_record: Dict[str, Any]) -> Dict[str, Any]:
    timeout_seconds = get_link_check_timeout(config)
    urls = get_public_hfh_urls(config, linked_record)

    checked = []
    errors = []
    for label, url in urls.items():
        logging.info("Checking public HFH URL without token: %s", url)
        result = check_public_url(url, timeout_seconds)
        result["label"] = label
        checked.append(result)
        if result["status"] != "passed":
            errors.append(
                f"HFH public URL check failed for {label}: {url} "
                f"({result.get('http_status')}, {result.get('reason')})"
            )

    return {
        "status": "passed" if not errors else "failed",
        "num_urls_checked": len(checked),
        "num_errors": len(errors),
        "errors": errors,
        "urls": checked,
    }


def verify_hfh_api_visibility(config: Dict[str, Any]) -> Dict[str, Any]:
    repo_id = get_repo_id(config)
    repo_type = get_repo_type(config)
    token_env_var = get_token_env_var(config)
    token = get_optional_hfh_token(config)

    result: Dict[str, Any] = {
        "status": "skipped", "repo_id": repo_id, "repo_type": repo_type,
        "token_env_var": token_env_var, "authenticated_as": None,
        "private": None, "public": None, "errors": [],
    }

    if not token:
        result["errors"].append(f"{token_env_var} is not defined; HFH API visibility check skipped.")
        return result

    try:
        user_info = authenticate(token)
        result["authenticated_as"] = user_info.get("name") or user_info.get("fullname")

        visibility = get_dataset_visibility(HfApi(token=token), repo_id, token)
        result["private"] = visibility["private"]
        result["public"] = visibility["public"]
        result["status"] = "passed" if visibility["public"] else "failed"
        if visibility["private"]:
            result["errors"].append(f"HFH repository is still private according to the API: {repo_id}")
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append(f"HFH API visibility check failed: {exc}")

    return result


def get_deposition_file_names(deposition: Dict[str, Any]) -> List[str]:
    names = []
    for item in deposition.get("files", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("filename") or item.get("key") or item.get("name")
        if name:
            names.append(str(name))
    return names


def verify_zenodo_deposition(
    config: Dict[str, Any], linked_record: Dict[str, Any], deposition: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    info = validate_and_extract_zenodo_info(linked_record)
    metadata = deposition.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    deposition_id = deposition.get("id")
    if int(deposition_id) != int(info["deposition_id"]):
        errors.append(f"Zenodo deposition id mismatch: linked_record={info['deposition_id']}, api={deposition_id}")

    prereserve = metadata.get("prereserve_doi")
    api_doi = prereserve.get("doi") if isinstance(prereserve, dict) else None
    api_doi = api_doi or metadata.get("doi") or deposition.get("doi")
    if str(api_doi) != info["doi"]:
        errors.append(f"Zenodo DOI mismatch: linked_record={info['doi']}, api={api_doi}")

    title = metadata.get("title")
    if not title:
        errors.append("Zenodo metadata title is empty.")

    creators = metadata.get("creators")
    if not isinstance(creators, list) or not creators:
        errors.append("Zenodo metadata creators list is empty.")

    license_value = metadata.get("license")
    if not license_value:
        errors.append("Zenodo metadata license is empty.")

    related_identifiers = metadata.get("related_identifiers")
    if not isinstance(related_identifiers, list):
        related_identifiers = []
    related_urls = [str(item.get("identifier", "")) for item in related_identifiers if isinstance(item, dict)]

    for url in get_public_hfh_urls(config, linked_record).values():
        if url not in related_urls:
            warnings.append(f"HFH URL is publicly accessible but not present in Zenodo related_identifiers: {url}")

    files = get_deposition_file_names(deposition)
    missing_expected_files = [name for name in EXPECTED_ZENODO_DEPOSITION_FILES if name not in files]
    if missing_expected_files:
        warnings.append(
            "Some expected evidence files are not present in Zenodo deposition: "
            + ", ".join(missing_expected_files)
        )

    tar_files = [name for name in files if name.endswith(".tar")]
    if tar_files:
        warnings.append(
            "Zenodo deposition contains tar shards. This is not expected for a linked dataset record: "
            + ", ".join(tar_files)
        )

    if bool(deposition.get("submitted", False)):
        warnings.append("Zenodo deposition appears to be already submitted/published.")

    return {
        "status": "passed" if not errors else "failed",
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "deposition_id": deposition_id,
        "state": deposition.get("state"),
        "submitted": bool(deposition.get("submitted", False)),
        "title": title,
        "creators": creators,
        "license": license_value,
        "doi": api_doi,
        "related_identifiers": related_identifiers,
        "files": files,
    }


def check_local_json_status(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file(), "status": None, "num_errors": None, "errors": []}

    if not path.is_file():
        result["errors"].append(f"Local report not found: {path}")
        return result

    try:
        data = read_json(path)
        result["status"] = data.get("status")
        result["num_errors"] = data.get("num_errors")
        if data.get("status") != "passed":
            result["errors"].append(f"Local report status is not passed: {path}")
        report_errors = data.get("errors")
        if isinstance(report_errors, list) and report_errors:
            result["errors"].extend(str(error) for error in report_errors)
    except Exception as exc:
        result["errors"].append(f"Could not read local report {path}: {exc}")

    return result


def verify_local_reports(config: Dict[str, Any]) -> Dict[str, Any]:
    reports = [
        get_zenodo_downloaded_report_path(config),
        get_output_filename(config, "file_verification_report_filename", "zenodo_file_verification_report.json"),
        get_public_visibility_report_path(config),
    ]
    checked = [check_local_json_status(path) for path in reports]

    # zenodo_link_verification_report.json may predate HFH being made public,
    # so it's not treated as authoritative — this check does fresh URL checks instead.
    stale_link_report = get_output_filename(
        config, "link_verification_report_filename", "zenodo_link_verification_report.json",
    )

    errors = [error for item in checked for error in item["errors"]]
    warnings = []
    if stale_link_report.is_file():
        warnings.append(
            f"Existing {stale_link_report} was not used as authoritative; "
            "fresh public link checks were performed instead."
        )

    return {
        "status": "passed" if not errors else "failed",
        "num_reports_checked": len(checked),
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "reports": checked,
    }


def run_check_release_readiness(config_path: Path, template_context: Optional[Dict[str, Any]] = None) -> None:
    if not config_path.is_file():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_config_source(config_path, **(template_context or {}))

    output_dir = get_hfh_export_dir(config)
    linked_record_path = get_linked_record_path(config)
    report_path = get_readiness_report_path(config)
    repo_id = get_repo_id(config)
    repo_type = get_repo_type(config)
    environment = get_zenodo_environment(config)

    logging.info("Target repo_id: %s", repo_id)
    logging.info("Target repo_type: %s", repo_type)
    logging.info("HFH output directory: %s", output_dir)
    logging.info("Zenodo environment: %s", environment)
    logging.info("Zenodo linked record: %s", linked_record_path)
    logging.info("Readiness report: %s", report_path)

    if not linked_record_path.is_file():
        fail(f"Zenodo linked dataset record not found: {linked_record_path}. Run 'zenodo prepare' first.")

    linked_record = read_json(linked_record_path)
    zenodo_info = validate_and_extract_zenodo_info(linked_record)

    logging.info("Zenodo deposition id: %s", zenodo_info["deposition_id"])
    logging.info("Zenodo DOI: %s", zenodo_info["doi"])
    logging.info("Zenodo DOI URL: %s", zenodo_info["doi_url"])
    logging.info("Zenodo record URL: %s", zenodo_info["record_url"])

    logging.info("Checking HFH API visibility...")
    hfh_api_visibility = verify_hfh_api_visibility(config)

    logging.info("Checking public HFH URLs without token...")
    hfh_public_urls = verify_hfh_public_urls(config, linked_record)

    logging.info("Reading Zenodo draft deposition through the API...")
    api_base_url = get_zenodo_api_base_url(config)
    token = get_zenodo_token(config)
    deposition = get_deposition(api_base_url, token, int(zenodo_info["deposition_id"]))

    logging.info("Checking Zenodo draft metadata, DOI, links and files...")
    zenodo_deposition_check = verify_zenodo_deposition(config, linked_record, deposition)

    logging.info("Checking local verification reports...")
    local_reports = verify_local_reports(config)

    all_errors: List[str] = []
    all_warnings: List[str] = []
    for section_name, section in [
        ("hfh_api_visibility", hfh_api_visibility),
        ("hfh_public_urls", hfh_public_urls),
        ("zenodo_deposition", zenodo_deposition_check),
        ("local_reports", local_reports),
    ]:
        all_errors.extend(f"{section_name}: {error}" for error in section.get("errors", []) or [])
        all_warnings.extend(f"{section_name}: {warning}" for warning in section.get("warnings", []) or [])

    ready_to_publish = not all_errors

    report = {
        "generated_at_utc": utc_now_iso(),
        "status": "passed" if ready_to_publish else "failed",
        "ready_to_publish_zenodo": ready_to_publish,
        "config_path": str(config_path),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "output_dir": str(output_dir),
        "zenodo": zenodo_info,
        "checks": {
            "hfh_api_visibility": hfh_api_visibility,
            "hfh_public_urls": hfh_public_urls,
            "zenodo_deposition": zenodo_deposition_check,
            "local_reports": local_reports,
        },
        "num_errors": len(all_errors),
        "num_warnings": len(all_warnings),
        "errors": all_errors,
        "warnings": all_warnings,
        "next_recommended_action": (
            "Review the Zenodo draft one last time and publish manually from the Zenodo web interface."
            if ready_to_publish
            else "Fix the reported errors before publishing the Zenodo draft."
        ),
    }

    write_json(report_path, report)
    logging.info("Report written: %s", report_path)

    if all_warnings:
        for warning in all_warnings:
            logging.warning(warning)

    if not ready_to_publish:
        for error in all_errors:
            logging.error(error)
        fail(f"Public release readiness check failed with {len(all_errors)} error(s).")

    logging.info("Public release readiness check passed.")
    logging.info("HFH public URLs are accessible without token.")
    logging.info("Zenodo draft metadata and evidence files were checked.")
    logging.info("Ready to publish Zenodo manually: yes")


# ═══════════════════════════════════════════════════════════════════════════
# Publicación definitiva del draft ("zenodo release")
# ═══════════════════════════════════════════════════════════════════════════
#
# Acción final e irreversible: una vez publicado en Zenodo, el depósito no se
# puede despublicar ni sus ficheros editar. Por eso exige que 'check-readiness'
# haya pasado (salvo --skip-readiness-check explícito) y tiene su propio
# --dry-run, que valida y consulta el estado sin llamar a actions/publish.

def get_publish_response_path(config: Dict[str, Any]) -> Path:
    return get_output_filename(config, "publish_response_filename", "zenodo_publish_response.json")


def get_deposition_response_path(config: Dict[str, Any]) -> Path:
    return get_output_filename(config, "deposition_response_filename", "zenodo_deposition_response.json")


def get_publication_report_path(config: Dict[str, Any]) -> Path:
    return get_output_filename(config, "publication_report_filename", "zenodo_publication_report.json")


def is_already_published(deposition: Dict[str, Any]) -> bool:
    submitted = deposition.get("submitted")
    state = str(deposition.get("state", "")).lower()
    links = deposition.get("links", {})

    if submitted is True:
        return True
    if state in {"done", "published"}:
        return True

    if isinstance(links, dict) and not links.get("publish"):
        # Published records commonly no longer expose the draft publish action.
        # Not sufficient on its own for every case, but a useful hint.
        if deposition.get("record_id") or deposition.get("doi"):
            return True

    return False


def collect_report_errors(report: Dict[str, Any]) -> List[str]:
    errors: List[str] = [str(item) for item in (report.get("errors") or []) if isinstance(report.get("errors"), list)]

    checks = report.get("checks")
    if isinstance(checks, dict):
        for check_name, check_data in checks.items():
            if not isinstance(check_data, dict):
                continue
            for item in check_data.get("errors") or []:
                errors.append(f"{check_name}: {item}")

    return errors


def readiness_report_passed(report: Dict[str, Any]) -> bool:
    status = str(report.get("status", "")).strip().lower()
    if status in {"passed", "pass", "ready", "success"}:
        return True

    for source in (report, report.get("summary") if isinstance(report.get("summary"), dict) else {}):
        for key in ("ready_to_publish", "ready_to_publish_zenodo", "ready_to_publish_zenodo_manually", "is_ready"):
            if isinstance(source.get(key), bool) and source[key]:
                return True

    return False


def validate_readiness_report(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        fail(
            f"Public release readiness report not found: {path}. "
            "Run 'zenodo check-readiness' first, or use --skip-readiness-check explicitly."
        )

    report = read_json(path)
    errors = collect_report_errors(report)
    if errors:
        fail(
            "Public release readiness report contains errors. Do not publish until "
            "'zenodo check-readiness' passes. Errors:\n" + "\n".join(f"  - {error}" for error in errors)
        )

    if not readiness_report_passed(report):
        fail(
            f"Public release readiness report does not indicate a passed/ready status: {path}. "
            "Run 'zenodo check-readiness' again."
        )

    return report


def verify_public_record_links(
    record_url: Optional[str], doi_url: Optional[str], timeout_seconds: int, environment: str,
) -> Dict[str, Any]:
    links = []
    errors: List[str] = []
    warnings: List[str] = []

    if record_url:
        result = check_public_url(record_url, timeout_seconds)
        result["label"] = "zenodo_record_url"
        links.append(result)
        if result["status"] != "passed":
            errors.append(
                f"Zenodo record URL is not publicly accessible: {record_url} "
                f"({result.get('http_status')}, {result.get('reason')})"
            )
    else:
        errors.append("Missing Zenodo record URL after publication.")

    if doi_url:
        result = check_public_url(doi_url, timeout_seconds)
        result["label"] = "doi_url"
        links.append(result)
        if result["status"] != "passed":
            if environment == "sandbox" and result.get("http_status") == 404:
                warnings.append("Zenodo Sandbox DOI URL returned 404. This can happen in Sandbox and is not fatal.")
            else:
                warnings.append(
                    f"DOI URL was not publicly resolvable immediately: {doi_url} "
                    f"({result.get('http_status')}, {result.get('reason')}). This can be a propagation delay; "
                    "the Zenodo record URL is the primary immediate check."
                )
    else:
        warnings.append("Missing DOI URL after publication.")

    return {"status": "passed" if not errors else "failed", "links": links, "errors": errors, "warnings": warnings}


def update_local_config_after_publication(
    config_path: Path, config: Dict[str, Any], deposition: Dict[str, Any],
    record_url: Optional[str], doi_url: Optional[str],
) -> None:
    zenodo = ensure_dict(config, "zenodo")
    doi = extract_reserved_doi(deposition)

    zenodo["published"] = True
    zenodo["published_at_utc"] = utc_now_iso()

    if deposition.get("id") is not None:
        zenodo["last_deposition_id"] = deposition.get("id")
    if deposition.get("record_id") is not None:
        zenodo["last_record_id"] = deposition.get("record_id")

    if doi:
        if get_zenodo_environment(config) == "sandbox":
            zenodo["sandbox_doi"] = doi
            zenodo["sandbox_doi_url"] = doi_url or build_doi_url(doi)
            if record_url:
                zenodo["sandbox_record_url"] = record_url
        else:
            zenodo["doi"] = doi
            zenodo["doi_url"] = doi_url or build_doi_url(doi)
            if record_url:
                zenodo["record_url"] = record_url

    write_yaml(config_path, config)


def run_release(
    config_path: Path,
    dry_run: bool = False,
    skip_readiness_check: bool = False,
    no_config_update: bool = False,
    template_context: Optional[Dict[str, Any]] = None,
) -> None:
    if not config_path.is_file():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_config_source(config_path, **(template_context or {}))

    environment = get_zenodo_environment(config)
    base_url = get_zenodo_base_url(config)
    api_base_url = get_zenodo_api_base_url(config)
    token_env_var = get_zenodo_token_env_var(config)
    linked_record_path = get_linked_record_path(config)
    readiness_report_path = get_readiness_report_path(config)
    publish_response_path = get_publish_response_path(config)
    deposition_response_path = get_deposition_response_path(config)
    publication_report_path = get_publication_report_path(config)

    logging.info("Zenodo environment: %s", environment)
    logging.info("Zenodo base URL: %s", base_url)
    logging.info("Zenodo API base URL: %s", api_base_url)
    logging.info("Zenodo token environment variable: %s", token_env_var)
    logging.info("Zenodo linked record: %s", linked_record_path)
    logging.info("Readiness report: %s", readiness_report_path)
    logging.info("Publication report: %s", publication_report_path)

    if not linked_record_path.is_file():
        fail(f"Zenodo linked dataset record not found: {linked_record_path}. Run 'zenodo prepare' first.")

    linked_record = read_json(linked_record_path)
    zenodo_info = validate_and_extract_zenodo_info(linked_record)
    deposition_id = int(zenodo_info["deposition_id"])

    logging.info("Target Zenodo deposition id: %s", deposition_id)
    logging.info("Linked record DOI: %s", linked_record.get("reserved_doi"))
    logging.info("Linked record DOI URL: %s", linked_record.get("doi_url"))
    logging.info("Linked record record URL: %s", linked_record.get("record_url"))

    readiness_report: Optional[Dict[str, Any]] = None
    if skip_readiness_check:
        logging.warning("Skipping public release readiness check because --skip-readiness-check was used.")
    else:
        logging.info("Validating public release readiness report...")
        readiness_report = validate_readiness_report(readiness_report_path)
        logging.info("Public release readiness report passed.")

    token = get_zenodo_token(config)

    logging.info("Reading Zenodo deposition before publication...")
    before = get_deposition(api_base_url, token, deposition_id)
    before_published = is_already_published(before)

    if before_published:
        logging.info("Zenodo deposition already appears to be published. No publish action will be performed.")
        publish_response: Dict[str, Any] = {
            "status": "already_published",
            "message": "No publish action was performed because the deposition already appears to be published.",
            "deposition_id": deposition_id,
        }
        after = before
    elif dry_run:
        logging.info("Dry run enabled. Zenodo deposition will not be published.")
        publish_response = {
            "status": "dry_run",
            "message": "No publish action was performed because dry-run is enabled.",
            "deposition_id": deposition_id,
        }
        after = before
    else:
        logging.info("Publishing Zenodo deposition...")
        publish_response = publish_deposition(api_base_url, token, deposition_id)
        logging.info("Publish request completed.")

        time.sleep(2)  # Zenodo can take a moment to expose updated record metadata.

        logging.info("Refreshing Zenodo deposition after publication...")
        after = get_deposition(api_base_url, token, deposition_id)

    doi = extract_reserved_doi(after) or str(linked_record.get("reserved_doi") or "") or None
    doi_url = build_doi_url(doi)
    record_url = build_record_url(base_url, after) or str(linked_record.get("record_url") or "") or None

    logging.info("DOI after publication: %s", doi or "not available")
    logging.info("DOI URL after publication: %s", doi_url or "not available")
    logging.info("Record URL after publication: %s", record_url or "not available")

    timeout_seconds = get_link_check_timeout(config)
    public_link_verification = verify_public_record_links(record_url, doi_url, timeout_seconds, environment)

    status = "passed"
    errors: List[str] = []
    warnings: List[str] = list(public_link_verification["warnings"])

    if dry_run:
        status = "dry_run"
    elif public_link_verification["errors"]:
        status = "failed"
        errors.extend(public_link_verification["errors"])

    if not dry_run and not before_published and not is_already_published(after):
        status = "failed"
        errors.append("Zenodo deposition does not appear to be published after publish action.")

    publication_report = {
        "generated_at_utc": utc_now_iso(),
        "status": status,
        "environment": environment,
        "deposition_id": deposition_id,
        "was_already_published_before_run": before_published,
        "dry_run": dry_run,
        "readiness_check": {
            "skipped": skip_readiness_check,
            "report_path": str(readiness_report_path),
            "status": readiness_report.get("status") if isinstance(readiness_report, dict) else None,
        },
        "doi": doi,
        "doi_url": doi_url,
        "record_url": record_url,
        "public_link_verification": public_link_verification,
        "num_errors": len(errors),
        "errors": errors,
        "warnings": warnings,
        "output_files": {
            "publish_response": str(publish_response_path),
            "deposition_response": str(deposition_response_path),
            "publication_report": str(publication_report_path),
        },
    }

    write_json(publish_response_path, publish_response)
    write_json(deposition_response_path, after)
    write_json(publication_report_path, publication_report)

    logging.info("Publish response written: %s", publish_response_path)
    logging.info("Deposition response written: %s", deposition_response_path)
    logging.info("Publication report written: %s", publication_report_path)

    if not dry_run and status == "passed" and not no_config_update:
        if config_path.suffix.lower() == ".j2":
            # Never overwrite a Jinja template with resolved YAML — write the
            # published metadata to a plain sibling copy instead.
            resolved_config_path = get_zenodo_output_dir(config) / "Zenodo_published.yaml"
            ensure_dir(resolved_config_path.parent)
            logging.info(
                "Config is a Jinja template (%s); writing publication metadata to %s instead.",
                config_path, resolved_config_path,
            )
            update_local_config_after_publication(resolved_config_path, config, after, record_url, doi_url)
        else:
            logging.info("Updating local configuration with publication metadata...")
            update_local_config_after_publication(config_path, config, after, record_url, doi_url)

    for warning in warnings:
        logging.warning(warning)

    if errors:
        for error in errors:
            logging.error(error)
        fail(f"Zenodo publication check failed with {len(errors)} error(s).")

    if dry_run:
        logging.info("Dry run completed successfully. Zenodo deposition was not published.")
    elif before_published:
        logging.info("Zenodo record was already published and public checks passed.")
    else:
        logging.info("Zenodo record published successfully.")

    logging.info("DOI: %s", doi_url or doi)
    logging.info("Record URL: %s", record_url)
