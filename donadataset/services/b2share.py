"""Lógica de interacción con B2SHARE (EUDAT) — sin nada de CLI/Typer.

Mismo patrón de "registro enlazado" que donadataset.services.zenodo:

- Los shards pesados del dataset se quedan en HuggingFace Hub.
- B2SHARE almacena metadata, manifests, checksums, reports de verificación y
  un enlace (alternate_identifier) al repo de HuggingFace Hub — no un
  archivo físico completo de todas las imágenes.
- A diferencia de Zenodo, B2SHARE NO reserva un PID/DOI al crear el draft
  (solo al publicar) — por eso no hay un paso "upload" (inserción temprana
  del identificador); en vez de eso, 'run_sync_b2share_pid' se ejecuta
  DESPUÉS de publicar, para leer el PID/DOI ya asignado e insertarlo en el
  CITATION.cff local.

Cuatro operaciones:

- `run_b2share_linked_dataset_creation`: descarga+verifica HuggingFace Hub en
  vivo, crea (o sincroniza) el draft de B2SHARE y sube los ficheros de
  evidencia.
- `run_check_release_readiness`: chequeo de solo lectura antes de publicar.
- `run_release`: publica el draft — acción irreversible (o pendiente de
  aprobación por un moderador de la comunidad, según cómo esté configurada).
- `run_sync_b2share_pid`: una vez publicado, inserta el PID/DOI en el
  CITATION.cff local del export de HuggingFace.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from huggingface_hub import HfApi

from donadataset.config import settings as global_settings
from donadataset.services.common import (
    as_bool,
    fail,
    format_size,
    get_nested,
    load_yaml,
    md5_file,
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
    check_public_url,
    download_and_verify_hfh,
    ensure_dir,
    get_dataset_visibility,
    get_output_dir,
    get_repo_id,
    get_repo_type,
    get_token as get_hfh_token,
    get_token_env_var as get_hfh_token_env_var,
    get_tree_url,
    load_config_source,
)

INTERNAL_CONFIG_FILENAME = "HuggingFaceHub.yaml"

# Same evidence files as Zenodo — whatever 'huggingface upload' pushed
# already contains them, fetched fresh from the live download below.
DEFAULT_B2SHARE_EVIDENCE_FILENAMES = [
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


# ── Configuración ─────────────────────────────────────────────────────────────

def get_b2share_environment(config: Dict[str, Any]) -> str:
    env = str(get_nested(config, ["b2share", "environment"], "sandbox")).strip().lower()
    if env not in {"sandbox", "production"}:
        fail("b2share.environment must be either 'sandbox' or 'production'.")
    return env


def get_b2share_base_url(config: Dict[str, Any]) -> str:
    return (
        "https://trng-b2share.eudat.eu" if get_b2share_environment(config) == "sandbox"
        else "https://b2share.eudat.eu"
    )


def get_b2share_api_base_url(config: Dict[str, Any]) -> str:
    return f"{get_b2share_base_url(config)}/api"


def get_b2share_token_env_var(config: Dict[str, Any]) -> str:
    token_env_var = str(get_nested(config, ["b2share", "token_env_var"], "")).strip()
    return token_env_var or "B2SHARE_TOKEN"


def get_b2share_token(config: Dict[str, Any]) -> str:
    """Resolves the B2SHARE token: the environment variable always wins if
    set; otherwise falls back to b2share.token in settings.toml (set via
    'donadataset publish b2share config set token')."""
    token_env_var = get_b2share_token_env_var(config)
    token = os.environ.get(token_env_var)
    if token:
        return token
    if global_settings.B2SHARE.token:
        return global_settings.B2SHARE.token
    base_url = get_b2share_base_url(config)
    fail(
        f"No B2SHARE token found. Get one at {base_url}/user/profile under "
        f"'Applications' -> 'Personal access tokens', then either set it with: "
        f"export {token_env_var}='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', or store it "
        "with 'donadataset publish b2share config set token'."
    )


def get_community_id(config: Dict[str, Any]) -> str:
    community_id = str(get_nested(config, ["b2share", "community"], "")).strip()
    if not community_id or "REPLACE_WITH" in community_id:
        fail(
            "Invalid b2share.community in the configuration. This must be the UUID of an "
            "existing EUDAT community — request one at https://b2share.eudat.eu if you "
            "don't have one yet, then set it with "
            "'donadataset publish b2share config set community_id=<uuid>'."
        )
    return community_id


def get_publish_flag(config: Dict[str, Any]) -> bool:
    return as_bool(get_nested(config, ["b2share", "publish"], False), False)


def get_hfh_export_dir(config: Dict[str, Any]) -> Path:
    """Same export directory 'huggingface prepare' created — only used by
    'sync-pid', to write the PID/DOI into the local CITATION.cff."""
    return get_output_dir(config)


def build_b2share_template_context(
    *,
    hfh_output_dir: Optional[str] = None,
    b2share_output_dir: Optional[str] = None,
    repo_id: Optional[str] = None,
    community_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    description: Optional[str] = None,
    license_id: Optional[str] = None,
    author_given_names: Optional[str] = None,
    author_family_names: Optional[str] = None,
    author_affiliation: Optional[str] = None,
    environment: Optional[str] = None,
) -> Dict[str, Any]:
    """Builds the Jinja context for templates/B2SHARE.yaml.j2 — same
    fallback-resolution pattern as build_zenodo_template_context."""
    if repo_id:
        huggingface_dataset_url = build_dataset_url(repo_id)
        huggingface_tree_url = get_tree_url(repo_id)
    else:
        huggingface_dataset_url = "https://huggingface.co/datasets/REPLACE_WITH_HF_USER/REPLACE_WITH_DATASET_SLUG"
        huggingface_tree_url = huggingface_dataset_url + "/tree/main"

    return {
        "hfh_output_dir": hfh_output_dir,
        "b2share_output_dir": b2share_output_dir or "REPLACE_WITH_B2SHARE_OUTPUT_DIR",
        "repo_id": repo_id,
        "community_id": community_id,
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


def get_b2share_output_dir(config: Dict[str, Any]) -> Path:
    """Directory where everything B2SHARE-related is staged — mirrors
    get_zenodo_output_dir. b2share.output_dir override wins; otherwise a
    sibling of the HFH export dir named B2SHARE_{dataset_slug}."""
    configured = get_nested(config, ["b2share", "output_dir"], None)
    if configured:
        return Path(str(configured))

    from donadataset.services.huggingface import get_dataset_slug
    hfh_export_dir = get_hfh_export_dir(config)
    return hfh_export_dir.parent / f"B2SHARE_{get_dataset_slug(config)}"


def get_b2share_hfh_download_dir(config: Dict[str, Any]) -> Path:
    return get_b2share_output_dir(config) / "hfh_download"


def get_b2share_downloaded_report_path(config: Dict[str, Any]) -> Path:
    return get_b2share_output_dir(config) / "verification_report_downloaded.json"


def ensure_fresh_hfh_download_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """Downloads the HuggingFace Hub repo right now and verifies it — mirrors
    Zenodo's own live-download-and-verify step, so what's staged for B2SHARE
    is guaranteed to match what's live on HuggingFace Hub."""
    download_dir = get_b2share_hfh_download_dir(config)
    report_path = get_b2share_downloaded_report_path(config)
    token = get_hfh_token(config)

    logging.info("Downloading HuggingFace Hub repository to verify it matches the local export...")
    logging.info("Download directory: %s", download_dir)
    return download_and_verify_hfh(config, token, download_dir, report_path, delete_after_success=False)


def get_default_files_to_upload(config: Dict[str, Any]) -> List[Path]:
    hfh_download_dir = get_b2share_hfh_download_dir(config)
    files = [hfh_download_dir / filename for filename in DEFAULT_B2SHARE_EVIDENCE_FILENAMES]
    files.append(get_b2share_downloaded_report_path(config))
    return files


def validate_files_to_upload(files: List[Path]) -> None:
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        fail("Some B2SHARE evidence files do not exist:\n" + "\n".join(f"  - {path}" for path in missing))


def get_linked_record_path(config: Dict[str, Any]) -> Path:
    return get_b2share_output_dir(config) / "b2share_linked_dataset_record.json"


def get_readiness_report_path(config: Dict[str, Any]) -> Path:
    return get_b2share_output_dir(config) / "b2share_public_release_readiness_report.json"


def get_link_check_timeout(config: Dict[str, Any]) -> int:
    return int(get_nested(config, ["b2share", "link_checking", "timeout_seconds"], 20))


# ── API de B2SHARE ────────────────────────────────────────────────────────────

def b2share_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def check_response(response: requests.Response, expected: Tuple[int, ...], context: str) -> None:
    if response.status_code in expected:
        return
    try:
        body = response.json()
    except Exception:
        body = response.text
    fail(f"{context} failed. HTTP status={response.status_code}. Response={body}")


def build_b2share_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "titles": get_nested(config, ["b2share", "titles"], []),
        "community": get_community_id(config),
        "open_access": as_bool(get_nested(config, ["b2share", "open_access"], True), True),
        "descriptions": get_nested(config, ["b2share", "descriptions"], []),
        "creators": get_nested(config, ["b2share", "creators"], []),
        "licence": get_nested(config, ["b2share", "licence"], None),
        "keywords": get_nested(config, ["b2share", "keywords"], []),
        "alternate_identifier": get_nested(config, ["b2share", "alternate_identifier"], None),
        "community_specific": get_nested(config, ["b2share", "community_specific"], {}),
    }
    if not metadata["titles"]:
        fail("b2share.titles must be a non-empty list.")
    return {k: v for k, v in metadata.items() if v not in (None, [], {})}


def create_draft_record(api_base_url: str, token: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{api_base_url}/records/", headers=b2share_headers(token), json=metadata, timeout=60)
    check_response(response, (200, 201), "Create B2SHARE draft record")
    return response.json()


def get_record(api_base_url: str, token: str, record_id: str, draft: bool = False) -> Dict[str, Any]:
    suffix = "/draft" if draft else ""
    url = f"{api_base_url}/records/{record_id}{suffix}"
    response = requests.get(url, headers=b2share_headers(token), timeout=60)
    check_response(response, (200,), "Get B2SHARE record")
    return response.json()


def get_bucket_id(record: Dict[str, Any]) -> str:
    links = record.get("links", {})
    bucket_url = links.get("files") if isinstance(links, dict) else None
    if not bucket_url:
        fail("B2SHARE record response does not contain a files bucket link.")
    return str(bucket_url).rstrip("/").rsplit("/", 1)[-1]


def upload_file_to_bucket(api_base_url: str, token: str, bucket_id: str, file_path: Path, remote_filename: str) -> Dict[str, Any]:
    url = f"{api_base_url}/files/{bucket_id}/{quote(remote_filename)}"
    with file_path.open("rb") as f:
        response = requests.put(url, headers={"Authorization": f"Bearer {token}"}, data=f, timeout=300)
    check_response(response, (200, 201), f"Upload file to B2SHARE bucket: {remote_filename}")
    return response.json()


def publish_draft_record(api_base_url: str, token: str, record_id: str) -> Dict[str, Any]:
    """B2SHARE has no dedicated 'publish' endpoint like Zenodo — publication
    is a JSON Patch changing publication_state on the (non-draft) record."""
    url = f"{api_base_url}/records/{record_id}"
    patch = [{"op": "add", "path": "/publication_state", "value": "submitted"}]
    response = requests.patch(
        url, headers={**b2share_headers(token), "Content-Type": "application/json-patch+json"},
        json=patch, timeout=60,
    )
    check_response(response, (200,), "Publish B2SHARE record")
    return response.json()


def extract_pid(record: Dict[str, Any]) -> Optional[str]:
    metadata = record.get("metadata", record)
    return metadata.get("DOI") or metadata.get("ePIC_PID")


def build_pid_url(pid: Optional[str]) -> Optional[str]:
    if not pid:
        return None
    return f"https://doi.org/{pid}" if not pid.startswith("http") else pid


def build_record_url(base_url: str, record: Dict[str, Any]) -> Optional[str]:
    links = record.get("links", {})
    if isinstance(links, dict):
        html = links.get("html")
        if html:
            return str(html)
    record_id = record.get("id")
    return f"{base_url}/records/{record_id}" if record_id else None


def create_linked_dataset_record(
    config: Dict[str, Any], record: Dict[str, Any], files: List[Path], downloaded_report: Dict[str, Any],
) -> Dict[str, Any]:
    base_url = get_b2share_base_url(config)
    file_entries = [
        {
            "path": str(path), "name": path.name, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "md5": md5_file(path),
        }
        for path in files
    ]

    return {
        "generated_at_utc": utc_now_iso(),
        "record_type": "b2share_linked_dataset_record",
        "record_scope": (
            "This record stores metadata, manifests, checksums, verification reports, "
            "and links. Heavy dataset shards are hosted on Hugging Face Hub."
        ),
        "b2share_environment": get_b2share_environment(config),
        "b2share_base_url": base_url,
        "record_id": record.get("id"),
        "pid": extract_pid(record),
        "pid_url": build_pid_url(extract_pid(record)),
        "record_url": build_record_url(base_url, record),
        "huggingface_verification": {
            "status": downloaded_report.get("status"),
            "repo_id": downloaded_report.get("repo_id"),
            "repo_type": downloaded_report.get("repo_type"),
            "global_files_verified": downloaded_report.get("global_files_verified"),
            "internal_tar_members_verified": downloaded_report.get("internal_tar_members_verified"),
            "num_errors": downloaded_report.get("num_errors"),
        },
        "files_to_upload": file_entries,
    }


def validate_and_extract_b2share_info(record: Dict[str, Any]) -> Dict[str, Any]:
    environment = record.get("b2share_environment")
    record_id = record.get("record_id")

    if environment not in {"sandbox", "production"}:
        fail("b2share_linked_dataset_record.json must contain 'b2share_environment' with value 'sandbox' or 'production'.")
    if not record_id:
        fail("b2share_linked_dataset_record.json does not contain record_id.")

    return {
        "environment": str(environment),
        "record_id": record_id,
        "pid": record.get("pid"),
        "pid_url": record.get("pid_url"),
        "record_url": record.get("record_url"),
    }


# ── Subida de ficheros de evidencia ───────────────────────────────────────────

def upload_evidence_files(
    api_base_url: str, token: str, record: Dict[str, Any], files_to_upload: List[Path],
) -> Dict[str, Any]:
    bucket_id = get_bucket_id(record)
    for path in files_to_upload:
        logging.info("Uploading %s (%s)...", path.name, format_size(path.stat().st_size))
        upload_file_to_bucket(api_base_url, token, bucket_id, path, path.name)
    return record


# ── Creación del registro enlazado (flujo principal) ─────────────────────────

def run_b2share_linked_dataset_creation(
    config_path: Path, dry_run: bool = False, template_context: Optional[Dict[str, Any]] = None,
) -> None:
    if not config_path.exists():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_config_source(config_path, **(template_context or {}))

    if not as_bool(get_nested(config, ["b2share", "enabled"], False), False):
        fail("b2share.enabled is false. Set b2share.enabled: true to use this command.")

    environment = get_b2share_environment(config)
    api_base_url = get_b2share_api_base_url(config)
    base_url = get_b2share_base_url(config)
    token_env_var = get_b2share_token_env_var(config)
    files_to_upload = get_default_files_to_upload(config)

    logging.info("B2SHARE environment: %s", environment)
    logging.info("B2SHARE base URL: %s", base_url)
    logging.info("B2SHARE API base URL: %s", api_base_url)
    logging.info("B2SHARE token environment variable: %s", token_env_var)
    logging.info("Files that will be uploaded (fetched from a live HuggingFace Hub download): %d", len(files_to_upload))

    if dry_run:
        logging.info("Dry run enabled. HuggingFace Hub will not be downloaded; no B2SHARE record will be created.")
        return

    downloaded_report = ensure_fresh_hfh_download_report(config)

    logging.info("Validating downloaded files configured for B2SHARE upload...")
    validate_files_to_upload(files_to_upload)

    total_upload_size = sum(path.stat().st_size for path in files_to_upload)
    logging.info("Total B2SHARE evidence upload size: %s", format_size(total_upload_size))

    token = get_b2share_token(config)
    metadata = build_b2share_metadata(config)

    logging.info("Creating B2SHARE draft record...")
    record = create_draft_record(api_base_url, token, metadata)
    record_id = record.get("id")
    if record_id is None:
        fail("B2SHARE draft record response does not contain an id.")
    logging.info("Created draft record id: %s", record_id)

    upload_evidence_files(api_base_url, token, record, files_to_upload)

    logging.info("Reading draft record after upload...")
    record = get_record(api_base_url, token, str(record_id), draft=True)

    linked_record = create_linked_dataset_record(config, record, files_to_upload, downloaded_report)
    linked_record_path = get_linked_record_path(config)
    write_json(linked_record_path, linked_record)
    logging.info("Linked dataset record written: %s", linked_record_path)
    logging.info(
        "B2SHARE draft created (id=%s). No PID/DOI is reserved until you publish "
        "with 'b2share release'. Note: publishing may require approval by a "
        "moderator of your EUDAT community, depending on how it's configured.",
        record_id,
    )


def get_existing_record_id_from_linked_record(config: Dict[str, Any]) -> str:
    linked_record_path = get_linked_record_path(config)
    if not linked_record_path.is_file():
        fail(
            f"B2SHARE linked dataset record not found: {linked_record_path}. "
            "Run 'b2share prepare' first, or provide an existing draft."
        )
    linked_record = read_json(linked_record_path)
    record_id = linked_record.get("record_id")
    if not record_id:
        fail(f"{linked_record_path} does not contain record_id.")
    return str(record_id)


def run_b2share_existing_draft_sync(
    config_path: Path, dry_run: bool = False, template_context: Optional[Dict[str, Any]] = None,
) -> None:
    if not config_path.exists():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_config_source(config_path, **(template_context or {}))

    if not as_bool(get_nested(config, ["b2share", "enabled"], False), False):
        fail("b2share.enabled is false. Set b2share.enabled: true to use this command.")

    environment = get_b2share_environment(config)
    api_base_url = get_b2share_api_base_url(config)
    base_url = get_b2share_base_url(config)
    token_env_var = get_b2share_token_env_var(config)
    files_to_upload = get_default_files_to_upload(config)
    record_id = get_existing_record_id_from_linked_record(config)

    logging.info("B2SHARE environment: %s", environment)
    logging.info("B2SHARE base URL: %s", base_url)
    logging.info("B2SHARE API base URL: %s", api_base_url)
    logging.info("B2SHARE token environment variable: %s", token_env_var)
    logging.info("Existing draft record id: %s", record_id)
    logging.info("Evidence files that will be synced (fetched from a live HuggingFace Hub download): %d", len(files_to_upload))

    if dry_run:
        logging.info("Dry run enabled. HuggingFace Hub will not be downloaded; existing B2SHARE draft will not be modified.")
        for path in files_to_upload:
            logging.info("  - %s", path)
        return

    downloaded_report = ensure_fresh_hfh_download_report(config)

    logging.info("Validating downloaded evidence files...")
    validate_files_to_upload(files_to_upload)

    total_upload_size = sum(path.stat().st_size for path in files_to_upload)
    logging.info("Total B2SHARE evidence upload size: %s", format_size(total_upload_size))

    token = get_b2share_token(config)

    logging.info("Reading existing B2SHARE draft record...")
    record = get_record(api_base_url, token, record_id, draft=True)

    upload_evidence_files(api_base_url, token, record, files_to_upload)

    record = get_record(api_base_url, token, record_id, draft=True)
    linked_record = create_linked_dataset_record(config, record, files_to_upload, downloaded_report)
    linked_record_path = get_linked_record_path(config)
    write_json(linked_record_path, linked_record)

    logging.info("Existing B2SHARE draft synchronization completed successfully.")
    logging.info("Record id: %s", record.get("id"))
    logging.info("Linked dataset record: %s", linked_record_path)


# ── Chequeo de preparación para publicar ─────────────────────────────────────

def check_local_json_status(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "missing", "errors": [f"Report not found: {path}"]}
    try:
        data = read_json(path)
    except Exception as exc:
        return {"path": str(path), "status": "unreadable", "errors": [f"Could not parse {path}: {exc}"]}
    status = data.get("status")
    if status != "passed":
        return {"path": str(path), "status": status, "errors": [f"{path} does not indicate status=passed (got {status!r})."]}
    return {"path": str(path), "status": "passed", "errors": []}


def verify_local_reports(config: Dict[str, Any]) -> Dict[str, Any]:
    reports = [get_b2share_downloaded_report_path(config)]
    checked = [check_local_json_status(path) for path in reports]
    errors = [error for item in checked for error in item["errors"]]
    return {
        "status": "passed" if not errors else "failed",
        "num_reports_checked": len(checked),
        "num_errors": len(errors),
        "errors": errors,
        "reports": checked,
    }


def verify_hfh_api_visibility(config: Dict[str, Any]) -> Dict[str, Any]:
    repo_id = get_repo_id(config)
    repo_type = get_repo_type(config)
    token_env_var = get_hfh_token_env_var(config)
    token = os.environ.get(token_env_var)

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


def verify_hfh_public_urls(config: Dict[str, Any]) -> Dict[str, Any]:
    timeout_seconds = get_link_check_timeout(config)
    repo_id = get_repo_id(config)
    urls = {"dataset_url": build_dataset_url(repo_id), "tree_url": get_tree_url(repo_id)}

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


def run_check_release_readiness(config_path: Path, template_context: Optional[Dict[str, Any]] = None) -> None:
    if not config_path.is_file():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_config_source(config_path, **(template_context or {}))

    linked_record_path = get_linked_record_path(config)
    report_path = get_readiness_report_path(config)
    repo_id = get_repo_id(config)
    environment = get_b2share_environment(config)

    logging.info("Target repo_id: %s", repo_id)
    logging.info("B2SHARE environment: %s", environment)
    logging.info("B2SHARE linked record: %s", linked_record_path)
    logging.info("Readiness report: %s", report_path)

    if not linked_record_path.is_file():
        fail(f"B2SHARE linked dataset record not found: {linked_record_path}. Run 'b2share prepare' first.")

    linked_record = read_json(linked_record_path)
    b2share_info = validate_and_extract_b2share_info(linked_record)

    logging.info("B2SHARE record id: %s", b2share_info["record_id"])

    logging.info("Checking HFH API visibility...")
    hfh_api_visibility = verify_hfh_api_visibility(config)

    logging.info("Checking public HFH URLs without token...")
    hfh_public_urls = verify_hfh_public_urls(config)

    logging.info("Checking local verification reports...")
    local_reports = verify_local_reports(config)

    all_errors: List[str] = []
    for section_name, section in [
        ("hfh_api_visibility", hfh_api_visibility),
        ("hfh_public_urls", hfh_public_urls),
        ("local_reports", local_reports),
    ]:
        all_errors.extend(f"{section_name}: {error}" for error in section.get("errors", []) or [])

    ready_to_publish = not all_errors

    report = {
        "generated_at_utc": utc_now_iso(),
        "status": "passed" if ready_to_publish else "failed",
        "ready_to_publish_b2share": ready_to_publish,
        "repo_id": repo_id,
        "b2share_record_id": b2share_info["record_id"],
        "hfh_api_visibility": hfh_api_visibility,
        "hfh_public_urls": hfh_public_urls,
        "local_reports": local_reports,
        "errors": all_errors,
    }
    write_json(report_path, report)

    if not ready_to_publish:
        for error in all_errors:
            logging.error(error)
        fail(f"B2SHARE release readiness check failed with {len(all_errors)} error(s).")

    logging.info("Ready to publish. Run 'donadataset publish b2share release' when you want to make it definitive.")


def validate_readiness_report(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        fail(f"Readiness report not found: {path}. Run 'b2share check-readiness' first.")
    report = read_json(path)
    if report.get("status") != "passed":
        fail(f"Readiness report at {path} does not indicate a passed status.")
    return report


# ── Publicación ───────────────────────────────────────────────────────────────

def run_release(
    config_path: Path, dry_run: bool = False, skip_readiness_check: bool = False,
    template_context: Optional[Dict[str, Any]] = None,
) -> None:
    if not config_path.is_file():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_config_source(config_path, **(template_context or {}))

    api_base_url = get_b2share_api_base_url(config)
    linked_record_path = get_linked_record_path(config)
    readiness_report_path = get_readiness_report_path(config)

    if not linked_record_path.is_file():
        fail(f"B2SHARE linked dataset record not found: {linked_record_path}. Run 'b2share prepare' first.")

    linked_record = read_json(linked_record_path)
    b2share_info = validate_and_extract_b2share_info(linked_record)
    record_id = str(b2share_info["record_id"])

    logging.info("Target B2SHARE record id: %s", record_id)

    if skip_readiness_check:
        logging.warning("Skipping public release readiness check because --skip-readiness-check was used.")
    else:
        logging.info("Validating public release readiness report...")
        validate_readiness_report(readiness_report_path)
        logging.info("Public release readiness report passed.")

    token = get_b2share_token(config)

    if dry_run:
        logging.info("Dry run enabled. B2SHARE record will not be published.")
        return

    logging.info("Publishing B2SHARE record (submitting for publication)...")
    publish_draft_record(api_base_url, token, record_id)

    logging.info("Reading record after publication...")
    record = get_record(api_base_url, token, record_id)

    updated_linked_record = create_linked_dataset_record(
        config, record, [Path(f["path"]) for f in linked_record.get("files_to_upload", [])],
        linked_record.get("huggingface_verification", {}),
    )
    write_json(linked_record_path, updated_linked_record)

    pid = updated_linked_record.get("pid")
    if pid:
        logging.info("B2SHARE record submitted for publication. PID/DOI: %s", pid)
    else:
        logging.info(
            "B2SHARE record submitted for publication. No PID/DOI returned yet — "
            "this may mean it's pending approval by a moderator of your EUDAT "
            "community. Check %s once approved, then run 'b2share sync-pid'.",
            updated_linked_record.get("record_url"),
        )


# ── Sincronizar el PID/DOI en la metadata local ("b2share sync-pid") ────────

def update_citation_cff_with_b2share_pid(citation_path: Path, pid: str, record_url: str) -> bool:
    """Writes the B2SHARE PID/DOI into CITATION.cff. Mirrors
    update_citation_cff_with_hfh_doi (huggingface.py): if citation.doi is
    already set (e.g. by Zenodo), this is added to identifiers instead of
    overwriting the main field — different platforms, different scope."""
    citation = load_yaml(citation_path)

    if citation.get("doi") == pid:
        return False

    if not citation.get("doi"):
        citation["doi"] = pid
        citation["url"] = record_url
    else:
        identifiers = citation.get("identifiers")
        if not isinstance(identifiers, list):
            identifiers = []
        identifiers = [
            item for item in identifiers
            if not (isinstance(item, dict) and item.get("description") == "B2SHARE (EUDAT) dataset record PID/DOI")
        ]
        identifiers.append({
            "type": "doi", "value": pid, "description": "B2SHARE (EUDAT) dataset record PID/DOI",
        })
        citation["identifiers"] = identifiers

    write_yaml(citation_path, citation)
    return True


def run_sync_b2share_pid(config_path: Path, template_context: Optional[Dict[str, Any]] = None) -> None:
    if not config_path.exists():
        fail(f"Configuration file not found: {config_path}")

    logging.info("Reading configuration: %s", config_path)
    config = load_config_source(config_path, **(template_context or {}))

    linked_record_path = get_linked_record_path(config)
    if not linked_record_path.is_file():
        fail(f"B2SHARE linked dataset record not found: {linked_record_path}. Run 'b2share prepare' first.")

    linked_record = read_json(linked_record_path)
    pid = linked_record.get("pid")
    record_url = linked_record.get("record_url")

    if not pid:
        logging.warning(
            "No PID/DOI found yet in %s. It's only assigned once the record is actually "
            "published (may require community moderator approval) — run 'b2share release' "
            "first, and check back later if it was pending approval.",
            linked_record_path,
        )
        return

    hfh_export_dir = get_hfh_export_dir(config)
    citation_path = hfh_export_dir / str(get_nested(config, ["output_files", "citation_filename"], "CITATION.cff"))
    if not citation_path.is_file():
        fail(f"CITATION.cff not found: {citation_path}. Run 'huggingface prepare' first.")

    changed = update_citation_cff_with_b2share_pid(citation_path, pid, record_url)

    if changed:
        from donadataset.services.huggingface import write_checksums
        logging.info("CITATION.cff changed — recomputing checksums...")
        write_checksums(hfh_export_dir, config)
        logging.info("Remember to run 'huggingface upload' again so the updated CITATION.cff is published.")
    else:
        logging.info("CITATION.cff already had this PID/DOI. Nothing to update.")
