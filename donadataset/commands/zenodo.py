"""Comandos CLI para publicar el dataset en Zenodo.

Gestiona únicamente los parámetros de entrada; toda la lógica de
preparación/actualización/descarga vive en donadataset.services.zenodo.
"""
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional

import typer
from rich.console import Console

from donadataset.commands import zenodo_config_commands
from donadataset.commands.huggingface import _resolved_config_path as _hf_resolved_config_path
from donadataset.commands.huggingface import _warn_if_token_missing as _hf_warn_if_token_missing
from donadataset.commands.huggingface import _wizard_resolve_repo_id
from donadataset.config import REPO_ROOT, ZenodoSettings, get_hfh_output_dir, get_zenodo_output_dir, settings
from donadataset.services import huggingface as hf_service
from donadataset.services import zenodo as zenodo_service

console = Console()
app     = typer.Typer(help="Publica el dataset en Zenodo.")
app.add_typer(zenodo_config_commands.app, name="config")


class ZenodoEnvironment(str, Enum):
    sandbox    = "sandbox"
    production = "production"


# Única fuente de configuración para prepare/upload/check-readiness/release —
# no hay --config, ni YAML personal alternativo. Siempre se rellena esta
# plantilla con los flags de abajo (cuyo valor por defecto sale de
# settings.toml ZENODO.*/HUGGINGFACE.*).
DEFAULT_TEMPLATE_FILE = REPO_ROOT / "templates" / "Zenodo.yaml.j2"

# El export de HuggingFace ya existente (ENTRADA para 'zenodo upload') — mismo
# default que 'huggingface prepare', así si usaste el suyo por defecto,
# zenodo lo encuentra sin que tengas que repetirlo.
HF_DEFAULTS = settings.HUGGINGFACE
ZENODO_DEFAULTS = settings.ZENODO
DEFAULT_HFH_OUTPUT_DIR = get_hfh_output_dir(HF_DEFAULTS.repo_id)

# Directorio PROPIO de zenodo (SALIDA): aquí descarga HuggingFace Hub en
# tiempo real y guarda todo lo que sube a Zenodo. Misma estructura que el
# de HuggingFace de arriba: <Documents>/donadataset/Zenodo/<repo_id>.
DEFAULT_ZENODO_OUTPUT_DIR = get_zenodo_output_dir(HF_DEFAULTS.repo_id)


def _zenodo_help(field: str) -> str:
    """Reutiliza la description de ZenodoSettings como help= del flag
    equivalente — un único sitio para mantener el texto (config.py), en vez
    de duplicarlo aquí a mano."""
    return ZenodoSettings.model_fields[field].description


def _parse_communities(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_communities(communities_flag: Optional[str]) -> List[str]:
    """--communities picks a subset of settings.toml ZENODO.communities for
    this run — it can never name a community that isn't already configured
    there, since that's meant to be the single source of truth (config set
    communities=... once, then optionally narrow it per invocation)."""
    configured = _parse_communities(ZENODO_DEFAULTS.communities)
    if communities_flag is None:
        return configured

    selected = _parse_communities(communities_flag)
    unknown = [slug for slug in selected if slug not in configured]
    if unknown:
        raise typer.BadParameter(
            f"Comunidad(es) no configurada(s) en settings.toml ZENODO.communities: {', '.join(unknown)}. "
            f"Configuradas: {', '.join(configured) or '(ninguna)'}. "
            "Añádelas primero con 'donadataset publish zenodo config set communities=...'."
        )
    return selected


def _build_template_context(
    zenodo_output_dir: Optional[str], repo_id: Optional[str],
    hfh_output_dir: Optional[str] = None, environment: Optional[str] = None,
    communities: Optional[List[str]] = None,
) -> dict:
    """--repo-id/--output-dir son los datos que hay que decidir por
    invocación; el resto de la identidad del dataset (nombre, descripción,
    licencia, autor) se reutiliza de settings.toml (HUGGINGFACE.*) — las
    mismas que ya usa 'huggingface prepare'. hfh_output_dir solo lo pasan
    'sync-doi' (necesita escribir el DOI en el export local de HuggingFace) y
    'check-readiness' (necesita localizar hfh_publication_report.json ahí);
    el resto de comandos leen todo desde la descarga en vivo de HuggingFace Hub.
    communities por defecto son TODAS las configuradas en settings.toml —
    solo 'prepare'/'pipeline' lo estrechan de verdad a un subconjunto vía
    --communities; el resto de comandos lo llevan solo porque siempre
    renderizan la plantilla entera, no porque lo usen."""
    return zenodo_service.build_zenodo_template_context(
        hfh_output_dir=hfh_output_dir,
        zenodo_output_dir=zenodo_output_dir,
        repo_id=repo_id,
        dataset_name=HF_DEFAULTS.dataset_name,
        description=HF_DEFAULTS.description,
        license_id=HF_DEFAULTS.license_id,
        author_given_names=HF_DEFAULTS.author_given_names,
        author_family_names=HF_DEFAULTS.author_family_names,
        author_affiliation=HF_DEFAULTS.author_affiliation,
        environment=environment or ZENODO_DEFAULTS.environment,
        communities=communities if communities is not None else _parse_communities(ZENODO_DEFAULTS.communities),
    )


@app.command("prepare")
def prepare(
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_ZENODO_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de Zenodo: aquí se descarga HuggingFace Hub en tiempo real y se guarda todo lo que se sube a Zenodo.",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub al que enlazar (el mismo --repo-id de 'huggingface prepare').",
    ),
    environment: ZenodoEnvironment = typer.Option(
        ZENODO_DEFAULTS.environment, "--environment", help=_zenodo_help("environment"),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Valida todo pero no crea ni modifica ningún depósito de Zenodo.",
    ),
    sync_existing_draft: bool = typer.Option(
        ZENODO_DEFAULTS.sync_existing_draft, "--sync-existing-draft", help=_zenodo_help("sync_existing_draft"),
    ),
    verify_data: bool = typer.Option(
        False, "--verify-data/--no-verify-data",
        help=(
            "Descarga también los shards .tar de HuggingFace Hub y verifica sus hashes "
            "internos (más lento, más ancho de banda y disco — se queda igualmente en "
            "local, nunca se sube a Zenodo). Por defecto (--no-verify-data) solo se "
            "descargan y verifican los ficheros pequeños de evidencia."
        ),
    ),
    communities: Optional[str] = typer.Option(
        None, "--communities",
        help=(
            "Subconjunto, separado por comas, de las comunidades configuradas en "
            "settings.toml (ZENODO.communities) al que enviar el depósito en esta "
            "ejecución. Por defecto, todas las configuradas."
        ),
    ),
) -> None:
    """Prepara EN LOCAL un registro Zenodo enlazado: no sube nada a Zenodo.

    Descarga en vivo el repo de HuggingFace Hub (README, LICENSE,
    CITATION.cff, manifests, checksums — no hace falta apuntar a ningún
    export local previo, solo --repo-id), crea (o relee, con
    --sync-existing-draft) el depósito de Zenodo solo para reservar/leer el
    DOI, y copia esos ficheros a --output-dir con el DOI ya inyectado en la
    copia de CITATION.cff. El siguiente paso es 'donadataset publish zenodo
    upload', que sube tal cual lo que queda aquí.

    Los shards pesados se quedan en HuggingFace Hub — Zenodo solo aloja
    metadata, manifests, checksums y reports de verificación. Por eso, por
    defecto, ni siquiera se descargan para verificarlos (usa --verify-data
    si quieres esa comprobación extra de todas formas).
    """
    zenodo_service.setup_logging()
    template_context = _build_template_context(
        output_dir, repo_id, environment=environment.value, communities=_resolve_communities(communities),
    )
    try:
        zenodo_service.run_zenodo_prepare(
            DEFAULT_TEMPLATE_FILE, dry_run=dry_run, template_context=template_context,
            verify_data=verify_data, sync_existing_draft=sync_existing_draft,
        )
    except Exception as exc:
        logging.error("Zenodo prepare failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("upload")
def upload(
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_ZENODO_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de Zenodo (el mismo usado en 'prepare').",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub enlazado (el mismo --repo-id de 'huggingface prepare').",
    ),
    environment: ZenodoEnvironment = typer.Option(
        ZENODO_DEFAULTS.environment, "--environment", help=_zenodo_help("environment"),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Muestra qué ficheros se subirían, sin contactar con Zenodo.",
    ),
) -> None:
    """Sube a Zenodo exactamente lo que 'prepare' ya dejó preparado en local.

    No descarga ni modifica nada de HuggingFace Hub — lee --output-dir tal
    cual (DOI ya inyectado en CITATION.cff por 'prepare') y lo sube al
    depósito. Publica automáticamente si zenodo.publish está activo en la
    plantilla (por defecto no). El siguiente paso recomendado es
    'donadataset publish zenodo sync-doi' para reflejar el DOI en HuggingFace Hub.
    """
    zenodo_service.setup_logging()
    template_context = _build_template_context(output_dir, repo_id, environment=environment.value)
    try:
        zenodo_service.run_zenodo_upload(
            DEFAULT_TEMPLATE_FILE, dry_run=dry_run, template_context=template_context,
        )
    except Exception as exc:
        logging.error("Zenodo upload failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("sync-doi")
def sync_doi(
    hfh_output_dir: Optional[str] = typer.Option(
        str(DEFAULT_HFH_OUTPUT_DIR), "--hfh-output-dir",
        help="Directorio del export de HuggingFace ya existente (el mismo --output-dir de 'huggingface prepare').",
    ),
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_ZENODO_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de Zenodo (el mismo usado en 'prepare').",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub enlazado (el mismo --repo-id de 'huggingface prepare').",
    ),
    environment: ZenodoEnvironment = typer.Option(
        ZENODO_DEFAULTS.environment, "--environment", help=_zenodo_help("environment"),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Muestra qué haría, sin modificar ni subir nada.",
    ),
    upload: bool = typer.Option(
        True, "--upload/--no-upload",
        help="Tras copiar el DOI a local, subirlo automáticamente a HuggingFace Hub (solo CITATION.cff, README.md y el fichero de checksums, no el resto del export).",
    ),
) -> None:
    """Refleja en HuggingFace Hub el DOI que 'prepare' ya reservó en Zenodo.

    Copia la copia local de CITATION.cff (ya con el DOI, preparada por
    'zenodo prepare') sobre el export de HuggingFace Hub, actualiza la
    sección "## Zenodo DOI" de su README.md, y regenera su
    checksums-sha256.txt. Por defecto también vuelve a subir esos TRES
    ficheros a HuggingFace Hub automáticamente (--no-upload para saltarte
    esa parte) — nunca el resto del export, así los shards .tar ya
    publicados no se vuelven a tocar.
    """
    zenodo_service.setup_logging()
    template_context = _build_template_context(
        output_dir, repo_id, hfh_output_dir=hfh_output_dir, environment=environment.value,
    )
    try:
        zenodo_service.run_zenodo_sync_doi(
            DEFAULT_TEMPLATE_FILE, dry_run=dry_run, template_context=template_context,
        )
    except Exception as exc:
        logging.error("Zenodo DOI sync failed: %s", exc)
        raise typer.Exit(1) from exc

    if dry_run or not upload:
        return

    hf_config_path = _hf_resolved_config_path(hfh_output_dir)
    _hf_warn_if_token_missing(hf_config_path, required_permission="write")
    checksums_filename = zenodo_service.get_checksums_filename(
        zenodo_service.load_config_source(DEFAULT_TEMPLATE_FILE, **template_context)
    )
    try:
        hf_service.run_upload(
            hf_config_path, dry_run=False, allow_patterns=["CITATION.cff", "README.md", checksums_filename],
        )
    except Exception as exc:
        logging.error("Re-upload to HuggingFace Hub failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("download")
def download(
    deploy_dir: Path = typer.Option(..., "--deploy-dir", help="Directorio donde desplegar el dataset en formato YOLO."),
    zenodo_record: Optional[str] = typer.Option(
        None, "--zenodo-record", help="Record id o URL de Zenodo (ej. 21136807 o https://zenodo.org/records/21136807).",
    ),
    doi: Optional[str] = typer.Option(
        None, "--doi", help="DOI o URL de DOI de Zenodo (ej. 10.5281/zenodo.21136807).",
    ),
    download_dir: Optional[Path] = typer.Option(
        None, "--download-dir", help="Directorio de descarga. Por defecto, uno temporal (se borra al terminar salvo --keep-download).",
    ),
    keep_download: bool = typer.Option(
        False, "--keep-download", help="Conservar el directorio de descarga tras desplegar.",
    ),
    clean_deploy_dir: bool = typer.Option(
        False, "--clean-deploy-dir", help="Borrar el directorio de despliegue antes de extraer.",
    ),
    verify_checksums: bool = typer.Option(
        True, "--verify-checksums/--no-verify-checksums", help="Verificar checksums-sha256.txt tras la descarga.",
    ),
    copy_metadata: bool = typer.Option(
        True, "--copy-metadata/--no-copy-metadata", help="Copiar README/LICENSE/manifests... al desplegar.",
    ),
    allow_linked_hfh: bool = typer.Option(
        True, "--linked-hfh/--no-linked-hfh",
        help="Permitir descargar los shards pesados desde el repo de HuggingFace Hub enlazado en Zenodo.",
    ),
    hf_token_env: Optional[str] = typer.Option(
        None, "--hf-token-env", help="Variable de entorno con el token de HuggingFace Hub (solo si el repo enlazado es privado).",
    ),
) -> None:
    """Descarga un registro Zenodo (completo, o enlazado a HuggingFace Hub) y lo despliega en formato YOLO."""
    zenodo_service.setup_logging()
    try:
        zenodo_service.run_download_and_deploy(
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
    except Exception as exc:
        logging.error("Zenodo download/deploy failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("check-readiness")
def check_readiness(
    hfh_output_dir: Optional[str] = typer.Option(
        str(DEFAULT_HFH_OUTPUT_DIR), "--hfh-output-dir",
        help="Directorio del export de HuggingFace ya existente (el mismo --output-dir de 'huggingface prepare'). Usado para localizar hfh_publication_report.json.",
    ),
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_ZENODO_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de Zenodo (el mismo usado en 'prepare').",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub enlazado (el mismo --repo-id de 'huggingface prepare').",
    ),
    environment: ZenodoEnvironment = typer.Option(
        ZENODO_DEFAULTS.environment, "--environment", help=_zenodo_help("environment"),
    ),
) -> None:
    """Chequeo final de solo lectura antes de publicar el draft de Zenodo a mano.

    Comprueba que HFH es público de verdad (URL sin token + API), que el
    draft de Zenodo tiene metadata completa y todos los ficheros de evidencia
    subidos, y que los reports locales previos dicen 'passed' (incluyendo
    hfh_publication_report.json, que escribe 'huggingface release' dentro de
    --hfh-output-dir). No publica nada ni cambia nada — solo te da luz verde
    (o no) para publicar tú mismo desde la web de Zenodo.
    """
    zenodo_service.setup_logging()
    template_context = _build_template_context(
        output_dir, repo_id, hfh_output_dir=hfh_output_dir, environment=environment.value,
    )
    try:
        zenodo_service.run_check_release_readiness(DEFAULT_TEMPLATE_FILE, template_context=template_context)
    except Exception as exc:
        logging.error("Public release readiness check failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("release")
def release(
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_ZENODO_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de Zenodo (el mismo usado en 'prepare').",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub enlazado (el mismo --repo-id de 'huggingface prepare').",
    ),
    environment: ZenodoEnvironment = typer.Option(
        ZENODO_DEFAULTS.environment, "--environment", help=_zenodo_help("environment"),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Valida y consulta el estado, pero no publica nada.",
    ),
    skip_readiness_check: bool = typer.Option(
        False, "--skip-readiness-check",
        help="Publica sin exigir que 'check-readiness' haya pasado. Úsalo con cuidado.",
    ),
    no_config_update: bool = typer.Option(
        False, "--no-config-update",
        help="No actualizar el YAML local con el DOI/estado tras publicar.",
    ),
) -> None:
    """Publica de forma DEFINITIVA el draft de Zenodo — acción irreversible.

    Exige que 'zenodo check-readiness' haya pasado (salvo
    --skip-readiness-check). Si el depósito ya estaba publicado, no hace nada
    y solo informa. Tras publicar, verifica que la URL del record y del DOI
    responden de verdad, y actualiza tu YAML local con el DOI definitivo.
    """
    zenodo_service.setup_logging()
    template_context = _build_template_context(output_dir, repo_id, environment=environment.value)
    try:
        zenodo_service.run_release(
            DEFAULT_TEMPLATE_FILE,
            dry_run=dry_run,
            skip_readiness_check=skip_readiness_check,
            no_config_update=no_config_update,
            template_context=template_context,
        )
    except Exception as exc:
        logging.error("Zenodo publication failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("pipeline")
def pipeline(
    hfh_output_dir: Optional[str] = typer.Option(
        str(DEFAULT_HFH_OUTPUT_DIR), "--hfh-output-dir",
        help="Directorio del export de HuggingFace ya existente (el mismo --output-dir de 'huggingface prepare').",
    ),
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_ZENODO_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de Zenodo: aquí se descarga HuggingFace Hub en tiempo real y se guarda todo lo que se sube a Zenodo.",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub al que enlazar (el mismo --repo-id de 'huggingface prepare').",
    ),
    environment: ZenodoEnvironment = typer.Option(
        ZENODO_DEFAULTS.environment, "--environment", help=_zenodo_help("environment"),
    ),
    sync_existing_draft: bool = typer.Option(
        ZENODO_DEFAULTS.sync_existing_draft, "--sync-existing-draft", help=_zenodo_help("sync_existing_draft"),
    ),
    skip_readiness_check: bool = typer.Option(
        False, "--skip-readiness-check",
        help="Publica sin exigir que 'check-readiness' haya pasado. Úsalo con cuidado.",
    ),
    no_config_update: bool = typer.Option(
        False, "--no-config-update",
        help="No actualizar el YAML local con el DOI/estado tras publicar.",
    ),
    verify_data: bool = typer.Option(
        False, "--verify-data/--no-verify-data",
        help=(
            "Descarga también los shards .tar de HuggingFace Hub y verifica sus hashes "
            "internos en el paso 'prepare' (más lento, más ancho de banda y disco). Por "
            "defecto (--no-verify-data) solo se descargan y verifican los ficheros "
            "pequeños de evidencia."
        ),
    ),
    communities: Optional[str] = typer.Option(
        None, "--communities",
        help=(
            "Subconjunto, separado por comas, de las comunidades configuradas en "
            "settings.toml (ZENODO.communities) al que enviar el depósito en el paso "
            "'prepare'. Por defecto, todas las configuradas."
        ),
    ),
) -> None:
    """Ejecuta de un tirón: prepare -> upload -> sync-doi -> check-readiness -> release.

    'sync-doi' vuelve a subir a HuggingFace Hub automáticamente (solo
    CITATION.cff, README.md y el fichero de checksums), así que no hace
    falta ningún paso manual entre medias.
    """
    console.print("[bold cyan]── Paso 1/5: prepare ──[/bold cyan]")
    prepare_context = _build_template_context(
        output_dir, repo_id, environment=environment.value, communities=_resolve_communities(communities),
    )
    try:
        zenodo_service.run_zenodo_prepare(
            DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=prepare_context,
            verify_data=verify_data, sync_existing_draft=sync_existing_draft,
        )
    except Exception as exc:
        logging.error("Zenodo prepare failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 2/5: upload ──[/bold cyan]")
    try:
        zenodo_service.run_zenodo_upload(
            DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=prepare_context,
        )
    except Exception as exc:
        logging.error("Zenodo upload failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 3/5: sync-doi ──[/bold cyan]")
    sync_doi_context = _build_template_context(
        output_dir, repo_id, hfh_output_dir=hfh_output_dir, environment=environment.value,
    )
    try:
        zenodo_service.run_zenodo_sync_doi(
            DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=sync_doi_context,
        )
        hf_config_path = _hf_resolved_config_path(hfh_output_dir)
        checksums_filename = zenodo_service.get_checksums_filename(
            zenodo_service.load_config_source(DEFAULT_TEMPLATE_FILE, **sync_doi_context)
        )
        hf_service.run_upload(
            hf_config_path, dry_run=False, allow_patterns=["CITATION.cff", "README.md", checksums_filename],
        )
    except Exception as exc:
        logging.error("Zenodo DOI sync failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 4/5: check-readiness ──[/bold cyan]")
    readiness_context = _build_template_context(
        output_dir, repo_id, hfh_output_dir=hfh_output_dir, environment=environment.value,
    )
    try:
        zenodo_service.run_check_release_readiness(DEFAULT_TEMPLATE_FILE, template_context=readiness_context)
    except Exception as exc:
        logging.error("Public release readiness check failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 5/5: release ──[/bold cyan]")
    release_context = _build_template_context(output_dir, repo_id, environment=environment.value)
    try:
        zenodo_service.run_release(
            DEFAULT_TEMPLATE_FILE,
            dry_run=False,
            skip_readiness_check=skip_readiness_check,
            no_config_update=no_config_update,
            template_context=release_context,
        )
    except Exception as exc:
        logging.error("Zenodo publication failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold green]✔  Pipeline completado.[/bold green]")


# ── wizard ────────────────────────────────────────────────────────────────

def _run_wizard_step(label: str, action: Callable[[], Any], *, allow_skip: bool = False) -> Any:
    """Runs `action`, retrying/skipping/aborting interactively on failure —
    unlike 'pipeline', which just crashes on the first exception. Returns
    whatever `action` returns (None if the step was skipped). Mirrors
    donadataset.commands.huggingface._run_wizard_step."""
    while True:
        console.print(f"\n[bold cyan]── {label} ──[/bold cyan]")
        try:
            return action()
        except Exception as exc:
            logging.error("%s falló: %s", label, exc)
            options = "[r]eintentar" + (" / [s]altar" if allow_skip else "") + " / [a]bortar"
            choice = typer.prompt(f"¿Qué quieres hacer? {options}", default="r").strip().lower()
            if choice.startswith("r"):
                continue
            if allow_skip and choice.startswith("s"):
                console.print("[yellow]Paso saltado — puedes completarlo más tarde a mano.[/yellow]")
                return None
            console.print("[red]Abortado.[/red]")
            raise typer.Exit(1) from exc


@app.command("wizard")
def wizard(
    hfh_output_dir: Optional[str] = typer.Option(
        str(DEFAULT_HFH_OUTPUT_DIR), "--hfh-output-dir",
        help="Directorio del export de HuggingFace ya existente (el mismo --output-dir de 'huggingface prepare').",
    ),
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_ZENODO_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de Zenodo: aquí se descarga HuggingFace Hub en tiempo real y se guarda todo lo que se sube a Zenodo.",
    ),
) -> None:
    """Asistente interactivo: te guía paso a paso por todo el proceso de publicación en Zenodo.

    A diferencia de 'pipeline' (que ejecuta todo de un tirón y se detiene en
    seco si algo falla), el wizard explica cada fase antes de ejecutarla, te
    pregunta el repo_id si todavía no lo has configurado, detecta si ya
    existe un depósito vinculado (para ofrecerte sincronizarlo en vez de
    crear uno nuevo), hace la re-subida a HuggingFace Hub por ti
    automáticamente al reflejar el DOI, te pide confirmación antes de
    publicar de forma DEFINITIVA e IRREVERSIBLE, y si un paso falla te deja
    reintentarlo o abortar en vez de terminar en seco. Los valores de
    identidad del dataset (nombre, licencia, autor...) salen de 'donadataset
    publish huggingface config' — el wizard no te los vuelve a preguntar.
    """
    console.print("[bold]Asistente de publicación en Zenodo[/bold]")
    console.print(
        "Fases: 1) preparar en local (descargar HFH, reservar el DOI en Zenodo, "
        "inyectarlo en la copia local)  2) subir lo preparado a Zenodo  3) reflejar el "
        "DOI en HuggingFace Hub (con re-subida automática)  4) comprobar que todo está "
        "listo  5) publicar de forma definitiva (irreversible).\n"
    )

    repo_id = _wizard_resolve_repo_id()
    environment = ZENODO_DEFAULTS.environment or "sandbox"
    console.print(
        f"Entorno: [bold]{environment}[/bold] "
        "(cambia con 'donadataset publish zenodo config set environment')."
    )
    if environment == "production":
        if not typer.confirm(
            "Estás en 'production' — esto reserva y puede publicar un DOI real de Zenodo "
            "(no de sandbox). ¿Continuar?",
            default=False,
        ):
            console.print(
                "[yellow]Abortado.[/yellow] Cambia a sandbox con "
                "[bold]donadataset publish zenodo config set environment=sandbox[/bold] "
                "si querías probar antes."
            )
            raise typer.Exit(0)

    prepare_context = _build_template_context(output_dir, repo_id, environment=environment)
    config = zenodo_service.load_config_source(DEFAULT_TEMPLATE_FILE, **prepare_context)

    token_env_var = zenodo_service.get_zenodo_token_env_var(config)
    if not (os.environ.get(token_env_var) or settings.ZENODO.token):
        console.print(f"[red]✘  No hay ningún token de Zenodo configurado ({token_env_var}).[/red]")
        console.print(
            "   Consigue uno en [bold]https://"
            f"{'sandbox.' if environment == 'sandbox' else ''}zenodo.org/account/settings/applications/tokens/new/"
            "[/bold] y expórtalo:"
        )
        console.print(f"   [bold]export {token_env_var}='...'[/bold]")
        console.print(
            "   o guárdalo de forma permanente: "
            "[bold]donadataset publish zenodo config set token[/bold]"
        )
        raise typer.Exit(1)

    # ── Fase 1/5: prepare ──────────────────────────────────────────────────
    linked_record_path = zenodo_service.get_linked_record_path(config)
    sync_existing_draft = False
    if linked_record_path.is_file():
        existing_record = zenodo_service.read_json(linked_record_path)
        console.print(
            f"\nYa existe un depósito Zenodo vinculado en [bold]{linked_record_path}[/bold] "
            f"(deposition_id={existing_record.get('deposition_id')}, "
            f"DOI reservado={existing_record.get('reserved_doi') or 'aún no'})."
        )
        sync_existing_draft = typer.confirm(
            "¿Sincronizarlo (en vez de crear un depósito nuevo)?", default=True,
        )

    def _do_prepare() -> None:
        zenodo_service.run_zenodo_prepare(
            DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=prepare_context,
            sync_existing_draft=sync_existing_draft,
        )

    _run_wizard_step("Fase 1/5: prepare", _do_prepare)

    linked_record = zenodo_service.read_json(linked_record_path)
    reserved_doi = linked_record.get("reserved_doi")
    if reserved_doi:
        console.print(f"[green]✔  DOI reservado: {reserved_doi}[/green]")
    else:
        console.print("[yellow]⚠  Todavía no se ha reservado un DOI — revisa el depósito en Zenodo.[/yellow]")

    # ── Fase 2/5: upload (subir a Zenodo lo ya preparado) ───────────────────
    _run_wizard_step(
        "Fase 2/5: upload",
        lambda: zenodo_service.run_zenodo_upload(
            DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=prepare_context,
        ),
    )

    # ── Fase 3/5: sync-doi (refleja el DOI en HuggingFace Hub) ──────────────
    sync_doi_context = _build_template_context(
        output_dir, repo_id, hfh_output_dir=hfh_output_dir, environment=environment,
    )
    hf_config_path = _hf_resolved_config_path(hfh_output_dir)
    console.print(
        f"\nSe va a copiar el CITATION.cff con el DOI y actualizar la sección de "
        f"Zenodo en README.md sobre la copia local de HuggingFace Hub en "
        f"[bold]{hfh_output_dir}[/bold], recalcular checksums, y volver a subir "
        f"automáticamente esos ficheros (nada más) al repo público de "
        f"[bold]{repo_id}[/bold]."
    )
    _hf_warn_if_token_missing(hf_config_path, required_permission="write")

    def _do_sync_doi() -> None:
        zenodo_service.run_zenodo_sync_doi(
            DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=sync_doi_context,
        )
        checksums_filename = zenodo_service.get_checksums_filename(
            zenodo_service.load_config_source(DEFAULT_TEMPLATE_FILE, **sync_doi_context)
        )
        hf_service.run_upload(
            hf_config_path, dry_run=False, allow_patterns=["CITATION.cff", "README.md", checksums_filename],
        )

    _run_wizard_step("Fase 3/5: sync-doi", _do_sync_doi)

    # ── Fase 4/5: check-readiness (solo lectura) ────────────────────────────
    readiness_context = _build_template_context(
        output_dir, repo_id, hfh_output_dir=hfh_output_dir, environment=environment,
    )
    _run_wizard_step(
        "Fase 4/5: check-readiness",
        lambda: zenodo_service.run_check_release_readiness(DEFAULT_TEMPLATE_FILE, template_context=readiness_context),
    )

    # ── Fase 5/5: release (irreversible) ─────────────────────────────────────
    console.print(
        "\nEl siguiente paso publica el depósito de Zenodo de forma "
        "[bold red]DEFINITIVA e IRREVERSIBLE[/bold red] — una vez publicado no se puede "
        "despublicar ni editar sus ficheros."
    )
    if not typer.confirm("¿Continuar y publicar ahora?", default=False):
        console.print(
            "\n[yellow]De acuerdo, lo dejo como draft.[/yellow] Cuando quieras publicarlo, "
            "vuelve a ejecutar el wizard o usa 'donadataset publish zenodo release'."
        )
        raise typer.Exit(0)

    release_context = _build_template_context(output_dir, repo_id, environment=environment)
    _run_wizard_step(
        "Fase 5/5: release",
        lambda: zenodo_service.run_release(
            DEFAULT_TEMPLATE_FILE, dry_run=False, skip_readiness_check=False,
            no_config_update=False, template_context=release_context,
        ),
    )

    publication_report = zenodo_service.read_json(zenodo_service.get_publication_report_path(config))
    record_url = publication_report.get("record_url")
    console.print(
        "\n[bold green]✔  Publicación en Zenodo completada"
        + (f": {record_url}" if record_url else "")
        + "[/bold green]"
    )
