"""Comandos CLI para publicar el dataset en B2SHARE (EUDAT).

Gestiona únicamente los parámetros de entrada; toda la lógica de
preparación/publicación vive en donadataset.services.b2share.
"""
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from donadataset.commands import b2share_config_commands
from donadataset.config import REPO_ROOT, B2ShareSettings, get_app_documents_dir, settings
from donadataset.services import b2share as b2share_service

console = Console()
app     = typer.Typer(help="Publica el dataset en B2SHARE (EUDAT).")
app.add_typer(b2share_config_commands.app, name="config")


class B2ShareEnvironment(str, Enum):
    sandbox    = "sandbox"
    production = "production"


# Única fuente de configuración para prepare/check-readiness/release/sync-pid
# — no hay --config, ni YAML personal alternativo. Siempre se rellena esta
# plantilla con los flags de abajo (cuyo valor por defecto sale de
# settings.toml B2SHARE.*/HUGGINGFACE.*).
DEFAULT_TEMPLATE_FILE = REPO_ROOT / "templates" / "B2SHARE.yaml.j2"

HF_DEFAULTS = settings.HUGGINGFACE
B2SHARE_DEFAULTS = settings.B2SHARE
DEFAULT_HFH_OUTPUT_DIR = get_app_documents_dir() / "HFH"

# Directorio PROPIO de B2SHARE (SALIDA): mismo patrón que Zenodo — hermano
# del de HuggingFace, con nombre propio.
DEFAULT_B2SHARE_OUTPUT_DIR = get_app_documents_dir() / f"B2SHARE_{HF_DEFAULTS.dataset_slug}"


def _b2share_help(field: str) -> str:
    """Reutiliza la description de B2ShareSettings como help= del flag
    equivalente — un único sitio para mantener el texto (config.py)."""
    return B2ShareSettings.model_fields[field].description


def _build_template_context(
    b2share_output_dir: Optional[str], repo_id: Optional[str], community_id: Optional[str],
    hfh_output_dir: Optional[str] = None, environment: Optional[str] = None,
) -> dict:
    """--repo-id/--output-dir/--community-id son los datos que hay que
    decidir por invocación; el resto de la identidad del dataset (nombre,
    descripción, licencia, autor) se reutiliza de settings.toml
    (HUGGINGFACE.*) — las mismas que ya usa 'huggingface prepare'.
    hfh_output_dir solo lo pasa 'sync-pid' (necesita escribir el PID/DOI en
    el export local); el resto de comandos leen todo desde la descarga en
    vivo de HuggingFace Hub."""
    return b2share_service.build_b2share_template_context(
        hfh_output_dir=hfh_output_dir,
        b2share_output_dir=b2share_output_dir,
        repo_id=repo_id,
        community_id=community_id or B2SHARE_DEFAULTS.community_id,
        dataset_name=HF_DEFAULTS.dataset_name,
        description=HF_DEFAULTS.description,
        license_id=HF_DEFAULTS.license_id,
        author_given_names=HF_DEFAULTS.author_given_names,
        author_family_names=HF_DEFAULTS.author_family_names,
        author_affiliation=HF_DEFAULTS.author_affiliation,
        environment=environment or B2SHARE_DEFAULTS.environment,
    )


@app.command("prepare")
def prepare(
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_B2SHARE_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de B2SHARE: aquí se descarga HuggingFace Hub en tiempo real y se guarda todo lo que se sube a B2SHARE.",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub al que enlazar (el mismo --repo-id de 'huggingface prepare').",
    ),
    community_id: Optional[str] = typer.Option(
        B2SHARE_DEFAULTS.community_id, "--community-id", help=_b2share_help("community_id"),
    ),
    environment: B2ShareEnvironment = typer.Option(
        B2SHARE_DEFAULTS.environment, "--environment", help=_b2share_help("environment"),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Valida todo pero no crea ni modifica ningún draft de B2SHARE.",
    ),
    sync_existing_draft: bool = typer.Option(
        B2SHARE_DEFAULTS.sync_existing_draft, "--sync-existing-draft", help=_b2share_help("sync_existing_draft"),
    ),
) -> None:
    """Crea (o sincroniza) un registro B2SHARE enlazado y sube los ficheros de evidencia.

    Todo lo que se sube (README, LICENSE, CITATION.cff, manifests, checksums)
    se obtiene de una descarga en vivo del repo de HuggingFace Hub — no hace
    falta apuntar a ningún export local previo, solo --repo-id.

    A diferencia de Zenodo, B2SHARE no reserva un PID/DOI en este paso —
    solo al publicar con 'b2share release'.
    """
    b2share_service.setup_logging()
    template_context = _build_template_context(output_dir, repo_id, community_id, environment=environment.value)
    try:
        if sync_existing_draft:
            b2share_service.run_b2share_existing_draft_sync(
                DEFAULT_TEMPLATE_FILE, dry_run=dry_run, template_context=template_context,
            )
        else:
            b2share_service.run_b2share_linked_dataset_creation(
                DEFAULT_TEMPLATE_FILE, dry_run=dry_run, template_context=template_context,
            )
    except Exception as exc:
        logging.error("B2SHARE linked dataset creation failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("check-readiness")
def check_readiness(
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_B2SHARE_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de B2SHARE (el mismo usado en 'prepare').",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub enlazado (el mismo --repo-id de 'huggingface prepare').",
    ),
    community_id: Optional[str] = typer.Option(
        B2SHARE_DEFAULTS.community_id, "--community-id", help=_b2share_help("community_id"),
    ),
    environment: B2ShareEnvironment = typer.Option(
        B2SHARE_DEFAULTS.environment, "--environment", help=_b2share_help("environment"),
    ),
) -> None:
    """Chequeo final de solo lectura antes de publicar el draft de B2SHARE a mano.

    Comprueba que HFH es público de verdad (URL sin token + API) y que los
    reports locales previos dicen 'passed'. No publica nada ni cambia nada —
    solo te da luz verde (o no) para publicar con 'b2share release'.
    """
    b2share_service.setup_logging()
    template_context = _build_template_context(output_dir, repo_id, community_id, environment=environment.value)
    try:
        b2share_service.run_check_release_readiness(DEFAULT_TEMPLATE_FILE, template_context=template_context)
    except Exception as exc:
        logging.error("Public release readiness check failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("release")
def release(
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_B2SHARE_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de B2SHARE (el mismo usado en 'prepare').",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub enlazado (el mismo --repo-id de 'huggingface prepare').",
    ),
    community_id: Optional[str] = typer.Option(
        B2SHARE_DEFAULTS.community_id, "--community-id", help=_b2share_help("community_id"),
    ),
    environment: B2ShareEnvironment = typer.Option(
        B2SHARE_DEFAULTS.environment, "--environment", help=_b2share_help("environment"),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Valida y consulta el estado, pero no publica nada.",
    ),
    skip_readiness_check: bool = typer.Option(
        False, "--skip-readiness-check",
        help="Publica sin exigir que 'check-readiness' haya pasado. Úsalo con cuidado.",
    ),
) -> None:
    """Publica el draft de B2SHARE — puede quedar pendiente de aprobación por tu comunidad EUDAT.

    Exige que 'b2share check-readiness' haya pasado (salvo
    --skip-readiness-check). A diferencia de Zenodo, publicar no garantiza un
    PID/DOI inmediato: según cómo esté configurada tu comunidad, puede
    requerir aprobación de un moderador antes de asignarlo — usa 'b2share
    sync-pid' más tarde para comprobarlo y reflejarlo en tu metadata local.
    """
    b2share_service.setup_logging()
    template_context = _build_template_context(output_dir, repo_id, community_id, environment=environment.value)
    try:
        b2share_service.run_release(
            DEFAULT_TEMPLATE_FILE,
            dry_run=dry_run,
            skip_readiness_check=skip_readiness_check,
            template_context=template_context,
        )
    except Exception as exc:
        logging.error("B2SHARE publication failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("sync-pid")
def sync_pid(
    hfh_output_dir: Optional[str] = typer.Option(
        str(DEFAULT_HFH_OUTPUT_DIR), "--hfh-output-dir",
        help="Directorio del export de HuggingFace ya existente (el mismo --output-dir de 'huggingface prepare').",
    ),
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_B2SHARE_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de B2SHARE (el mismo usado en 'prepare').",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub enlazado (el mismo --repo-id de 'huggingface prepare').",
    ),
    community_id: Optional[str] = typer.Option(
        B2SHARE_DEFAULTS.community_id, "--community-id", help=_b2share_help("community_id"),
    ),
    environment: B2ShareEnvironment = typer.Option(
        B2SHARE_DEFAULTS.environment, "--environment", help=_b2share_help("environment"),
    ),
) -> None:
    """Lee el PID/DOI ya asignado por B2SHARE (tras publicar) y lo refleja en CITATION.cff.

    B2SHARE no reserva el PID/DOI hasta que el registro queda realmente
    publicado (posiblemente tras aprobación de un moderador de tu comunidad).
    Este comando NO publica nada; solo detecta el identificador una vez
    existe y actualiza tu CITATION.cff local y checksums-sha256.txt. El
    siguiente paso recomendado es volver a subir con 'huggingface upload'.
    """
    b2share_service.setup_logging()
    template_context = _build_template_context(
        output_dir, repo_id, community_id, hfh_output_dir=hfh_output_dir, environment=environment.value,
    )
    try:
        b2share_service.run_sync_b2share_pid(DEFAULT_TEMPLATE_FILE, template_context=template_context)
    except Exception as exc:
        logging.error("B2SHARE PID/DOI sync failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("pipeline")
def pipeline(
    hfh_output_dir: Optional[str] = typer.Option(
        str(DEFAULT_HFH_OUTPUT_DIR), "--hfh-output-dir",
        help="Directorio del export de HuggingFace ya existente (el mismo --output-dir de 'huggingface prepare').",
    ),
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_B2SHARE_OUTPUT_DIR), "--output-dir",
        help="Directorio propio de B2SHARE: aquí se descarga HuggingFace Hub en tiempo real y se guarda todo lo que se sube a B2SHARE.",
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help="Repo de HuggingFace Hub al que enlazar (el mismo --repo-id de 'huggingface prepare').",
    ),
    community_id: Optional[str] = typer.Option(
        B2SHARE_DEFAULTS.community_id, "--community-id", help=_b2share_help("community_id"),
    ),
    environment: B2ShareEnvironment = typer.Option(
        B2SHARE_DEFAULTS.environment, "--environment", help=_b2share_help("environment"),
    ),
    sync_existing_draft: bool = typer.Option(
        B2SHARE_DEFAULTS.sync_existing_draft, "--sync-existing-draft", help=_b2share_help("sync_existing_draft"),
    ),
    skip_readiness_check: bool = typer.Option(
        False, "--skip-readiness-check",
        help="Publica sin exigir que 'check-readiness' haya pasado. Úsalo con cuidado.",
    ),
) -> None:
    """Ejecuta de un tirón: prepare -> check-readiness -> release -> sync-pid.

    A diferencia del pipeline de Zenodo, aquí no hay pausa manual entre
    pasos: el PID/DOI de B2SHARE no se reserva hasta publicar, así que
    'sync-pid' simplemente detecta si ya está disponible al final (si tu
    comunidad exige aprobación de un moderador, puede que todavía no lo
    esté — vuelve a ejecutar 'donadataset publish b2share sync-pid' más
    tarde para comprobarlo).
    """
    console.print("[bold cyan]── Paso 1/3: prepare ──[/bold cyan]")
    prepare_context = _build_template_context(output_dir, repo_id, community_id, environment=environment.value)
    try:
        if sync_existing_draft:
            b2share_service.run_b2share_existing_draft_sync(
                DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=prepare_context,
            )
        else:
            b2share_service.run_b2share_linked_dataset_creation(
                DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=prepare_context,
            )
    except Exception as exc:
        logging.error("B2SHARE linked dataset creation failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 2/3: check-readiness ──[/bold cyan]")
    readiness_context = _build_template_context(output_dir, repo_id, community_id, environment=environment.value)
    try:
        b2share_service.run_check_release_readiness(DEFAULT_TEMPLATE_FILE, template_context=readiness_context)
    except Exception as exc:
        logging.error("Public release readiness check failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 3/3: release + sync-pid ──[/bold cyan]")
    release_context = _build_template_context(output_dir, repo_id, community_id, environment=environment.value)
    try:
        b2share_service.run_release(
            DEFAULT_TEMPLATE_FILE, dry_run=False, skip_readiness_check=skip_readiness_check,
            template_context=release_context,
        )
    except Exception as exc:
        logging.error("B2SHARE publication failed: %s", exc)
        raise typer.Exit(1) from exc

    sync_context = _build_template_context(
        output_dir, repo_id, community_id, hfh_output_dir=hfh_output_dir, environment=environment.value,
    )
    try:
        b2share_service.run_sync_b2share_pid(DEFAULT_TEMPLATE_FILE, template_context=sync_context)
    except Exception as exc:
        logging.error("B2SHARE PID/DOI sync failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold green]✔  Pipeline completado.[/bold green]")
