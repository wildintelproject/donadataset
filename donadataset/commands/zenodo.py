"""Comandos CLI para publicar el dataset en Zenodo.

Gestiona únicamente los parámetros de entrada; toda la lógica de
preparación/actualización/descarga vive en donadataset.services.zenodo.
"""
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from rich.console import Console

from donadataset.commands import zenodo_config_commands
from donadataset.commands.huggingface import _resolved_config_path as _hf_resolved_config_path
from donadataset.commands.huggingface import _warn_if_token_missing as _hf_warn_if_token_missing
from donadataset.commands.huggingface import _wizard_resolve_repo_id
from donadataset.config import REPO_ROOT, ZenodoSettings, get_app_documents_dir, get_hfh_output_dir, settings
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
# tiempo real y guarda todo lo que sube a Zenodo. Por defecto, hermano del
# de HuggingFace de arriba (misma convención que get_zenodo_output_dir()).
DEFAULT_ZENODO_OUTPUT_DIR = get_app_documents_dir() / f"Zenodo_{HF_DEFAULTS.dataset_slug}"


def _zenodo_help(field: str) -> str:
    """Reutiliza la description de ZenodoSettings como help= del flag
    equivalente — un único sitio para mantener el texto (config.py), en vez
    de duplicarlo aquí a mano."""
    return ZenodoSettings.model_fields[field].description


def _build_template_context(
    zenodo_output_dir: Optional[str], repo_id: Optional[str],
    hfh_output_dir: Optional[str] = None, environment: Optional[str] = None,
) -> dict:
    """--repo-id/--output-dir son los datos que hay que decidir por
    invocación; el resto de la identidad del dataset (nombre, descripción,
    licencia, autor) se reutiliza de settings.toml (HUGGINGFACE.*) — las
    mismas que ya usa 'huggingface prepare'. hfh_output_dir solo lo pasa
    'upload' (necesita escribir el DOI en el pre-upload local); el resto de
    comandos leen todo desde la descarga en vivo de HuggingFace Hub."""
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
) -> None:
    """Crea (o sincroniza) un registro Zenodo enlazado y sube los ficheros de evidencia.

    Todo lo que se sube (README, LICENSE, CITATION.cff, manifests, checksums)
    se obtiene de una descarga en vivo del repo de HuggingFace Hub — no hace
    falta apuntar a ningún export local previo, solo --repo-id.

    Los shards pesados se quedan en HuggingFace Hub — Zenodo solo aloja
    metadata, manifests, checksums y reports de verificación, y reserva un DOI.
    Por eso, por defecto, ni siquiera se descargan para verificarlos (usa
    --verify-data si quieres esa comprobación extra de todas formas).
    """
    zenodo_service.setup_logging()
    template_context = _build_template_context(output_dir, repo_id, environment=environment.value)
    try:
        if sync_existing_draft:
            zenodo_service.run_zenodo_existing_draft_sync(
                DEFAULT_TEMPLATE_FILE, dry_run=dry_run, template_context=template_context,
                verify_data=verify_data,
            )
        else:
            zenodo_service.run_zenodo_linked_dataset_creation(
                DEFAULT_TEMPLATE_FILE, dry_run=dry_run, template_context=template_context,
                verify_data=verify_data,
            )
    except Exception as exc:
        logging.error("Zenodo linked dataset creation failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("upload")
def upload(
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
        False, "--dry-run", help="Muestra qué ficheros se actualizarían, sin modificar nada.",
    ),
    no_backup: bool = typer.Option(
        False, "--no-backup", help="No crear copias .bak antes de modificar la metadata local.",
    ),
) -> None:
    """Inserta el DOI ya reservado por 'prepare' en la metadata local y regenera checksums.

    No sube nada a Zenodo — actualiza HuggingFaceHub.yaml, dataset_info.json,
    CITATION.cff y README.md con la información del DOI, y recalcula
    checksums-sha256.txt porque esos ficheros cambiaron. El siguiente paso
    recomendado es volver a subir con 'huggingface upload'.
    """
    zenodo_service.setup_logging()
    template_context = _build_template_context(
        output_dir, repo_id, hfh_output_dir=hfh_output_dir, environment=environment.value,
    )
    try:
        zenodo_service.run_update_local_metadata_with_doi(
            DEFAULT_TEMPLATE_FILE, dry_run=dry_run, no_backup=no_backup, template_context=template_context,
        )
    except Exception as exc:
        logging.error("Metadata update failed: %s", exc)
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
    subidos, y que los reports locales previos dicen 'passed'. No publica
    nada ni cambia nada — solo te da luz verde (o no) para publicar tú mismo
    desde la web de Zenodo.
    """
    zenodo_service.setup_logging()
    template_context = _build_template_context(output_dir, repo_id, environment=environment.value)
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
    no_backup: bool = typer.Option(
        False, "--no-backup", help="No crear copias .bak antes de modificar la metadata local.",
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
) -> None:
    """Ejecuta de un tirón: prepare -> upload -> check-readiness -> release.

    Antes de check-readiness se detiene: el DOI ya está en tu metadata local
    (paso 'upload'), pero todavía no en el repo público de HuggingFace — hace
    falta volver a subirlo con 'huggingface upload' (o 'huggingface
    pipeline') antes de que 'check-readiness' pueda pasar. Te lo explica y
    espera a que pulses Enter para continuar.
    """
    console.print("[bold cyan]── Paso 1/4: prepare ──[/bold cyan]")
    prepare_context = _build_template_context(output_dir, repo_id, environment=environment.value)
    try:
        if sync_existing_draft:
            zenodo_service.run_zenodo_existing_draft_sync(
                DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=prepare_context,
                verify_data=verify_data,
            )
        else:
            zenodo_service.run_zenodo_linked_dataset_creation(
                DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=prepare_context,
                verify_data=verify_data,
            )
    except Exception as exc:
        logging.error("Zenodo linked dataset creation failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 2/4: upload ──[/bold cyan]")
    upload_context = _build_template_context(
        output_dir, repo_id, hfh_output_dir=hfh_output_dir, environment=environment.value,
    )
    try:
        zenodo_service.run_update_local_metadata_with_doi(
            DEFAULT_TEMPLATE_FILE, dry_run=False, no_backup=no_backup, template_context=upload_context,
        )
    except Exception as exc:
        logging.error("Metadata update failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold yellow]── Antes de continuar ──[/bold yellow]")
    console.print(
        "El DOI ya está en tu metadata local (CITATION.cff, dataset_info.json, README.md), "
        "pero el repo público de HuggingFace todavía no lo refleja:\n\n"
        "  1. Ejecuta [bold]donadataset publish huggingface upload[/bold]\n"
        "     (o [bold]donadataset publish huggingface release[/bold] si aún no lo hiciste público)\n"
    )
    typer.prompt("Cuando lo hayas subido, pulsa Enter para continuar", default="", show_default=False)

    console.print("\n[bold cyan]── Paso 3/4: check-readiness ──[/bold cyan]")
    readiness_context = _build_template_context(output_dir, repo_id, environment=environment.value)
    try:
        zenodo_service.run_check_release_readiness(DEFAULT_TEMPLATE_FILE, template_context=readiness_context)
    except Exception as exc:
        logging.error("Public release readiness check failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 4/4: release ──[/bold cyan]")
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

    A diferencia de 'pipeline' (que ejecuta todo de un tirón, se detiene en
    seco si algo falla, y solo pausa para que reflejes el DOI en HuggingFace
    a mano), el wizard explica cada fase antes de ejecutarla, te pregunta el
    repo_id si todavía no lo has configurado, detecta si ya existe un
    depósito vinculado (para ofrecerte sincronizarlo en vez de crear uno
    nuevo), hace la re-subida a HuggingFace Hub por ti automáticamente, te
    pide confirmación antes de publicar de forma DEFINITIVA e IRREVERSIBLE,
    y si un paso falla te deja reintentarlo o abortar en vez de terminar en
    seco. Los valores de identidad del dataset (nombre, licencia, autor...)
    salen de 'donadataset publish huggingface config' — el wizard no te los
    vuelve a preguntar.
    """
    console.print("[bold]Asistente de publicación en Zenodo[/bold]")
    console.print(
        "Fases: 1) crear/sincronizar el depósito Zenodo enlazado y reservar el DOI  "
        "2) reflejar el DOI en la metadata local  3) volver a subir a HuggingFace Hub "
        "(para publicar el DOI)  4) comprobar que todo está listo  5) publicar de forma "
        "definitiva (irreversible).\n"
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
        if sync_existing_draft:
            zenodo_service.run_zenodo_existing_draft_sync(
                DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=prepare_context,
            )
        else:
            zenodo_service.run_zenodo_linked_dataset_creation(
                DEFAULT_TEMPLATE_FILE, dry_run=False, template_context=prepare_context,
            )

    _run_wizard_step("Fase 1/5: prepare", _do_prepare)

    linked_record = zenodo_service.read_json(linked_record_path)
    reserved_doi = linked_record.get("reserved_doi")
    if reserved_doi:
        console.print(f"[green]✔  DOI reservado: {reserved_doi}[/green]")
    else:
        console.print("[yellow]⚠  Todavía no se ha reservado un DOI — revisa el depósito en Zenodo.[/yellow]")

    # ── Fase 2/5: upload (DOI -> metadata local) ────────────────────────────
    upload_context = _build_template_context(
        output_dir, repo_id, hfh_output_dir=hfh_output_dir, environment=environment,
    )
    console.print(
        f"\nSe va a insertar el DOI en la copia local de la metadata de HuggingFace Hub en "
        f"[bold]{hfh_output_dir}[/bold] (README, CITATION.cff, dataset_info.json) y a "
        "recalcular checksums."
    )
    _run_wizard_step(
        "Fase 2/5: upload (DOI -> metadata local)",
        lambda: zenodo_service.run_update_local_metadata_with_doi(
            DEFAULT_TEMPLATE_FILE, dry_run=False, no_backup=False, template_context=upload_context,
        ),
    )

    # ── Fase 3/5: re-subida a HuggingFace Hub (publica el DOI) ──────────────
    hf_config_path = _hf_resolved_config_path(hfh_output_dir)
    console.print(
        f"\nEl DOI ya está en tu metadata local — falta reflejarlo en el repo público de "
        f"[bold]{repo_id}[/bold] en HuggingFace Hub."
    )
    _hf_warn_if_token_missing(hf_config_path, required_permission="write")
    _run_wizard_step(
        "Fase 3/5: re-subida a HuggingFace Hub",
        lambda: hf_service.run_upload(hf_config_path, dry_run=False),
    )

    # ── Fase 4/5: check-readiness (solo lectura) ────────────────────────────
    readiness_context = _build_template_context(output_dir, repo_id, environment=environment)
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
