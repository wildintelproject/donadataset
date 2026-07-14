"""Comandos CLI para publicar el dataset en HuggingFace Hub.

Gestiona únicamente los parámetros de entrada; toda la lógica de
preparación/subida/descarga vive en donadataset.services.huggingface.
"""
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from dynaconf import loaders
from huggingface_hub import HfApi
from rich.console import Console

from donadataset.commands import hf_config_commands
from donadataset.commands.config_commands import _apply_update
from donadataset.config import DEFAULT_CONFIG_FILE, REPO_ROOT, HuggingFaceSettings, get_app_documents_dir, load_settings, settings
from donadataset.services import huggingface as hf_service
from donadataset.services.common import setup_logging

console = Console()
app     = typer.Typer(help="Publica el dataset en HuggingFace Hub.")
app.add_typer(hf_config_commands.app, name="config")

# Única fuente de configuración para todos los comandos — no hay --config,
# ni YAML personal alternativo. 'prepare' siempre rellena esta plantilla con
# los flags de abajo (cuyo valor por defecto sale de settings.toml
# HUGGINGFACE.*); los otros 4 comandos leen la copia YA resuelta que
# 'prepare' deja dentro de --output-dir (HuggingFaceHub.yaml).
DEFAULT_TEMPLATE_FILE = REPO_ROOT / "templates" / "hfh" / "HuggingFaceHub.yaml.j2"

# Defaults de los flags — igual que 'generate real' con settings.GENERATE.*
# (se fijan al importar el módulo, por eso --help ya muestra el valor real:
# donadataset config set HUGGINGFACE.<campo>=... para cambiarlos).
HF_DEFAULTS = settings.HUGGINGFACE
DEFAULT_OUTPUT_DIR = get_app_documents_dir() / "HFH"


def _git_tag_version() -> Optional[str]:
    """If HEAD is exactly on a git tag (e.g. right after 'git tag v2.0.0'),
    return it with any leading 'v' stripped, so --version defaults to the
    same tag you're about to use for a GitHub release — no need to type the
    version twice. Returns None outside a git repo, without a tag, or if git
    itself isn't installed (mirrors wildintel-trapverify/cli.py's own
    _get_version pattern)."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip().lstrip("v") or None


DEFAULT_VERSION = _git_tag_version()


def _resolved_config_path(output_dir: str) -> Path:
    """La copia ya resuelta que 'prepare' escribió dentro de --output-dir."""
    return Path(output_dir) / hf_service.INTERNAL_CONFIG_FILENAME


def _hf_help(field: str) -> str:
    """Reutiliza la description de HuggingFaceSettings como help= del flag
    equivalente — un único sitio para mantener el texto (config.py), en vez
    de duplicarlo aquí a mano."""
    return HuggingFaceSettings.model_fields[field].description


def _warn_if_token_missing(config_path: Path, required_permission: str = "write") -> None:
    """Aviso claro y temprano si falta el token, antes de que la operación
    real (autenticación/subida/descarga) falle más adentro con un mensaje
    envuelto en 'X failed: ...'. Si el YAML ni siquiera se puede leer, o
    repo_id no es válido, no hace nada aquí — deja que el propio comando
    reporte ese error primero, en el mismo orden que sin este aviso."""
    try:
        config = hf_service.load_yaml(config_path)
        hf_service.get_repo_id(config)
        token_env_var = hf_service.get_token_env_var(config)
    except Exception:
        return

    if os.environ.get(token_env_var) or settings.HUGGINGFACE.token:
        return

    console.print(f"[red]✘  No hay ningún token de HuggingFace Hub configurado.[/red]")
    console.print(
        f"   Consigue uno en [bold]https://huggingface.co/settings/tokens[/bold] "
        f"(con permiso de [bold]{required_permission}[/bold]) y expórtalo:"
    )
    console.print(f"   [bold]export {token_env_var}='hf_xxxxxxxxxxxxxxxxxxxxxxxxx'[/bold]")
    console.print(
        "   o guárdalo de forma permanente: "
        "[bold]donadataset publish huggingface config set token[/bold]"
    )
    raise typer.Exit(1)


@app.command("prepare")
def prepare(
    source_dataset_dir: Optional[str] = typer.Option(
        str(settings.GENERATE.output), "--source-dataset-dir",
        help=(
            "Directorio del dataset YOLO ya limpio del que se genera el export "
            "(debe contener images/<split>/ y labels/<split>/) — normalmente la "
            "salida de 'generate real'. (paths.source_dataset_dir)"
        ),
    ),
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_OUTPUT_DIR), "--output-dir",
        help=(
            "Directorio donde se escribe el export completo (shards .tar, manifests, "
            "checksums, README, LICENSE, CITATION.cff...), listo para 'huggingface upload'. "
            "(paths.output_dir)"
        ),
    ),
    dataset_slug: Optional[str] = typer.Option(
        HF_DEFAULTS.dataset_slug, "--dataset-slug", help=_hf_help("dataset_slug"),
    ),
    dataset_name: Optional[str] = typer.Option(
        HF_DEFAULTS.dataset_name, "--dataset-name", help=_hf_help("dataset_name"),
    ),
    version: Optional[str] = typer.Option(
        DEFAULT_VERSION, "--version",
        help=(
            "Versión del dataset (ej. 1.0.0) — se escribe en dataset_info.json y "
            "CITATION.cff. Si HEAD está justo en un tag de git (ej. tras 'git tag "
            "v2.0.0'), se usa ese tag automáticamente (sin la 'v'); si no, y no se "
            "pasa el flag, queda como REPLACE_WITH_VERSION. (project.version)"
        ),
    ),
    description: Optional[str] = typer.Option(
        HF_DEFAULTS.description, "--description", help=_hf_help("description"),
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help=_hf_help("repo_id"),
    ),
    license_id: Optional[str] = typer.Option(
        HF_DEFAULTS.license_id, "--license-id", help=_hf_help("license_id"),
    ),
    license_name: Optional[str] = typer.Option(
        HF_DEFAULTS.license_name, "--license-name", help=_hf_help("license_name"),
    ),
    license_url: Optional[str] = typer.Option(
        HF_DEFAULTS.license_url, "--license-url", help=_hf_help("license_url"),
    ),
    author_given_names: Optional[str] = typer.Option(
        HF_DEFAULTS.author_given_names, "--author-given-names", help=_hf_help("author_given_names"),
    ),
    author_family_names: Optional[str] = typer.Option(
        HF_DEFAULTS.author_family_names, "--author-family-names", help=_hf_help("author_family_names"),
    ),
    author_affiliation: Optional[str] = typer.Option(
        HF_DEFAULTS.author_affiliation, "--author-affiliation", help=_hf_help("author_affiliation"),
    ),
    message: Optional[str] = typer.Option(
        HF_DEFAULTS.message, "--message", help=_hf_help("message"),
    ),
    repository_code: Optional[str] = typer.Option(
        HF_DEFAULTS.repository_code, "--repository-code", help=_hf_help("repository_code"),
    ),
    overwrite: Optional[bool] = typer.Option(
        None, "--overwrite/--no-overwrite",
        help=(
            "Si --output-dir ya existe, bórralo y vuelve a crearlo en vez de fallar "
            "(útil para repetir 'prepare' sin borrarlo tú a mano). Por defecto falla "
            "si ya existe. (export.overwrite_output_dir)"
        ),
    ),
) -> None:
    """Prepara un export local (shards, manifests, checksums, docs) listo para subir a HuggingFace Hub.

    Siempre rellena la plantilla incluida en el proyecto
    (templates/hfh/HuggingFaceHub.yaml.j2) con estos flags. El valor por
    defecto de cada uno (visible en --help) sale de
    'donadataset publish huggingface config set <campo>=...' — si no pasas el
    flag explícitamente, se usa ese valor guardado.
    """
    try:
        hf_service.run_export(
            DEFAULT_TEMPLATE_FILE,
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
            overwrite_output_dir=overwrite,
        )
    except Exception as exc:
        logging.error("Export failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("upload")
def upload(
    output_dir: str = typer.Option(
        str(DEFAULT_OUTPUT_DIR), "--output-dir", help="Directorio del export ya preparado (el mismo de 'prepare').",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Valida todo pero no crea el repo ni sube ficheros.",
    ),
) -> None:
    """Sube el export preparado (HFH/) a un repositorio dataset de HuggingFace Hub."""
    config = _resolved_config_path(output_dir)
    _warn_if_token_missing(config, required_permission="write")
    setup_logging()
    try:
        hf_service.run_upload(config, dry_run=dry_run)
    except Exception as exc:
        logging.error("Upload failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("download")
def download(
    output_dir: str = typer.Option(
        str(DEFAULT_OUTPUT_DIR), "--output-dir", help="Directorio del export ya preparado (el mismo de 'prepare').",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Valida la configuración pero no descarga nada.",
    ),
) -> None:
    """Descarga el dataset publicado desde HuggingFace Hub y verifica su integridad."""
    config = _resolved_config_path(output_dir)
    _warn_if_token_missing(config, required_permission="read")
    setup_logging()
    try:
        hf_service.run_download_and_verify(config, dry_run=dry_run)
    except Exception as exc:
        logging.error("Downloaded verification failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("release")
def release(
    output_dir: str = typer.Option(
        str(DEFAULT_OUTPUT_DIR), "--output-dir", help="Directorio del export ya preparado (el mismo de 'prepare').",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Muestra qué cambiaría, sin modificar la visibilidad del repo.",
    ),
    verify_only: bool = typer.Option(
        False, "--verify-only",
        help="Solo verifica la visibilidad/accesibilidad actual, sin cambiar nada.",
    ),
) -> None:
    """Hace público el repositorio de HuggingFace Hub y verifica que es accesible sin token.

    Pensado para ejecutarse al final del flujo, una vez el DOI de Zenodo ya
    está en la metadata local y se ha vuelto a subir con 'huggingface upload'.
    """
    if dry_run and verify_only:
        console.print("[red]✘  Usa --dry-run o --verify-only, no ambos.[/red]")
        raise typer.Exit(1)
    config = _resolved_config_path(output_dir)
    _warn_if_token_missing(config, required_permission="write")
    setup_logging()
    try:
        hf_service.run_release(config, dry_run=dry_run, verify_only=verify_only)
    except Exception as exc:
        logging.error("HFH public visibility operation failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("sync-doi")
def sync_doi(
    output_dir: str = typer.Option(
        str(DEFAULT_OUTPUT_DIR), "--output-dir", help="Directorio del export ya preparado (el mismo de 'prepare').",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Solo comprueba si ya existe un DOI, sin modificar CITATION.cff.",
    ),
) -> None:
    """Lee el DOI nativo de HuggingFace Hub (si ya lo generaste a mano en la web) y lo refleja en CITATION.cff.

    HuggingFace Hub no permite generar el DOI por API — solo con el botón
    "Generate DOI" en Settings del repo. Este comando NO lo genera; solo lo
    detecta una vez existe (aparece como tag 'doi:...' en el repo) y actualiza
    tu CITATION.cff local y checksums-sha256.txt. El siguiente paso
    recomendado es volver a subir con 'huggingface upload'.
    """
    config = _resolved_config_path(output_dir)
    _warn_if_token_missing(config, required_permission="read")
    setup_logging()
    try:
        hf_service.run_sync_hfh_doi(config, dry_run=dry_run)
    except Exception as exc:
        logging.error("Hugging Face Hub DOI sync failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("pipeline")
def pipeline(
    source_dataset_dir: Optional[str] = typer.Option(
        str(settings.GENERATE.output), "--source-dataset-dir",
        help=(
            "Directorio del dataset YOLO ya limpio del que se genera el export "
            "(debe contener images/<split>/ y labels/<split>/) — normalmente la "
            "salida de 'generate real'. (paths.source_dataset_dir)"
        ),
    ),
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_OUTPUT_DIR), "--output-dir",
        help=(
            "Directorio donde se escribe el export completo (shards .tar, manifests, "
            "checksums, README, LICENSE, CITATION.cff...), listo para 'huggingface upload'. "
            "(paths.output_dir)"
        ),
    ),
    dataset_slug: Optional[str] = typer.Option(
        HF_DEFAULTS.dataset_slug, "--dataset-slug", help=_hf_help("dataset_slug"),
    ),
    dataset_name: Optional[str] = typer.Option(
        HF_DEFAULTS.dataset_name, "--dataset-name", help=_hf_help("dataset_name"),
    ),
    version: Optional[str] = typer.Option(
        DEFAULT_VERSION, "--version",
        help=(
            "Versión del dataset (ej. 1.0.0) — se escribe en dataset_info.json y "
            "CITATION.cff. Si HEAD está justo en un tag de git (ej. tras 'git tag "
            "v2.0.0'), se usa ese tag automáticamente (sin la 'v'); si no, y no se "
            "pasa el flag, queda como REPLACE_WITH_VERSION. (project.version)"
        ),
    ),
    description: Optional[str] = typer.Option(
        HF_DEFAULTS.description, "--description", help=_hf_help("description"),
    ),
    repo_id: Optional[str] = typer.Option(
        HF_DEFAULTS.repo_id, "--repo-id", help=_hf_help("repo_id"),
    ),
    license_id: Optional[str] = typer.Option(
        HF_DEFAULTS.license_id, "--license-id", help=_hf_help("license_id"),
    ),
    license_name: Optional[str] = typer.Option(
        HF_DEFAULTS.license_name, "--license-name", help=_hf_help("license_name"),
    ),
    license_url: Optional[str] = typer.Option(
        HF_DEFAULTS.license_url, "--license-url", help=_hf_help("license_url"),
    ),
    author_given_names: Optional[str] = typer.Option(
        HF_DEFAULTS.author_given_names, "--author-given-names", help=_hf_help("author_given_names"),
    ),
    author_family_names: Optional[str] = typer.Option(
        HF_DEFAULTS.author_family_names, "--author-family-names", help=_hf_help("author_family_names"),
    ),
    author_affiliation: Optional[str] = typer.Option(
        HF_DEFAULTS.author_affiliation, "--author-affiliation", help=_hf_help("author_affiliation"),
    ),
    message: Optional[str] = typer.Option(
        HF_DEFAULTS.message, "--message", help=_hf_help("message"),
    ),
    repository_code: Optional[str] = typer.Option(
        HF_DEFAULTS.repository_code, "--repository-code", help=_hf_help("repository_code"),
    ),
    overwrite: Optional[bool] = typer.Option(
        None, "--overwrite/--no-overwrite",
        help=(
            "Si --output-dir ya existe, bórralo y vuelve a crearlo en vez de fallar "
            "(útil para repetir 'prepare' sin borrarlo tú a mano). Por defecto falla "
            "si ya existe. (export.overwrite_output_dir)"
        ),
    ),
) -> None:
    """Ejecuta de un tirón: prepare -> upload -> release -> sync-doi.

    Antes del último paso se detiene: HuggingFace Hub no permite generar el
    DOI por API, así que te explica cómo hacerlo a mano en la web y espera a
    que pulses Enter para continuar con 'sync-doi'.
    """
    console.print("[bold cyan]── Paso 1/4: prepare ──[/bold cyan]")
    try:
        hf_service.run_export(
            DEFAULT_TEMPLATE_FILE,
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
            overwrite_output_dir=overwrite,
        )
    except Exception as exc:
        logging.error("Export failed: %s", exc)
        raise typer.Exit(1) from exc

    config = _resolved_config_path(output_dir)

    console.print("\n[bold cyan]── Paso 2/4: upload ──[/bold cyan]")
    _warn_if_token_missing(config, required_permission="write")
    setup_logging()
    try:
        hf_service.run_upload(config, dry_run=False)
    except Exception as exc:
        logging.error("Upload failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 3/4: release ──[/bold cyan]")
    _warn_if_token_missing(config, required_permission="write")
    setup_logging()
    try:
        hf_service.run_release(config, dry_run=False, verify_only=False)
    except Exception as exc:
        logging.error("HFH public visibility operation failed: %s", exc)
        raise typer.Exit(1) from exc

    resolved_repo_id = hf_service.get_repo_id(hf_service.load_yaml(config))

    console.print("\n[bold yellow]── Antes de continuar ──[/bold yellow]")
    console.print(
        "HuggingFace Hub no permite generar el DOI por API — solo desde la web:\n\n"
        f"  1. Abre [bold]https://huggingface.co/datasets/{resolved_repo_id}/settings[/bold]\n"
        "  2. Baja hasta la sección [bold]'Digital Object Identifier (DOI)'[/bold]\n"
        "  3. Pulsa [bold]'Generate DOI'[/bold] y acepta las condiciones\n"
    )
    typer.prompt("Cuando lo hayas generado, pulsa Enter para continuar", default="", show_default=False)

    console.print("\n[bold cyan]── Paso 4/4: sync-doi ──[/bold cyan]")
    _warn_if_token_missing(config, required_permission="read")
    setup_logging()
    try:
        hf_service.run_sync_hfh_doi(config, dry_run=False)
    except Exception as exc:
        logging.error("Hugging Face Hub DOI sync failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold green]✔  Pipeline completado.[/bold green]")


# ── wizard ────────────────────────────────────────────────────────────────

def _run_wizard_step(label: str, action: Callable[[], Any], *, allow_skip: bool = False) -> Any:
    """Runs `action`, retrying/skipping/aborting interactively on failure —
    unlike 'pipeline', which just crashes on the first exception. Returns
    whatever `action` returns (None if the step was skipped)."""
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


def _wizard_resolve_repo_id() -> str:
    """Reads huggingface.repo_id from settings.toml, or asks for it once
    (and offers to save it) if it's still unset — the one piece of identity
    this integration truly cannot guess, same reasoning as
    HuggingFaceSettings.repo_id's own docstring."""
    settings_now = load_settings()
    repo_id = settings_now.HUGGINGFACE.repo_id
    if repo_id and "REPLACE_WITH" not in repo_id:
        return repo_id

    console.print(
        "\nTodavía no has configurado en qué repositorio de HuggingFace Hub publicar "
        "(huggingface.repo_id)."
    )
    while True:
        repo_id = typer.prompt("Repositorio (formato usuario_o_org/dataset)").strip()
        if "/" in repo_id and len(repo_id.split("/")) == 2 and all(repo_id.split("/")):
            break
        console.print("[red]Formato inválido — debe ser 'usuario_o_org/dataset'.[/red]")

    if typer.confirm(f"¿Guardar '{repo_id}' en la configuración para no volver a preguntarlo?", default=True):
        new_settings = _apply_update(settings_now, "HUGGINGFACE", "repo_id", repo_id)
        loaders.toml_loader.write(str(DEFAULT_CONFIG_FILE), new_settings.model_dump(mode="json"), merge=False)
        console.print(f"[green]✔  HUGGINGFACE.repo_id = {repo_id}[/green]")

    return repo_id


@app.command("wizard")
def wizard(
    source_dataset_dir: Optional[str] = typer.Option(
        str(settings.GENERATE.output), "--source-dataset-dir",
        help="Directorio del dataset YOLO ya limpio (normalmente la salida de 'generate real').",
    ),
    output_dir: Optional[str] = typer.Option(
        str(DEFAULT_OUTPUT_DIR), "--output-dir",
        help="Directorio donde se escribe (o ya vive) el export de HuggingFace Hub.",
    ),
) -> None:
    """Asistente interactivo: te guía paso a paso por todo el proceso de publicación en HuggingFace Hub.

    A diferencia de 'pipeline' (que ejecuta todo de un tirón y solo se
    detiene en el DOI), el wizard explica cada fase antes de ejecutarla, te
    pregunta el repo_id si todavía no lo has configurado, detecta si ya
    generaste el DOI en una ejecución anterior (para no hacerte esperar sin
    necesidad), te pide confirmación antes de hacer público el dataset, y
    si un paso falla te deja reintentarlo, saltarlo o abortar en vez de
    terminar en seco. Los valores de identidad del dataset (nombre,
    licencia, autor...) salen de 'donadataset publish huggingface config' —
    el wizard no te los vuelve a preguntar.
    """
    console.print("[bold]Asistente de publicación en HuggingFace Hub[/bold]")
    console.print(
        "Fases: 1) preparar export local  2) subir  3) hacer público  "
        "4) generar DOI (manual, en la web)  5) reflejar el DOI localmente.\n"
    )

    if not (os.environ.get("HF_TOKEN") or settings.HUGGINGFACE.token):
        console.print("[red]✘  No hay ningún token de HuggingFace Hub configurado.[/red]")
        console.print(
            "   Consigue uno en [bold]https://huggingface.co/settings/tokens[/bold] "
            "(con permiso de [bold]write[/bold]) y expórtalo:"
        )
        console.print("   [bold]export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxxxxxxx'[/bold]")
        console.print(
            "   o guárdalo de forma permanente: "
            "[bold]donadataset publish huggingface config set token[/bold]"
        )
        raise typer.Exit(1)

    repo_id = _wizard_resolve_repo_id()
    hf_defaults = load_settings().HUGGINGFACE
    output_path = Path(output_dir)
    config = _resolved_config_path(output_dir)

    # ── Fase 1/5: prepare ──────────────────────────────────────────────────
    reuse_existing = False
    if output_path.exists():
        console.print(f"\nYa existe un export en [bold]{output_path}[/bold].")
        reuse_existing = typer.confirm(
            "¿Reutilizarlo tal cual (en vez de regenerarlo desde cero)?", default=True,
        )

    if reuse_existing:
        console.print("\n[bold cyan]── Fase 1/5: prepare (reutilizando el export existente) ──[/bold cyan]")
    else:
        console.print(
            f"\nSe va a generar el export local (shards .tar, manifests, checksums, README, "
            f"LICENSE, CITATION.cff...) en [bold]{output_path}[/bold], listo para subir."
        )

        def _do_prepare() -> None:
            hf_service.run_export(
                DEFAULT_TEMPLATE_FILE,
                source_dataset_dir=source_dataset_dir,
                output_dir=output_dir,
                dataset_slug=hf_defaults.dataset_slug,
                dataset_name=hf_defaults.dataset_name,
                version=DEFAULT_VERSION,
                description=hf_defaults.description,
                repo_id=repo_id,
                license_id=hf_defaults.license_id,
                license_name=hf_defaults.license_name,
                license_url=hf_defaults.license_url,
                author_given_names=hf_defaults.author_given_names,
                author_family_names=hf_defaults.author_family_names,
                author_affiliation=hf_defaults.author_affiliation,
                message=hf_defaults.message,
                repository_code=hf_defaults.repository_code,
                overwrite_output_dir=output_path.exists(),
            )

        _run_wizard_step("Fase 1/5: prepare", _do_prepare)

    # ── Fase 2/5: upload ───────────────────────────────────────────────────
    console.print(
        f"\nSe va a subir el export a [bold]{repo_id}[/bold] "
        f"({'privado' if hf_service.get_private(hf_service.load_yaml(config)) else 'público'} por ahora)."
    )
    _run_wizard_step("Fase 2/5: upload", lambda: hf_service.run_upload(config, dry_run=False))

    # ── Fase 3/5: release (hacerlo público) ────────────────────────────────
    console.print(
        f"\nEl siguiente paso hace [bold red]público[/bold red] el dataset en "
        f"https://huggingface.co/datasets/{repo_id} — visible para cualquiera."
    )
    if not typer.confirm("¿Continuar y hacerlo público ahora?", default=False):
        console.print(
            "\n[yellow]De acuerdo, lo dejo en privado.[/yellow] Cuando quieras continuar, "
            "vuelve a ejecutar el wizard o usa 'donadataset publish huggingface release'."
        )
        raise typer.Exit(0)

    _run_wizard_step("Fase 3/5: release", lambda: hf_service.run_release(config, dry_run=False, verify_only=False))

    # ── Fase 4/5: DOI (paso manual inevitable) ─────────────────────────────
    def _check_doi() -> Optional[str]:
        doi_config = hf_service.load_yaml(config)
        api = HfApi()
        return hf_service.get_repo_doi(
            api,
            hf_service.get_repo_id(doi_config),
            hf_service.get_repo_type(doi_config),
            hf_service.get_token(doi_config),
        )

    console.print("\n[bold cyan]── Fase 4/5: DOI ──[/bold cyan]")
    doi = _run_wizard_step("Comprobar si ya existe un DOI", _check_doi)

    if doi:
        console.print(f"[green]✔  Ya existe un DOI generado: {doi}[/green] — no hace falta generarlo de nuevo.")
    else:
        console.print(
            "HuggingFace Hub no permite generar el DOI por API — solo desde la web:\n\n"
            f"  1. Abre [bold]https://huggingface.co/datasets/{repo_id}/settings[/bold]\n"
            "  2. Baja hasta la sección [bold]'Digital Object Identifier (DOI)'[/bold]\n"
            "  3. Pulsa [bold]'Generate DOI'[/bold] y acepta las condiciones\n"
        )
        while True:
            typer.prompt("Cuando lo hayas generado, pulsa Enter para continuar", default="", show_default=False)
            doi = _check_doi()
            if doi:
                console.print(f"[green]✔  DOI detectado: {doi}[/green]")
                break
            if not typer.confirm("Todavía no veo ningún DOI. ¿Comprobar de nuevo?", default=True):
                console.print("[yellow]De acuerdo — puedes terminar esto más tarde con 'huggingface sync-doi'.[/yellow]")
                break

    # ── Fase 5/5: reflejar el DOI localmente y volver a subir ──────────────
    if doi:
        _run_wizard_step("Fase 5/5: sync-doi", lambda: hf_service.run_sync_hfh_doi(config, dry_run=False))
        console.print(
            "\nEl DOI ya está en tu CITATION.cff local — falta reflejarlo en el repo público."
        )
        _run_wizard_step(
            "Fase 5/5: re-subida (para publicar el DOI actualizado)",
            lambda: hf_service.run_upload(config, dry_run=False),
        )

    console.print(f"\n[bold green]✔  Publicación en HuggingFace Hub completada: https://huggingface.co/datasets/{repo_id}[/bold green]")
