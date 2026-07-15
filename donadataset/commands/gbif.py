"""Comandos CLI para publicar el dataset en GBIF.

Gestiona únicamente los parámetros de entrada; toda la lógica vive en
donadataset.services.gbif.
"""
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from donadataset.commands import gbif_config_commands
from donadataset.commands.huggingface import _resolved_config_path as _hf_resolved_config_path
from donadataset.commands.huggingface import _warn_if_token_missing as _hf_warn_if_token_missing
from donadataset.config import GBIFSettings, get_gbif_output_dir, settings
from donadataset.services import gbif as gbif_service
from donadataset.services import huggingface as hf_service
from donadataset.services.common import setup_logging

console = Console()
app     = typer.Typer(help="Publica el dataset en GBIF (Camtrap DP).")
app.add_typer(gbif_config_commands.app, name="config")

GBIF_DEFAULTS      = settings.GBIF
HF_DEFAULTS        = settings.HUGGINGFACE
DEFAULT_OUTPUT_DIR = get_gbif_output_dir(settings.HUGGINGFACE.repo_id)


def _gbif_help(field: str) -> str:
    """Reutiliza la description de GBIFSettings como help= del flag equivalente."""
    return GBIFSettings.model_fields[field].description


def _derive_dataset_slug(repo_id: Optional[str]) -> str:
    """El paquete Camtrap DP ya no tiene su propio --dataset-slug — se
    deriva del segmento de dataset de --hf-repo-id (user_or_org/dataset ->
    dataset), que es la identidad real de lo que se está empaquetando. Solo
    recurre a HUGGINGFACE.dataset_slug si no hay repo_id disponible en
    absoluto (--source-dataset-dir sin repo_id configurado)."""
    if repo_id and "REPLACE_WITH" not in repo_id and "/" in repo_id:
        return repo_id.split("/", 1)[1]
    return HF_DEFAULTS.dataset_slug or "donadataset"


@app.command("prepare")
def prepare(
    source_dataset_dir: Optional[str] = typer.Option(
        None, "--source-dataset-dir",
        help=(
            "Directorio LOCAL del dataset YOLO ya limpio, para saltarte HuggingFace Hub por "
            "completo (útil para probar antes de haber publicado nada). Por defecto (sin "
            "este flag) se usa --hf-repo-id: primero se reutiliza el export que ya dejó "
            "'huggingface prepare' en <Documents>/donadataset/HFH/<repo_id>, y solo si no "
            "está ahí se descarga de HuggingFace Hub."
        ),
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir",
        help="Directorio donde escribir el paquete Camtrap DP (deployments/media/observations.csv, datapackage.json y el .zip).",
    ),
    institution_code: str = typer.Option(
        GBIF_DEFAULTS.institution_code, "--institution-code", help=_gbif_help("institution_code"),
    ),
    contact_email: Optional[str] = typer.Option(
        GBIF_DEFAULTS.contact_email, "--contact-email", help=_gbif_help("contact_email"),
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite",
        help="Si --output-dir ya existe, bórralo y vuelve a crearlo en vez de fallar.",
    ),
    hf_repo_id: Optional[str] = typer.Option(
        settings.HUGGINGFACE.repo_id, "--hf-repo-id",
        help=(
            "Repositorio de HuggingFace Hub (user_or_org/dataset). Usado para leer el "
            "dataset fuente cuando no se pasa --source-dataset-dir, y para construir la URL de "
            "media.filePath (siempre enlaza al .tar de HuggingFace Hub, nunca a una ruta "
            "local). (huggingface.repo_id)"
        ),
    ),
) -> None:
    """Convierte el dataset YOLO ya limpio en un paquete Camtrap DP, sin nada que rellenar a mano.

    Sin --source-dataset-dir, el dataset se obtiene de HuggingFace Hub
    (--hf-repo-id): reutiliza en local el export de 'huggingface prepare' si
    ya está en disco, o lo descarga si no.

    media.filePath siempre apunta a la URL persistente del `.tar` de
    HuggingFace Hub en el que 'huggingface prepare' empaquetó cada imagen
    (el de train para las de train, el de val para las de val...) — nunca a
    una ruta local. No es una URL por imagen individual: varias imágenes de
    un mismo shard comparten la misma URL (queda anotado en
    media.mediaComments). Requiere que el dataset ya tenga manifest.csv
    (huggingface prepare ya ejecutado); si el `--source-dataset-dir` que usas
    no coincide con lo publicado, falla en vez de adivinar en silencio.

    Este pipeline no rastrea GPS ni fecha de despliegue por cámara en
    ningún punto, así que se asume un deployment por split (train/val/test)
    con coordenadas fijas ilustrativas dentro de Doñana. La fecha de cada
    imagen se lee de su EXIF si existe; si no, se reparte dentro del rango
    EXIF real del propio split (o, si el split entero no tiene EXIF, dentro
    de un año-placeholder fijo) — todo queda anotado en
    deploymentComments/mediaComments. Cada caja detectada se agrupa por
    imagen+especie (una observación con count = número de cajas de esa
    especie en esa imagen); las imágenes cuya única clase es 'Empty' (o sin
    cajas) generan una observación 'blank'. El resultado es un .zip
    (datapackage.json + deployments/media/observations.csv) que puedes
    subir a mano a un IPT v3+, o subir con 'donadataset publish gbif upload'
    al repo de HuggingFace Hub ya publicado para tener una URL persistente
    y registrar con 'donadataset publish gbif register'. Puramente local —
    no sube nada.
    """
    setup_logging()
    try:
        gbif_service.run_prepare(
            Path(source_dataset_dir) if source_dataset_dir else None,
            output_dir,
            repo_id=hf_repo_id,
            dataset_slug=_derive_dataset_slug(hf_repo_id),
            dataset_name=HF_DEFAULTS.dataset_name,
            description=HF_DEFAULTS.description,
            license_id=HF_DEFAULTS.license_id,
            license_name=HF_DEFAULTS.license_name,
            license_url=HF_DEFAULTS.license_url,
            rights_holder=HF_DEFAULTS.author_affiliation,
            institution_code=institution_code,
            contact_name=HF_DEFAULTS.author_family_names,
            contact_email=contact_email,
            overwrite=overwrite,
        )
    except Exception as exc:
        logging.error("GBIF prepare failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("upload")
def upload(
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir",
        help="Directorio donde 'gbif prepare' generó el paquete Camtrap DP (el .zip a subir).",
    ),
    hf_repo_id: Optional[str] = typer.Option(
        settings.HUGGINGFACE.repo_id, "--hf-repo-id",
        help=(
            "Repositorio de HuggingFace Hub (user_or_org/dataset) ya publicado, donde añadir "
            "el .zip. (huggingface.repo_id)"
        ),
    ),
    hfh_output_dir: Optional[str] = typer.Option(
        None, "--hfh-output-dir",
        help=(
            "Directorio LOCAL del export de HuggingFace Hub, para saltarte la resolución "
            "automática. Por defecto, se busca en <Documents>/donadataset/HFH/<repo_id> "
            "(el mismo --output-dir de 'huggingface prepare'), y si no está ahí, se "
            "descarga el repo ya publicado como caché en "
            "<Documents>/donadataset/GBIF/<repo_id>/hfh_download."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Muestra qué se copiaría/subiría, sin tocar nada (nunca descarga).",
    ),
) -> None:
    """Sube el .zip que generó 'gbif prepare' al repo de HuggingFace Hub ya publicado.

    Lo copia al export local de HuggingFace Hub (--hfh-output-dir, resuelto
    automáticamente si no se pasa) y regenera su checksums-sha256.txt, luego
    sube AMBOS ficheros con un 'huggingface upload' acotado (allow_patterns)
    — nunca toca el resto del export ya publicado (shards .tar, README...).
    Requiere que el repo ya exista (huggingface prepare + upload ya
    ejecutados) y HF_TOKEN con permiso de escritura. Da la URL persistente
    lista para 'donadataset publish gbif register --archive-url'.
    """
    setup_logging()
    try:
        archive_filename, checksums_filename, resolved_hfh_output_dir = gbif_service.run_upload(
            output_dir, hf_repo_id, Path(hfh_output_dir) if hfh_output_dir else None, dry_run=dry_run,
        )
    except Exception as exc:
        logging.error("GBIF upload failed: %s", exc)
        raise typer.Exit(1) from exc

    if dry_run or resolved_hfh_output_dir is None:
        return

    hf_config_path = _hf_resolved_config_path(str(resolved_hfh_output_dir))
    _hf_warn_if_token_missing(hf_config_path, required_permission="write")
    try:
        hf_service.run_upload(
            hf_config_path, dry_run=False, allow_patterns=[archive_filename, checksums_filename],
        )
    except Exception as exc:
        logging.error("Upload to HuggingFace Hub failed: %s", exc)
        raise typer.Exit(1) from exc

    persistent_url = f"https://huggingface.co/datasets/{hf_repo_id}/resolve/main/{archive_filename}"
    console.print(f"[bold green]Subido:[/bold green] {persistent_url}")
    console.print(f"Regístralo con: donadataset publish gbif register --archive-url {persistent_url}")


@app.command("register")
def register(
    archive_url: str = typer.Option(
        ..., "--archive-url",
        help=(
            "URL pública donde TÚ has alojado el .zip generado por 'gbif prepare' "
            "(tu propio servidor, un bucket, HuggingFace...) — GBIF lo rastreará desde ahí. "
            "No sube ningún fichero; solo registra esta URL."
        ),
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir",
        help=(
            "Directorio del export ya preparado (el mismo --output-dir de 'prepare') — ahí se "
            "lee/escribe gbif_linked_dataset_record.json para saber si ya existe un dataset "
            "registrado y hay que actualizarlo en vez de crear uno nuevo."
        ),
    ),
    environment: str = typer.Option(
        GBIF_DEFAULTS.environment, "--environment", help=_gbif_help("environment"),
    ),
    publishing_organization_key: Optional[str] = typer.Option(
        GBIF_DEFAULTS.publishing_organization_key, "--publishing-organization-key",
        help=_gbif_help("publishing_organization_key"),
    ),
    installation_key: Optional[str] = typer.Option(
        GBIF_DEFAULTS.installation_key, "--installation-key", help=_gbif_help("installation_key"),
    ),
    registry_language: str = typer.Option(
        GBIF_DEFAULTS.registry_language, "--registry-language", help=_gbif_help("registry_language"),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Muestra qué se registraría/actualizaría, sin llamar a la Registry API.",
    ),
) -> None:
    """Registra (o actualiza) el dataset en el Registry de GBIF por API — sin pasar por un IPT.

    Requiere que ya hayas alojado tú mismo el .zip de 'gbif prepare' en una
    URL pública (--archive-url) y que gbif.publishing_organization_key /
    gbif.installation_key estén configurados (créalos una vez, a mano, en
    gbif.org o su sandbox gbif-test.org). Usa las credenciales de tu cuenta
    de gbif.org vía las variables de entorno GBIF_USERNAME/GBIF_PASSWORD —
    la Registry API usa Basic Auth, no un token. La primera vez crea el
    dataset; en las siguientes actualiza su metadata y reemplaza el
    endpoint CAMTRAP_DP por el --archive-url actual.
    """
    setup_logging()
    try:
        gbif_service.run_register(
            archive_url,
            output_dir,
            environment=environment,
            publishing_organization_key=publishing_organization_key,
            installation_key=installation_key,
            dataset_name=HF_DEFAULTS.dataset_name,
            description=HF_DEFAULTS.description,
            license_url=HF_DEFAULTS.license_url,
            registry_language=registry_language,
            dry_run=dry_run,
        )
    except Exception as exc:
        logging.error("GBIF register failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("pipeline")
def pipeline(
    source_dataset_dir: Optional[str] = typer.Option(
        None, "--source-dataset-dir",
        help=(
            "Directorio LOCAL del dataset YOLO ya limpio, para saltarte HuggingFace Hub por "
            "completo. Por defecto (sin este flag) se usa --hf-repo-id: reutiliza el export "
            "local de 'huggingface prepare' si ya está en disco, o lo descarga si no."
        ),
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir",
        help="Directorio donde escribir el paquete Camtrap DP y gbif_linked_dataset_record.json.",
    ),
    institution_code: str = typer.Option(
        GBIF_DEFAULTS.institution_code, "--institution-code", help=_gbif_help("institution_code"),
    ),
    contact_email: Optional[str] = typer.Option(
        GBIF_DEFAULTS.contact_email, "--contact-email", help=_gbif_help("contact_email"),
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite",
        help="Si --output-dir ya existe, bórralo y vuelve a crearlo en vez de fallar.",
    ),
    hf_repo_id: Optional[str] = typer.Option(
        settings.HUGGINGFACE.repo_id, "--hf-repo-id",
        help=(
            "Repositorio de HuggingFace Hub (user_or_org/dataset). Usado para leer el "
            "dataset fuente cuando no se pasa --source-dataset-dir, y para alojar el .zip. "
            "(huggingface.repo_id)"
        ),
    ),
    hfh_output_dir: Optional[str] = typer.Option(
        None, "--hfh-output-dir",
        help=(
            "Directorio LOCAL del export de HuggingFace Hub, para saltarte la resolución "
            "automática. Por defecto, se busca en <Documents>/donadataset/HFH/<repo_id>, y "
            "si no está ahí, se descarga el repo ya publicado como caché en "
            "<Documents>/donadataset/GBIF/<repo_id>/hfh_download."
        ),
    ),
    environment: str = typer.Option(
        GBIF_DEFAULTS.environment, "--environment", help=_gbif_help("environment"),
    ),
    publishing_organization_key: Optional[str] = typer.Option(
        GBIF_DEFAULTS.publishing_organization_key, "--publishing-organization-key",
        help=_gbif_help("publishing_organization_key"),
    ),
    installation_key: Optional[str] = typer.Option(
        GBIF_DEFAULTS.installation_key, "--installation-key", help=_gbif_help("installation_key"),
    ),
    registry_language: str = typer.Option(
        GBIF_DEFAULTS.registry_language, "--registry-language", help=_gbif_help("registry_language"),
    ),
) -> None:
    """Ejecuta de un tirón: prepare -> upload (a HuggingFace Hub) -> register.

    Encadena los tres comandos por separado sin que tengas que copiar la
    URL a mano entre pasos (media.filePath ya enlaza siempre a HuggingFace
    Hub en ambos comandos). Requiere que el repo de HuggingFace Hub
    (--hf-repo-id) ya esté publicado (huggingface prepare + upload ya
    ejecutados), HF_TOKEN con permiso de escritura, GBIF_USERNAME/
    GBIF_PASSWORD, y gbif.publishing_organization_key/installation_key ya
    configurados.
    """
    setup_logging()
    try:
        console.print("[bold cyan]── Paso 1/3: prepare ──[/bold cyan]")
        gbif_service.run_prepare(
            Path(source_dataset_dir) if source_dataset_dir else None,
            output_dir,
            repo_id=hf_repo_id,
            dataset_slug=_derive_dataset_slug(hf_repo_id),
            dataset_name=HF_DEFAULTS.dataset_name,
            description=HF_DEFAULTS.description,
            license_id=HF_DEFAULTS.license_id,
            license_name=HF_DEFAULTS.license_name,
            license_url=HF_DEFAULTS.license_url,
            rights_holder=HF_DEFAULTS.author_affiliation,
            institution_code=institution_code,
            contact_name=HF_DEFAULTS.author_family_names,
            contact_email=contact_email,
            overwrite=overwrite,
        )
    except Exception as exc:
        logging.error("GBIF prepare failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 2/3: upload ──[/bold cyan]")
    try:
        archive_filename, checksums_filename, resolved_hfh_output_dir = gbif_service.run_upload(
            output_dir, hf_repo_id, Path(hfh_output_dir) if hfh_output_dir else None,
        )
    except Exception as exc:
        logging.error("GBIF upload failed: %s", exc)
        raise typer.Exit(1) from exc

    hf_config_path = _hf_resolved_config_path(str(resolved_hfh_output_dir))
    _hf_warn_if_token_missing(hf_config_path, required_permission="write")
    try:
        hf_service.run_upload(
            hf_config_path, dry_run=False, allow_patterns=[archive_filename, checksums_filename],
        )
    except Exception as exc:
        logging.error("Upload to HuggingFace Hub failed: %s", exc)
        raise typer.Exit(1) from exc
    persistent_url = f"https://huggingface.co/datasets/{hf_repo_id}/resolve/main/{archive_filename}"
    console.print(f"[bold green]Subido:[/bold green] {persistent_url}")

    console.print("\n[bold cyan]── Paso 3/3: register ──[/bold cyan]")
    try:
        gbif_service.run_register(
            persistent_url,
            output_dir,
            environment=environment,
            publishing_organization_key=publishing_organization_key,
            installation_key=installation_key,
            dataset_name=HF_DEFAULTS.dataset_name,
            description=HF_DEFAULTS.description,
            license_url=HF_DEFAULTS.license_url,
            registry_language=registry_language,
            dry_run=False,
        )
    except Exception as exc:
        logging.error("GBIF register failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold green]✔  Pipeline completado.[/bold green]")
