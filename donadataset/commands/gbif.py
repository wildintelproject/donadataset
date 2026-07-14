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
from donadataset.config import GBIFSettings, get_app_documents_dir, settings
from donadataset.services import gbif as gbif_service
from donadataset.services.common import setup_logging

console = Console()
app     = typer.Typer(help="Publica el dataset en GBIF (Camtrap DP).")
app.add_typer(gbif_config_commands.app, name="config")

GBIF_DEFAULTS      = settings.GBIF
DEFAULT_OUTPUT_DIR = get_app_documents_dir() / "GBIF"


def _gbif_help(field: str) -> str:
    """Reutiliza la description de GBIFSettings como help= del flag equivalente."""
    return GBIFSettings.model_fields[field].description


@app.command("prepare")
def prepare(
    source_dataset_dir: str = typer.Option(
        str(settings.GENERATE.output), "--source-dataset-dir",
        help="Directorio del dataset YOLO ya limpio (el mismo que usa 'huggingface prepare').",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir",
        help="Directorio donde escribir el paquete Camtrap DP (deployments/media/observations.csv, datapackage.json y el .zip).",
    ),
    dataset_slug: str = typer.Option(
        "donadataset", "--dataset-slug", help="Slug usado en el nombre del .zip generado y en el 'name'/'id' del datapackage.json.",
    ),
    dataset_name: str = typer.Option(
        GBIF_DEFAULTS.dataset_name, "--dataset-name", help=_gbif_help("dataset_name"),
    ),
    description: str = typer.Option(
        GBIF_DEFAULTS.description, "--description", help=_gbif_help("description"),
    ),
    license_id: str = typer.Option(
        GBIF_DEFAULTS.license_id, "--license-id", help=_gbif_help("license_id"),
    ),
    license_name: str = typer.Option(
        GBIF_DEFAULTS.license_name, "--license-name", help=_gbif_help("license_name"),
    ),
    license_url: str = typer.Option(
        GBIF_DEFAULTS.license_url, "--license-url", help=_gbif_help("license_url"),
    ),
    rights_holder: str = typer.Option(
        GBIF_DEFAULTS.rights_holder, "--rights-holder", help=_gbif_help("rights_holder"),
    ),
    institution_code: str = typer.Option(
        GBIF_DEFAULTS.institution_code, "--institution-code", help=_gbif_help("institution_code"),
    ),
    contact_name: str = typer.Option(
        GBIF_DEFAULTS.contact_name, "--contact-name", help=_gbif_help("contact_name"),
    ),
    contact_email: Optional[str] = typer.Option(
        GBIF_DEFAULTS.contact_email, "--contact-email", help=_gbif_help("contact_email"),
    ),
    classified_by: str = typer.Option(
        GBIF_DEFAULTS.classified_by, "--classified-by", help=_gbif_help("classified_by"),
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite",
        help="Si --output-dir ya existe, bórralo y vuelve a crearlo en vez de fallar.",
    ),
    upload_to_huggingface: bool = typer.Option(
        False, "--upload-to-huggingface/--no-upload-to-huggingface",
        help=(
            "Tras generar el .zip, súbelo como fichero suelto al repo de HuggingFace Hub "
            "(--hf-repo-id) para tener una URL persistente lista para 'gbif register "
            "--archive-url'. Requiere que el repo ya exista (huggingface prepare + upload "
            "ya ejecutados) y HF_TOKEN con permiso de escritura."
        ),
    ),
    link_media_to_huggingface: bool = typer.Option(
        False, "--link-media-to-huggingface/--no-link-media-to-huggingface",
        help=(
            "En vez de una ruta relativa local, pon en media.filePath la URL persistente del "
            ".tar de HuggingFace Hub (--hf-repo-id) en el que 'huggingface prepare' empaquetó "
            "cada imagen (el de train para las de train, el de val para las de val...). Requiere "
            "que el repo ya exista con manifest.csv subido (huggingface prepare + upload ya "
            "ejecutados con el MISMO dataset fuente) — descarga solo ese manifest.csv, no los "
            ".tar. No da una URL por imagen individual: varias imágenes de un mismo shard "
            "comparten la misma URL (queda anotado en media.mediaComments)."
        ),
    ),
    hf_repo_id: Optional[str] = typer.Option(
        settings.HUGGINGFACE.repo_id, "--hf-repo-id",
        help="Repositorio de HuggingFace Hub (user_or_org/dataset) usado con --upload-to-huggingface/--link-media-to-huggingface. (huggingface.repo_id)",
    ),
) -> None:
    """Convierte el dataset YOLO ya limpio en un paquete Camtrap DP, sin nada que rellenar a mano.

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
    subir a mano a un IPT v3+, o (con --upload-to-huggingface) subir tú
    mismo al repo de HuggingFace Hub ya publicado para tener una URL
    persistente y registrar con 'donadataset publish gbif register'.
    """
    setup_logging()
    try:
        gbif_service.run_prepare(
            Path(source_dataset_dir),
            output_dir,
            dataset_slug=dataset_slug,
            dataset_name=dataset_name,
            description=description,
            license_id=license_id,
            license_name=license_name,
            license_url=license_url,
            rights_holder=rights_holder,
            institution_code=institution_code,
            contact_name=contact_name,
            contact_email=contact_email,
            classified_by=classified_by,
            overwrite=overwrite,
            upload_to_huggingface=upload_to_huggingface,
            link_media_to_huggingface=link_media_to_huggingface,
            hf_repo_id=hf_repo_id,
        )
    except Exception as exc:
        logging.error("GBIF prepare failed: %s", exc)
        raise typer.Exit(1) from exc


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
    dataset_name: str = typer.Option(
        GBIF_DEFAULTS.dataset_name, "--dataset-name", help=_gbif_help("dataset_name"),
    ),
    description: str = typer.Option(
        GBIF_DEFAULTS.description, "--description", help=_gbif_help("description"),
    ),
    license_url: str = typer.Option(
        GBIF_DEFAULTS.license_url, "--license-url", help=_gbif_help("license_url"),
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
            dataset_name=dataset_name,
            description=description,
            license_url=license_url,
            registry_language=registry_language,
            dry_run=dry_run,
        )
    except Exception as exc:
        logging.error("GBIF register failed: %s", exc)
        raise typer.Exit(1) from exc


@app.command("pipeline")
def pipeline(
    source_dataset_dir: str = typer.Option(
        str(settings.GENERATE.output), "--source-dataset-dir",
        help="Directorio del dataset YOLO ya limpio (el mismo que usa 'huggingface prepare').",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir",
        help="Directorio donde escribir el paquete Camtrap DP y gbif_linked_dataset_record.json.",
    ),
    dataset_slug: str = typer.Option(
        "donadataset", "--dataset-slug", help="Slug usado en el nombre del .zip y en el 'name'/'id' del datapackage.json.",
    ),
    dataset_name: str = typer.Option(
        GBIF_DEFAULTS.dataset_name, "--dataset-name", help=_gbif_help("dataset_name"),
    ),
    description: str = typer.Option(
        GBIF_DEFAULTS.description, "--description", help=_gbif_help("description"),
    ),
    license_id: str = typer.Option(
        GBIF_DEFAULTS.license_id, "--license-id", help=_gbif_help("license_id"),
    ),
    license_name: str = typer.Option(
        GBIF_DEFAULTS.license_name, "--license-name", help=_gbif_help("license_name"),
    ),
    license_url: str = typer.Option(
        GBIF_DEFAULTS.license_url, "--license-url", help=_gbif_help("license_url"),
    ),
    rights_holder: str = typer.Option(
        GBIF_DEFAULTS.rights_holder, "--rights-holder", help=_gbif_help("rights_holder"),
    ),
    institution_code: str = typer.Option(
        GBIF_DEFAULTS.institution_code, "--institution-code", help=_gbif_help("institution_code"),
    ),
    contact_name: str = typer.Option(
        GBIF_DEFAULTS.contact_name, "--contact-name", help=_gbif_help("contact_name"),
    ),
    contact_email: Optional[str] = typer.Option(
        GBIF_DEFAULTS.contact_email, "--contact-email", help=_gbif_help("contact_email"),
    ),
    classified_by: str = typer.Option(
        GBIF_DEFAULTS.classified_by, "--classified-by", help=_gbif_help("classified_by"),
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite",
        help="Si --output-dir ya existe, bórralo y vuelve a crearlo en vez de fallar.",
    ),
    hf_repo_id: Optional[str] = typer.Option(
        settings.HUGGINGFACE.repo_id, "--hf-repo-id",
        help="Repositorio de HuggingFace Hub (user_or_org/dataset) donde alojar el .zip. (huggingface.repo_id)",
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
    """Ejecuta de un tirón: prepare (subiendo el .zip a HuggingFace Hub) -> register.

    A diferencia de 'gbif prepare' a secas, aquí --upload-to-huggingface y
    --link-media-to-huggingface van siempre activados — es la única forma de
    encadenar automáticamente con 'register' sin que tengas que copiar la URL
    a mano. Requiere que el repo de HuggingFace Hub (--hf-repo-id) ya esté
    publicado (huggingface prepare + upload ya ejecutados), HF_TOKEN con
    permiso de escritura, GBIF_USERNAME/GBIF_PASSWORD, y
    gbif.publishing_organization_key/installation_key ya configurados.
    """
    setup_logging()
    try:
        console.print("[bold cyan]── Paso 1/2: prepare ──[/bold cyan]")
        persistent_url = gbif_service.run_prepare(
            Path(source_dataset_dir),
            output_dir,
            dataset_slug=dataset_slug,
            dataset_name=dataset_name,
            description=description,
            license_id=license_id,
            license_name=license_name,
            license_url=license_url,
            rights_holder=rights_holder,
            institution_code=institution_code,
            contact_name=contact_name,
            contact_email=contact_email,
            classified_by=classified_by,
            overwrite=overwrite,
            upload_to_huggingface=True,
            link_media_to_huggingface=True,
            hf_repo_id=hf_repo_id,
        )
    except Exception as exc:
        logging.error("GBIF prepare failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold cyan]── Paso 2/2: register ──[/bold cyan]")
    try:
        gbif_service.run_register(
            persistent_url,
            output_dir,
            environment=environment,
            publishing_organization_key=publishing_organization_key,
            installation_key=installation_key,
            dataset_name=dataset_name,
            description=description,
            license_url=license_url,
            registry_language=registry_language,
            dry_run=False,
        )
    except Exception as exc:
        logging.error("GBIF register failed: %s", exc)
        raise typer.Exit(1) from exc

    console.print("\n[bold green]✔  Pipeline completado.[/bold green]")
