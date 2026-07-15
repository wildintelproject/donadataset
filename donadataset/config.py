"""Configuración de la aplicación — cargada con Dynaconf y validada con Pydantic.

El fichero de configuración vive en el directorio de app del usuario
(no en el repo), porque source/output son específicos de cada máquina:
  - Linux:   ~/.config/donadataset/settings.toml
  - macOS:   ~/Library/Application Support/donadataset/settings.toml
  - Windows: %APPDATA%\\donadataset\\settings.toml
Se genera automáticamente con los valores por defecto la primera vez que se usa.
"""
from pathlib import Path
from typing import Literal, Optional

import platformdirs
import typer
from dynaconf import Dynaconf, loaders
from pydantic import BaseModel, Field, field_validator, model_validator

APP_NAME = "donadataset"

REPO_ROOT           = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_DIR = Path(typer.get_app_dir(APP_NAME))
DEFAULT_CONFIG_FILE  = DEFAULT_SETTINGS_DIR / "settings.toml"


def get_documents_dir() -> Path:
    """Directorio "Documents" del usuario, portable entre Linux/macOS/Windows."""
    return Path(platformdirs.user_documents_dir())


def get_app_documents_dir() -> Path:
    """<Documents>/donadataset — donde viven el dataset fuente y el generado por defecto."""
    return get_documents_dir() / APP_NAME


def get_hfh_output_dir(repo_id: Optional[str] = None) -> Path:
    """Directorio por defecto del export local de HuggingFace Hub —
    <Documents>/donadataset/HFH/<repo_id> cuando repo_id ya está configurado
    (para no mezclar exports de distintos repos en la misma carpeta), o
    <Documents>/donadataset/HFH a secas si todavía no lo está. Usado como
    default de --output-dir en 'huggingface prepare'/etc., y de
    --hfh-output-dir en 'zenodo'/'b2share' — mismo cálculo en los tres sitios
    para que sus defaults sigan coincidiendo sin tener que pasarlo a mano."""
    base = get_app_documents_dir() / "HFH"
    return (base / repo_id) if repo_id else base


class GenerateSettings(BaseModel):
    source: Path = Field(
        default_factory=lambda: get_app_documents_dir() / "source",
        description="Directorio raíz del dataset original (contiene images/<split>/ y labels/<split>/).",
    )
    output: Path = Field(
        default_factory=lambda: get_app_documents_dir() / "output",
        description="Directorio de salida. Se vacía por completo antes de generar.",
    )
    splits: list[Literal["train", "val", "test"]] = Field(
        default_factory=lambda: ["train", "val", "test"],
        description="Splits a procesar.",
    )
    remove_class_ids: list[int] = Field(
        default_factory=lambda: [10, 17],
        description=(
            "IDs de clase (esquema original) a eliminar por completo. "
            "Por defecto: Homo sapiens (10) y Vehicle (17)."
        ),
    )
    duplicate_key_mode: Literal["stem", "relative_stem"] = Field(
        default="stem",
        description="Criterio para detectar imágenes duplicadas.",
    )
    classes_map: Path = Field(
        default_factory=lambda: REPO_ROOT / "metadata" / "source_classes.yaml",
        description="YAML con el esquema de clases original (id -> nombre) del dataset fuente.",
    )
    image_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
        description="Extensiones de imagen reconocidas.",
    )
    extension_priority: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"],
        description="Orden de preferencia de extensión al resolver imágenes duplicadas.",
    )


class GenerateToySettings(BaseModel):
    source: Path = Field(
        default_factory=lambda: get_app_documents_dir() / "output",
        description="Dataset ya generado (salida de 'generate real'), usado como fuente del toy.",
    )
    output: Path = Field(
        default_factory=lambda: get_app_documents_dir() / "toy",
        description="Directorio de salida del dataset toy. Se vacía por completo antes de generar.",
    )
    splits: list[Literal["train", "val", "test"]] = Field(
        default_factory=lambda: ["train", "val", "test"],
        description="Splits a procesar.",
    )
    samples_per_class: dict[Literal["train", "val", "test"], int] = Field(
        default_factory=lambda: {"train": 25, "val": 10, "test": 10},
        description="Número máximo de imágenes a muestrear por clase, para cada split.",
    )
    random_seed: int = Field(
        default=42,
        description="Semilla aleatoria para que el muestreo sea reproducible.",
    )

    @field_validator("samples_per_class")
    @classmethod
    def _require_all_splits(cls, v: dict[str, int]) -> dict[str, int]:
        missing = {"train", "val", "test"} - v.keys()
        if missing:
            raise ValueError(
                f"faltan splits en samples_per_class: {sorted(missing)} "
                "(se requieren 'train', 'val' y 'test')"
            )
        return v


def _slug_to_dataset_name(slug: str) -> str:
    """Deriva un nombre legible ('camera-trap-mammals' -> 'Camera Trap Mammals')
    a partir de un slug, usado como valor de arranque de dataset_name cuando no
    se ha fijado explícitamente — ver HuggingFaceSettings más abajo."""
    return slug.replace("-", " ").replace("_", " ").title()


class HuggingFaceSettings(BaseModel):
    """Valores por defecto de identidad de proyecto, reutilizados entre
    ejecuciones de 'publish huggingface prepare' (plantilla .j2 o flags),
    para no tener que repetirlos ni editarlos en cada YAML/plantilla.

    Los que tienen un valor de proyecto razonable (licencia, descripción,
    autor) vienen ya rellenos; repo_id se queda en None a propósito — depende
    de tu cuenta/organización de HuggingFace, así que no hay un valor
    correcto que valga para cualquiera que clone este código (fíjalo tú en tu
    propio settings.toml con 'config set repo_id=...'). dataset_name tampoco
    tiene un valor fijo propio: si no lo fijas a mano, se deriva de
    dataset_slug (ver _derive_dataset_name más abajo) — evita mantener dos
    campos casi redundantes (uno para nombrar carpetas locales, otro para
    mostrar) cuando lo único que de verdad no se puede adivinar es repo_id."""
    dataset_slug: Optional[str] = Field(
        default="donadataset",
        description=(
            "Identificador corto interno (minúsculas, sin espacios) usado solo para nombrar "
            "carpetas/ficheros locales de este proyecto — la carpeta de export si no fijas "
            "--output-dir (HFH_Z_{slug}), y lo reutilizan Zenodo/B2SHARE/GBIF para nombrar las "
            "suyas (Zenodo_{slug}, B2SHARE_{slug}, {slug}-camtrap-dp.zip). No tiene nada que ver "
            "con el repo de HuggingFace Hub — eso es repo_id. (project.dataset_slug)"
        ),
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description=(
            "Nombre legible para mostrar en README.md, dataset_info.json y el título de "
            "CITATION.cff — solo texto de presentación, no nombra ningún fichero/carpeta ni "
            "afecta al repo de HuggingFace Hub. Si no lo fijas, se deriva automáticamente de "
            "dataset_slug ('donadataset' -> 'Donadataset'). (project.dataset_name)"
        ),
    )
    description: Optional[str] = Field(
        default="Camera-trap image dataset for object detection, from Doñana National Park, Spain.",
        description="Descripción del dataset (project.description).",
    )
    license_id: Optional[str] = Field(
        default="CC-BY-4.0", description="Identificador de licencia, ej. CC-BY-4.0 (license.license_id).",
    )
    license_name: Optional[str] = Field(
        default="Creative Commons Attribution 4.0 International",
        description="Nombre completo de la licencia (license.license_name).",
    )
    license_url: Optional[str] = Field(
        default="https://creativecommons.org/licenses/by/4.0/",
        description="URL de la licencia (license.license_url).",
    )
    author_given_names: Optional[str] = Field(
        default="WildINTEL Spain",
        description="Nombre del autor (citation.authors[0].given_names).",
    )
    author_family_names: Optional[str] = Field(
        default="WildINTEL",
        description="Apellidos del autor (citation.authors[0].family_names).",
    )
    author_affiliation: Optional[str] = Field(
        default="University of Huelva",
        description="Afiliación del autor (citation.authors[0].affiliation).",
    )
    message: Optional[str] = Field(
        default="If you use this dataset, please cite it as below.",
        description="Mensaje de cita en CITATION.cff (citation.message).",
    )
    repository_code: Optional[str] = Field(
        default="https://github.com/wildintelproject/donadataset",
        description="URL del repositorio de código fuente (citation.repository_code).",
    )
    repo_id: Optional[str] = Field(
        default=None,
        description="Repositorio de HuggingFace Hub, formato user_or_org/dataset (huggingface.repo_id).",
    )
    token: Optional[str] = Field(
        default=None,
        description=(
            "Token de acceso de HuggingFace Hub (permiso write) — "
            "https://huggingface.co/settings/tokens. La variable de entorno HF_TOKEN, si "
            "está definida, tiene prioridad sobre este valor. (huggingface.token)"
        ),
        json_schema_extra={"secret": True},
    )

    @model_validator(mode="after")
    def _derive_dataset_name(self) -> "HuggingFaceSettings":
        """Si dataset_name no se ha fijado (a mano, o ya en un settings.toml
        existente), se rellena a partir de dataset_slug — así solo hay que
        elegir un valor (el slug) para tener ambos, en vez de mantener dos
        campos casi redundantes en sincronía."""
        if not self.dataset_name and self.dataset_slug:
            self.dataset_name = _slug_to_dataset_name(self.dataset_slug)
        return self


class ZenodoSettings(BaseModel):
    """Valores por defecto reutilizados entre ejecuciones de 'publish zenodo
    prepare/upload/check-readiness/release'. El nombre de la variable de
    entorno (ZENODO_TOKEN) sigue fijo en templates/Zenodo.yaml.j2, no es un
    setting — pero su VALOR puede guardarse aquí como alternativa a
    exportarla cada sesión (ver el campo 'token' más abajo; la variable de
    entorno, si está definida, siempre gana). output_dir tampoco está aquí:
    sigue el mismo patrón que HuggingFace, cuyo --output-dir se calcula con
    una fórmula fija, no se lee de settings.toml."""
    environment: Optional[Literal["sandbox", "production"]] = Field(
        default="sandbox",
        description="Entorno de Zenodo: 'sandbox' (sandbox.zenodo.org, pruebas) o 'production' (DOI real). (zenodo.environment)",
    )
    sync_existing_draft: Optional[bool] = Field(
        default=False,
        description=(
            "No crear un depósito nuevo en 'zenodo prepare': sincronizar los ficheros de "
            "evidencia con un draft ya existente (lee el deposition_id de "
            "zenodo_linked_dataset_record.json)."
        ),
    )
    token: Optional[str] = Field(
        default=None,
        description=(
            "Token de acceso de Zenodo (o de Zenodo Sandbox si environment=sandbox) — "
            "https://zenodo.org/account/settings/applications/tokens/new/. La variable de "
            "entorno ZENODO_TOKEN, si está definida, tiene prioridad sobre este valor. "
            "(zenodo.token)"
        ),
        json_schema_extra={"secret": True},
    )


class B2ShareSettings(BaseModel):
    """Valores por defecto reutilizados entre ejecuciones de 'publish b2share
    prepare/check-readiness/release/sync-pid'. El nombre de la variable de
    entorno (B2SHARE_TOKEN) sigue fijo en templates/B2SHARE.yaml.j2, no es un
    setting — pero su VALOR puede guardarse aquí como alternativa a
    exportarla cada sesión (ver el campo 'token' más abajo; la variable de
    entorno, si está definida, siempre gana). output_dir tampoco está aquí:
    mismo patrón que HuggingFace/Zenodo."""
    environment: Optional[Literal["sandbox", "production"]] = Field(
        default="sandbox",
        description=(
            "Entorno de B2SHARE: 'sandbox' (trng-b2share.eudat.eu, pruebas) o "
            "'production' (b2share.eudat.eu, PID/DOI real). (b2share.environment)"
        ),
    )
    community_id: Optional[str] = Field(
        default=None,
        description=(
            "UUID de la comunidad EUDAT B2SHARE a la que pertenece el registro. "
            "Solicítala a EUDAT (no se puede adivinar). (b2share.community)"
        ),
    )
    sync_existing_draft: Optional[bool] = Field(
        default=False,
        description=(
            "No crear un draft nuevo en 'b2share prepare': sincronizar los ficheros de "
            "evidencia con uno ya existente (lee el record_id de "
            "b2share_linked_dataset_record.json)."
        ),
    )
    token: Optional[str] = Field(
        default=None,
        description=(
            "Token de acceso de B2SHARE (o de su sandbox si environment=sandbox) — "
            "generado desde tu perfil en b2share.eudat.eu. La variable de entorno "
            "B2SHARE_TOKEN, si está definida, tiene prioridad sobre este valor. (b2share.token)"
        ),
        json_schema_extra={"secret": True},
    )


class GBIFSettings(BaseModel):
    """Valores por defecto reutilizados entre ejecuciones de 'publish gbif
    prepare/register'. 'prepare' genera un paquete Camtrap DP (datapackage.json
    + deployments/media/observations.csv) — estos campos alimentan su
    metadata (title/description/license/contributors...); 'register' además
    usa environment/*_key/registry_language para hablar con la Registry API
    de GBIF directamente (sin pasar por un IPT — ver
    donadataset.services.gbif). Los nombres de las variables de entorno
    (GBIF_USERNAME/GBIF_PASSWORD) se quedan fijos en services/gbif.py, igual
    que HF_TOKEN — pero sus VALORES pueden guardarse aquí como alternativa a
    exportarlas cada sesión (campos 'username'/'password' más abajo; las
    variables de entorno, si están definidas, siempre ganan). La Registry
    API usa Basic Auth, no un token único."""
    environment: Optional[Literal["sandbox", "production"]] = Field(
        default="sandbox",
        description=(
            "Entorno de la Registry API de GBIF usado por 'gbif register': 'sandbox' "
            "(api.gbif-test.org, pruebas) o 'production' (api.gbif.org, dataset real). "
            "(gbif.environment)"
        ),
    )
    publishing_organization_key: Optional[str] = Field(
        default=None,
        description=(
            "UUID de tu organización ya registrada en GBIF (gbif.org o gbif-test.org). "
            "No se puede adivinar. (gbif.publishing_organization_key)"
        ),
    )
    installation_key: Optional[str] = Field(
        default=None,
        description=(
            "UUID de tu instalación ya registrada en GBIF (no tiene por qué ser un IPT — "
            "vale una 'Test/HTTP installation'). No se puede adivinar. (gbif.installation_key)"
        ),
    )
    registry_language: Optional[str] = Field(
        default="eng",
        description=(
            "Código de idioma (ISO 639-2/T, ej. eng/spa) que exige el campo 'language' "
            "de la Registry API al registrar el dataset. (gbif.registry_language)"
        ),
    )
    dataset_name: Optional[str] = Field(
        default="DonaDataset",
        description="Nombre del dataset, usado como 'title' del datapackage.json (gbif.dataset_name).",
    )
    description: Optional[str] = Field(
        default=(
            "Camera-trap biodiversity data for mammal species from Doñana National Park, "
            "Spain, in Camtrap DP format."
        ),
        description="Descripción del dataset en el datapackage.json (gbif.description).",
    )
    license_id: Optional[str] = Field(
        default="CC-BY-4.0", description="Identificador de licencia, ej. CC-BY-4.0 (gbif.license_id).",
    )
    license_name: Optional[str] = Field(
        default="Creative Commons Attribution 4.0 International",
        description="Nombre completo de la licencia (gbif.license_name).",
    )
    license_url: Optional[str] = Field(
        default="https://creativecommons.org/licenses/by/4.0/",
        description="URL de la licencia, usada en licenses[].path del datapackage.json (gbif.license_url).",
    )
    rights_holder: Optional[str] = Field(
        default="University of Huelva",
        description="Titular de los derechos sobre los datos, usado como contributors[].organization (gbif.rights_holder).",
    )
    institution_code: Optional[str] = Field(
        default="UHU",
        description="Código de institución publicadora (gbif.institution_code).",
    )
    contact_name: Optional[str] = Field(
        default="WildINTEL",
        description="Nombre del contacto técnico del dataset, usado como contributors[].title (gbif.contact_name).",
    )
    contact_email: Optional[str] = Field(
        default=None,
        description=(
            "Email del contacto técnico del dataset — no se puede adivinar, hay que "
            "rellenarlo. (gbif.contact_email)"
        ),
    )
    classified_by: Optional[str] = Field(
        default="DonaDataset YOLO pipeline",
        description=(
            "Quién/qué generó las clasificaciones (observations.classifiedBy) — un "
            "modelo, no una persona, así que classificationMethod siempre es 'machine'. "
            "(gbif.classified_by)"
        ),
    )
    username: Optional[str] = Field(
        default=None,
        description=(
            "Usuario de tu cuenta en gbif.org (o gbif-test.org si environment=sandbox), "
            "para la Registry API (Basic Auth). La variable de entorno GBIF_USERNAME, si "
            "está definida, tiene prioridad sobre este valor. (gbif.username)"
        ),
        json_schema_extra={"secret": True},
    )
    password: Optional[str] = Field(
        default=None,
        description=(
            "Contraseña de esa misma cuenta de gbif.org, para la Registry API (Basic "
            "Auth). La variable de entorno GBIF_PASSWORD, si está definida, tiene "
            "prioridad sobre este valor. (gbif.password)"
        ),
        json_schema_extra={"secret": True},
    )


class Settings(BaseModel):
    GENERATE: GenerateSettings = Field(default_factory=GenerateSettings)
    GENERATE_TOY: GenerateToySettings = Field(default_factory=GenerateToySettings)
    HUGGINGFACE: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    ZENODO: ZenodoSettings = Field(default_factory=ZenodoSettings)
    B2SHARE: B2ShareSettings = Field(default_factory=B2ShareSettings)
    GBIF: GBIFSettings = Field(default_factory=GBIFSettings)


def _ensure_config_file(config_file: Path) -> None:
    """Crea config_file con los valores por defecto si todavía no existe."""
    if config_file.exists():
        return
    config_file.parent.mkdir(parents=True, exist_ok=True)
    defaults = Settings().model_dump(mode="json")
    loaders.toml_loader.write(str(config_file), defaults, merge=False)


def load_settings(config_file: Path = DEFAULT_CONFIG_FILE) -> Settings:
    _ensure_config_file(config_file)
    dynaconf_settings = Dynaconf(settings_files=[str(config_file)], envvar_prefix="DONADATASET")
    return Settings.model_validate(dynaconf_settings.to_dict())


settings = load_settings()
