"""CLI para inspeccionar y editar el fichero de configuración (settings.toml)."""
from typing import Any, get_origin

import typer
from dynaconf import loaders
from pydantic import TypeAdapter, ValidationError
from rich.console import Console
from rich.table import Table

from donadataset.config import DEFAULT_CONFIG_FILE, Settings, load_settings

console = Console()
app     = typer.Typer(help="Inspecciona o edita el fichero de configuración (settings.toml).")


def _split_key(key: str) -> tuple[str, str]:
    if "." not in key:
        raise typer.BadParameter(
            f"'{key}' debe tener el formato SECCION.CLAVE (ej. GENERATE.source)."
        )
    section, field = key.split(".", 1)
    return section, field


CONFIG_SECTIONS = ("GENERATE", "GENERATE_TOY", "HUGGINGFACE", "ZENODO", "B2SHARE", "GBIF")


def _get_section_model(settings: Settings, section: str):
    if not hasattr(settings, section):
        raise typer.BadParameter(f"Sección desconocida: '{section}'. Usa una de: {', '.join(CONFIG_SECTIONS)}.")
    return getattr(settings, section)


def _resolve_field(section_model, section: str, field: str):
    """Valida que `field` existe en la sección y devuelve su FieldInfo (tipo, descripción...)."""
    if field not in type(section_model).model_fields:
        raise typer.BadParameter(f"Parámetro desconocido: '{section}.{field}'.")
    return type(section_model).model_fields[field]


def _parse_value(annotation: Any, raw: str) -> Any:
    """Convierte el string recibido por CLI al tipo del campo (list/dict admiten CSV)."""
    origin = get_origin(annotation)
    if origin is list:
        items = [v.strip() for v in raw.split(",") if v.strip()]
        return TypeAdapter(annotation).validate_python(items)
    if origin is dict:
        pairs = (pair.split("=", 1) for pair in raw.split(",") if pair.strip())
        raw_dict = {k.strip(): v.strip() for k, v in pairs}
        return TypeAdapter(annotation).validate_python(raw_dict)
    return TypeAdapter(annotation).validate_python(raw)


def _apply_update(settings: Settings, section: str, field: str, value: Any) -> Settings:
    """Devuelve una nueva Settings con section.field = value, totalmente revalidada."""
    section_model = _get_section_model(settings, section)
    new_section = section_model.model_copy(update={field: value})
    new_settings = settings.model_copy(update={section: new_section})
    return Settings.model_validate(new_settings.model_dump(mode="json"))


SECRET_MASK = "••••••••"


def _is_secret_field(section_model, field: str) -> bool:
    """True para campos marcados con json_schema_extra={"secret": True} en
    config.py (tokens, usuario/contraseña) — nunca se muestran en claro por
    'show'/el resumen del wizard, ni se piden con eco en pantalla."""
    field_info = type(section_model).model_fields[field]
    extra = field_info.json_schema_extra
    return bool(isinstance(extra, dict) and extra.get("secret"))


def _format_value(value: Any, secret: bool = False) -> str:
    if secret:
        return SECRET_MASK if value else "(sin definir)"
    return str(value)


def _prompt_field(section_model, field: str) -> Any:
    """Pide por consola el valor de un campo, mostrando el actual como default.

    Los campos list/dict se piden como una única línea CSV (mismo formato que
    admite `config set`); reintenta si el valor no valida. Los campos
    secretos (ver _is_secret_field) son un caso aparte: nunca se muestra el
    valor actual (ni siquiera como default, que Typer mostraría en claro
    entre corchetes), la entrada no hace eco en pantalla, y dejarlo en blanco
    significa "no cambiarlo" en vez de "borrarlo".
    """
    field_info = type(section_model).model_fields[field]
    current = getattr(section_model, field)
    annotation = field_info.annotation
    origin = get_origin(annotation)
    label = field_info.description or field

    if _is_secret_field(section_model, field):
        console.print(f"\n[bold]{field}[/bold] — {label}")
        console.print(f"  Valor actual: {'definido (no se muestra)' if current else 'sin definir'}")
        raw = typer.prompt(
            "  Nuevo valor (Enter para no cambiarlo)", default="", show_default=False, hide_input=True,
        )
        return current if raw == "" else raw

    if origin is list:
        default_str = ",".join(str(v) for v in current)
    elif origin is dict:
        default_str = ",".join(f"{k}={v}" for k, v in current.items())
    elif current is None:
        default_str = ""   # campos Optional sin valor — no mostrar el string "None" como default
    else:
        default_str = str(current)

    while True:
        console.print(f"\n[bold]{field}[/bold] — {label}")
        raw = typer.prompt("  Valor", default=default_str)
        if raw == "" and current is None:
            return None    # no se ha escrito nada y no había valor previo — se deja sin definir
        try:
            return _parse_value(annotation, raw)
        except ValidationError as e:
            console.print(f"  [red]Valor inválido:[/red] {e}")


def _print_section(title: str) -> None:
    console.print()
    console.print(f"[cyan]{'─' * 60}[/cyan]")
    console.print(f"[bold cyan]  {title}[/bold cyan]")
    console.print(f"[cyan]{'─' * 60}[/cyan]")


@app.command("show")
def config_show() -> None:
    """Muestra el contenido del fichero de configuración (los valores secretos aparecen enmascarados)."""
    settings = load_settings()  # lo crea con los valores por defecto si aún no existe
    console.print(f"[bold]{DEFAULT_CONFIG_FILE}[/bold]")
    for section in CONFIG_SECTIONS:
        section_model = getattr(settings, section)
        table = Table(title=f"[{section}]")
        table.add_column("Campo")
        table.add_column("Valor")
        for field in type(section_model).model_fields:
            secret = _is_secret_field(section_model, field)
            table.add_row(field, _format_value(getattr(section_model, field), secret))
        console.print(table)


@app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Parámetro en formato SECCION.CLAVE (ej. GENERATE.source)."),
) -> None:
    """Muestra el valor de un único parámetro."""
    settings = load_settings()
    section, field = _split_key(key)
    section_model = _get_section_model(settings, section)
    _resolve_field(section_model, section, field)
    console.print(getattr(section_model, field))


@app.command("set")
def config_set(
    assignment: str = typer.Argument(
        ...,
        help=(
            "Asignación en formato SECCION.CLAVE=VALOR (ej. GENERATE.duplicate_key_mode=relative_stem), "
            "o SECCION.CLAVE a secas para campos secretos (ej. HUGGINGFACE.token) — pide el valor con "
            "entrada oculta en vez de tenerlo que escribir en claro en el comando."
        ),
    ),
) -> None:
    """Cambia el valor de un parámetro y lo guarda en el fichero de configuración."""
    settings = load_settings()

    if "=" in assignment:
        key, raw_value = assignment.split("=", 1)
        section, field = _split_key(key)
        section_model = _get_section_model(settings, section)
        field_info = _resolve_field(section_model, section, field)
    else:
        section, field = _split_key(assignment)
        section_model = _get_section_model(settings, section)
        field_info = _resolve_field(section_model, section, field)
        if not _is_secret_field(section_model, field):
            raise typer.BadParameter("Formato esperado: SECCION.CLAVE=VALOR (ej. GENERATE.random_seed=7).")
        raw_value = typer.prompt(f"Valor de {section}.{field}", hide_input=True)

    try:
        parsed_value = _parse_value(field_info.annotation, raw_value)
        new_settings = _apply_update(settings, section, field, parsed_value)
    except ValidationError as e:
        console.print(f"[red]✘  Valor inválido para {section}.{field}:[/red]\n{e}")
        raise typer.Exit(1)

    loaders.toml_loader.write(
        str(DEFAULT_CONFIG_FILE), new_settings.model_dump(mode="json"), merge=False,
    )
    secret = _is_secret_field(section_model, field)
    new_value = getattr(getattr(new_settings, section), field)
    console.print(f"[green]✔  {section}.{field} = {_format_value(new_value, secret)}[/green]")


@app.command("wizard")
def config_wizard() -> None:
    """Asistente interactivo: pregunta por cada parámetro y guarda los cambios."""
    settings = load_settings()

    console.print(
        "[bold]Asistente de configuración de DonaDataset[/bold]\n"
        "Pulsa Enter en cualquier pregunta para mantener el valor actual."
    )

    new_settings = settings
    for section in CONFIG_SECTIONS:
        _print_section(section)
        section_model = getattr(new_settings, section)
        for field in type(section_model).model_fields:
            value = _prompt_field(section_model, field)
            new_settings = _apply_update(new_settings, section, field, value)
            section_model = getattr(new_settings, section)  # refleja el cambio ya aplicado

    _print_section("Resumen de cambios")
    table = Table()
    table.add_column("Parámetro")
    table.add_column("Valor anterior")
    table.add_column("Valor nuevo")
    changed = False
    for section in CONFIG_SECTIONS:
        old_model = getattr(settings, section)
        new_model = getattr(new_settings, section)
        for field in type(old_model).model_fields:
            old_value = getattr(old_model, field)
            new_value = getattr(new_model, field)
            if old_value != new_value:
                changed = True
                secret = _is_secret_field(old_model, field)
                table.add_row(f"{section}.{field}", _format_value(old_value, secret), _format_value(new_value, secret))

    if not changed:
        console.print("[yellow]No se ha cambiado ningún valor.[/yellow]")
        raise typer.Exit(0)

    console.print(table)
    if not typer.confirm("\n¿Guardar esta configuración?", default=True):
        console.print("[yellow]Cancelado. No se ha guardado nada.[/yellow]")
        raise typer.Exit(0)

    loaders.toml_loader.write(
        str(DEFAULT_CONFIG_FILE), new_settings.model_dump(mode="json"), merge=False,
    )
    console.print(f"[green]✔  Configuración guardada en {DEFAULT_CONFIG_FILE}[/green]")
