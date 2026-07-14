"""CLI para inspeccionar y editar la sección B2SHARE de settings.toml.

Envoltorio fino sobre donadataset.commands.config_commands (mismo motor:
Settings de Pydantic, misma validación, mismo fichero) — evita tener que
escribir el prefijo 'B2SHARE.' en cada get/set, y el wizard solo pregunta por
estos campos. No gestiona ningún YAML propio — 'b2share
prepare'/'check-readiness'/'release'/'sync-pid' siempre leen la plantilla
incluida en el proyecto (templates/B2SHARE.yaml.j2), rellenada con estos
mismos valores.
"""
import typer
from dynaconf import loaders
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from donadataset.commands.config_commands import (
    _apply_update,
    _format_value,
    _is_secret_field,
    _parse_value,
    _prompt_field,
    _resolve_field,
)
from donadataset.config import DEFAULT_CONFIG_FILE, load_settings

console = Console()
app     = typer.Typer(help="Inspecciona o edita la sección B2SHARE de settings.toml.")

SECTION = "B2SHARE"


@app.command("show")
def config_show() -> None:
    """Muestra los valores actuales de B2SHARE en settings.toml."""
    settings = load_settings()
    section_model = settings.B2SHARE

    table = Table(title=f"settings.toml [{SECTION}]")
    table.add_column("Campo")
    table.add_column("Valor")
    for field in type(section_model).model_fields:
        secret = _is_secret_field(section_model, field)
        table.add_row(field, _format_value(getattr(section_model, field), secret))
    console.print(table)


@app.command("get")
def config_get(
    field: str = typer.Argument(..., help="Nombre del campo, ej. community_id (sin prefijo B2SHARE.)."),
) -> None:
    """Muestra el valor de un único campo."""
    settings = load_settings()
    section_model = settings.B2SHARE
    _resolve_field(section_model, SECTION, field)
    console.print(getattr(section_model, field))


@app.command("set")
def config_set(
    assignment: str = typer.Argument(
        ...,
        help=(
            "Asignación CAMPO=VALOR (ej. community_id=<uuid>), o CAMPO a secas para "
            "campos secretos — pide el valor con entrada oculta en vez de "
            "escribirlo en claro en el comando."
        ),
    ),
) -> None:
    """Cambia el valor de un campo de B2SHARE y lo guarda en settings.toml."""
    settings = load_settings()
    section_model = settings.B2SHARE

    if "=" in assignment:
        field, raw_value = assignment.split("=", 1)
        field_info = _resolve_field(section_model, SECTION, field)
    else:
        field = assignment
        field_info = _resolve_field(section_model, SECTION, field)
        if not _is_secret_field(section_model, field):
            raise typer.BadParameter("Formato esperado: CAMPO=VALOR (ej. community_id=<uuid>).")
        raw_value = typer.prompt(f"Valor de {field}", hide_input=True)

    try:
        parsed_value = _parse_value(field_info.annotation, raw_value)
        new_settings = _apply_update(settings, SECTION, field, parsed_value)
    except ValidationError as e:
        console.print(f"[red]✘  Valor inválido para {field}:[/red]\n{e}")
        raise typer.Exit(1)

    loaders.toml_loader.write(str(DEFAULT_CONFIG_FILE), new_settings.model_dump(mode="json"), merge=False)
    secret = _is_secret_field(section_model, field)
    console.print(f"[green]✔  {SECTION}.{field} = {_format_value(getattr(new_settings.B2SHARE, field), secret)}[/green]")


@app.command("wizard")
def config_wizard() -> None:
    """Asistente interactivo: pregunta solo por los campos de B2SHARE."""
    settings = load_settings()

    console.print(
        "[bold]Asistente de configuración de B2SHARE[/bold]\n"
        "Pulsa Enter en cualquier pregunta para mantener el valor actual."
    )

    new_settings = settings
    section_model = new_settings.B2SHARE
    for field in type(section_model).model_fields:
        value = _prompt_field(section_model, field)
        new_settings = _apply_update(new_settings, SECTION, field, value)
        section_model = new_settings.B2SHARE

    table = Table(title="Resumen de cambios")
    table.add_column("Campo")
    table.add_column("Valor anterior")
    table.add_column("Valor nuevo")
    changed = False
    old_model = settings.B2SHARE
    new_model = new_settings.B2SHARE
    for field in type(old_model).model_fields:
        old_value = getattr(old_model, field)
        new_value = getattr(new_model, field)
        if old_value != new_value:
            changed = True
            secret = _is_secret_field(old_model, field)
            table.add_row(field, _format_value(old_value, secret), _format_value(new_value, secret))

    if not changed:
        console.print("[yellow]No se ha cambiado ningún valor.[/yellow]")
        raise typer.Exit(0)

    console.print(table)
    if not typer.confirm("\n¿Guardar esta configuración?", default=True):
        console.print("[yellow]Cancelado. No se ha guardado nada.[/yellow]")
        raise typer.Exit(0)

    loaders.toml_loader.write(str(DEFAULT_CONFIG_FILE), new_settings.model_dump(mode="json"), merge=False)
    console.print(f"[green]✔  Configuración guardada en {DEFAULT_CONFIG_FILE}[/green]")
