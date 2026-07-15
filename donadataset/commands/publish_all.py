"""Comando CLI 'publish all' — publica en todos los repositorios soportados, en orden.

Orquesta los 'pipeline' (o pasos individuales) que cada integración ya
expone, re-invocando el propio CLI como subproceso (mismo intérprete,
`python -m donadataset.main ...`) en vez de llamar a las funciones Typer
directamente — así cada paso resuelve sus valores por defecto exactamente
igual que si el usuario lo tecleara a mano (settings.toml ya configurado con
'donadataset publish <repo> config set ...'), sin duplicar esa lógica aquí.

HuggingFace Hub va siempre primero: Zenodo/B2SHARE se enlazan a una copia
pública de HFH ya subida, y GBIF puede alojar su .zip ahí también. El paso
'zenodo sync-doi' ya re-sube a HuggingFace Hub por sí solo (solo
CITATION.cff y el fichero de checksums), así que Zenodo no necesita ningún
paso extra aquí para reflejar su DOI — a diferencia de B2SHARE, que sí
requiere una re-subida completa aparte tras publicar su PID.
"""
import subprocess
import sys
from typing import List, Optional

import typer
from rich.console import Console

console = Console()

ALL_REPOS = ["huggingface", "zenodo", "b2share", "gbif"]


def _parse_repo_list(value: Optional[str]) -> set:
    if not value:
        return set()
    names = {item.strip() for item in value.split(",") if item.strip()}
    invalid = names - set(ALL_REPOS)
    if invalid:
        raise typer.BadParameter(
            f"Repositorio(s) desconocido(s): {sorted(invalid)}. Usa: {', '.join(ALL_REPOS)}."
        )
    return names


def _resolve_repos(include: Optional[str], exclude: Optional[str]) -> List[str]:
    """Por defecto, todos. --exclude quita de la lista; --include siempre
    gana — un repo en ambas listas a la vez se publica igualmente."""
    whitelist = _parse_repo_list(include)
    blacklist = _parse_repo_list(exclude)
    selected = (set(ALL_REPOS) - blacklist) | whitelist
    return [repo for repo in ALL_REPOS if repo in selected]


def _run_step(step_label: str, args: List[str], dry_run: bool) -> None:
    command = [sys.executable, "-m", "donadataset.main", *args]
    console.print(f"\n[bold cyan]── {step_label} ──[/bold cyan]")
    if dry_run:
        console.print(f"  {' '.join(command)}")
        return

    result = subprocess.run(command)
    if result.returncode != 0:
        console.print(f"[red]✘  {step_label} falló (código {result.returncode}).[/red]")
        raise typer.Exit(result.returncode)


def publish_all(
    include: Optional[str] = typer.Option(
        None, "--include",
        help=(
            "Lista blanca separada por comas (huggingface,zenodo,b2share,gbif) — estos "
            "SIEMPRE se publican, incluso si también están en --exclude."
        ),
    ),
    exclude: Optional[str] = typer.Option(
        None, "--exclude",
        help=(
            "Lista negra separada por comas — estos NUNCA se publican (salvo que también "
            "estén en --include). Por defecto no se excluye nada: se publica en todos."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Muestra el orden y los comandos exactos que se ejecutarían, sin ejecutar nada.",
    ),
) -> None:
    """Publica el dataset en todos los repositorios soportados, en un único pipeline.

    Orden fijo: HuggingFace Hub -> Zenodo -> B2SHARE -> GBIF (los tres
    últimos dependen de que HFH ya esté publicado). Cada paso reutiliza los
    valores ya guardados con 'donadataset publish <repo> config set ...' —
    no hace falta pasar ningún flag aquí si ya configuraste cada integración
    una vez.

    Intervención manual: HuggingFace Hub no tiene API para generar su propio
    DOI, así que su 'pipeline' se detiene una vez, pidiéndote que lo generes
    a mano en la web y pulses Enter — es la única pausa de todo el proceso.
    """
    repos = _resolve_repos(include, exclude)
    if not repos:
        console.print("[yellow]No hay ningún repositorio seleccionado para publicar.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Orden de publicación:[/bold] {' -> '.join(repos)}")

    if "huggingface" in repos:
        _run_step("HuggingFace Hub", ["publish", "huggingface", "pipeline"], dry_run)

    if "zenodo" in repos:
        _run_step("Zenodo · prepare", ["publish", "zenodo", "prepare"], dry_run)
        _run_step("Zenodo · upload", ["publish", "zenodo", "upload"], dry_run)
        _run_step(
            "Zenodo · sync-doi (refleja el DOI en HuggingFace Hub, re-sube automáticamente)",
            ["publish", "zenodo", "sync-doi"], dry_run,
        )
        _run_step("Zenodo · check-readiness", ["publish", "zenodo", "check-readiness"], dry_run)
        _run_step("Zenodo · release", ["publish", "zenodo", "release"], dry_run)

    if "b2share" in repos:
        _run_step("B2SHARE", ["publish", "b2share", "pipeline"], dry_run)
        _run_step(
            "HuggingFace Hub · re-subida automática (refleja el PID de B2SHARE)",
            ["publish", "huggingface", "upload"], dry_run,
        )

    if "gbif" in repos:
        _run_step("GBIF", ["publish", "gbif", "pipeline"], dry_run)

    console.print("\n[bold green]✔  Publicación completa en todos los repositorios seleccionados.[/bold green]")
