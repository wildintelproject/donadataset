"""
DonaDataset — CLI de gestión.

Uso:
    uv run cli docs serve
    uv run cli docs build
    uv run cli docs pdf
    uv run cli test [unit|integration|all]
"""
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

ROOT_DIR   = Path(__file__).parent
MKDOCS_CFG = ROOT_DIR / "mkdocs.yml"
SITE_DIR   = ROOT_DIR / "site"

# Tiene que coincidir con plugins.with-pdf.enabled_if_env y .output_path en
# mkdocs.yml — el plugin genera el PDF durante 'mkdocs build' solo si esta
# variable vale "1"; el resto de builds (docs serve/build normales) no pagan
# el coste de renderizar cada página con WeasyPrint.
PDF_ENV_VAR  = "ENABLE_PDF_EXPORT"
PDF_REL_PATH = "pdf/donadataset.pdf"

console = Console()
app     = typer.Typer(help="DonaDataset — herramienta de gestión.")


class TestSuite(str, Enum):
    unit        = "unit"
    integration = "integration"
    all         = "all"


def _run(*args: str) -> None:
    result = subprocess.run(list(args), cwd=ROOT_DIR)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


# ── docs (manual del dataset) ────────────────────────────────────────────────

docs_app = typer.Typer(help="Genera o sirve la documentación del dataset.")
app.add_typer(docs_app, name="docs")


@docs_app.command("serve")
def docs_serve(
    port: int = typer.Option(8000, "--port", "-p", help="Puerto del servidor de documentación."),
) -> None:
    """Sirve la documentación en local con recarga automática (http://127.0.0.1:<port>)."""
    console.print(f"[green]Documentación en http://127.0.0.1:{port}[/green]\n")
    _run("mkdocs", "serve", "--config-file", str(MKDOCS_CFG), "--dev-addr", f"127.0.0.1:{port}")


@docs_app.command("build")
def docs_build() -> None:
    """Genera el sitio estático de la documentación en site/."""
    console.print("[green]Generando documentación...[/green]")
    _run("mkdocs", "build", "--config-file", str(MKDOCS_CFG))
    console.print(f"[green]✔  Sitio generado en {SITE_DIR}[/green]")


@docs_app.command("pdf")
def docs_pdf() -> None:
    """Genera un único PDF con toda la documentación (site/pdf/donadataset.pdf).

    Es el mismo 'mkdocs build' de siempre, con el plugin mkdocs-with-pdf
    activado solo para esta ejecución (vía la variable de entorno
    ENABLE_PDF_EXPORT) — recorre el nav de mkdocs.yml en orden y renderiza
    cada página con WeasyPrint, con portada e índice incluidos.
    """
    console.print("[green]Generando documentación en PDF (WeasyPrint)...[/green]")
    env = {**os.environ, PDF_ENV_VAR: "1"}
    result = subprocess.run(["mkdocs", "build", "--config-file", str(MKDOCS_CFG)], cwd=ROOT_DIR, env=env)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)

    pdf_path = SITE_DIR / PDF_REL_PATH
    if not pdf_path.is_file():
        console.print(f"[red]✘  mkdocs build terminó bien pero no encuentro {pdf_path}.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✔  PDF generado en {pdf_path}[/green]")


# ── test ──────────────────────────────────────────────────────────────────────

@app.command("test")
def test(
    suite: TestSuite = typer.Argument(TestSuite.all, help="Suite a ejecutar."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Salida detallada (-v de pytest)."),
    keyword: Optional[str] = typer.Option(None, "--keyword", "-k", help="Filtro de tests por nombre (-k de pytest)."),
) -> None:
    """Ejecuta los tests del proyecto con pytest."""
    paths = {
        TestSuite.unit: ["tests/unit/"],
        TestSuite.integration: ["tests/integration/"],
        TestSuite.all: ["tests/unit/", "tests/integration/"],
    }[suite]

    console.print(Panel(
        f"[bold]Suite:[/bold] {suite.value}   [bold]Paths:[/bold] {', '.join(paths)}",
        title="DonaDataset — tests",
    ))

    cmd = [sys.executable, "-m", "pytest", *paths]
    if verbose:
        cmd.append("-v")
    if keyword:
        cmd.extend(["-k", keyword])

    _run(*cmd)


if __name__ == "__main__":
    app()
