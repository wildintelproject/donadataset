"""
DonaDataset — CLI de generación y publicación.

Uso:
    uv run donadataset generate real --source ... --output ...
    uv run donadataset generate toy
    uv run donadataset publish huggingface prepare --config HuggingFaceHub.yaml
    uv run donadataset publish zenodo prepare --config HuggingFaceHub.yaml
    uv run donadataset publish all                          # todo, en orden, de un tirón
    uv run donadataset publish all --exclude b2share,gbif    # todo menos estos
"""
import typer

from donadataset.commands import b2share as b2share_cmd
from donadataset.commands import config_commands
from donadataset.commands import gbif as gbif_cmd
from donadataset.commands import generate as generate_cmd
from donadataset.commands import huggingface as huggingface_cmd
from donadataset.commands import publish_all as publish_all_cmd
from donadataset.commands import zenodo as zenodo_cmd

app = typer.Typer(help="DonaDataset — genera y publica el dataset.")
app.add_typer(generate_cmd.app, name="generate")
# Mismo comando que 'generate config' (settings.toml cubre GENERATE/GENERATE_TOY/
# HUGGINGFACE, no solo 'generate') — se deja aquí también para no depender de
# 'generate' para editar ajustes que ya no son solo suyos.
app.add_typer(config_commands.app, name="config")

publish_app = typer.Typer(help="Publica el dataset en distintos repositorios.")
app.add_typer(publish_app, name="publish")
publish_app.add_typer(huggingface_cmd.app, name="huggingface")
publish_app.add_typer(huggingface_cmd.app, name="hfh", help="Alias de 'huggingface'.", hidden=True)
publish_app.add_typer(zenodo_cmd.app, name="zenodo")
publish_app.add_typer(b2share_cmd.app, name="b2share")
publish_app.add_typer(gbif_cmd.app, name="gbif")
publish_app.command("all")(publish_all_cmd.publish_all)


if __name__ == "__main__":
    app()
