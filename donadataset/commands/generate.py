"""Genera el dataset (imágenes + labels) a partir de las fuentes originales."""
import random
import shutil
from collections import defaultdict
from enum import Enum
from pathlib import Path

import typer
import yaml
from rich.console import Console

from donadataset.commands import config_commands
from donadataset.config import settings

console = Console()
app     = typer.Typer(help="Genera el dataset (real o toy) a partir de las fuentes originales.")
app.add_typer(config_commands.app, name="config")

# Extensiones de imagen reconocidas / orden de preferencia ante duplicados.
IMAGE_EXTENSIONS   = set(settings.GENERATE.image_extensions)
EXTENSION_PRIORITY = settings.GENERATE.extension_priority


class Split(str, Enum):
    train = "train"
    val   = "val"
    test  = "test"


class DuplicateKeyMode(str, Enum):
    # Considera duplicadas las imágenes con el mismo nombre base dentro del split,
    # aunque estén en subdirectorios distintos.
    stem = "stem"
    # Considera duplicadas solo si tienen la misma ruta relativa sin extensión.
    relative_stem = "relative_stem"


DEFAULT_SOURCE      = settings.GENERATE.source
DEFAULT_OUTPUT      = settings.GENERATE.output
DEFAULT_SPLITS      = [Split(s) for s in settings.GENERATE.splits]
DEFAULT_REMOVE_IDS  = settings.GENERATE.remove_class_ids
DEFAULT_DUP_MODE    = DuplicateKeyMode(settings.GENERATE.duplicate_key_mode)
DEFAULT_CLASSES_MAP = settings.GENERATE.classes_map

# Nombre fijo del YAML que "real" escribe en su salida y que "toy" espera como fuente.
SOURCE_YAML_FILENAME = "donana_filtered.yaml"
# Nombre fijo del YAML que "toy" escribe en su propia salida.
TOY_YAML_FILENAME    = "donadatasetToy.yaml"

DEFAULT_TOY_SOURCE     = settings.GENERATE_TOY.source
DEFAULT_TOY_OUTPUT     = settings.GENERATE_TOY.output
DEFAULT_TOY_SPLITS     = [Split(s) for s in settings.GENERATE_TOY.splits]
DEFAULT_SAMPLES        = settings.GENERATE_TOY.samples_per_class
DEFAULT_RANDOM_SEED    = settings.GENERATE_TOY.random_seed


# ── Esquema de clases ────────────────────────────────────────────────────────

def load_original_names(classes_map: Path) -> dict[int, str]:
    with classes_map.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data["classes"].items()}


def build_class_mapping(
    original_names: dict[int, str], remove_ids: set[int]
) -> tuple[dict[int, int], dict[int, str]]:
    """Mapea índice antiguo -> índice nuevo, eliminando remove_ids y dejando ids consecutivos."""
    kept_classes = [old_id for old_id in sorted(original_names) if old_id not in remove_ids]
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_classes)}
    new_names  = {new_id: original_names[old_id] for old_id, new_id in old_to_new.items()}
    return old_to_new, new_names


# ── Labels YOLO ───────────────────────────────────────────────────────────────

def read_yolo_label(label_path: Path) -> list[list[str]]:
    if not label_path.exists():
        return []
    lines = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line.split())
    return lines


def label_contains_removed_class(label_lines: list[list[str]], remove_ids: set[int]) -> bool:
    return any(int(tokens[0]) in remove_ids for tokens in label_lines)


def remap_label_lines(label_lines: list[list[str]], old_to_new: dict[int, int]) -> list[str]:
    remapped = []
    for tokens in label_lines:
        old_class_id = int(tokens[0])
        if old_class_id not in old_to_new:
            raise ValueError(
                f"Clase {old_class_id} no encontrada en el mapeo. "
                "Esto no debería ocurrir si las clases eliminadas ya fueron filtradas."
            )
        tokens[0] = str(old_to_new[old_class_id])
        remapped.append(" ".join(tokens))
    return remapped


# ── Selección de imágenes ────────────────────────────────────────────────────

def find_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        (p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )


def get_label_path_for_image(image_path: Path, src_images_dir: Path, src_labels_dir: Path) -> Path:
    relative_image_path = image_path.relative_to(src_images_dir)
    return src_labels_dir / relative_image_path.with_suffix(".txt")


def get_duplicate_key(image_path: Path, src_images_dir: Path, mode: DuplicateKeyMode) -> str:
    if mode == DuplicateKeyMode.stem:
        return image_path.stem.lower()
    relative_image_path = image_path.relative_to(src_images_dir)
    return relative_image_path.with_suffix("").as_posix().lower()


def extension_rank(image_path: Path) -> int:
    suffix = image_path.suffix.lower()
    if suffix in EXTENSION_PRIORITY:
        return EXTENSION_PRIORITY.index(suffix)
    return len(EXTENSION_PRIORITY)


def select_unique_images(
    images: list[Path],
    src_images_dir: Path,
    src_labels_dir: Path,
    split: str,
    duplicate_key_mode: DuplicateKeyMode,
    quiet: bool = False,
) -> tuple[list[Path], int, int]:
    """Agrupa imágenes duplicadas y conserva solamente una por grupo.

    Criterios: 1) prioriza una imagen con label existente, 2) prioriza extensión
    según EXTENSION_PRIORITY, 3) orden alfabético como desempate.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        key = get_duplicate_key(image_path, src_images_dir, duplicate_key_mode)
        groups[key].append(image_path)

    selected_images: list[Path] = []
    duplicate_groups = 0
    duplicated_images_removed = 0

    def candidate_score(path: Path) -> tuple[int, int, str]:
        label_path = get_label_path_for_image(path, src_images_dir, src_labels_dir)
        has_label_penalty = 0 if label_path.exists() else 1
        return (has_label_penalty, extension_rank(path), str(path).lower())

    for key, group in sorted(groups.items()):
        # Same ordering used to pick the winner, so the printed candidate list
        # reflects the actual priority (label existence, then extension).
        ranked = sorted(group, key=candidate_score)
        selected = ranked[0]

        if len(group) > 1:
            duplicate_groups += 1
            duplicated_images_removed += len(group) - 1
            if not quiet:
                console.print(f"[yellow]\\[duplicadas][/yellow] split={split} key={key}")
                for candidate in ranked:
                    console.print(f"  candidate: {candidate}")
                console.print(f"  selected:  {selected}")

        selected_images.append(selected)

    return selected_images, duplicate_groups, duplicated_images_removed


# ── Pipeline principal ────────────────────────────────────────────────────────

def clean_output_dir(output_dir: Path, splits: list[str]) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def process_dataset(
    source: Path,
    output: Path,
    splits: list[str],
    remove_class_ids: set[int],
    duplicate_key_mode: DuplicateKeyMode,
    original_names: dict[int, str],
    quiet: bool = False,
) -> None:
    old_to_new, new_names = build_class_mapping(original_names, remove_class_ids)

    console.print(f"[yellow]⚠  Se va a vaciar el directorio de salida: {output.resolve()}[/yellow]")
    clean_output_dir(output, splits)

    stats = {
        split: {
            "total_images": 0,
            "duplicate_groups": 0,
            "duplicated_images_removed": 0,
            "kept_images": 0,
            "removed_images": 0,
            "missing_labels": 0,
            "removed_missing_labels": 0,
        }
        for split in splits
    }

    for split in splits:
        src_images_dir = source / "images" / split
        src_labels_dir = source / "labels" / split
        dst_images_dir = output / "images" / split
        dst_labels_dir = output / "labels" / split

        images = find_images(src_images_dir)
        stats[split]["total_images"] = len(images)

        images, duplicate_groups, duplicated_images_removed = select_unique_images(
            images, src_images_dir, src_labels_dir, split, duplicate_key_mode, quiet=quiet,
        )
        stats[split]["duplicate_groups"] = duplicate_groups
        stats[split]["duplicated_images_removed"] = duplicated_images_removed

        for image_path in images:
            relative_image_path = image_path.relative_to(src_images_dir)
            label_relative_path = relative_image_path.with_suffix(".txt")
            label_path = src_labels_dir / label_relative_path

            if not label_path.exists():
                stats[split]["missing_labels"] += 1
                stats[split]["removed_missing_labels"] += 1
                if not quiet:
                    console.print(
                        f"[yellow]\\[missing label][/yellow] split={split} "
                        f"image={image_path} expected={label_path}"
                    )
                continue

            label_lines = read_yolo_label(label_path)

            if label_contains_removed_class(label_lines, remove_class_ids):
                stats[split]["removed_images"] += 1
                continue

            remapped_lines = remap_label_lines(label_lines, old_to_new)

            dst_image_path = dst_images_dir / relative_image_path
            dst_label_path = dst_labels_dir / label_relative_path
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            dst_label_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(image_path, dst_image_path)
            with dst_label_path.open("w", encoding="utf-8") as f:
                if remapped_lines:
                    f.write("\n".join(remapped_lines) + "\n")

            stats[split]["kept_images"] += 1

    yaml_data = {
        **{split: str((output / "images" / split).resolve()) for split in splits},
        "nc": len(new_names),
        "names": new_names,
    }
    yaml_path = output / SOURCE_YAML_FILENAME
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    console.print("\n[green]✔  Dataset generado correctamente.[/green]")
    console.print(f"  Salida: {output.resolve()}")
    console.print(f"  YAML:   {yaml_path.resolve()}")

    console.print("\n[bold]Mapeo de clases (antiguo → nuevo):[/bold]")
    for old_id, new_id in old_to_new.items():
        console.print(f"  {old_id:2d} -> {new_id:2d} | {original_names[old_id]}")

    console.print("\n[bold]Estadísticas:[/bold]")
    for split, split_stats in stats.items():
        console.print(f"\n  [cyan]{split}[/cyan]:")
        for key, value in split_stats.items():
            console.print(f"    {key}: {value}")

        total_check = (
            split_stats["kept_images"]
            + split_stats["removed_images"]
            + split_stats["removed_missing_labels"]
            + split_stats["duplicated_images_removed"]
        )
        console.print(f"    total_control: {total_check}")
        if total_check != split_stats["total_images"]:
            console.print(
                "    [red]\\[aviso][/red] la suma kept + removed_images + "
                "removed_missing_labels + duplicated_images_removed no coincide "
                "con total_images."
            )


# ── Comandos ──────────────────────────────────────────────────────────────────

@app.command("real")
def generate_real(
    source: Path = typer.Option(
        DEFAULT_SOURCE, "--source", "-s", exists=True, file_okay=False, dir_okay=True,
        help="Directorio raíz del dataset original (contiene images/<split>/ y labels/<split>/).",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT, "--output", "-o",
        help="Directorio de salida. Se vacía por completo antes de generar (¡cuidado!).",
    ),
    splits: list[Split] = typer.Option(
        DEFAULT_SPLITS, "--split", help="Splits a procesar (repetir para varios).",
    ),
    remove_class_id: list[int] = typer.Option(
        DEFAULT_REMOVE_IDS, "--remove-class-id",
        help="ID de clase (esquema original) a eliminar por completo (repetir para varios).",
    ),
    duplicate_key_mode: DuplicateKeyMode = typer.Option(
        DEFAULT_DUP_MODE, "--duplicate-key-mode",
        help="Criterio para detectar imágenes duplicadas.",
    ),
    classes_map: Path = typer.Option(
        DEFAULT_CLASSES_MAP, "--classes-map", exists=True,
        help="YAML con el esquema de clases original (id -> nombre) del dataset fuente.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Oculta el detalle por imagen (duplicados, labels faltantes); deja solo el resumen final.",
    ),
) -> None:
    """Genera el dataset real completo a partir de las fuentes originales."""
    original_names = load_original_names(classes_map)
    process_dataset(
        source=source,
        output=output,
        splits=[s.value for s in splits],
        remove_class_ids=set(remove_class_id),
        duplicate_key_mode=duplicate_key_mode,
        original_names=original_names,
        quiet=quiet,
    )


# ── Dataset toy ───────────────────────────────────────────────────────────────

def load_yaml(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        raise FileNotFoundError(f"No existe el archivo YAML: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_class_to_images_index(source: Path, split: str) -> dict[int, list[Path]]:
    """Índice clase -> lista de imágenes de ese split que contienen esa clase."""
    images_split_dir = source / "images" / split
    labels_split_dir = source / "labels" / split

    class_to_images: dict[int, list[Path]] = defaultdict(list)

    for image_path in find_images(images_split_dir):
        label_path = get_label_path_for_image(image_path, images_split_dir, labels_split_dir)
        classes_in_image = {int(tokens[0]) for tokens in read_yolo_label(label_path)}
        for class_id in classes_in_image:
            class_to_images[class_id].append(image_path)

    return class_to_images


def copy_image_and_label(image_path: Path, source: Path, output: Path, split: str) -> None:
    src_images_dir = source / "images" / split
    src_labels_dir = source / "labels" / split
    dst_images_dir = output / "images" / split
    dst_labels_dir = output / "labels" / split

    relative_image_path = image_path.relative_to(src_images_dir)
    src_label_path = src_labels_dir / relative_image_path.with_suffix(".txt")
    dst_image_path = dst_images_dir / relative_image_path
    dst_label_path = dst_labels_dir / relative_image_path.with_suffix(".txt")

    dst_image_path.parent.mkdir(parents=True, exist_ok=True)
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, dst_image_path)
    if src_label_path.exists():
        shutil.copy2(src_label_path, dst_label_path)
    else:
        dst_label_path.touch()


def create_toy_yaml(source_yaml_data: dict, output: Path, splits: list[str], output_yaml: Path) -> None:
    toy_yaml_data = dict(source_yaml_data)
    for split in splits:
        toy_yaml_data[split] = str((output / "images" / split).resolve())

    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(toy_yaml_data, f, sort_keys=False, allow_unicode=True)


def process_toy_dataset(
    source: Path,
    output: Path,
    splits: list[str],
    samples_per_class: dict[str, int],
    random_seed: int,
    quiet: bool = False,
) -> None:
    random.seed(random_seed)

    source_yaml = source / SOURCE_YAML_FILENAME
    yaml_data = load_yaml(source_yaml)

    nc = int(yaml_data["nc"])
    names = yaml_data["names"]
    class_ids = list(range(nc))

    console.print(f"[yellow]⚠  Se va a vaciar el directorio de salida: {output.resolve()}[/yellow]")
    clean_output_dir(output, splits)

    global_stats = {}

    for split in splits:
        console.print(f"\n[bold]Procesando split:[/bold] {split}")
        n_samples = samples_per_class[split]

        class_to_images = build_class_to_images_index(source, split)
        selected_images: set[Path] = set()
        split_stats = {}

        for class_id in class_ids:
            available_images = class_to_images.get(class_id, [])

            if len(available_images) <= n_samples:
                selected_for_class = available_images
            else:
                selected_for_class = random.sample(available_images, n_samples)

            selected_images.update(selected_for_class)

            class_name = names[class_id]
            split_stats[class_id] = {
                "class_name": class_name,
                "available_images": len(available_images),
                "requested_images": n_samples,
                "selected_images": len(selected_for_class),
            }

            if len(available_images) < n_samples and not quiet:
                console.print(
                    f"  [yellow]Aviso:[/yellow] clase {class_id} ({class_name}) solo tiene "
                    f"{len(available_images)} imágenes en {split}. Se copiarán todas."
                )

        for image_path in selected_images:
            copy_image_and_label(image_path, source, output, split)

        global_stats[split] = {
            "selected_unique_images": len(selected_images),
            "per_class": split_stats,
        }
        console.print(f"  Imágenes únicas copiadas en {split}: {len(selected_images)}")

    output_yaml = output / TOY_YAML_FILENAME
    create_toy_yaml(yaml_data, output, splits, output_yaml)

    console.print("\n[green]✔  Dataset toy creado correctamente.[/green]")
    console.print(f"  Salida: {output.resolve()}")
    console.print(f"  YAML:   {output_yaml.resolve()}")

    console.print("\n[bold]Resumen por split:[/bold]")
    for split in splits:
        console.print(f"\n  [cyan]{split}[/cyan]:")
        console.print(f"    Imágenes únicas copiadas: {global_stats[split]['selected_unique_images']}")
        for class_id, stats in global_stats[split]["per_class"].items():
            console.print(
                f"    Clase {class_id:2d} - {stats['class_name']}: "
                f"{stats['selected_images']}/{stats['requested_images']} seleccionadas "
                f"({stats['available_images']} disponibles)"
            )


@app.command("toy")
def generate_toy(
    source: Path = typer.Option(
        DEFAULT_TOY_SOURCE, "--source", "-s", exists=True, file_okay=False, dir_okay=True,
        help="Dataset ya generado (salida de 'generate real'), usado como fuente del toy.",
    ),
    output: Path = typer.Option(
        DEFAULT_TOY_OUTPUT, "--output", "-o",
        help="Directorio de salida del dataset toy. Se vacía por completo antes de generar (¡cuidado!).",
    ),
    splits: list[Split] = typer.Option(
        DEFAULT_TOY_SPLITS, "--split", help="Splits a procesar (repetir para varios).",
    ),
    samples_train: int = typer.Option(
        DEFAULT_SAMPLES["train"], "--samples-train", help="Máximo de imágenes por clase en train.",
    ),
    samples_val: int = typer.Option(
        DEFAULT_SAMPLES["val"], "--samples-val", help="Máximo de imágenes por clase en val.",
    ),
    samples_test: int = typer.Option(
        DEFAULT_SAMPLES["test"], "--samples-test", help="Máximo de imágenes por clase en test.",
    ),
    random_seed: int = typer.Option(
        DEFAULT_RANDOM_SEED, "--random-seed", help="Semilla aleatoria para un muestreo reproducible.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Oculta los avisos por clase durante el proceso; deja solo el resumen final.",
    ),
) -> None:
    """Genera un dataset toy (subconjunto reducido) a partir de un dataset ya generado."""
    samples_per_class = {"train": samples_train, "val": samples_val, "test": samples_test}
    process_toy_dataset(
        source=source,
        output=output,
        splits=[s.value for s in splits],
        samples_per_class=samples_per_class,
        random_seed=random_seed,
        quiet=quiet,
    )
