"""Utilidades genéricas compartidas por los distintos servicios de publicación
(donadataset.services.huggingface, donadataset.services.zenodo...).

Nada aquí es específico de HuggingFace ni de Zenodo — son helpers de
YAML/JSON, hashing y formateo reutilizados por ambos para no duplicar la
misma lógica en cada integración (como ya pasó una vez con upload/download
de HuggingFace antes de deduplicarlos).
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml
from jinja2 import Environment


def render_text_template(template_path: Path, **context: Any) -> str:
    """Renders a prose/markdown .j2 template (README.md.j2, LICENSE.j2...),
    trimming the whitespace {% for %}/{% if %} tags would otherwise leave
    behind — unlike load_config_source's YAML templates, blank lines here
    would be visible in the generated file."""
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    return template.render(**context)


def setup_logging() -> None:
    """Basic stdout logging for commands that don't need their own log file
    (unlike donadataset.services.huggingface's config-aware setup_logging,
    used only by 'generate'/'prepare' to also write a log inside the export
    dir). Without this, the root logger defaults to WARNING and every
    logging.info() call in a service function is silently swallowed."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_nested(config: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in parent or not isinstance(parent[key], dict):
        parent[key] = {}
    return parent[key]


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def sha256_file(path: Path, chunk_size_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream, chunk_size_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_size_bytes), b""):
        digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - only used to compare Zenodo file checksums
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def read_checksums(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(maxsplit=1)
            result[rel.strip()] = digest.strip()
    return result


def count_files(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*") if path.is_file())


def total_size_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def remove_dir_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
