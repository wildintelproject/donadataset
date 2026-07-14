"""Shared pytest fixtures.

Redirects HOME / XDG_CONFIG_HOME / APPDATA to a throwaway temp directory
*before* anything imports ``donadataset.config`` (which auto-creates a
settings file in the user's real app-config dir on first import). This keeps
the test suite from touching the developer's actual ~/.config/donadataset or
~/Documents.
"""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

_FAKE_HOME = Path(tempfile.mkdtemp(prefix="donadataset-test-home-"))
os.environ["HOME"] = str(_FAKE_HOME)
os.environ["XDG_CONFIG_HOME"] = str(_FAKE_HOME / ".config")
os.environ["APPDATA"] = str(_FAKE_HOME / "AppData" / "Roaming")
atexit.register(shutil.rmtree, _FAKE_HOME, True)

import pytest  # noqa: E402

REPO_ROOT             = Path(__file__).resolve().parent.parent
EXAMPLE_SOURCE_DATASET = REPO_ROOT / "examples" / "source_dataset"
SOURCE_CLASSES_MAP     = REPO_ROOT / "metadata" / "source_classes.yaml"


@pytest.fixture
def example_source_dataset() -> Path:
    """The bundled example source dataset (examples/source_dataset)."""
    return EXAMPLE_SOURCE_DATASET


@pytest.fixture
def source_classes_map() -> Path:
    """The repo's 18-class original label scheme."""
    return SOURCE_CLASSES_MAP


@pytest.fixture
def make_source_dataset(tmp_path: Path):
    """Factory building a minimal images/+labels/ tree for a single split.

    Usage::

        source = make_source_dataset("train", {
            "a.jpg": [0],          # kept, class 0
            "b.jpg": [10],         # removed class -> whole image dropped
            "c.jpg": None,         # no label file -> missing label
        })
    """
    def _make(split: str, images: dict[str, list[int] | None]) -> Path:
        root = tmp_path / "source"
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for filename, class_ids in images.items():
            (images_dir / filename).touch()
            if class_ids is None:
                continue
            label_path = labels_dir / Path(filename).with_suffix(".txt").name
            lines = [f"{cid} 0.5 0.5 0.2 0.2" for cid in class_ids]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        return root

    return _make
