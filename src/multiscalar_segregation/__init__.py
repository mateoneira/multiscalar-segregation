"""Multiscalar socio-spatial segregation via percolation hierarchies and the GJSD."""

from pathlib import Path

import yaml

__version__ = "1.0.0"


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load the pipeline configuration, resolving paths relative to the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    """Repo root, located as the ancestor of this file containing pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found in any ancestor directory")
