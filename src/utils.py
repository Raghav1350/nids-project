"""Small shared helpers."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> None:
    for key in ("model_dir", "reports_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
