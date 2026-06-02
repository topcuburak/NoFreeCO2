"""Minimal YAML config loader for ford.yaml / coefficients.yaml."""
from __future__ import annotations

import os

import yaml

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(_HERE, "config")


def load(name: str) -> dict:
    path = name if os.path.isabs(name) else os.path.join(CONFIG_DIR, name)
    with open(path) as f:
        return yaml.safe_load(f)
