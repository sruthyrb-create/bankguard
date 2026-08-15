"""Load YAML config and environment variables in one place."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_env() -> None:
    """Load variables from the repo's .env file (if present) into os.environ."""
    load_dotenv(REPO_ROOT / ".env", override=False)


def load_yaml(name: str) -> dict[str, Any]:
    """Read configs/<name>.yaml and return it as a dict."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def get_secret(env_var: str) -> str | None:
    """Return an environment variable, treating empty strings as missing."""
    value = os.getenv(env_var, "").strip()
    return value or None
