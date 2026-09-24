"""Paths, environment, and config loading shared by every core module."""

from __future__ import annotations

import os
import tomllib
from functools import cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
SCHEMAS_DIR = REPO_ROOT / "schemas"

DEFAULT_MAX_RUN_USD = 0.50


def _path_from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def publish_dir() -> Path:
    """Root of the site data tree (`site/public/data`)."""
    return _path_from_env("PUBLISH_DIR", REPO_ROOT / "site" / "public" / "data")


def costs_path() -> Path:
    return _path_from_env("COSTS_PATH", REPO_ROOT / "data" / "costs.jsonl")


def http_cache_dir() -> Path:
    return _path_from_env("HTTP_CACHE_DIR", REPO_ROOT / ".cache" / "http")


def http_cache_ttl_seconds() -> int:
    return int(os.environ.get("HTTP_CACHE_TTL_SECONDS", 6 * 3600))


def max_run_usd() -> float:
    value = os.environ.get("MAX_RUN_USD")
    return float(value) if value else DEFAULT_MAX_RUN_USD


def require_env(name: str) -> str:
    """Return a secret from the environment or fail with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable {name} (see .env.example)")
    return value


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from `.env` without overriding variables already set."""
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key, value)


@cache
def load_config(name: str) -> dict[str, Any]:
    """Load `config/<name>.toml`. Cached for the life of the process."""
    path = CONFIG_DIR / f"{name}.toml"
    with path.open("rb") as f:
        return tomllib.load(f)
