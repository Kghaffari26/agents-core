"""Configurable paths and env vars, shared by every module in this package.

An agent repo doesn't need to set anything: `data_dir()` defaults to `data/` (local
run state: costs.jsonl, guard_failures.jsonl, the HTTP cache) and `publish_dir()`
defaults to `public-data/` (published output: latest.json, manifest-entry.json,
costs-summary.json, schema.json — see the data-branch contract in the README).
Both are overridable via env var so CI can point them elsewhere without code changes.
Nothing here assumes a particular website's directory layout.
"""

from __future__ import annotations

import os
import tomllib
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path("data")
DEFAULT_PUBLISH_DIR = Path("public-data")
DEFAULT_MAX_RUN_USD = 0.50
DEFAULT_MODELS_PATH = Path("config/models.toml")


def _path_from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else default


def data_dir() -> Path:
    return _path_from_env("AGENTS_CORE_DATA_DIR", DEFAULT_DATA_DIR)


def publish_dir() -> Path:
    return _path_from_env("AGENTS_CORE_PUBLISH_DIR", DEFAULT_PUBLISH_DIR)


def costs_path() -> Path:
    return _path_from_env("AGENTS_CORE_COSTS_PATH", data_dir() / "costs.jsonl")


def guard_failures_path() -> Path:
    return _path_from_env("AGENTS_CORE_GUARD_FAILURES_PATH", data_dir() / "guard_failures.jsonl")


def http_cache_dir() -> Path:
    # Deliberately outside data_dir(): CI commits data/ back to the caller's default
    # branch as run state, and the HTTP cache is a large, purely local dev convenience
    # that shouldn't bloat that history.
    return _path_from_env("AGENTS_CORE_HTTP_CACHE_DIR", Path(".cache") / "http")


def http_cache_ttl_seconds() -> int:
    return int(os.environ.get("AGENTS_CORE_HTTP_CACHE_TTL_SECONDS", 6 * 3600))


def max_run_usd() -> float:
    value = os.environ.get("AGENTS_CORE_MAX_RUN_USD")
    return float(value) if value else DEFAULT_MAX_RUN_USD


def require_env(name: str) -> str:
    """Return a secret from the environment or fail with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable {name}")
    return value


def anthropic_api_key() -> str:
    """ANTHROPIC_API_KEY, falling back to AGENTS_ANTHROPIC_API_KEY.

    Some cloud dev environments reserve the ANTHROPIC_API_KEY name for their own use,
    so runs there can set AGENTS_ANTHROPIC_API_KEY instead.
    """
    value = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("AGENTS_ANTHROPIC_API_KEY")
    if not value:
        raise RuntimeError(
            "Missing required environment variable ANTHROPIC_API_KEY (or AGENTS_ANTHROPIC_API_KEY)"
        )
    return value


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from `.env` without overriding variables already set."""
    path = path or Path(".env")
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@cache
def load_config(name: str, *, path: Path | str | None = None) -> dict[str, Any]:
    """Load `config/<name>.toml`, merged over the package's shipped default of the
    same name (an agent repo's file only needs to override what it wants to change).
    Cached for the life of the process.
    """
    default_text = resources.files("agents_core.config").joinpath(f"{name}.toml").read_text()
    merged = tomllib.loads(default_text)
    override_path = Path(path) if path is not None else Path(f"config/{name}.toml")
    if override_path.is_file():
        merged = _deep_merge(merged, tomllib.loads(override_path.read_text()))
    return merged
