"""Writes an agent's validated JSON to its publish dir, with dated history.

Layout (see the README's data-branch contract — this is everything a CI workflow
force-pushes to the agent's own `data` branch as a single orphan commit):

    <publish_dir>/
    ├── latest.json              (+ agent-specific files, e.g. metros/<slug>.json)
    ├── history/YYYY-MM-DD.json
    ├── manifest-entry.json
    ├── costs-summary.json       (written by agents_core.costs.publish_costs_summary)
    └── schema.json              (written by agents_core.export_schemas.write_schema)

`publish_dir` defaults to `agents_core.settings.publish_dir()` (itself `public-data/`,
or `$AGENTS_CORE_PUBLISH_DIR`) — nothing here assumes any particular website's layout,
and there is no cross-agent manifest: each agent publishes its own single
`manifest-entry.json`, and a consuming website assembles a manifest across agents.

Files are written atomically. `latest.json` is written last, so a failure part-way
through never leaves a new latest.json pointing at missing files.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel

from agents_core import settings
from agents_core.schema import AgentOutput, ManifestEntry

log = logging.getLogger(__name__)

# The largest data file fetched before first paint should stay small; this is a
# warning, not a hard limit — agents on a bigger site may legitimately exceed it.
LATEST_SIZE_WARN_BYTES = 300_000
_HISTORY_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
_RESERVED_NAMES = frozenset(
    {"latest.json", "manifest-entry.json", "costs-summary.json", "schema.json"}
)


def _dump(obj: BaseModel | dict[str, Any]) -> bytes:
    data = obj.model_dump(mode="json", exclude_none=False) if isinstance(obj, BaseModel) else obj
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()


def write_json(path: Path, obj: BaseModel | dict[str, Any]) -> int:
    """Atomically write compact JSON. Returns the byte size."""
    payload = _dump(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return len(payload)


def _safe_relpath(rel: str) -> PurePosixPath:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or p.suffix != ".json":
        raise ValueError(f"extra file path must be a relative .json path: {rel!r}")
    if p.parts[0] == "history" or str(p) in _RESERVED_NAMES:
        raise ValueError(f"{rel!r} is reserved")
    return p


def publish_output(
    output: AgentOutput,
    *,
    files: dict[str, BaseModel] | None = None,
    history_keep: int,
    publish_dir: Path | str | None = None,
) -> Path:
    """Write extra files, today's history snapshot, then latest.json; trim history.

    Returns the path to latest.json.
    """
    base = Path(publish_dir) if publish_dir is not None else settings.publish_dir()
    for rel, model in (files or {}).items():
        write_json(base / _safe_relpath(rel), model)

    run_date = output.meta.finished_at.astimezone(UTC).date().isoformat()
    write_json(base / "history" / f"{run_date}.json", output)
    latest = base / "latest.json"
    size = write_json(latest, output)
    if size > LATEST_SIZE_WARN_BYTES:
        log.warning(
            "%s is %d bytes (budget %d); pre-aggregate or lazy-load more",
            latest,
            size,
            LATEST_SIZE_WARN_BYTES,
        )
    trim_history(base / "history", history_keep)
    return latest


def trim_history(history_dir: Path, keep: int) -> list[Path]:
    """Delete the oldest dated snapshots beyond `keep`. Returns removed paths."""
    if not history_dir.is_dir():
        return []
    snapshots = sorted(p for p in history_dir.iterdir() if _HISTORY_NAME.match(p.name))
    removed = snapshots[:-keep] if keep > 0 else snapshots
    for p in removed:
        p.unlink()
    return removed


def write_manifest_entry(entry: ManifestEntry, publish_dir: Path | str | None = None) -> Path:
    """Write `<publish_dir>/manifest-entry.json`. Returns its path."""
    base = Path(publish_dir) if publish_dir is not None else settings.publish_dir()
    path = base / "manifest-entry.json"
    write_json(path, entry)
    return path


def read_previous_manifest_entry(publish_dir: Path | str | None = None) -> ManifestEntry | None:
    """The previous run's manifest entry, or None if there isn't one locally (see
    `Agent.previous_latest`'s note on when a fresh CI checkout has one)."""
    base = Path(publish_dir) if publish_dir is not None else settings.publish_dir()
    path = base / "manifest-entry.json"
    if not path.is_file():
        return None
    try:
        return ManifestEntry.model_validate_json(path.read_bytes())
    except ValueError as e:  # json.JSONDecodeError and pydantic.ValidationError are both this
        log.warning("previous manifest-entry.json is invalid and will be ignored: %s", e)
        return None
