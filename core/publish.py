"""Writes validated JSON into the site data tree, maintains the manifest and history.

Layout (docs/specs/SPEC_WEBSITE.md §3):

    site/public/data/
    ├── manifest.json
    ├── costs/summary.json
    └── <agent>/latest.json, history/YYYY-MM-DD.json, plus agent-specific files

Files are written atomically. `latest.json` is written last, so a failure part-way
through never leaves a new latest.json pointing at missing files.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ValidationError

from core import costs, registry, settings
from core.schema import AgentOutput, Manifest, ManifestEntry

log = logging.getLogger(__name__)

# Spec §11: the largest data file fetched before first paint stays under 300KB.
LATEST_SIZE_WARN_BYTES = 300_000
_HISTORY_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


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


def agent_dir(agent_id: str, root: Path | None = None) -> Path:
    return (root or settings.publish_dir()) / agent_id


def _safe_relpath(rel: str) -> PurePosixPath:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or p.suffix != ".json":
        raise ValueError(f"extra file path must be a relative .json path: {rel!r}")
    if p.parts[0] == "history" or str(p) == "latest.json":
        raise ValueError(f"{rel!r} is reserved")
    return p


def publish_output(
    agent_id: str,
    output: AgentOutput,
    *,
    files: dict[str, BaseModel] | None = None,
    history_keep: int,
    root: Path | None = None,
) -> Path:
    """Write extra files, today's history snapshot, then latest.json; trim history."""
    base = agent_dir(agent_id, root)
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


# ---- manifest -------------------------------------------------------------


def read_manifest(root: Path | None = None) -> Manifest | None:
    path = (root or settings.publish_dir()) / "manifest.json"
    if not path.is_file():
        return None
    try:
        return Manifest.model_validate_json(path.read_bytes())
    except ValidationError as e:
        # A corrupt manifest is rebuilt entry by entry rather than blocking every agent.
        log.error("manifest.json is invalid and will be rebuilt: %s", e)
        return None


def previous_entry(agent_id: str, root: Path | None = None) -> ManifestEntry | None:
    manifest = read_manifest(root)
    if manifest is None:
        return None
    return next((a for a in manifest.agents if a.id == agent_id), None)


def upsert_manifest(entry: ManifestEntry, root: Path | None = None) -> Manifest:
    manifest = read_manifest(root)
    entries = {a.id: a for a in (manifest.agents if manifest else [])}
    entries[entry.id] = entry
    order = {agent_id: i for i, agent_id in enumerate(registry.AGENT_IDS)}
    ordered = sorted(entries.values(), key=lambda a: (order.get(a.id, len(order)), a.id))
    updated = Manifest(generated_at=datetime.now(UTC), agents=ordered)
    write_json((root or settings.publish_dir()) / "manifest.json", updated)
    return updated


def write_cost_summary(root: Path | None = None, costs_path: Path | None = None) -> None:
    summary = costs.summarize(costs.read_log(costs_path))
    write_json((root or settings.publish_dir()) / "costs" / "summary.json", summary)
