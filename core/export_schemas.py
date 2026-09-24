"""Export JSON Schema for every published file: `python -m core.export_schemas`.

Writes `schemas/*.schema.json` at the repo root. The site's `npm run gen:types` turns
these into TypeScript types and zod validators (docs/specs/SPEC_WEBSITE.md §3).
Shared files always export; agent files export once the agent is implemented.

`--check` exits 1 if any schema on disk is out of date (for CI).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel

from core import registry, settings
from core.schema import CostSummary, Manifest

log = logging.getLogger(__name__)

SHARED: dict[str, type[BaseModel]] = {
    "manifest": Manifest,
    "costs_summary": CostSummary,
}


def collect() -> dict[str, type[BaseModel]]:
    models = dict(SHARED)
    for agent in registry.load_available():
        models[agent.id] = agent.output_model
        for name, model in agent.extra_models().items():
            models[f"{agent.id}.{name}"] = model
    return models


def render(name: str, model: type[BaseModel]) -> str:
    schema = model.model_json_schema(mode="serialization")
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": name, **schema}
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def export(out_dir: Path | None = None, *, check: bool = False) -> list[Path]:
    """Write (or with check=True, compare) schemas. Returns paths that changed."""
    out_dir = out_dir or settings.SCHEMAS_DIR
    changed = []
    for name, model in collect().items():
        path = out_dir / f"{name}.schema.json"
        text = render(name, model)
        if path.is_file() and path.read_text() == text:
            continue
        changed.append(path)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.export_schemas")
    parser.add_argument("--check", action="store_true", help="fail if schemas are out of date")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    changed = export(check=args.check)
    for path in changed:
        root = settings.REPO_ROOT
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        print(f"{'out of date' if args.check else 'wrote'}: {rel}")
    if args.check and changed:
        print("run `uv run python -m core.export_schemas` and commit the result")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
