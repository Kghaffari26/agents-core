"""Exports an agent's `latest.json` model as JSON Schema, published as `schema.json`,
and the (agent-independent) `trace.json` schema, published as `trace.schema.json`.

`agents_core.runner` calls `write_schema` and `write_trace_schema` automatically after
every run, so an agent never has to remember to do it. `python -m
agents_core.export_schemas <agent>` is the same thing by hand, for a website repo's
codegen step or local inspection; `--trace` writes only `trace.schema.json`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel

from agents_core import registry, settings
from agents_core.publish import write_json
from agents_core.schema import Trace

TRACE_SCHEMA_FILE = "trace.schema.json"


def schema_dict(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema(mode="serialization")
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}


def write_schema(model: type[BaseModel], publish_dir: Path | str | None = None) -> Path:
    """Write `<publish_dir>/schema.json` for `model`. Returns its path."""
    base = Path(publish_dir) if publish_dir is not None else settings.publish_dir()
    path = base / "schema.json"
    write_json(path, schema_dict(model))
    return path


def trace_schema_dict() -> dict:
    return schema_dict(Trace)


def write_trace_schema(publish_dir: Path | str | None = None) -> Path:
    """Write `<publish_dir>/trace.schema.json`. Returns its path."""
    base = Path(publish_dir) if publish_dir is not None else settings.publish_dir()
    path = base / TRACE_SCHEMA_FILE
    write_json(path, trace_schema_dict())
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agents_core.export_schemas")
    parser.add_argument("agent", nargs="?", help="Registered agent name; omit to list agents")
    parser.add_argument("--trace", action="store_true", help="write trace.schema.json only")
    args = parser.parse_args(argv)

    if args.trace:
        print(f"wrote {write_trace_schema()}")
        return 0

    if not args.agent:
        for name, target in sorted(registry.discover_agents().items()):
            print(f"{name} -> {target}")
        return 0

    try:
        agent = registry.load(args.agent)
    except registry.AgentNotFound as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    path = write_schema(agent.output_model)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
