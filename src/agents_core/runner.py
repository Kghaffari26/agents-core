"""`agents-run <agent> [--dry-run] [--apply] [extra args...]` — the console command
agents-core gives every agent repo for free, plus `agents_core.runner.main` for
`python -m agents_core.runner`.

fetch -> transform -> analyze -> validate -> publish -> export schema. Any exception
(including a failed validation or BudgetExceeded) fails the run: the previous
latest.json is left untouched and manifest-entry.json is marked `failed`. Exit code 0
on success, 1 on failure, 2 if the named agent isn't registered.

Unrecognized arguments are forwarded verbatim as `ctx.extra_args`, so an agent can
define flags of its own (e.g. a `--force-briefs`) without this package knowing about
them.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import sys
from datetime import UTC, datetime

from pydantic import BaseModel

from agents_core import costs, export_schemas, publish, registry, settings
from agents_core.agent import Agent, AgentResult, RunContext
from agents_core.costs import CostTracker
from agents_core.http import Http
from agents_core.llm import LLM
from agents_core.schema import AgentOutput, ManifestEntry, RunMeta

log = logging.getLogger("agents_core.runner")


def new_run_id(now: datetime) -> str:
    """e.g. `2026-09-23T14-00-05Z-a1b2c3`"""
    return f"{now.astimezone(UTC).strftime('%Y-%m-%dT%H-%M-%SZ')}-{secrets.token_hex(3)}"


def build_output(
    agent: Agent, result: AgentResult, ctx: RunContext, finished_at: datetime
) -> AgentOutput:
    meta = RunMeta(
        agent=agent.id,
        schema_version=agent.schema_version,
        run_id=ctx.run_id,
        started_at=ctx.started_at,
        finished_at=finished_at,
        status=result.status,
        data_changed=result.data_changed,
        cost_usd=round(ctx.costs.total_usd, 6),
        model_usage=ctx.costs.model_usage(),
        sources=result.sources,
    )
    body = (
        result.body.model_dump(mode="json")
        if isinstance(result.body, BaseModel)
        else dict(result.body)
    )
    body.pop("meta", None)
    return agent.output_model.model_validate({**body, "meta": meta.model_dump(mode="json")})


def manifest_entry_for(
    agent: Agent,
    result: AgentResult,
    output: AgentOutput,
    previous: ManifestEntry | None,
) -> ManifestEntry:
    finished = output.meta.finished_at
    if result.data_changed:
        last_change = finished
    else:
        last_change = previous.last_data_change_at if previous else None
    return ManifestEntry(
        id=agent.id,
        name=agent.name,
        route=agent.route,
        status=result.status,
        last_run_at=finished,
        last_data_change_at=last_change,
        expected_interval_hours=agent.expected_interval_hours,
        next_run_hint=agent.next_run_hint,
        headline=result.headline,
        key_stats=result.key_stats,
        run_cost_usd=round(output.meta.cost_usd, 4),
        items_count=result.items_count,
    )


def failed_entry(
    agent: Agent, previous: ManifestEntry | None, finished_at: datetime, cost_usd: float
) -> ManifestEntry:
    """Keep the last good headline and stats; a consuming site shows a failure banner."""
    if previous is not None:
        return previous.model_copy(
            update={"status": "failed", "last_run_at": finished_at, "run_cost_usd": cost_usd}
        )
    return ManifestEntry(
        id=agent.id,
        name=agent.name,
        route=agent.route,
        status="failed",
        last_run_at=finished_at,
        last_data_change_at=None,
        expected_interval_hours=agent.expected_interval_hours,
        next_run_hint=agent.next_run_hint,
        headline="No successful run yet.",
        key_stats=[],
        run_cost_usd=cost_usd,
    )


def run(
    agent: Agent,
    *,
    dry_run: bool = False,
    apply: bool = False,
    extra_args: list[str] | None = None,
    http: Http | None = None,
    llm_client: object | None = None,
) -> int:
    started = datetime.now(UTC)
    run_id = new_run_id(started)
    tracker = CostTracker(agent=agent.id, run_id=run_id)
    own_http = http is None
    http = http or Http()
    agent.configure_http(http)
    ctx = RunContext(
        agent_id=agent.id,
        run_id=run_id,
        started_at=started,
        http=http,
        llm=LLM(tracker, client=llm_client),
        costs=tracker,
        dry_run=dry_run,
        apply=apply,
        extra_args=extra_args or [],
        log=logging.getLogger(f"agents.{agent.id}"),
    )
    log.info(
        "run %s: %s%s%s",
        run_id,
        agent.id,
        " (dry run)" if dry_run else "",
        " (apply)" if apply else "",
    )

    status = "failed"
    try:
        raw = agent.fetch(ctx)
        data = agent.transform(ctx, raw)
        if dry_run:
            log.info("dry run: %s", agent.summarize_dry_run(data))
            log.info(
                "dry run: %d network requests; skipping LLM and publish", http.network_requests
            )
            status = "dry_run"
            return 0

        result = agent.analyze(ctx, data)
        finished = datetime.now(UTC)
        output = build_output(agent, result, ctx, finished)
        entry = manifest_entry_for(agent, result, output, publish.read_previous_manifest_entry())
        publish.publish_output(output, files=result.files, history_keep=agent.history_keep)
        publish.write_manifest_entry(entry)
        status = result.status
        log.info("run %s published: status=%s cost=$%.4f", run_id, status, tracker.total_usd)
        return 0
    except Exception:
        log.exception("run %s failed", run_id)
        if not dry_run:
            try:
                previous = publish.read_previous_manifest_entry()
                publish.write_manifest_entry(
                    failed_entry(agent, previous, datetime.now(UTC), round(tracker.total_usd, 4))
                )
            except Exception:
                log.exception("could not write manifest-entry.json for %s", agent.id)
        return 1
    finally:
        if not dry_run:
            tracker.record_run(status)
            try:
                export_schemas.write_schema(agent.output_model)
            except Exception:
                log.exception("could not write schema.json")
            try:
                costs.publish_costs_summary(agent.id, costs_path=tracker.path)
            except Exception:
                log.exception("could not write costs-summary.json")
        if own_http:
            http.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agents-run", description=__doc__)
    parser.add_argument("agent", nargs="?", help="Registered agent name")
    parser.add_argument(
        "--dry-run", action="store_true", help="fetch + transform only; skip the LLM and publishing"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="allow live external changes; semantics are agent-specific",
    )
    parser.add_argument("--list", action="store_true", help="list registered agents and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, extra = parser.parse_known_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its logs include full URLs
    settings.load_dotenv()

    if args.list or not args.agent:
        for name, target in sorted(registry.discover_agents().items()):
            print(f"{name} -> {target}")
        return 0

    try:
        agent = registry.load(args.agent)
    except registry.AgentNotFound as e:
        log.error("%s", e)
        return 2
    return run(agent, dry_run=args.dry_run, apply=args.apply, extra_args=extra)


if __name__ == "__main__":
    sys.exit(main())
