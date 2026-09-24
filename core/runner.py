"""Entry point: `python -m core.runner <agent> [--dry-run] [--apply]`.

fetch -> transform -> analyze -> validate -> publish. Any exception (including a
failed validation or BudgetExceeded) fails the run: the previous latest.json is left
untouched and the manifest entry is marked `failed`. Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import sys
from datetime import UTC, datetime

from pydantic import BaseModel

from core import publish, registry, settings
from core.agent import Agent, AgentResult, RunContext
from core.costs import CostTracker
from core.http import Http
from core.llm import LLM
from core.schema import AgentOutput, ManifestEntry, RunMeta

log = logging.getLogger("core.runner")


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
    """Keep the last good headline and stats; the site shows a failure banner."""
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
        entry = manifest_entry_for(agent, result, output, publish.previous_entry(agent.id))
        publish.publish_output(
            agent.id, output, files=result.files, history_keep=agent.history_keep
        )
        publish.upsert_manifest(entry)
        status = result.status
        log.info("run %s published: status=%s cost=$%.4f", run_id, status, tracker.total_usd)
        return 0
    except Exception:
        log.exception("run %s failed", run_id)
        if not dry_run:
            try:
                previous = publish.previous_entry(agent.id)
                publish.upsert_manifest(
                    failed_entry(agent, previous, datetime.now(UTC), round(tracker.total_usd, 4))
                )
            except Exception:
                log.exception("could not mark %s failed in manifest", agent.id)
        return 1
    finally:
        if not dry_run or tracker.calls:
            tracker.record_run(status)
            try:
                publish.write_cost_summary()
            except Exception:
                log.exception("could not write cost summary")
        if own_http:
            http.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.runner", description=__doc__)
    parser.add_argument("agent", choices=registry.AGENT_IDS)
    parser.add_argument(
        "--dry-run", action="store_true", help="fetch + transform only; skip the LLM and publishing"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="allow write actions (repo_maint only, allowlisted repos)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its logs include full URLs
    settings.load_dotenv()

    try:
        agent = registry.load(args.agent)
    except registry.UnknownAgent as e:
        log.error("%s", e)
        return 2
    return run(agent, dry_run=args.dry_run, apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
