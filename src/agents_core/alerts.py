"""Ops alerts: open or update one GitHub issue per alert title, deduplicated.

For problems a human needs to fix that don't fail the run — a rejected API key, a
source that changed shape, an extraction that fell back to last week's data:

    ctx.alert("SAM.gov API key rejected", "SAM returned 403 for the opportunities search...")

At most one alert per title per `min_interval` (default 7 days):

1. `data/ops_alerts.json` (run state the reusable workflow commits back) remembers
   when each title last alerted; a title alerted more recently is skipped without
   touching the network.
2. Otherwise the repo's open issues labelled `ops-alert` are searched for the exact
   title. An open one updated within the interval is left alone; an older one gets
   a comment; if there is none, a new issue is opened.

It's active only when a token (`GITHUB_TOKEN`) and a repo (`GITHUB_REPOSITORY`,
which GitHub Actions always sets) are both available — in local runs it's a no-op
that just logs. It never raises: a failed alert is logged and reported as "failed",
and must never fail the agent's run. The token needs `issues: write`.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from agents_core import settings
from agents_core.http import Http
from agents_core.schema import iso_z

log = logging.getLogger(__name__)

AlertOutcome = Literal["created", "commented", "skipped_recent", "disabled", "failed"]

ALERT_LABEL = "ops-alert"
DEFAULT_MIN_INTERVAL = timedelta(days=7)
GITHUB_API = "https://api.github.com"
_MAX_PAGES = 10
_FOOTER = (
    "\n\n---\n_Opened by agents-core `ops_alert`. Repeats of this alert title are "
    "suppressed for {days} days; close this issue once it's fixed._"
)


def _read_state() -> dict[str, str]:
    path = settings.ops_alerts_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(state: dict[str, str]) -> None:
    path = settings.ops_alerts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def ops_alert(
    title: str,
    body: str,
    *,
    http: Http | None = None,
    repo: str | None = None,
    token: str | None = None,
    label: str = ALERT_LABEL,
    min_interval: timedelta = DEFAULT_MIN_INTERVAL,
    now: datetime | None = None,
) -> AlertOutcome:
    """Open or update the GitHub issue for `title`. See the module docstring."""
    token = token or os.environ.get("GITHUB_TOKEN")
    repo = repo or os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        log.warning("ops alert (not sent: no GITHUB_TOKEN/GITHUB_REPOSITORY): %s", title)
        return "disabled"

    now = now or datetime.now(UTC)
    state = _read_state()
    last = _parse_ts(state.get(title))
    if last is not None and now - last < min_interval:
        log.info("ops alert %r already sent at %s; skipping", title, iso_z(last))
        return "skipped_recent"

    own_http = http is None
    client = http or Http()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        outcome = _send(client, repo, headers, title, body, label, min_interval, now)
    except Exception as e:  # never fail a run over an alert
        log.error("ops alert %r could not be sent: %s", title, e)
        return "failed"
    finally:
        if own_http:
            client.close()

    state[title] = iso_z(now)
    try:
        _write_state(state)
    except OSError as e:
        log.warning("could not record ops alert state: %s", e)
    log.warning("ops alert %r: %s", title, outcome)
    return outcome


def _send(
    http: Http,
    repo: str,
    headers: dict[str, str],
    title: str,
    body: str,
    label: str,
    min_interval: timedelta,
    now: datetime,
) -> AlertOutcome:
    issues_url = f"{GITHUB_API}/repos/{repo}/issues"
    existing = _find_open_issue(http, issues_url, headers, title, label)
    if existing is not None:
        updated = _parse_ts(existing.get("updated_at"))
        if updated is not None and now - updated < min_interval:
            return "skipped_recent"
        http.request(
            "POST",
            f"{issues_url}/{existing['number']}/comments",
            headers=headers,
            json_body={"body": f"Still happening as of {iso_z(now)}.\n\n{body}"},
            ttl_seconds=0,
        )
        return "commented"
    footer = _FOOTER.format(days=min_interval.days)
    http.request(
        "POST",
        issues_url,
        headers=headers,
        json_body={"title": title, "body": body + footer, "labels": [label]},
        ttl_seconds=0,
    )
    return "created"


def _find_open_issue(
    http: Http, issues_url: str, headers: dict[str, str], title: str, label: str
) -> dict[str, Any] | None:
    for page in range(1, _MAX_PAGES + 1):
        issues = http.request(
            "GET",
            issues_url,
            params={"state": "open", "labels": label, "per_page": 100, "page": page},
            headers=headers,
            ttl_seconds=0,
        ).json()
        for issue in issues:
            if issue.get("title") == title and "pull_request" not in issue:
                return issue
        if len(issues) < 100:
            return None
    return None
