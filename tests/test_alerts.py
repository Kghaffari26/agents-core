"""agents_core.alerts: one deduplicated GitHub issue per alert title."""

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from agents_core import alerts
from agents_core.agent import RunContext
from agents_core.costs import CostTracker
from agents_core.http import Http
from agents_core.llm import LLM

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
TITLE = "SAM.gov API key rejected"


class FakeGitHub:
    def __init__(self, issues=None, fail=None):
        self.issues = list(issues or [])
        self.fail = fail
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            return httpx.Response(self.fail, json={"message": "nope"})
        path = request.url.path
        if request.method == "GET" and path == "/repos/o/r/issues":
            page = int(request.url.params.get("page", 1))
            return httpx.Response(200, json=self.issues[(page - 1) * 100 : page * 100])
        if request.method == "POST" and path == "/repos/o/r/issues":
            return httpx.Response(201, json={"number": 99})
        if request.method == "POST" and path.endswith("/comments"):
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(404)

    def posts(self):
        return [(r.url.path, json.loads(r.content)) for r in self.requests if r.method == "POST"]


def send(gh, **kw):
    http = Http(transport=httpx.MockTransport(gh), sleep=lambda s: None)
    kw.setdefault("repo", "o/r")
    kw.setdefault("token", "t0k")
    return alerts.ops_alert(TITLE, "SAM returned 403.", http=http, now=kw.pop("now", NOW), **kw)


@pytest.fixture(autouse=True)
def no_ambient_github(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)


def issue(number, title=TITLE, updated=NOW - timedelta(days=30), **extra):
    return {"number": number, "title": title, "updated_at": updated.isoformat(), **extra}


def test_disabled_without_token_or_repo(caplog):
    gh = FakeGitHub()
    assert send(gh, token="", repo="") == "disabled"
    assert send(gh, repo="") == "disabled"
    assert gh.requests == []
    assert TITLE in caplog.text


def test_reads_token_and_repo_from_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "envtok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    gh = FakeGitHub()
    http = Http(transport=httpx.MockTransport(gh))
    assert alerts.ops_alert(TITLE, "b", http=http, now=NOW) == "created"
    assert gh.requests[0].headers["authorization"] == "Bearer envtok"


def test_creates_labelled_issue_when_none_open(isolated_paths):
    gh = FakeGitHub(issues=[issue(1, title="Something else")])
    assert send(gh) == "created"
    ((path, body),) = gh.posts()
    assert path == "/repos/o/r/issues"
    assert body["title"] == TITLE and body["labels"] == ["ops-alert"]
    assert body["body"].startswith("SAM returned 403.") and "7 days" in body["body"]
    listing = gh.requests[0]
    assert listing.url.params["labels"] == "ops-alert" and listing.url.params["state"] == "open"
    state = json.loads((isolated_paths / "data" / "ops_alerts.json").read_text())
    assert state == {TITLE: "2026-09-26T12:00:00Z"}


def test_comments_on_stale_open_issue():
    gh = FakeGitHub(issues=[issue(7)])
    assert send(gh) == "commented"
    ((path, body),) = gh.posts()
    assert path == "/repos/o/r/issues/7/comments"
    assert "SAM returned 403." in body["body"]


def test_skips_open_issue_updated_within_interval():
    gh = FakeGitHub(issues=[issue(7, updated=NOW - timedelta(days=2))])
    assert send(gh) == "skipped_recent"
    assert gh.posts() == []


def test_ignores_pull_requests_with_same_title():
    gh = FakeGitHub(issues=[issue(3, pull_request={"url": "x"})])
    assert send(gh) == "created"


def test_at_most_once_per_interval_per_title_without_network():
    gh = FakeGitHub()
    assert send(gh) == "created"
    n = len(gh.requests)
    assert send(gh, now=NOW + timedelta(days=6)) == "skipped_recent"
    assert len(gh.requests) == n  # decided from local state alone
    gh.issues = [issue(99, updated=NOW)]
    assert send(gh, now=NOW + timedelta(days=8)) == "commented"


def test_titles_are_deduplicated_independently():
    gh = FakeGitHub()
    assert send(gh) == "created"
    http = Http(transport=httpx.MockTransport(gh))
    other = alerts.ops_alert(
        "FOMC extraction failed", "b", http=http, repo="o/r", token="t", now=NOW
    )
    assert other == "created"


def test_finds_issue_on_a_later_page():
    filler = [issue(i, title=f"other {i}") for i in range(100)]
    gh = FakeGitHub(issues=[*filler, issue(150)])
    assert send(gh) == "commented"
    assert gh.posts()[0][0] == "/repos/o/r/issues/150/comments"


def test_api_failure_is_logged_not_raised(isolated_paths, caplog):
    gh = FakeGitHub(fail=403)
    assert send(gh) == "failed"
    assert "could not be sent" in caplog.text
    assert not (isolated_paths / "data" / "ops_alerts.json").exists()  # retried next run


def test_token_never_logged_or_stored(isolated_paths, caplog):
    send(FakeGitHub())
    assert "t0k" not in caplog.text
    assert "t0k" not in (isolated_paths / "data" / "ops_alerts.json").read_text()
    for path in (isolated_paths / "http_cache").rglob("*"):
        if path.is_file():
            assert "t0k" not in path.read_text()


def test_ctx_alert_uses_run_http(monkeypatch):
    gh = FakeGitHub()
    http = Http(transport=httpx.MockTransport(gh))
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    tracker = CostTracker(agent="a", run_id="r")
    ctx = RunContext(
        agent_id="a", run_id="r", started_at=NOW, http=http, llm=LLM(tracker), costs=tracker
    )
    assert ctx.alert(TITLE, "body") == "created"
