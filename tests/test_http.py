import json

import httpx
import pytest

from agents_core.http import HostPolicy, Http, HttpError, RequestBudgetExceeded, redact_url


class Recorder:
    def __init__(self, statuses=None, body=b'{"ok": true}'):
        self.statuses = list(statuses or [])
        self.body = body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self.statuses.pop(0) if self.statuses else 200
        headers = {"content-type": "application/json"}
        if status == 429:
            headers["retry-after"] = "2"
        return httpx.Response(status, content=self.body, headers=headers)


def make_http(handler, **kwargs):
    sleeps: list[float] = []
    http = Http(transport=httpx.MockTransport(handler), sleep=sleeps.append, **kwargs)
    return http, sleeps


def test_second_get_is_served_from_cache_without_leaking_secret(isolated_paths):
    rec = Recorder()
    http, _ = make_http(rec)
    first = http.get("https://api.example.gov/series", params={"id": "CPI", "api_key": "SECRET"})
    second = http.get("https://api.example.gov/series", params={"id": "CPI", "api_key": "OTHER"})
    assert first.json() == {"ok": True} and not first.from_cache
    assert second.from_cache and second.json() == {"ok": True}
    assert len(rec.requests) == 1
    assert rec.requests[0].url.params["api_key"] == "SECRET"
    for path in (isolated_paths / "http_cache").rglob("*.json"):
        assert "SECRET" not in path.read_text()
    assert "SECRET" not in first.url


def test_ttl_zero_and_refresh_bypass_cache():
    rec = Recorder()
    http, _ = make_http(rec)
    http.get("https://x.test/a", ttl_seconds=0)
    http.get("https://x.test/a", ttl_seconds=0)
    http.get("https://x.test/b")
    http.get("https://x.test/b", refresh=True)
    assert len(rec.requests) == 4


def test_retries_transient_errors_with_backoff_and_retry_after():
    rec = Recorder(statuses=[503, 429, 200])
    http, sleeps = make_http(rec, backoff_seconds=1.0)
    assert http.get("https://x.test/a").status == 200
    assert sleeps == [1.0, 2.0]  # backoff for 503, Retry-After for 429


def test_client_errors_are_not_retried():
    rec = Recorder(statuses=[404])
    http, sleeps = make_http(rec)
    with pytest.raises(HttpError) as e:
        http.get("https://x.test/a?api_key=SECRET")
    assert e.value.status == 404
    assert "SECRET" not in str(e.value)
    assert len(rec.requests) == 1 and sleeps == []


def test_gives_up_after_max_attempts():
    http, _ = make_http(Recorder(statuses=[500] * 10), max_attempts=3)
    with pytest.raises(HttpError, match="after 3 attempts"):
        http.get("https://x.test/a")


def test_daily_budget_counts_network_requests_only():
    rec = Recorder()
    http, _ = make_http(rec)
    http.set_policy("api.sam.gov", HostPolicy(daily_budget=2))
    http.get("https://api.sam.gov/opps", params={"q": "1"})
    http.get("https://api.sam.gov/opps", params={"q": "1"})  # cache hit, free
    http.get("https://api.sam.gov/opps", params={"q": "2"})
    assert http.budget_remaining("api.sam.gov") == 0
    with pytest.raises(RequestBudgetExceeded):
        http.get("https://api.sam.gov/opps", params={"q": "3"})
    # Budget persists across Http instances within the day.
    http2, _ = make_http(rec)
    http2.set_policy("api.sam.gov", HostPolicy(daily_budget=2))
    assert http2.budget_remaining("api.sam.gov") == 0


def test_min_interval_throttles_same_host():
    now = [100.0]
    rec = Recorder()
    http, sleeps = make_http(rec, clock=lambda: now[0])
    http.set_policy("x.test", HostPolicy(min_interval_seconds=1.5))
    http.get("https://x.test/a", ttl_seconds=0)
    now[0] += 0.5
    http.get("https://x.test/b", ttl_seconds=0)
    assert sleeps == [pytest.approx(1.0)]


def test_redact_url():
    assert redact_url("https://h.test/p?api_key=abc&x=1") == "https://h.test/p?api_key=***&x=1"
    assert redact_url("https://h.test/p", {"token": "t0k", "a": 1}) == (
        "https://h.test/p?token=***&a=1"
    )


def test_text_and_json_helpers():
    http, _ = make_http(Recorder(body=json.dumps({"a": "é"}).encode()))
    r = http.get("https://x.test/a")
    assert r.json() == {"a": "é"}
    assert "a" in r.text
