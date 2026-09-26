"""Cache correctness, honest request budgets, and Http.download (v0.2.0)."""

import json
from datetime import UTC, datetime

import httpx
import pytest

from agents_core.http import HostPolicy, Http, HttpError, RequestBudgetExceeded


def make_http(handler, **kwargs):
    sleeps: list[float] = []
    client = Http(transport=httpx.MockTransport(handler), sleep=sleeps.append, **kwargs)
    return client, sleeps


class Script:
    """Replays a list of (status, body, headers) or exceptions, recording requests."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        step = self.steps.pop(0) if self.steps else (200, b'{"ok": true}', {})
        if isinstance(step, Exception):
            raise step
        status, body, headers = step
        return httpx.Response(status, content=body, headers=headers)


JSON = {"content-type": "application/json"}


# ---- caching ----------------------------------------------------------------


def test_non_2xx_responses_are_never_cached():
    script = Script((404, b"missing key", {}), (200, b'{"v": 1}', JSON))
    client, _ = make_http(script)
    with pytest.raises(HttpError):
        client.get("https://api.census.gov/data")
    assert client.get("https://api.census.gov/data").json() == {"v": 1}
    assert client.get("https://api.census.gov/data").from_cache
    assert len(script.requests) == 2


def test_failed_retries_leave_no_cache_entry(isolated_paths):
    client, _ = make_http(Script(*[(503, b"down", {})] * 4))
    with pytest.raises(HttpError):
        client.get("https://x.test/a")
    assert not list((isolated_paths / "http_cache").rglob("*.json"))


def test_get_json_never_caches_a_non_json_200():
    script = Script(
        (200, b"<html>error: key required</html>", {"content-type": "text/html"}),
        (200, b'{"rows": [1]}', JSON),
    )
    client, _ = make_http(script)
    with pytest.raises(ValueError):
        client.get_json("https://api.census.gov/data", params={"get": "B01001"})
    assert client.get_json("https://api.census.gov/data", params={"get": "B01001"}) == {"rows": [1]}
    assert len(script.requests) == 2


def test_cached_non_json_entry_is_ignored_by_get_json():
    script = Script((200, b"<html>oops</html>", {}), (200, b'{"a": 1}', JSON))
    client, _ = make_http(script)
    assert client.get("https://x.test/a").text.startswith("<html>")  # plain get caches it
    assert client.get_json("https://x.test/a") == {"a": 1}  # but get_json refetches
    assert len(script.requests) == 2


def test_response_fetched_without_key_is_not_served_after_key_added(isolated_paths):
    script = Script((200, b'{"error": "no key"}', JSON), (200, b'{"rows": []}', JSON))
    client, _ = make_http(script)
    assert client.get_json("https://api.census.gov/data", params={"get": "x"}) == {
        "error": "no key"
    }
    with_key = client.get_json("https://api.census.gov/data", params={"get": "x", "key": "K"})
    assert with_key == {"rows": []}
    assert len(script.requests) == 2
    for path in (isolated_paths / "http_cache").rglob("*.json"):
        assert 'K"' not in path.read_text()


def test_secret_value_change_still_hits_cache():
    script = Script()
    client, _ = make_http(script)
    client.get("https://x.test/a", params={"api_key": "one"})
    assert client.get("https://x.test/a", params={"api_key": "two"}).from_cache
    assert len(script.requests) == 1


def test_secret_header_presence_is_part_of_cache_key():
    script = Script()
    client, _ = make_http(script)
    client.get("https://x.test/a")
    assert not client.get("https://x.test/a", headers={"Authorization": "Bearer t"}).from_cache
    assert client.get("https://x.test/a", headers={"Authorization": "Bearer u"}).from_cache
    assert len(script.requests) == 2


# ---- request budgets ---------------------------------------------------------


def budgeted(script, budget=3, **policy):
    client, sleeps = make_http(script)
    client.set_policy("api.sam.gov", HostPolicy(daily_budget=budget, **policy))
    return client, sleeps


def test_capped_host_does_not_retry_by_default():
    script = Script((503, b"", {}), (200, b"{}", JSON))
    client, sleeps = budgeted(script)
    with pytest.raises(HttpError, match="after 1 attempt:"):
        client.get("https://api.sam.gov/opps")
    assert len(script.requests) == 1 and sleeps == []
    assert client.budget_remaining("api.sam.gov") == 2


def test_capped_host_retries_when_policy_allows():
    script = Script((503, b"", {}), (200, b"{}", JSON))
    client, sleeps = budgeted(script, max_attempts=2)
    assert client.get("https://api.sam.gov/opps").status == 200
    assert len(sleeps) == 1
    assert client.budget_remaining("api.sam.gov") == 1  # both sent requests counted


def test_retries_stop_at_the_budget():
    client, _ = budgeted(Script(*[(503, b"", {})] * 5), budget=2, max_attempts=5)
    with pytest.raises(RequestBudgetExceeded):
        client.get("https://api.sam.gov/opps")
    assert client.budget_remaining("api.sam.gov") == 0


def test_uncapped_host_keeps_default_retries():
    script = Script((503, b"", {}), (503, b"", {}), (200, b"{}", JSON))
    client, _ = make_http(script)
    assert client.get("https://x.test/a").status == 200
    assert len(script.requests) == 3


def test_connection_failures_never_reach_the_budget():
    script = Script(httpx.ConnectError("refused"), httpx.ConnectTimeout("slow"))
    client, _ = budgeted(script, max_attempts=2)
    with pytest.raises(HttpError):
        client.get("https://api.sam.gov/opps")
    assert client.budget_remaining("api.sam.gov") == 3
    assert client.network_requests == 0


def test_requests_that_were_sent_count_even_without_a_response():
    client, _ = budgeted(Script(httpx.ReadTimeout("no reply")))
    with pytest.raises(HttpError):
        client.get("https://api.sam.gov/opps")
    assert client.budget_remaining("api.sam.gov") == 2


def test_client_errors_count_toward_budget():
    client, _ = budgeted(Script((403, b"bad key", {})))
    with pytest.raises(HttpError) as e:
        client.get("https://api.sam.gov/opps")
    assert e.value.status == 403
    assert client.budget_remaining("api.sam.gov") == 2


def test_cache_hits_never_count_toward_budget():
    client, _ = budgeted(Script(), budget=1)
    client.get("https://api.sam.gov/opps", params={"q": "1"})
    for _ in range(5):
        assert client.get("https://api.sam.gov/opps", params={"q": "1"}).from_cache
    assert client.budget_remaining("api.sam.gov") == 0
    with pytest.raises(RequestBudgetExceeded):
        client.get("https://api.sam.gov/opps", params={"q": "2"})


def test_exhausted_budget_still_serves_cache():
    client, _ = budgeted(Script(), budget=1)
    client.get("https://api.sam.gov/opps")
    assert client.get("https://api.sam.gov/opps").from_cache


def test_budget_day_is_utc(isolated_paths, monkeypatch):
    client, _ = budgeted(Script(), budget=1)
    client.get("https://api.sam.gov/opps")
    data = json.loads((isolated_paths / "http_cache" / "_budget.json").read_text())
    assert data["date"] == datetime.now(UTC).date().isoformat()
    # A new UTC day resets the count.
    monkeypatch.setattr(Http, "_today", staticmethod(lambda: "2999-01-01"))
    assert client.budget_remaining("api.sam.gov") == 1


# ---- download ------------------------------------------------------------------


class FileServer:
    def __init__(
        self, body=b"x" * 10_000, etag='"v1"', last_modified="Wed, 01 Jan 2026 00:00:00 GMT"
    ):
        self.body = body
        self.etag = etag
        self.last_modified = last_modified
        self.requests: list[httpx.Request] = []
        self.fail_first: list[int] = []

    def __call__(self, request):
        self.requests.append(request)
        if self.fail_first:
            return httpx.Response(self.fail_first.pop(0))
        if request.headers.get("if-none-match") == self.etag:
            return httpx.Response(304)
        return httpx.Response(
            200,
            content=self.body,
            headers={"etag": self.etag, "last-modified": self.last_modified},
        )


def test_download_writes_file_and_sidecar(tmp_path):
    server = FileServer()
    client, _ = make_http(server)
    dest = tmp_path / "raw" / "tracker.tsv.gz"
    result = client.download("https://redfin.test/tracker.tsv.gz?api_key=S", dest, chunk_size=1024)
    assert result.modified and result.status == 200 and result.bytes == 10_000
    assert dest.read_bytes() == server.body
    meta = json.loads((tmp_path / "raw" / "tracker.tsv.gz.meta.json").read_text())
    assert meta["etag"] == '"v1"' and meta["last_modified"].startswith("Wed")
    assert meta["url"] == "https://redfin.test/tracker.tsv.gz?api_key=***"
    assert not list((tmp_path / "raw").glob("*.part"))


def test_download_not_modified_keeps_file(tmp_path):
    server = FileServer()
    client, _ = make_http(server)
    dest = tmp_path / "f.bin"
    client.download("https://h.test/f", dest)
    result = client.download("https://h.test/f", dest)
    assert not result.modified and result.status == 304 and result.etag == '"v1"'
    assert result.bytes == 10_000 and dest.read_bytes() == server.body
    assert server.requests[1].headers["if-none-match"] == '"v1"'
    assert server.requests[1].headers["if-modified-since"].startswith("Wed")


def test_download_changed_file_replaces_it(tmp_path):
    server = FileServer()
    client, _ = make_http(server)
    dest = tmp_path / "f.bin"
    client.download("https://h.test/f", dest)
    server.body, server.etag = b"new", '"v2"'
    result = client.download("https://h.test/f", dest)
    assert result.modified and dest.read_bytes() == b"new" and result.etag == '"v2"'


def test_download_force_skips_conditional_headers(tmp_path):
    server = FileServer()
    client, _ = make_http(server)
    dest = tmp_path / "f.bin"
    client.download("https://h.test/f", dest)
    assert client.download("https://h.test/f", dest, force=True).modified
    assert "if-none-match" not in server.requests[1].headers


def test_download_without_previous_file_sends_no_conditional_headers(tmp_path):
    server = FileServer()
    client, _ = make_http(server)
    (tmp_path / "f.bin.meta.json").write_text(json.dumps({"etag": '"v1"'}))
    assert client.download("https://h.test/f", tmp_path / "f.bin").modified
    assert "if-none-match" not in server.requests[0].headers


def test_download_retries_transient_errors_and_bypasses_json_cache(tmp_path, isolated_paths):
    server = FileServer()
    server.fail_first = [503]
    client, sleeps = make_http(server)
    assert client.download("https://h.test/f", tmp_path / "f.bin").modified
    assert len(sleeps) == 1
    assert not list((isolated_paths / "http_cache").rglob("*.json"))


def test_download_error_leaves_previous_file(tmp_path):
    server = FileServer()
    client, _ = make_http(server)
    dest = tmp_path / "f.bin"
    client.download("https://h.test/f", dest)
    server.fail_first = [404]
    with pytest.raises(HttpError) as e:
        client.download("https://h.test/f", dest, force=True)
    assert e.value.status == 404
    assert dest.read_bytes() == server.body and not list(tmp_path.glob("*.part"))


def test_download_counts_toward_budget(tmp_path):
    server = FileServer()
    client, _ = make_http(server)
    client.set_policy("h.test", HostPolicy(daily_budget=2))
    client.download("https://h.test/f", tmp_path / "f.bin")
    client.download("https://h.test/f", tmp_path / "f.bin")  # a 304 was still a request
    with pytest.raises(RequestBudgetExceeded):
        client.download("https://h.test/f", tmp_path / "f.bin")
