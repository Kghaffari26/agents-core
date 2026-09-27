"""Requests built by the real Anthropic SDK (the version in uv.lock), sent to a local
stub HTTP server: `temperature` must reach the request body for every call type
without the SDK rejecting it (v0.3.0 raised `TypeError: Messages.parse() got an
unexpected keyword argument 'temperature'`)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel

from agents_core import llm as llm_module
from agents_core.costs import CostTracker
from agents_core.llm import LLM, BatchItem, tier_config


class Brief(BaseModel):
    summary: str


def _message(text: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": tier_config("fast").model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _batch(status: str, base_url: str) -> dict:
    return {
        "id": "msgbatch_1",
        "type": "message_batch",
        "processing_status": status,
        "request_counts": {
            "processing": 0,
            "succeeded": 1,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "created_at": "2026-09-27T00:00:00Z",
        "expires_at": "2026-09-28T00:00:00Z",
        "ended_at": "2026-09-27T00:01:00Z",
        "archived_at": None,
        "cancel_initiated_at": None,
        "results_url": f"{base_url}/v1/messages/batches/msgbatch_1/results",
    }


class StubAPI:
    """Records each request's JSON body and answers like the Messages API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.text = "hello"
        self.requests: list[tuple[str, str, dict | None]] = []

    def respond(self, method: str, path: str, body: dict | None) -> tuple[int, str, str]:
        self.requests.append((method, path, body))
        if path == "/v1/messages":
            return 200, "application/json", json.dumps(_message(self.text))
        if path == "/v1/messages/batches" and method == "POST":
            return 200, "application/json", json.dumps(_batch("in_progress", self.base_url))
        if path == "/v1/messages/batches/msgbatch_1":
            return 200, "application/json", json.dumps(_batch("ended", self.base_url))
        if path.endswith("/results"):
            line = {
                "custom_id": "a",
                "result": {"type": "succeeded", "message": _message(self.text)},
            }
            return 200, "application/x-jsonl", json.dumps(line) + "\n"
        error = {"type": "error", "error": {"type": "not_found_error", "message": path}}
        return 404, "application/json", json.dumps(error)

    def bodies(self, path: str = "/v1/messages") -> list[dict]:
        return [b for m, p, b in self.requests if p == path and m == "POST" and b is not None]


@pytest.fixture
def stub(isolated_paths, monkeypatch):
    api: StubAPI

    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            path = self.path.split("?")[0]
            status, ctype, text = api.respond(self.command, path, json.loads(raw) if raw else None)
            data = text.encode()
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _handle

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    api = StubAPI(base_url)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    # The SDK client comes from agents_core.llm (the only module importing anthropic).
    client = llm_module.anthropic.Anthropic(api_key="test-key", base_url=base_url, max_retries=0)
    tracker = CostTracker(agent="macro", run_id="r1", max_usd=1.0)
    try:
        yield api, LLM(tracker, client=client, sleep=lambda s: None)
    finally:
        server.shutdown()
        server.server_close()


def test_structured_sends_temperature_through_real_sdk(stub):
    api, llm = stub
    api.text = '{"summary": "rates held"}'
    out = llm.structured("fast", "p", Brief, system="s", temperature=0)
    assert out == Brief(summary="rates held")
    [body] = api.bodies()
    assert body["temperature"] == 0
    assert body["output_config"]["format"]["type"] == "json_schema"


def test_structured_with_tier_temperature(stub, monkeypatch):
    api, llm = stub
    api.text = '{"summary": "ok"}'
    cfg = tier_config("fast")
    monkeypatch.setattr(
        llm_module,
        "tier_config",
        lambda tier: llm_module.TierConfig(cfg.model, cfg.max_tokens, temperature=0.2),
    )
    llm.structured("fast", "p", Brief, system="s")
    assert api.bodies()[0]["temperature"] == 0.2


def test_complete_and_converse_send_temperature(stub):
    api, llm = stub
    assert llm.complete("fast", "p", system="s", temperature=0.5) == "hello"
    turn = llm.converse("fast", [{"role": "user", "content": "hi"}], system="s", temperature=0.25)
    assert turn.stop_reason == "end_turn"
    assert [b["temperature"] for b in api.bodies()] == [0.5, 0.25]


def test_no_temperature_key_when_unset(stub):
    api, llm = stub
    llm.complete("fast", "p", system="s")
    assert "temperature" not in api.bodies()[0]


def test_batch_sends_temperature_through_real_sdk(stub):
    api, llm = stub
    api.text = '{"summary": "batched"}'
    results = llm.batch(
        "fast", [BatchItem("a", "x")], system="s", output_model=Brief, temperature=0
    )
    assert results["a"].value == Brief(summary="batched")
    [body] = api.bodies("/v1/messages/batches")
    params = body["requests"][0]["params"]
    assert params["temperature"] == 0
    assert "extra_body" not in params
