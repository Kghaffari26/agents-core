from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every writable path at tmp_path so tests never touch the repo."""
    monkeypatch.setenv("PUBLISH_DIR", str(tmp_path / "site_data"))
    monkeypatch.setenv("COSTS_PATH", str(tmp_path / "costs.jsonl"))
    monkeypatch.setenv("HTTP_CACHE_DIR", str(tmp_path / "http_cache"))
    monkeypatch.setenv("GUARD_FAILURES_PATH", str(tmp_path / "guard_failures.jsonl"))
    monkeypatch.setenv("MAX_RUN_USD", "0.50")
    return tmp_path


def fake_message(
    text: str = "hello",
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read: int = 0,
    cache_write: int = 0,
    parsed: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
        parsed_output=parsed,
    )


class FakeBatches:
    def __init__(self, results: list[SimpleNamespace], polls_before_end: int = 1) -> None:
        self.results_list = results
        self.polls_before_end = polls_before_end
        self.created: list[Any] = []
        self.cancelled: list[str] = []

    def create(self, requests: list[Any]) -> SimpleNamespace:
        self.created.append(requests)
        return SimpleNamespace(id="batch_1")

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        if self.polls_before_end > 0:
            self.polls_before_end -= 1
            return SimpleNamespace(processing_status="in_progress")
        return SimpleNamespace(processing_status="ended")

    def cancel(self, batch_id: str) -> None:
        self.cancelled.append(batch_id)

    def results(self, batch_id: str) -> list[SimpleNamespace]:
        return self.results_list


class FakeMessages:
    def __init__(self, responses: list[SimpleNamespace], batches: FakeBatches | None = None):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.batches = batches or FakeBatches([])

    def create(self, **params: Any) -> SimpleNamespace:
        self.calls.append(params)
        return self.responses.pop(0)

    def parse(self, **params: Any) -> SimpleNamespace:
        self.calls.append(params)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[SimpleNamespace] | None = None, batches=None) -> None:
        self.messages = FakeMessages(responses or [], batches)
