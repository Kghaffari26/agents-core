"""meta.warnings, meta_schema_version, and agent-specific meta fields (v0.2.0)."""

import json

import pytest

from agents_core import export_schemas, runner
from agents_core.agent import AgentResult
from agents_core.schema import META_SCHEMA_VERSION, AgentOutput, RunMeta
from tests.conftest import FakeClient, fake_message
from tests.fake_agent import FakeAgent, FakeOutput


class GrantsMeta(RunMeta):
    sam_budget_exhausted: bool = False
    sam_requests_used: int = 0


class GrantsOutput(AgentOutput):
    meta: GrantsMeta
    headline_value: float
    brief: str


class GrantsAgent(FakeAgent):
    output_model = GrantsOutput

    def __init__(self, meta_fields=None, **kw):
        super().__init__(**kw)
        self.meta_fields = meta_fields or {}

    def analyze(self, ctx, data) -> AgentResult:
        result = super().analyze(ctx, data)
        result.meta_fields = self.meta_fields
        return result


def run(agent):
    return runner.run(agent, llm_client=FakeClient([fake_message("Index rose.")]))


def latest(root):
    return json.loads((root / "public_data" / "latest.json").read_text())


def test_meta_schema_version_is_1_1_0():
    assert META_SCHEMA_VERSION == "1.1.0"


def test_v0_1_0_agent_publishes_empty_warnings_and_meta_version(isolated_paths):
    assert run(FakeAgent()) == 0
    meta = latest(isolated_paths)["meta"]
    assert meta["warnings"] == []
    assert meta["meta_schema_version"] == "1.1.0"


def test_v0_1_0_latest_json_without_new_fields_still_validates(isolated_paths):
    assert run(FakeAgent()) == 0
    data = latest(isolated_paths)
    del data["meta"]["warnings"], data["meta"]["meta_schema_version"]
    FakeOutput.model_validate(data)


def test_result_warnings_and_ctx_warn_are_published_in_order(isolated_paths):
    class Warner(FakeAgent):
        def fetch(self, ctx):
            ctx.warn("FRED series CPIAUCSL is 3 days late")
            ctx.warn("dup")
            return super().fetch(ctx)

        def analyze(self, ctx, data):
            result = super().analyze(ctx, data)
            result.warnings = ["batch timed out; briefs ran synchronously", "dup"]
            return result

    assert run(Warner()) == 0
    assert latest(isolated_paths)["meta"]["warnings"] == [
        "batch timed out; briefs ran synchronously",
        "dup",
        "FRED series CPIAUCSL is 3 days late",
    ]


def test_agent_specific_meta_fields_are_published(isolated_paths):
    agent = GrantsAgent({"sam_budget_exhausted": True, "sam_requests_used": 3})
    assert run(agent) == 0
    meta = latest(isolated_paths)["meta"]
    assert meta["sam_budget_exhausted"] is True and meta["sam_requests_used"] == 3
    assert meta["agent"] == "macro"


def test_agent_specific_meta_fields_default_when_not_returned(isolated_paths):
    assert run(GrantsAgent()) == 0
    meta = latest(isolated_paths)["meta"]
    assert meta["sam_budget_exhausted"] is False and meta["sam_requests_used"] == 0


def test_agent_specific_meta_fields_appear_in_schema_json(isolated_paths):
    assert run(GrantsAgent()) == 0
    schema = json.loads((isolated_paths / "public_data" / "schema.json").read_text())
    meta_schema = schema["$defs"]["GrantsMeta"]
    assert {"sam_budget_exhausted", "sam_requests_used", "warnings"} <= set(
        meta_schema["properties"]
    )
    assert "warnings" in meta_schema["required"]
    assert meta_schema["additionalProperties"] is False


def test_undeclared_meta_field_fails_the_run(isolated_paths):
    assert run(FakeAgent()) == 0
    before = latest(isolated_paths)
    assert run(GrantsAgent({"not_declared": 1})) == 1
    assert latest(isolated_paths) == before


@pytest.mark.parametrize("key", ["agent", "cost_usd", "warnings"])
def test_meta_fields_cannot_override_shared_fields(key, isolated_paths):
    assert run(GrantsAgent({key: "spoofed"})) == 1
    assert not (isolated_paths / "public_data" / "latest.json").exists()


def test_run_meta_schema_lists_warnings_as_required_array():
    schema = export_schemas.schema_dict(FakeOutput)
    props = schema["$defs"]["RunMeta"]["properties"]
    assert props["warnings"]["type"] == "array"
    assert props["warnings"]["items"] == {"type": "string"}
    assert "warnings" in schema["$defs"]["RunMeta"]["required"]
