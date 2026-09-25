import json

import pytest

from agents_core import export_schemas
from tests.fake_agent import FakeOutput


def test_schema_dict_shape():
    schema = export_schemas.schema_dict(FakeOutput)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert "meta" in schema["required"]
    assert schema["properties"]["headline_value"]["type"] == "number"


def test_write_schema(isolated_paths):
    path = export_schemas.write_schema(FakeOutput)
    assert path == isolated_paths / "public_data" / "schema.json"
    data = json.loads(path.read_text())
    assert data["required"] == ["meta", "headline_value", "brief"]


def test_write_schema_explicit_publish_dir(tmp_path):
    path = export_schemas.write_schema(FakeOutput, publish_dir=tmp_path / "out")
    assert path == tmp_path / "out" / "schema.json"
    assert path.is_file()


def test_cli_lists_agents_when_none_named(monkeypatch, capsys):
    monkeypatch.setattr(
        export_schemas.registry, "discover_agents", lambda: {"macro": "pkg.agent:AGENT"}
    )
    assert export_schemas.main([]) == 0
    assert "macro -> pkg.agent:AGENT" in capsys.readouterr().out


def test_cli_writes_named_agent_schema(monkeypatch, isolated_paths, capsys):
    from tests.fake_agent import AGENT

    monkeypatch.setattr(export_schemas.registry, "load", lambda name: AGENT)
    assert export_schemas.main(["macro"]) == 0
    assert (isolated_paths / "public_data" / "schema.json").is_file()
    assert "wrote" in capsys.readouterr().out


def test_cli_unknown_agent_exits_2(capsys):
    assert export_schemas.main(["nope"]) == 2
    assert "nope" in capsys.readouterr().err


@pytest.mark.parametrize("value", [1, "x", None])
def test_write_schema_rejects_non_model_gracefully(value):
    with pytest.raises(AttributeError):
        export_schemas.schema_dict(value)  # not a BaseModel subclass
