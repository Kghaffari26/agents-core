import json

from core import export_schemas
from tests.fake_agent import FakeAgent


def test_exports_shared_schemas_and_check_mode(tmp_path):
    changed = export_schemas.export(tmp_path)
    names = sorted(p.name for p in changed)
    assert "manifest.schema.json" in names and "costs_summary.schema.json" in names
    manifest = json.loads((tmp_path / "manifest.schema.json").read_text())
    assert manifest["$id"] == "manifest"
    assert manifest["properties"]["generated_at"]["type"] == "string"
    assert export_schemas.export(tmp_path, check=True) == []
    (tmp_path / "manifest.schema.json").write_text("{}")
    assert [p.name for p in export_schemas.export(tmp_path, check=True)] == ["manifest.schema.json"]


def test_agent_schemas_include_extra_models(tmp_path, monkeypatch):
    monkeypatch.setattr(export_schemas.registry, "load_available", lambda: [FakeAgent()])
    export_schemas.export(tmp_path)
    assert (tmp_path / "macro.schema.json").is_file()
    assert (tmp_path / "macro.detail.schema.json").is_file()
    schema = json.loads((tmp_path / "macro.schema.json").read_text())
    assert schema["additionalProperties"] is False
    assert "meta" in schema["required"]
