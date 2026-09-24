import json
from datetime import UTC, datetime

import pytest

from core import publish
from core.schema import ManifestEntry


def entry(agent_id: str, **kw) -> ManifestEntry:
    base = dict(
        id=agent_id,
        name=agent_id,
        route=f"/{agent_id}",
        status="ok",
        last_run_at=datetime(2026, 9, 23, tzinfo=UTC),
        last_data_change_at=None,
        expected_interval_hours=24,
        next_run_hint="daily",
        headline="h",
        key_stats=[],
        run_cost_usd=0.0,
    )
    return ManifestEntry(**{**base, **kw})


def test_trim_history_keeps_newest(tmp_path):
    h = tmp_path / "history"
    h.mkdir()
    for d in ["2026-09-01", "2026-09-03", "2026-09-02", "2026-09-04"]:
        (h / f"{d}.json").write_text("{}")
    (h / "notes.txt").write_text("keep me")
    removed = publish.trim_history(h, keep=2)
    assert sorted(p.name for p in removed) == ["2026-09-01.json", "2026-09-02.json"]
    assert sorted(p.name for p in h.iterdir()) == [
        "2026-09-03.json",
        "2026-09-04.json",
        "notes.txt",
    ]


def test_manifest_upsert_orders_by_registry(isolated_paths):
    publish.upsert_manifest(entry("repo_maint"))
    publish.upsert_manifest(entry("macro"))
    publish.upsert_manifest(entry("real_estate"))
    m = publish.upsert_manifest(entry("macro", headline="updated"))
    assert [a.id for a in m.agents] == ["real_estate", "macro", "repo_maint"]
    on_disk = json.loads((isolated_paths / "site_data" / "manifest.json").read_text())
    assert on_disk["agents"][1]["headline"] == "updated"
    assert on_disk["generated_at"].endswith("Z")


def test_corrupt_manifest_is_rebuilt(isolated_paths):
    path = isolated_paths / "site_data" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"nope": 1}')
    m = publish.upsert_manifest(entry("grants"))
    assert [a.id for a in m.agents] == ["grants"]


@pytest.mark.parametrize(
    "rel", ["../x.json", "/abs.json", "latest.json", "history/2026-01-01.json", "data.csv"]
)
def test_extra_file_paths_are_restricted(rel):
    with pytest.raises(ValueError):
        publish._safe_relpath(rel)


def test_write_json_is_compact(tmp_path):
    size = publish.write_json(tmp_path / "a.json", {"a": [1, 2], "b": "é"})
    assert (tmp_path / "a.json").read_text() == '{"a":[1,2],"b":"é"}'
    assert size == len('{"a":[1,2],"b":"é"}'.encode())
    assert not list(tmp_path.glob("*.tmp"))
