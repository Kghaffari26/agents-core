import json
from datetime import UTC, datetime

import pytest

from agents_core import publish
from agents_core.schema import ManifestEntry


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


def test_write_and_read_manifest_entry(isolated_paths):
    publish.write_manifest_entry(entry("macro", headline="first"))
    path = isolated_paths / "public_data" / "manifest-entry.json"
    assert json.loads(path.read_text())["headline"] == "first"

    publish.write_manifest_entry(entry("macro", headline="second"))
    got = publish.read_previous_manifest_entry()
    assert got is not None
    assert got.headline == "second"


def test_read_previous_manifest_entry_missing_returns_none(isolated_paths):
    assert publish.read_previous_manifest_entry() is None


def test_read_previous_manifest_entry_corrupt_returns_none(isolated_paths):
    path = isolated_paths / "public_data" / "manifest-entry.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"nope": 1}')
    assert publish.read_previous_manifest_entry() is None


@pytest.mark.parametrize(
    "rel",
    [
        "../x.json",
        "/abs.json",
        "latest.json",
        "manifest-entry.json",
        "costs-summary.json",
        "schema.json",
        "history/2026-01-01.json",
        "data.csv",
    ],
)
def test_extra_file_paths_are_restricted(rel):
    with pytest.raises(ValueError):
        publish._safe_relpath(rel)


def test_write_json_is_compact(tmp_path):
    size = publish.write_json(tmp_path / "a.json", {"a": [1, 2], "b": "é"})
    assert (tmp_path / "a.json").read_text() == '{"a":[1,2],"b":"é"}'
    assert size == len('{"a":[1,2],"b":"é"}'.encode())
    assert not list(tmp_path.glob("*.tmp"))
