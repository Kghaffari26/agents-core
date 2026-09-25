from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from agents_core.schema import KeyStat, ManifestEntry, RunMeta, Source


def meta(**overrides):
    base = dict(
        agent="macro",
        schema_version="1.0.0",
        run_id="2026-09-23T14-00-05Z-a1b2c3",
        started_at=datetime(2026, 9, 23, 14, 0, 5, 123456, tzinfo=UTC),
        finished_at=datetime(2026, 9, 23, 7, 1, 12, tzinfo=timezone(timedelta(hours=-7))),
        status="ok",
        data_changed=True,
        cost_usd=0.041,
        model_usage={"smart": {"input_tokens": 6120, "output_tokens": 540}},
        sources=[
            {
                "name": "FRED",
                "url": "https://fred.stlouisfed.org/",
                "retrieved_at": "2026-09-23T14:00:09Z",
            }
        ],
    )
    return RunMeta(**{**base, **overrides})


def test_timestamps_publish_as_utc_z_seconds():
    dumped = meta().model_dump(mode="json")
    assert dumped["started_at"] == "2026-09-23T14:00:05Z"
    assert dumped["finished_at"] == "2026-09-23T14:01:12Z"
    assert dumped["sources"][0]["retrieved_at"] == "2026-09-23T14:00:09Z"
    assert dumped["model_usage"]["fast"] == {"input_tokens": 0, "output_tokens": 0}


def test_naive_timestamps_rejected():
    with pytest.raises(ValidationError):
        meta(started_at=datetime(2026, 9, 23, 14, 0, 5))


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        Source(
            name="FRED",
            url="https://fred.stlouisfed.org/",
            retrieved_at="2026-09-23T14:00:09Z",
            extra="x",
        )


def test_schema_version_must_be_semver():
    with pytest.raises(ValidationError):
        meta(schema_version="1.0")


def test_key_stat_format_is_constrained():
    KeyStat(
        label="x",
        value=1,
        format="currency_compact",
        delta=0.02,
        delta_format="percent_signed",
        good_direction="up",
    )
    with pytest.raises(ValidationError):
        KeyStat(label="x", value=1, format="dollars")


def test_manifest_entry_limits_key_stats():
    stat = KeyStat(label="x", value=1, format="count")
    with pytest.raises(ValidationError):
        ManifestEntry(
            id="macro",
            name="Macro",
            route="/macro",
            status="ok",
            last_run_at="2026-09-23T14:00:00Z",
            last_data_change_at=None,
            expected_interval_hours=24,
            next_run_hint="weekdays",
            headline="h",
            key_stats=[stat] * 5,
            run_cost_usd=0.01,
        )
