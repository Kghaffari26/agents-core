import pytest
from pydantic import BaseModel

from core.guards import (
    collect_numbers,
    extract_numbers,
    fields_guard,
    text_guard,
    verify_numbers,
)


def passes(text, facts, **kw):
    return verify_numbers(text, facts, **kw).ok


def unsupported(text, facts, **kw):
    return verify_numbers(text, facts, **kw).unsupported


# ---- matching rule ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "facts"),
    [
        ("The target range is 4-1/4 to 4-1/2 percent.", [4.25, 4.5]),
        ("The Fed cut by 25 bp.", [25]),
        ("The Fed cut by 25 bp.", [0.25]),  # fact in pp, text in bp
        ("Unemployment rose +0.25 pp.", [0.25]),
        ("Deals worth $1.2M closed.", [1_200_000]),
        ("Payrolls rose 142K.", [142000]),
        ("Payrolls rose 142K.", [142]),  # fact already in thousands
        ("Payrolls rose +142K.", [142_300]),  # rounds to the token's precision
        ("Core inflation fell 0.3 points.", [-0.3]),
        ("Core inflation was -0.3.", [0.3]),
        ("CPI ran at 2.9%.", [2.94]),
        ("CPI ran at 2.9%.", [2.85]),  # tie at the boundary is accepted
        ("Sale-to-list was 98.7%.", [0.987]),  # ratio fact shown as a percent
        ("Median price $1,234.5 thousand", [1234.5]),
        ("Awards total $4.2B.", [4_200_000_000]),
        ("Awards total $4.2 billion.", [4_249_999_999]),
        ("Inventory is 12,431 homes.", [12431]),
        ("Rates were 6.84 percent.", [6.84]),
        ("A range of 4.25-4.50% held.", [4.25, 4.5]),
        ("It rose 3 points.", [2.6]),  # integer token, fact rounds to 3
    ],
)
def test_supported_numbers_pass(text, facts):
    assert verify_numbers(text, facts).ok, verify_numbers(text, facts)


@pytest.mark.parametrize(
    ("text", "facts", "bad"),
    [
        ("CPI ran at 2.9%.", [2.96], ["2.9%"]),
        ("Payrolls rose 150K.", [142000], ["150K"]),
        ("Deals worth $1.3M closed.", [1_200_000], ["$1.3M"]),
        ("Unemployment was 4.4% and GDP grew 3.1%.", [4.4], ["3.1%"]),  # invented number
        ("The target range is 4-1/4 to 4-1/2 percent.", [4.25], ["4-1/2 percent"]),
        ("Sold 12,431 homes.", [12430], ["12,431"]),
        ("Cut by 50 bp.", [0.25, 25], ["50 bp"]),
        ("Prices rose 2.9%, then 2.9% again, then 7%.", [2.9], ["7%"]),
    ],
)
def test_unsupported_numbers_fail(text, facts, bad):
    result = verify_numbers(text, facts)
    assert not result.ok
    assert result.unsupported == bad


def test_unsupported_tokens_are_deduplicated():
    assert unsupported("It was 7%, yes 7%.", []) == ["7%"]


# ---- ignored tokens ---------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The 2-year yield rose.",
        "Watch the 10-year and 30-year yields and the 3-month bill.",
        "It is at a 52-week high over the 12-month window.",
        "The next release is on Sep 19.",
        "Minutes come out October 3rd.",
        "Data through Sep 19, 2026.",
        "Prices peaked in 2026.",
        "Since 1999 and through 2100.",
        "Growth slowed in Q3 after Q1.",
        "Tagged v1.2.0 and v2.",
        "Released 1.4.3-rc1 today.",
        "Their 401(k) plans.",
        "Marked P0 and P3 issues.",
        "The 1st and 3rd meetings.",
        "Released 9/19/2026 at 8:30.",
        "Period 2026-09-19 ended.",
        "A text with no numbers at all.",
        "",
    ],
)
def test_ignored_tokens_need_no_facts(text):
    assert verify_numbers(text, []).ok, extract_numbers(text)


def test_year_rule_is_narrow():
    # Currency, decimals, signs and out-of-range values are not years.
    assert unsupported("$2026 and 2026.5 and +2026 and 2200", []) == [
        "$2026",
        "2026.5",
        "+2026",
        "2200",
    ]


def test_allow_list_skips_exact_terms_only():
    text = "The S&P 500 rose while the 2% target held; 12% is new."
    assert unsupported(text, [], allow=["S&P 500", "2%"]) == ["12%"]


# ---- tokenization -----------------------------------------------------------


def test_extract_records_value_decimals_scale_and_unit():
    tokens = extract_numbers("$1,234.5 · 2.9% · -0.3 · +142K · 1.2M · $4.2B · 25 bp · 0.25 pp")
    got = [(t.raw, t.value, t.decimals, t.scale, t.unit) for t in tokens]
    assert got == [
        ("$1,234.5", 1234.5, 1, 1.0, None),
        ("2.9%", 2.9, 1, 1.0, "%"),
        ("-0.3", 0.3, 1, 1.0, None),
        ("+142K", 142.0, 0, 1e3, None),
        ("1.2M", 1.2, 1, 1e6, None),
        ("$4.2B", 4.2, 1, 1e9, None),
        ("25 bp", 25.0, 0, 1.0, "bp"),
        ("0.25 pp", 0.25, 2, 1.0, "pp"),
    ]


def test_mixed_fraction_value():
    (t,) = extract_numbers("4-1/4")
    assert (t.value, t.decimals) == (4.25, 2)


# ---- unrecognized letter suffixes (scale=1, must match exactly) -------------


@pytest.mark.parametrize(
    ("text", "facts", "ok"),
    [
        ("It grew 5m.", [5], True),
        ("It grew 5m.", [5_000_000], False),
        ("It grew 5m.", [6], False),
        ("Up 3x from last year.", [3], True),
        ("Up 3x from last year.", [4], False),
    ],
)
def test_unrecognized_letter_suffix_requires_exact_match(text, facts, ok):
    assert verify_numbers(text, facts).ok is ok


def test_unrecognized_suffix_token_has_scale_one():
    (t5,) = extract_numbers("5m")
    (t3,) = extract_numbers("3x")
    assert (t5.value, t5.decimals, t5.scale, t5.unit) == (5.0, 0, 1.0, None)
    assert (t3.value, t3.decimals, t3.scale, t3.unit) == (3.0, 0, 1.0, None)


def test_known_scale_and_unit_suffixes_still_take_priority_over_unk():
    (t,) = extract_numbers("142K")
    assert (t.scale, t.unit) == (1e3, None)
    (t,) = extract_numbers("25bp")
    assert (t.scale, t.unit) == (1.0, "bp")


# ---- facts ------------------------------------------------------------------


def test_collect_numbers_recurses_and_skips_non_numbers():
    class Stat(BaseModel):
        value: float
        label: str

    facts = {
        "cpi": {"yoy": 2.94, "flags": [True, False], "note": "12", "missing": None},
        "series": [1, 2.5, (3,), {7}],
        "stat": Stat(value=9.5, label="x"),
        "nan": float("nan"),
        4: "keys are not facts",
    }
    assert sorted(collect_numbers(facts)) == [1.0, 2.5, 2.94, 3.0, 7.0, 9.5]


def test_nested_dict_facts():
    facts = {"labor": {"payrolls": {"latest": 142_000, "prior": 89_000}, "u3": 4.4}}
    assert passes("Payrolls rose 142K after 89K; unemployment 4.4%.", facts)
    assert not passes("Payrolls rose 150K.", facts)


def test_text_guard_and_fields_guard():
    class Brief(BaseModel):
        summary: str
        bullets: list[str]
        score: int  # not a narrative field; not checked

    facts = {"cpi": 2.9}
    assert text_guard(facts)("CPI 2.9%").ok
    guard = fields_guard(facts, ["summary", "bullets"])
    assert guard(Brief(summary="CPI 2.9%", bullets=["held at 2.9%"], score=77)).ok
    assert guard(Brief(summary="CPI 2.9%", bullets=["core 3.4%"], score=0)).unsupported == ["3.4%"]


def test_fields_guard_walks_lists_of_models_and_rejects_bad_paths():
    class Item(BaseModel):
        summary: str

    class Out(BaseModel):
        items: list[Item]

    guard = fields_guard([5], ["items.summary"])
    assert guard(Out(items=[Item(summary="5 bids"), Item(summary="none")])).ok
    assert not guard(Out(items=[Item(summary="6 bids")])).ok
    with pytest.raises(KeyError):
        fields_guard([], ["nope"])(Out(items=[]))
