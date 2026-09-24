"""Number guard: checks that model narrative uses only numbers that came from data.

    result = verify_numbers(text, facts)          # facts: numbers, or a nested dict/list
    if not result.ok:
        print(result.unsupported)                 # e.g. ["3.1%", "$4.2B"]

A numeric token in the text passes when some fact f satisfies

    round(abs(f) * s, d) == abs(token)

where d is the token's decimal places and s is 1 or the token's scale adjustment
("142K" matches a fact of 142 or 142000; "$1.2M" matches 1_200_000). Percent and
basis-point tokens also match a fact x100 (0.987 -> "98.7%", 0.25 pp -> "25 bp"). A tie
at the rounding boundary is accepted either way, since Python formatting rounds half-even.

Ignored: years 1900-2100, day numbers after a month name, numeric dates, clock times,
ordinals, fixed terms (2-year, 3-month, 52-week, Q1-Q4, 401(k), P0-P3), version strings,
and anything passed in `allow`.

`llm.complete`/`llm.structured` take a guard (see `text_guard` and `fields_guard`) and
retry once, then fall back to a template, when it fails.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Real
from typing import Any

from pydantic import BaseModel

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

# Spans blanked out before tokenizing. Each is replaced by spaces of equal length.
_IGNORE = re.compile(
    "|".join(
        [
            rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b",  # Sep 19, October 3rd
            r"\b\d+(?:st|nd|rd|th)\b",  # ordinals
            r"\b\d+-(?:year|yr|month|mo|week|wk|day)s?\b",  # 2-year, 3-month, 52-week
            r"\b\d{4}-\d{2}(?:-\d{2})?\b",  # 2026-09, 2026-09-19
            r"(?<![-\d])\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",  # 9/19, 9/19/2026 (not 4-1/4)
            r"\b\d{1,2}:\d{2}\b",  # 8:30
            r"\bv?\d+\.\d+\.\d+(?:[-+][\w.]+)?\b",  # 1.2.3, v1.2.3-rc1
            r"\bv\d+(?:\.\d+)?\b",  # v1, v1.2
            r"\b40[13]\([kb]\)",  # 401(k), 403(b)
        ]
    ),
    re.IGNORECASE,
)

_TOKEN = re.compile(
    r"""
    (?<![\w./])                                # not glued to a word, decimal, or slash
    (?P<sign>[+\-−])?
    (?P<cur>\$)?
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)
    (?:\.(?P<dec>\d+))?
    (?:-(?P<fnum>\d+)/(?P<fden>\d+))?          # mixed fraction: 4-1/4
    (?:
        (?P<suffix>(?-i:[KkMBT]))(?![A-Za-z])  # 142K, $1.2M, $4.2B (case-sensitive)
      | \s?(?P<word>thousand|million|billion|trillion)\b
    )?
    (?:\s?(?P<unit>%|percentage\s+points?\b|percent\b|pp\b|bps?\b|basis\s+points?\b))?
    (?![\w])
    """,
    re.VERBOSE | re.IGNORECASE,
)

_SCALES = {
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}


@dataclass(frozen=True)
class NumberToken:
    raw: str
    value: float  # magnitude as written, before scale: "142K" -> 142.0
    decimals: int
    scale: float = 1.0  # "142K" -> 1e3
    unit: str | None = None  # "%", "pp", "bp" or None


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    unsupported: list[str] = field(default_factory=list)


def collect_numbers(obj: Any) -> list[float]:
    """Every int/float in `obj`, recursing into dicts, lists, tuples, sets, pydantic models.

    Skips bools, None, NaN/inf, strings, and dict keys.
    """
    out: list[float] = []

    def walk(x: Any) -> None:
        if x is None or isinstance(x, bool | str | bytes):
            return
        if isinstance(x, Real):
            f = float(x)
            if math.isfinite(f):
                out.append(f)
        elif isinstance(x, BaseModel):
            walk(x.model_dump())
        elif isinstance(x, Mapping):
            for v in x.values():
                walk(v)
        elif isinstance(x, Iterable):
            for v in x:
                walk(v)
        elif isinstance(x, Decimal):
            walk(float(x))

    walk(obj)
    return out


def _decimals_of(value: float) -> int:
    exp = Decimal(repr(value)).normalize().as_tuple().exponent
    return max(-exp, 0) if isinstance(exp, int) else 0


def _unit(raw_unit: str | None) -> str | None:
    if not raw_unit:
        return None
    u = raw_unit.lower()
    if u == "%" or u == "percent":
        return "%"
    if u == "pp" or u.startswith("percentage"):
        return "pp"
    return "bp"


def _mask(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def _allow_pattern(allow: Iterable[str]) -> re.Pattern[str] | None:
    terms = sorted({a for a in allow if a}, key=len, reverse=True)
    if not terms:
        return None
    alternation = "|".join(re.escape(t) for t in terms)
    return re.compile(rf"(?<![\w.])(?:{alternation})(?![\w])")


def extract_numbers(text: str, *, allow: Iterable[str] = ()) -> list[NumberToken]:
    """Numeric tokens in `text` that need a supporting fact, in order of appearance."""
    allowed = _allow_pattern(allow)
    if allowed is not None:
        text = _mask(text, allowed)
    text = _mask(text, _IGNORE)

    tokens = []
    for m in _TOKEN.finditer(text):
        int_part = m["int"].replace(",", "")
        suffix = (m["suffix"] or m["word"] or "").lower()
        unit = _unit(m["unit"])
        plain = not (m["cur"] or m["dec"] or m["fnum"] or suffix or unit or m["sign"])
        if plain and "," not in m["int"] and len(int_part) == 4 and 1900 <= int(int_part) <= 2100:
            continue  # a year
        if m["fnum"]:
            den = int(m["fden"])
            if den == 0:
                continue
            value = int(int_part) + int(m["fnum"]) / den
            decimals = min(_decimals_of(value), 6)
        else:
            value = float(f"{int_part}.{m['dec']}" if m["dec"] else int_part)
            decimals = len(m["dec"] or "")
        tokens.append(
            NumberToken(
                raw=m.group(0).strip(),
                value=value,
                decimals=decimals,
                scale=_SCALES.get(suffix, 1.0),
                unit=unit,
            )
        )
    return tokens


def _matches(token: NumberToken, facts: Sequence[float]) -> bool:
    factors = [1.0]
    if token.scale != 1.0:
        factors.append(1.0 / token.scale)
    if token.unit in ("%", "bp"):
        factors.append(100.0)
    # Accept any fact that rounds to the token at its precision, ties either way.
    tolerance = 0.5 * 10 ** (-token.decimals) + 1e-9 * max(1.0, token.value)
    for f in facts:
        magnitude = abs(f)
        for s in factors:
            if abs(magnitude * s - token.value) <= tolerance:
                return True
    return False


def verify_numbers(text: str, facts: Any, *, allow: Iterable[str] = ()) -> GuardResult:
    """Check every number in `text` against `facts` (numbers or any nested structure)."""
    fact_values = facts if _is_float_list(facts) else collect_numbers(facts)
    unsupported: list[str] = []
    for token in extract_numbers(text, allow=allow):
        if not _matches(token, fact_values) and token.raw not in unsupported:
            unsupported.append(token.raw)
    return GuardResult(ok=not unsupported, unsupported=unsupported)


def _is_float_list(x: Any) -> bool:
    return isinstance(x, list) and all(type(v) is float for v in x)


# ---- guards for core.llm -----------------------------------------------------


def text_guard(facts: Any, *, allow: Iterable[str] = ()) -> Callable[[str], GuardResult]:
    """Guard for `llm.complete`: checks the whole returned text."""
    values = collect_numbers(facts)
    allow = tuple(allow)
    return lambda text: verify_numbers(text, values, allow=allow)


def fields_guard(
    facts: Any, fields: Sequence[str], *, allow: Iterable[str] = ()
) -> Callable[[Any], GuardResult]:
    """Guard for `llm.structured`: checks only the named narrative fields.

    Fields are dotted paths; lists are walked element-wise, so "bullets" checks a
    list[str] and "items.summary" checks `summary` on every element of `items`.
    Numeric fields are not checked here: they should be copied from data in code.
    """
    values = collect_numbers(facts)
    allow = tuple(allow)

    def guard(output: Any) -> GuardResult:
        text = "\n".join(_strings_at(output, fields))
        return verify_numbers(text, values, allow=allow)

    return guard


def _strings_at(obj: Any, paths: Sequence[str]) -> list[str]:
    out: list[str] = []
    for path in paths:
        _walk_path(obj, path.split("."), out)
    return out


def _walk_path(obj: Any, parts: list[str], out: list[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, list | tuple):
        for item in obj:
            _walk_path(item, parts, out)
        return
    if not parts:
        if isinstance(obj, str):
            out.append(obj)
        return
    head, rest = parts[0], parts[1:]
    if isinstance(obj, Mapping):
        if head not in obj:
            raise KeyError(f"guard field {head!r} not found")
        _walk_path(obj[head], rest, out)
    elif hasattr(obj, head):
        _walk_path(getattr(obj, head), rest, out)
    else:
        raise KeyError(f"guard field {head!r} not found on {type(obj).__name__}")
