"""Run an agent's evals: `python -m evals.run <agent>`.

Each agent provides `evals/<agent>.py` (or `evals/<agent>/__init__.py`) with a
`CHECKS` list of zero-argument callables. A check passes by returning, and fails by
raising (usually AssertionError). Checks run on committed fixtures, never live data.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback

from core import registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run")
    parser.add_argument("agent", choices=registry.AGENT_IDS)
    args = parser.parse_args(argv)

    try:
        module = importlib.import_module(f"evals.{args.agent}")
    except ModuleNotFoundError as e:
        if e.name == f"evals.{args.agent}":
            print(f"no evals for {args.agent} yet (evals/{args.agent}.py)")
            return 2
        raise

    checks = getattr(module, "CHECKS", [])
    failed = 0
    for check in checks:
        name = getattr(check, "__name__", repr(check))
        try:
            check()
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc(limit=3)
        else:
            print(f"ok   {name}")
    print(f"{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
