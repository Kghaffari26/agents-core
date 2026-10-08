"""agents-core: shared framework for scheduled data agents.

HTTP caching, cost tracking with a per-run budget cap, a number guard for LLM
narrative, prompt-cached/tiered Claude calls, a budgeted tool-use loop
(`agent_loop`), run tracing (`tracing`), an eval harness (`evals`), a JSON publisher
with dated history, and agent registration via the `agents_core.agents` entry-point
group. See the README for the full data-branch contract.
"""

__version__ = "0.3.2"
