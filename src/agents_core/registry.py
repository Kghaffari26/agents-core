"""Agent discovery via the `agents_core.agents` entry-point group.

An agent repo registers itself by adding, in its own `pyproject.toml`:

    [project.entry-points."agents_core.agents"]
    real_estate = "real_estate_agent.agent:AGENT"

where `AGENT` is a module-level instance of `agents_core.agent.Agent`. This package
ships no agents of its own — it's a dependency, not a standalone tool.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from agents_core.agent import Agent

log = logging.getLogger(__name__)

GROUP = "agents_core.agents"


class AgentNotFound(LookupError):
    pass


def discover_agents() -> dict[str, str]:
    """`{agent name: "module:attr"}` for every registered agent."""
    return {ep.name: ep.value for ep in entry_points(group=GROUP)}


def load(name: str) -> Agent:
    matches = [ep for ep in entry_points(group=GROUP) if ep.name == name]
    if not matches:
        known = sorted(discover_agents())
        raise AgentNotFound(f"no agent named {name!r} registered under {GROUP!r}; known: {known}")
    agent = matches[0].load()
    if not isinstance(agent, Agent):
        raise TypeError(
            f"entry point {name!r} ({matches[0].value}) is {type(agent).__name__}, "
            "expected an agents_core.agent.Agent instance"
        )
    if agent.id != name:
        raise TypeError(f"entry point {name!r}'s Agent.id is {agent.id!r}, expected {name!r}")
    return agent


def load_available() -> list[Agent]:
    """Every registered agent that loads cleanly. Broken registrations are skipped."""
    agents = []
    for name in discover_agents():
        try:
            agents.append(load(name))
        except (AgentNotFound, TypeError, ImportError) as e:
            log.warning("skipping agent %r: %s", name, e)
    return agents
