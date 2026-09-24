"""Known agents. Each lives in `agents/<id>/agent.py` with a module-level `AGENT`."""

from __future__ import annotations

import importlib
import logging

from core.agent import Agent

log = logging.getLogger(__name__)

# Order here is the order of cards on the overview page.
AGENT_IDS: tuple[str, ...] = ("real_estate", "macro", "grants", "repo_maint")


class UnknownAgent(LookupError):
    pass


def load(agent_id: str) -> Agent:
    if agent_id not in AGENT_IDS:
        raise UnknownAgent(f"unknown agent {agent_id!r}; expected one of {', '.join(AGENT_IDS)}")
    module_name = f"agents.{agent_id}.agent"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name in {"agents", f"agents.{agent_id}", module_name}:
            raise UnknownAgent(f"agent {agent_id!r} is not implemented yet ({module_name})") from e
        raise
    agent = getattr(module, "AGENT", None)
    if not isinstance(agent, Agent):
        raise TypeError(f"{module_name}.AGENT must be an instance of core.agent.Agent")
    if agent.id != agent_id:
        raise TypeError(f"{module_name}.AGENT.id is {agent.id!r}, expected {agent_id!r}")
    return agent


def load_available() -> list[Agent]:
    """Every agent that is implemented so far; unimplemented ones are skipped."""
    agents = []
    for agent_id in AGENT_IDS:
        try:
            agents.append(load(agent_id))
        except UnknownAgent as e:
            log.debug("%s", e)
    return agents
