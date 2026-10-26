"""
llmswitch.agentic

Chat model and tool loop for agentic workflows. Install the extra first:

    pip install "llmswitch-sdk[agentic]"

    from llmswitch.agentic import ChatLLMSwitch, run_tool_loop

Nothing here is imported by ``import llmswitch``. The names load on first use, so the
extra's dependencies are needed only by code that uses them.
"""
from __future__ import annotations

from typing import Any

_EXPORTS = {
    "ChatLLMSwitch": ".chat",
    "run_tool_loop": ".chat",
    "arun_tool_loop": ".chat",
    "Agent": ".agent",
    "AgentConfig": ".agent",
    "AgentStep": ".agent",
    "ToolExecutionResult": ".execution",
    "idempotent": ".execution",
    "AgentEventHandler": ".events",
    "AgentEvent": ".events",
    "CollectingEventHandler": ".events",
    "LoggingEventHandler": ".events",
    "CheckpointStore": ".checkpoints",
    "FileCheckpointStore": ".checkpoints",
    "InMemoryCheckpointStore": ".checkpoints",
    "BaseRoutingPolicy": ".routing",
    "RoundRobinRoutingPolicy": ".routing",
    "EscalationRoutingPolicy": ".routing",
    "CostAwareRoutingPolicy": ".routing",
    "RoutingDecision": ".routing",
    "SharedState": ".contracts",
    "SpecialistContract": ".contracts",
    "filter_specialists_by_capabilities": ".contracts",
    "attach_contract_weights": ".contracts",
    "invoke_specialists": ".specialists",
    "ainvoke_specialists": ".specialists",
    "make_subagent_tool": ".specialists",
    "make_specialist_tools": ".specialists",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)
