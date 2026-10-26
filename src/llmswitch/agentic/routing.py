"""
llmswitch.agentic.routing

Choosing which model handles each step of an agent run.

A policy receives the step number, whether the previous step had a tool error, the primary and fallback
models, and a state dict it may keep between steps. It returns a :class:`RoutingDecision` with the model to
use and the reason. Three policies are included:

- :class:`RoundRobinRoutingPolicy` alternates between the primary and the fallback model.
- :class:`EscalationRoutingPolicy` uses the fallback after a tool error, and the primary otherwise.
- :class:`CostAwareRoutingPolicy` uses the primary, switching to the fallback after a tool error or from
  a chosen step onwards.

Routing state, such as a round-robin position, belongs to the agent and carries over from one run to the
next. It is also saved in checkpoints, so a resumed run continues its own rotation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoutingDecision:
    """The model chosen for a step, the policy that chose it, and why."""

    llm: Any
    policy_name: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRoutingPolicy:
    """Base class for routing policies. Subclasses implement :meth:`select_llm`."""

    def select_llm(
        self,
        *,
        messages: list[Any],
        step: int,
        had_tool_error: bool,
        primary_llm: Any,
        fallback_llm: Any | None,
        state: dict[str, Any],
    ) -> RoutingDecision:
        raise NotImplementedError


class RoundRobinRoutingPolicy(BaseRoutingPolicy):
    """Cycle through the primary and, when set, the fallback model in order."""

    def select_llm(self, *, messages: list[Any], step: int, had_tool_error: bool, primary_llm: Any,
                   fallback_llm: Any | None, state: dict[str, Any]) -> RoutingDecision:
        available: list[tuple[str, Any]] = [("primary", primary_llm)]
        if fallback_llm is not None:
            available.append(("fallback", fallback_llm))
        index = int(state.get("round_robin_index", 0)) % len(available)
        state["round_robin_index"] = index + 1
        name, llm = available[index]
        return RoutingDecision(llm=llm, policy_name="round_robin", reason=f"selected_{name}",
                               metadata={"index": index, "pool_size": len(available), "step": step})


class EscalationRoutingPolicy(BaseRoutingPolicy):
    """Use the fallback model after a tool error, and the primary model otherwise."""

    def select_llm(self, *, messages: list[Any], step: int, had_tool_error: bool, primary_llm: Any,
                   fallback_llm: Any | None, state: dict[str, Any]) -> RoutingDecision:
        if had_tool_error and fallback_llm is not None:
            return RoutingDecision(llm=fallback_llm, policy_name="escalation", reason="tool_error_fallback",
                                   metadata={"step": step})
        return RoutingDecision(llm=primary_llm, policy_name="escalation", reason="primary_default",
                               metadata={"step": step})


class CostAwareRoutingPolicy(BaseRoutingPolicy):
    """Use the primary model, switching to the fallback after a tool error or from ``fallback_after_step``."""

    def __init__(self, fallback_after_step: int | None = None) -> None:
        if fallback_after_step is not None and fallback_after_step < 1:
            raise ValueError("fallback_after_step must be at least 1 when set")
        self.fallback_after_step = fallback_after_step

    def select_llm(self, *, messages: list[Any], step: int, had_tool_error: bool, primary_llm: Any,
                   fallback_llm: Any | None, state: dict[str, Any]) -> RoutingDecision:
        if had_tool_error and fallback_llm is not None:
            return RoutingDecision(llm=fallback_llm, policy_name="cost_aware", reason="tool_error_fallback",
                                   metadata={"step": step})
        if (fallback_llm is not None and self.fallback_after_step is not None
                and step >= self.fallback_after_step):
            return RoutingDecision(llm=fallback_llm, policy_name="cost_aware", reason="late_step_switch",
                                   metadata={"step": step, "fallback_after_step": self.fallback_after_step})
        return RoutingDecision(llm=primary_llm, policy_name="cost_aware", reason="primary_default",
                               metadata={"step": step, "fallback_after_step": self.fallback_after_step})
