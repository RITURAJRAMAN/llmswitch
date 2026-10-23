"""
core.planning

Request planning shared by the synchronous and asynchronous clients: stream
options, model selection, the execution plan, and the token budget that is
reserved for output and request overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .execution import RequestExecutionPlan, build_execution_plan
from .preprocess import apply_model_priority, estimate_request_overhead, resolve_reserved_output_tokens


@dataclass(frozen=True)
class RequestPlan:
    plan: RequestExecutionPlan
    model_name: Optional[str]
    reserved_output_tokens: int
    overhead_tokens: int


def plan_chat(config: Any, rate_limiter: Any, messages: list[dict[str, Any]], kwargs: dict[str, Any],
              stream: bool) -> RequestPlan:
    """Plan a chat completion. ``kwargs`` is updated in place with the stream options and model."""
    if stream and "stream_options" not in kwargs:
        kwargs["stream_options"] = {"include_usage": True}
    apply_model_priority(config, kwargs)
    plan = build_execution_plan(
        config,
        operation="chat.completions",
        normalized_request={"messages": messages, **kwargs},
    )
    model_name = plan.target.resolved_model
    reserved = resolve_reserved_output_tokens(kwargs, config.context_output_reserve_tokens)
    overhead = estimate_request_overhead(
        rate_limiter,
        model_name,
        kwargs,
        message_count=len(messages),
        configured_overhead_tokens=config.context_overhead_tokens,
    )
    return RequestPlan(plan, model_name, reserved, overhead)


def plan_responses(config: Any, rate_limiter: Any, input: Any, kwargs: dict[str, Any]) -> RequestPlan:  # noqa: A002
    """Plan a Responses create request. ``kwargs`` is updated in place with the model."""
    apply_model_priority(config, kwargs)
    plan = build_execution_plan(
        config,
        operation="responses",
        normalized_request={"input": input, **kwargs},
    )
    model_name = plan.target.resolved_model
    reserved = resolve_reserved_output_tokens(kwargs, config.context_output_reserve_tokens)
    overhead = estimate_request_overhead(
        rate_limiter,
        model_name,
        kwargs,
        configured_overhead_tokens=config.context_overhead_tokens,
    )
    return RequestPlan(plan, model_name, reserved, overhead)
