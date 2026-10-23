"""Shared execution primitives for provider dispatch, failover, and workflows."""
from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ..constants import DEFAULT_ANTHROPIC_MODEL
from ..errors import RateLimitExceededError


class ExecutionStatus(str, Enum):
    """Lifecycle state for a provider, stream, tool, or agent execution."""

    NOT_STARTED = "not_started"
    DISPATCHED = "dispatched"
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class NormalizedRequest:
    """Immutable provider-neutral intent before provider request transformation."""

    operation: str
    messages: tuple[Mapping[str, Any], ...] = ()
    input: Any = None
    embedding_input: tuple[str, ...] = ()
    tools: tuple[Any, ...] = ()
    structured_output: Any = None
    generation: Mapping[str, Any] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetDescriptor:
    """Resolved provider target used for capability, quota, and breaker isolation."""

    provider: str
    operation: str
    endpoint: str | None
    deployment_name: str | None
    model: str | None
    api_version: str | None

    @property
    def resolved_model(self) -> str | None:
        return self.deployment_name or self.model

    @property
    def endpoint_identity(self) -> str:
        source = self.endpoint or "default"
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    @property
    def breaker_key(self) -> tuple[str, str, str, str]:
        return (
            self.provider,
            self.operation,
            self.endpoint_identity,
            self.resolved_model or "<default>",
        )

    @property
    def limiter_scope(self) -> tuple[str, str, str]:
        return self.provider, self.endpoint_identity, self.resolved_model or "<default>"

    @property
    def limiter_scope_key(self) -> str:
        return hashlib.sha256("\x1f".join(self.limiter_scope).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ExecutionOutcome:
    """Explicit result classification for retry, failover, and caller reconciliation."""

    status: ExecutionStatus
    provider_dispatched: bool = False
    output_visible: bool = False
    retryable: bool = False
    error: Exception | None = None


@dataclass(frozen=True)
class RequestExecutionPlan:
    """A provider-target-specific dispatch plan built from a normalized user request."""

    target: TargetDescriptor
    normalized_request: NormalizedRequest
    provider_request: Mapping[str, Any]


def provider_request_from_normalized_request(
    request: NormalizedRequest,
    target: TargetDescriptor,
) -> Mapping[str, Any]:
    """Rebuild canonical provider kwargs from a normalized request and target."""
    payload: dict[str, Any] = {}
    payload.update(dict(request.generation))
    payload.update(dict(request.state))
    payload.update(dict(request.extensions))
    if request.tools:
        payload["tools"] = list(request.tools)
    if request.structured_output is not None:
        key = "text_format" if request.operation == "responses" else "response_format"
        payload[key] = request.structured_output
    if target.resolved_model is not None:
        payload["model"] = target.resolved_model
    return MappingProxyType(payload)


def provider_request_with_payload(
    plan: RequestExecutionPlan,
    *,
    input_key: str | None = None,
    messages_key: str | None = None,
    embedding_input_key: str | None = None,
) -> Mapping[str, Any]:
    """Augment a plan's canonical provider kwargs with explicit payload fields when needed."""
    payload = dict(plan.provider_request)
    if input_key is not None and plan.normalized_request.input is not None:
        payload[input_key] = plan.normalized_request.input
    if messages_key is not None and plan.normalized_request.messages:
        payload[messages_key] = [dict(message) for message in plan.normalized_request.messages]
    if embedding_input_key is not None and plan.normalized_request.embedding_input:
        payload[embedding_input_key] = list(plan.normalized_request.embedding_input)
    return MappingProxyType(payload)


def build_execution_plan(
    config: Any,
    *,
    operation: str,
    normalized_request: Mapping[str, Any] | NormalizedRequest,
    provider_request: Mapping[str, Any] | None = None,
) -> RequestExecutionPlan:
    """Build an immutable target-specific plan without mutating caller payloads."""
    request = normalized_request if isinstance(normalized_request, NormalizedRequest) else normalize_request(
        operation=operation,
        input=normalized_request.get("input"),
        messages=normalized_request.get("messages"),
        embedding_input=normalized_request.get("embedding_input"),
        request_kwargs=normalized_request,
    )
    deployment_name = getattr(config, "deployment_name", None)
    provider = config.provider
    if provider == "anthropic" and operation in ("chat.completions", "responses"):
        model = (deployment_name or request.generation.get("model") or getattr(config, "model", None)
                 or DEFAULT_ANTHROPIC_MODEL)
    else:
        model = deployment_name or request.generation.get("model") or getattr(config, "model", None)
    target = TargetDescriptor(
        provider=provider,
        operation=operation,
        endpoint=getattr(config, "endpoint", None),
        deployment_name=deployment_name,
        model=model,
        api_version=getattr(config, "responses_api_version", None) or getattr(config, "api_version", None),
    )
    return RequestExecutionPlan(
        target=target,
        normalized_request=request,
        provider_request=(
            MappingProxyType(dict(provider_request))
            if provider_request is not None
            else provider_request_from_normalized_request(request, target)
        ),
    )


def normalize_request(
    *,
    operation: str,
    input: Any = None,
    messages: Any = None,
    embedding_input: Any = None,
    request_kwargs: Mapping[str, Any] | None = None,
) -> NormalizedRequest:
    """Split common request intent from generation, state, and extension fields."""
    kwargs = dict(request_kwargs or {})
    kwargs.pop("input", None)
    kwargs.pop("messages", None)
    kwargs.pop("embedding_input", None)
    message_items = tuple(dict(message) for message in (messages or ()))
    if isinstance(embedding_input, str):
        embedding_items = (embedding_input,)
    else:
        embedding_items = tuple(embedding_input or ())
    tools = tuple(kwargs.pop("tools", ()) or ())
    structured_output = kwargs.pop("response_format", kwargs.pop("text_format", None))
    state_keys = {"previous_response_id", "conversation", "store", "background", "metadata"}
    state = {key: kwargs.pop(key) for key in tuple(kwargs) if key in state_keys}
    generation_keys = {
        "model", "temperature", "top_p", "max_tokens", "max_completion_tokens", "max_output_tokens",
        "tool_choice", "parallel_tool_calls", "max_tool_calls", "reasoning", "instructions", "text",
    }
    generation = {key: kwargs.pop(key) for key in tuple(kwargs) if key in generation_keys}
    return NormalizedRequest(
        operation=operation,
        messages=message_items,
        input=input,
        embedding_input=embedding_items,
        tools=tools,
        structured_output=structured_output,
        generation=MappingProxyType(generation),
        state=MappingProxyType(state),
        extensions=MappingProxyType(kwargs),
    )


def _should_wait(limiter: Any, delay: float | None) -> bool:
    """Whether a request that hit its limit should sleep for ``delay`` seconds rather than fail now."""
    if delay is None or delay <= 0 or getattr(limiter, "on_limit", "wait") != "wait":
        return False
    max_wait = getattr(limiter, "max_wait", None)
    return max_wait is None or delay <= max_wait


def reserve_rate_limit(
    limiter: Any,
    key: str,
    rpm: int,
    tpm: int,
    window_seconds: int,
    tokens: int,
    *,
    request_type: str = "completion",
    suppress_log: bool = False,
    scope: str | None = None,
) -> None:
    options = {"request_type": request_type, "suppress_log": suppress_log, "scope": scope}
    allowed, _ = limiter.is_allowed(key, rpm, tpm, window_seconds, tokens, **options)
    if allowed:
        return
    remaining = limiter.get_remaining(key, rpm, tpm, window_seconds, request_type=request_type, scope=scope)
    delay = remaining.get("remaining_time")
    if _should_wait(limiter, delay):
        time.sleep(delay)
        allowed, _ = limiter.is_allowed(key, rpm, tpm, window_seconds, tokens, **options)
        if allowed:
            return
        remaining = limiter.get_remaining(key, rpm, tpm, window_seconds, request_type=request_type, scope=scope)
        delay = remaining.get("remaining_time")
    raise RateLimitExceededError(max(1, math.ceil(delay or 1)))


async def areserve_rate_limit(
    limiter: Any,
    key: str,
    rpm: int,
    tpm: int,
    window_seconds: int,
    tokens: int,
    *,
    request_type: str = "completion",
    suppress_log: bool = False,
    scope: str | None = None,
) -> None:
    options = {"request_type": request_type, "suppress_log": suppress_log, "scope": scope}
    allowed, _ = await asyncio.to_thread(limiter.is_allowed, key, rpm, tpm, window_seconds, tokens, **options)
    if allowed:
        return
    remaining = await asyncio.to_thread(
        limiter.get_remaining, key, rpm, tpm, window_seconds, request_type=request_type, scope=scope,
    )
    delay = remaining.get("remaining_time")
    if _should_wait(limiter, delay):
        await asyncio.sleep(delay)
        allowed, _ = await asyncio.to_thread(
            limiter.is_allowed, key, rpm, tpm, window_seconds, tokens, **options,
        )
        if allowed:
            return
        remaining = await asyncio.to_thread(
            limiter.get_remaining, key, rpm, tpm, window_seconds, request_type=request_type, scope=scope,
        )
        delay = remaining.get("remaining_time")
    raise RateLimitExceededError(max(1, math.ceil(delay or 1)))
