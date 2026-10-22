"""
providers._shared

Response-mapping helpers shared by the OpenAI and Azure OpenAI providers, whose
SDKs return the same response shape. Centralizes the usage / choice / logprob /
tool-call(-delta) translation used by the streaming and non-streaming paths of
both providers.

Provider-specific fields (Azure's ``content_filter_results`` and message
``context``) stay in the provider modules; only the identical OpenAI-shaped
mapping lives here.
"""


from __future__ import annotations

import inspect
import math
from typing import Any, Optional, overload

from ..models import (
    ChatCompletionAudio,
    ChatCompletionMessageToolCall,
    ChatCompletionTokenLogprob,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChoiceLogprobs,
    CompletionTokensDetails,
    CompletionUsage,
    Function,
    PromptTokensDetails,
    TopLogprob,
)


def merge_client_options(defaults: dict[str, Any], client_options: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Native-SDK constructor kwargs: our defaults, overridden by the caller's ``client_options``.

    Reserved unified fields are rejected when the config is built, so the caller
    cannot replace the credentials or endpoint chosen by the unified constructor.
    """
    return {**defaults, **(client_options or {})}


def close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        close()


async def close_async_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if not callable(close):
        close = getattr(stream, "aclose", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


def get_retry_after_seconds(exc: Any, default: int = 60) -> int:
    """Read ``retry-after`` off a rate-limit error's response headers."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return default
    try:
        return int(headers.get("retry-after", default))
    except (TypeError, ValueError):
        return default


def configured_timeout_seconds(timeout: Any, default: int = 60) -> int:
    """Return a positive whole-second timeout for user-facing timeout errors."""
    try:
        if timeout is None:
            return default
        timeout_value = float(timeout)
        if timeout_value <= 0:
            return default
        return max(1, int(math.ceil(timeout_value)))
    except (TypeError, ValueError):
        return default


def build_openai_usage(usage: Any) -> CompletionUsage:
    """Map an OpenAI/Azure ``usage`` object to the unified ``CompletionUsage``.

    Works for both the non-streaming response usage and the streaming final-usage
    chunk (``latency_checkpoint`` is absent on stream usage and resolves to None,
    matching the previous inline behavior).
    """
    prompt_tokens_details = PromptTokensDetails(
        audio_tokens=getattr(usage.prompt_tokens_details, "audio_tokens", 0),
        cached_tokens=getattr(usage.prompt_tokens_details, "cached_tokens", 0),
    )
    completion_tokens_details = CompletionTokensDetails(
        accepted_prediction_tokens=getattr(usage.completion_tokens_details, "accepted_prediction_tokens", 0),
        audio_tokens=getattr(usage.completion_tokens_details, "audio_tokens", 0),
        reasoning_tokens=getattr(usage.completion_tokens_details, "reasoning_tokens", 0),
        rejected_prediction_tokens=getattr(usage.completion_tokens_details, "rejected_prediction_tokens", 0),
    )
    return CompletionUsage(
        completion_tokens=usage.completion_tokens,
        prompt_tokens=usage.prompt_tokens,
        total_tokens=usage.total_tokens,
        completion_tokens_details=completion_tokens_details,
        prompt_tokens_details=prompt_tokens_details,
        latency_checkpoint=getattr(usage, "latency_checkpoint", None),
    )


def map_tool_calls(message: Any) -> Optional[list[ChatCompletionMessageToolCall]]:
    """Convert SDK message ``tool_calls`` into unified tool-call models (or None)."""
    raw_tool_calls = getattr(message, "tool_calls", None) or []
    return [
        ChatCompletionMessageToolCall(
            id=tc.id,
            type="function",
            function=map_function_call(tc.function),
        )
        for tc in raw_tool_calls
    ] or None


@overload
def map_function_call(raw_call: None) -> None: ...


@overload
def map_function_call(raw_call: object) -> Function: ...


def map_function_call(raw_call: Any) -> Optional[Function]:
    if raw_call is None:
        return None
    return Function(
        name=raw_call.name,
        arguments=raw_call.arguments,
        parsed_arguments=getattr(raw_call, "parsed_arguments", None),
    )


def map_function_call_delta(raw_call: Any) -> Optional[ChoiceDeltaToolCallFunction]:
    if raw_call is None:
        return None
    return ChoiceDeltaToolCallFunction(name=raw_call.name, arguments=raw_call.arguments)


def audio_from(raw_audio: Any) -> Optional[ChatCompletionAudio]:
    """Build a unified ``ChatCompletionAudio`` from an SDK audio object (or None)."""
    if not raw_audio:
        return None
    return ChatCompletionAudio(
        id=getattr(raw_audio, "id", ""),
        data=getattr(raw_audio, "data", ""),
        expires_at=getattr(raw_audio, "expires_at", 0),
        transcript=getattr(raw_audio, "transcript", ""),
    )


def _token_logprob(t: Any) -> ChatCompletionTokenLogprob:
    return ChatCompletionTokenLogprob(
        token=t.token,
        bytes=t.bytes,
        logprob=t.logprob,
        top_logprobs=[
            TopLogprob(token=tp.token, bytes=tp.bytes, logprob=tp.logprob)
            for tp in (t.top_logprobs or [])
        ],
    )


def map_message_logprobs(logprobs: Any) -> Optional[ChoiceLogprobs]:
    """Map non-streaming choice logprobs (content + refusal)."""
    if not logprobs:
        return None
    return ChoiceLogprobs(
        content=[_token_logprob(t) for t in (logprobs.content or [])] or None,
        refusal=[_token_logprob(t) for t in (logprobs.refusal or [])] or None,
    )


def map_delta_logprobs(logprobs: Any) -> Optional[ChoiceLogprobs]:
    """Map streaming delta logprobs (content only; refusal omitted, as before)."""
    if not logprobs:
        return None
    return ChoiceLogprobs(
        content=[_token_logprob(t) for t in (logprobs.content or [])] or None,
        refusal=None,
    )


def map_tool_call_deltas(delta: Any) -> Optional[list[ChoiceDeltaToolCall]]:
    """Convert streaming delta ``tool_calls`` into unified delta tool-call models."""
    raw_tc_deltas = getattr(delta, "tool_calls", None) or []
    return [
        ChoiceDeltaToolCall(
            index=tc.index,
            id=tc.id,
            type=tc.type,
            function=ChoiceDeltaToolCallFunction(
                name=getattr(tc.function, "name", None),
                arguments=getattr(tc.function, "arguments", None),
            ) if tc.function else None,
        )
        for tc in raw_tc_deltas
    ] or None
