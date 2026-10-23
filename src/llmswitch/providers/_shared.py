"""
providers._shared

Response-mapping helpers shared by the OpenAI and Azure OpenAI providers, whose
SDKs return the same response shape. Centralizes the usage / choice / logprob /
tool-call(-delta) translation used by the streaming and non-streaming paths of
both providers.

Provider-specific fields (Azure's ``content_filter_results`` and message
``context``) stay in the provider modules; only the identical OpenAI-shaped
mapping lives here.

Also hosts the Responses API mappers and the OpenAI SDK exception translator
used by both providers' chat and Responses surfaces.
"""


from __future__ import annotations

import inspect
import math
from typing import Any, Optional, overload
from urllib.parse import urlparse

from ..errors import (
    APIConnectionError,
    AuthenticationError,
    InvalidResponseError,
    RateLimitExceededError,
    ResponseNotFoundError,
    TimeOutError,
)
from ..models import (
    ChatCompletionAudio,
    ChatCompletionMessageToolCall,
    ChatCompletionTokenLogprob,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChoiceLogprobs,
    CompactedResponse,
    CompletionTokensDetails,
    CompletionUsage,
    Conversation,
    ConversationDeleted,
    Function,
    InputTokenCountResponse,
    PromptTokensDetails,
    Response,
    ResponseDeleted,
    ResponseItem,
    ResponseItemList,
    ResponseOutputItem,
    ResponseStreamEvent,
    ResponseUsage,
    TopLogprob,
    response_output_item_from_data,
    response_stream_event_from_data,
)
from ..structured import provider_extensions_for, provider_extensions_from_raw, to_jsonable


def merge_client_options(defaults: dict[str, Any], client_options: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Native-SDK constructor kwargs: our defaults, overridden by the caller's ``client_options``.

    Reserved unified fields are rejected when the config is built, so the caller
    cannot replace the credentials or endpoint chosen by the unified constructor.
    """
    return {**defaults, **(client_options or {})}


TOKEN_LIMIT_NAMES = ("max_output_tokens", "max_completion_tokens", "max_tokens")


def normalize_token_limit(kwargs: dict[str, Any], target: str) -> None:
    """Accept any of the three token-limit names and keep only ``target``, in place.

    If several are given, ``target`` wins, then the order of ``TOKEN_LIMIT_NAMES``.
    """
    found = {name: kwargs.pop(name) for name in TOKEN_LIMIT_NAMES if name in kwargs}
    if not found:
        return
    value = found.get(target)
    if value is None:
        value = next((v for v in found.values() if v is not None), None)
    if value is not None:
        kwargs[target] = value


def openai_token_limit_name(endpoint: Optional[str]) -> str:
    """Token-limit name for the OpenAI chat API.

    OpenAI and Azure hosts take ``max_completion_tokens``. Other OpenAI-compatible
    servers often only know ``max_tokens``, so that name is kept for them.
    """
    host = (urlparse(endpoint).hostname or "") if endpoint else ""
    if not host or host == "openai.com" or host.endswith(".openai.com") or host.endswith(".azure.com"):
        return "max_completion_tokens"
    return "max_tokens"


def azure_token_limit_name(api_version: Optional[str]) -> str:
    """Azure OpenAI accepts ``max_completion_tokens`` from API version 2024-09-01 onward."""
    if api_version and api_version[:10] < "2024-09-01":
        return "max_tokens"
    return "max_completion_tokens"


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


def translate_openai_error(exc: Exception, endpoint: Optional[str], provider: str,
                           error_cls: type, operation: Optional[str] = None,
                           timeout_seconds: Any = None,
                           resource_id: Optional[str] = None) -> Exception:
    """
    Translate an OpenAI SDK exception into the library's typed error hierarchy.

    ``error_cls`` is the surface-specific fallback (``ChatCompletionError`` for
    chat, ``ResponsesError`` for the Responses API) so the same mapping serves
    both without collapsing their error codes.
    """
    import openai

    kwargs: dict[str, Any] = {"provider": provider}
    if operation is not None:
        kwargs["operation"] = operation

    if isinstance(exc, openai.RateLimitError):
        return RateLimitExceededError(get_retry_after_seconds(exc))
    if isinstance(exc, openai.AuthenticationError):
        return AuthenticationError(provider=provider)
    if isinstance(exc, openai.APITimeoutError):
        return TimeOutError(configured_timeout_seconds(timeout_seconds))
    if isinstance(exc, openai.NotFoundError) and resource_id:
        return ResponseNotFoundError(resource_id, provider=provider)
    if isinstance(exc, openai.BadRequestError):
        return error_cls(str(exc), status_code=400, **kwargs)
    if isinstance(exc, openai.APIConnectionError | ConnectionError):
        return APIConnectionError(endpoint or "https://api.openai.com", provider=provider)
    if isinstance(exc, PermissionError):
        return AuthenticationError(provider=provider)
    if isinstance(exc, ValueError):
        return InvalidResponseError(str(exc), provider=provider)
    return error_cls(str(exc), status_code=getattr(exc, "status_code", 500), **kwargs)


def _attach_provider_extensions(model: Any, provider: str, raw: Any, mapped_fields: set[str]) -> Any:
    extensions = provider_extensions_from_raw(provider, raw, mapped_fields)
    if hasattr(model, "extensions"):
        model.extensions = extensions
    return model


def build_response_usage(usage: Any) -> Optional[ResponseUsage]:
    """Map an SDK ``ResponseUsage`` to the unified :class:`ResponseUsage`."""
    if usage is None:
        return None
    return ResponseUsage.model_validate(to_jsonable(usage))


def map_response_output(output: Any, provider: str = "openai") -> list[ResponseOutputItem]:
    """Map an SDK response ``output`` array to unified output items."""
    items: list[ResponseOutputItem] = []
    for raw in output or []:
        data = to_jsonable(raw)
        if not isinstance(data, dict):
            continue
        item = response_output_item_from_data(data)
        _attach_provider_extensions(item, provider, data, set(type(item).model_fields))
        # ``parsed`` is a live Pydantic instance from .parse() and must not be JSON-flattened.
        parsed = getattr(raw, "parsed", None)
        if parsed is None:
            for part in getattr(raw, "content", None) or []:
                parsed = getattr(part, "parsed", None)
                if parsed is not None:
                    break
        if parsed is not None:
            item.parsed = parsed
        items.append(item)
    return items


def map_response(response: Any, provider: str = "openai") -> Response:
    """Map an SDK ``Response`` to the unified :class:`Response`."""
    data = to_jsonable(response)
    if not isinstance(data, dict):
        raise InvalidResponseError("Responses API returned a non-object payload")
    raw_output = getattr(response, "output", None)
    raw_usage = getattr(response, "usage", None)
    if isinstance(response, dict):
        raw_output = data.get("output")
        raw_usage = data.get("usage")
    extensions = provider_extensions_from_raw(
        provider,
        response,
        set(Response.model_fields) | {"output", "usage"},
    )
    data.pop("output", None)
    data.pop("usage", None)
    mapped = Response.model_validate(data)
    mapped.output = map_response_output(raw_output, provider=provider)
    mapped.usage = build_response_usage(raw_usage)
    mapped.extensions = extensions
    return mapped


def map_compacted_response(response: Any, provider: str = "openai") -> CompactedResponse:
    """Map an SDK ``CompactedResponse`` to the unified model."""
    data = to_jsonable(response)
    if not isinstance(data, dict):
        raise InvalidResponseError("responses.compact returned a non-object payload")
    extensions = provider_extensions_from_raw(
        provider,
        response,
        set(CompactedResponse.model_fields) | {"output", "usage"},
    )
    data.pop("output", None)
    data.pop("usage", None)
    mapped = CompactedResponse.model_validate(data)
    mapped.output = map_response_output(getattr(response, "output", None), provider=provider)
    mapped.usage = build_response_usage(getattr(response, "usage", None))
    mapped.extensions = extensions
    return mapped


def map_response_item(item: Any, provider: str = "openai") -> ResponseItem:
    """Map a single SDK response item to the unified :class:`ResponseItem`."""
    data = to_jsonable(item)
    if not isinstance(data, dict):
        raise InvalidResponseError("Responses API returned a non-object item payload")
    mapped = ResponseItem.model_validate(data)
    return _attach_provider_extensions(mapped, provider, data, set(ResponseItem.model_fields))


def map_response_item_page(page: Any, provider: str = "openai") -> ResponseItemList:
    """Map an SDK ``SyncCursorPage[ResponseItem]`` to a unified :class:`ResponseItemList`."""
    raw_items = getattr(page, "data", None) or []
    items = [map_response_item(i, provider=provider) for i in raw_items]
    mapped = ResponseItemList(
        object=getattr(page, "object", "list") or "list",
        data=items,
        first_id=getattr(page, "first_id", None),
        last_id=getattr(page, "last_id", None),
        has_more=bool(getattr(page, "has_more", False)),
    )
    return _attach_provider_extensions(mapped, provider, page, set(ResponseItemList.model_fields))


def map_input_token_count(count: Any, provider: str = "openai") -> InputTokenCountResponse:
    """Map an SDK ``InputTokenCountResponse`` to the unified model."""
    mapped = InputTokenCountResponse.model_validate(to_jsonable(count))
    return _attach_provider_extensions(mapped, provider, count, set(InputTokenCountResponse.model_fields))


def map_response_deleted(deleted: Any, response_id: str, provider: str = "openai") -> ResponseDeleted:
    """Map an SDK delete result (which may be ``None``) to :class:`ResponseDeleted`."""
    data = to_jsonable(deleted)
    if isinstance(data, dict) and data.get("id"):
        mapped = ResponseDeleted.model_validate(data)
        return _attach_provider_extensions(mapped, provider, data, set(ResponseDeleted.model_fields))
    return ResponseDeleted(id=response_id, deleted=True, extensions=provider_extensions_for(provider))


def map_conversation(conversation: Any, provider: str = "openai") -> Conversation:
    """Map an SDK conversation object to the unified model."""
    mapped = Conversation.model_validate(to_jsonable(conversation))
    return _attach_provider_extensions(mapped, provider, conversation, set(Conversation.model_fields))


def map_conversation_deleted(deleted: Any, conversation_id: str, provider: str = "openai") -> ConversationDeleted:
    """Map an SDK conversation delete result, including SDK variants returning None."""
    data = to_jsonable(deleted)
    if isinstance(data, dict) and data.get("id"):
        mapped = ConversationDeleted.model_validate(data)
        return _attach_provider_extensions(mapped, provider, data, set(ConversationDeleted.model_fields))
    return ConversationDeleted(id=conversation_id, deleted=True, extensions=provider_extensions_for(provider))


def map_response_stream_event(event: Any, provider: str = "openai") -> ResponseStreamEvent:
    """Map an SDK streaming event to the unified :class:`ResponseStreamEvent`."""
    data = to_jsonable(event)
    if not isinstance(data, dict):
        return ResponseStreamEvent(
            type="unknown",
            delta=data,
            extensions=provider_extensions_for(provider),
        )
    nested = data.pop("response", None)
    mapped = response_stream_event_from_data(data)
    _attach_provider_extensions(mapped, provider, data, set(type(mapped).model_fields))
    if nested is not None:
        raw_response = getattr(event, "response", None)
        mapped.response = map_response(raw_response if raw_response is not None else nested, provider=provider)
    return mapped


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
