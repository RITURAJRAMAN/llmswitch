"""
llmswitch.agentic.messages

Pure conversions between the agentic message objects and llmswitch's chat shapes.
Nothing here makes a network call, so each function can be tested on its own.

- Agentic message objects -> chat message dicts (``to_chat_messages``)
- ``ChatCompletion`` -> ``AIMessage`` (``to_ai_message``)
- ``ChatCompletionChunk`` -> ``AIMessageChunk`` (``to_ai_chunk``)
- Agentic message objects -> Responses input items (``to_response_input``)
- ``Response`` -> ``AIMessage`` (``response_to_ai_message``)
- Responses stream events -> ``AIMessageChunk`` or ``None`` (``response_event_to_chunk``)
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
    UsageMetadata,
)

from ..errors import ResponsesError
from ..models import (
    ChatCompletion,
    ChatCompletionChunk,
    ProviderExtensions,
    Response,
    ResponseStreamEvent,
)

_TEXT_BLOCK = "text"


def _image_url(block: dict[str, Any]) -> str:
    """Return a URL for an image block, from either the OpenAI shape or the standard shape."""
    if block.get("type") == "image_url":
        image = block.get("image_url")
        url = image.get("url") if isinstance(image, dict) else image
        if url:
            return url
    if block.get("type") == "image":
        if block.get("url"):
            return block["url"]
        if block.get("base64"):
            mime = block.get("mime_type") or "image/png"
            return f"data:{mime};base64,{block['base64']}"
    raise ValueError(f"Unsupported image content block: {sorted(block)}")


# Content blocks that describe a provider's own reasoning or tool use. They are not replayed from
# message content: tool calls are sent through ``tool_calls``. The data a provider needs back (Anthropic
# thinking blocks, Vertex function-call signatures) is kept in ``additional_kwargs`` by ``_kept_data``.
_NOT_REPLAYED = frozenset({
    "thinking", "redacted_thinking", "reasoning", "tool_use", "tool_call", "function_call", "server_tool_use",
})


def _content_parts(content: Any) -> list[dict[str, Any]]:
    """Normalize message content into chat content parts, skipping provider-only reasoning blocks."""
    if isinstance(content, str):
        return [{"type": _TEXT_BLOCK, "text": content}]
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            parts.append({"type": _TEXT_BLOCK, "text": block})
        elif isinstance(block, dict) and block.get("type") == _TEXT_BLOCK:
            parts.append({"type": _TEXT_BLOCK, "text": block.get("text", "")})
        elif isinstance(block, dict) and block.get("type") in ("image", "image_url"):
            parts.append({"type": "image_url", "image_url": {"url": _image_url(block)}})
        elif isinstance(block, dict) and block.get("type") in _NOT_REPLAYED:
            continue
        else:
            kind = block.get("type") if isinstance(block, dict) else type(block).__name__
            raise ValueError(f"Unsupported message content block: {kind}")
    return parts


def _text_only(parts: list[dict[str, Any]]) -> str:
    return "".join(part["text"] for part in parts)


def _content_for_role(content: Any, role: str) -> Any:
    """Keep images as parts for user messages; other roles take text only."""
    parts = _content_parts(content)
    has_images = any(part["type"] != _TEXT_BLOCK for part in parts)
    if role == "user" and has_images:
        return parts
    if has_images:
        raise ValueError(f"Images are only supported in user messages, not {role} messages.")
    return _text_only(parts)


def _kept_data(extensions: ProviderExtensions | None) -> dict[str, Any]:
    """The data a provider answer must send back on the next turn, as message additional kwargs.

    Each kind has its own key and no provider name, so chunks of one answer can be joined: lists of
    thinking blocks concatenate, and signature maps merge.
    """
    if extensions is None or not extensions.fields:
        return {}
    if extensions.provider == "anthropic" and extensions.fields.get("thinking_blocks"):
        return {"anthropic_thinking_blocks": json.loads(json.dumps(extensions.fields["thinking_blocks"]))}
    if extensions.provider == "vertexai" and extensions.fields.get("thought_signatures"):
        return {"vertexai_thought_signatures": json.loads(json.dumps(extensions.fields["thought_signatures"]))}
    return {}


def _tool_call_dict(call_id: str | None, name: str | None, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _assistant_dict(message: AIMessage, provider: str | None) -> dict[str, Any]:
    text = _content_for_role(message.content, "assistant") if message.content else ""
    calls = [
        _tool_call_dict(call["id"], call["name"], json.dumps(call["args"]))
        for call in message.tool_calls
    ]
    calls += [
        _tool_call_dict(call.get("id"), call.get("name"), call.get("args") or "")
        for call in message.invalid_tool_calls
    ]
    result: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        result["tool_calls"] = calls
    kept = message.additional_kwargs
    if provider == "anthropic" and kept.get("anthropic_thinking_blocks"):
        # Sent back unchanged, first in the assistant turn, as Anthropic requires with extended thinking.
        result["extensions"] = {"provider": "anthropic", "fields": {"thinking_blocks": kept["anthropic_thinking_blocks"]}}
    elif provider == "vertexai" and kept.get("vertexai_thought_signatures"):
        # Sent back with each function call, as Gemini requires.
        result["extensions"] = {"provider": "vertexai",
                                "fields": {"thought_signatures": kept["vertexai_thought_signatures"]}}
    return result


def to_chat_messages(messages: Sequence[BaseMessage], provider: str | None = None) -> list[dict[str, Any]]:
    """Convert agentic message objects to the chat message dicts that llmswitch sends.

    ``provider`` decides which provider-specific data is kept: the signed thinking blocks of an Anthropic
    answer are sent back only to Anthropic.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            converted.append({"role": "system", "content": _content_for_role(message.content, "system")})
        elif isinstance(message, HumanMessage):
            converted.append({"role": "user", "content": _content_for_role(message.content, "user")})
        elif isinstance(message, AIMessage):
            converted.append(_assistant_dict(message, provider))
        elif isinstance(message, ToolMessage):
            converted.append({"role": "tool", "tool_call_id": message.tool_call_id,
                              "content": _content_for_role(message.content, "tool")})
        else:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")
    return converted


def _parse_arguments(raw: str | None) -> tuple[bool, Any]:
    """Return ``(True, args)`` for a JSON object, or ``(False, error)`` otherwise."""
    if raw is None or raw.strip() == "":
        return True, {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, str(exc)
    if not isinstance(value, dict):
        return False, "tool arguments must be a JSON object"
    return True, value


def _usage_metadata(usage: Any) -> UsageMetadata | None:
    if usage is None:
        return None
    metadata = UsageMetadata(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )
    cached = getattr(usage, "cached_tokens", 0) or 0
    if cached:
        metadata["input_token_details"] = {"cache_read": cached}
    reasoning = getattr(usage, "thinking_tokens", 0) or 0
    if reasoning:
        metadata["output_token_details"] = {"reasoning": reasoning}
    return metadata


def to_ai_message(completion: ChatCompletion) -> AIMessage:
    """Convert a non-streaming completion to an ``AIMessage``.

    Tool calls whose arguments are not a JSON object go to ``invalid_tool_calls``
    with the raw string and the parse error, so the caller can decide what to do.
    """
    choice = completion.choices[0]
    message = choice.message
    tool_calls: list[ToolCall] = []
    invalid: list[InvalidToolCall] = []
    for call in message.tool_calls or []:
        ok, value = _parse_arguments(call.function.arguments)
        if ok:
            tool_calls.append(ToolCall(name=call.function.name, args=value, id=call.id))
        else:
            invalid.append(InvalidToolCall(type="invalid_tool_call", name=call.function.name,
                                           args=call.function.arguments, id=call.id, error=value))
    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
        invalid_tool_calls=invalid,
        additional_kwargs=_kept_data(message.extensions),
        usage_metadata=_usage_metadata(completion.usage),
        response_metadata={
            "model_name": completion.model,
            "finish_reason": choice.finish_reason,
            "id": completion.id,
        },
    )


def to_ai_chunk(chunk: ChatCompletionChunk) -> AIMessageChunk:
    """Convert one streamed chunk to an ``AIMessageChunk``.

    Tool-call fragments become ``tool_call_chunks`` with their raw argument text, so
    the agentic model can join them across chunks, as it does for other providers.
    """
    content = ""
    fragments: list[ToolCallChunk] = []
    finish_reason = None
    kept: dict[str, Any] = {}
    for choice in chunk.choices:
        delta = choice.delta
        kept.update(_kept_data(delta.extensions))
        if delta.content:
            content += delta.content
        for call in delta.tool_calls or []:
            name = call.function.name if call.function else None
            arguments = call.function.arguments if call.function else None
            fragments.append(ToolCallChunk(name=name, args=arguments, id=call.id, index=call.index))
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason
    return AIMessageChunk(
        content=content,
        tool_call_chunks=fragments,
        additional_kwargs=kept,
        usage_metadata=_usage_metadata(chunk.usage),
        response_metadata={"model_name": chunk.model, "finish_reason": finish_reason, "id": chunk.id},
    )


# --- Responses API ----------------------------------------------------------------------------------

def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from a model object or a plain dict, whichever the SDK layer produced."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_content(content: Any, role: str) -> Any:
    """Input content for a Responses message: text, or input parts for user images."""
    if isinstance(content, str) or content is None:
        return content or ""
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            parts.append({"type": "input_text", "text": block})
        elif isinstance(block, dict) and block.get("type") == _TEXT_BLOCK:
            parts.append({"type": "input_text", "text": block.get("text", "")})
        elif isinstance(block, dict) and block.get("type") in ("image", "image_url"):
            if role != "user":
                raise ValueError(f"Images are only supported in user messages, not {role} messages.")
            parts.append({"type": "input_image", "image_url": _image_url(block)})
        elif isinstance(block, dict) and block.get("type") in _NOT_REPLAYED:
            continue
        else:
            kind = block.get("type") if isinstance(block, dict) else type(block).__name__
            raise ValueError(f"Unsupported message content block: {kind}")
    if role != "user" or all(part["type"] == "input_text" for part in parts):
        return "".join(part["text"] for part in parts if part["type"] == "input_text")
    return parts


def to_response_input(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Convert agentic messages to Responses input items.

    Each call is stateless: the whole history is sent every time, and nothing is stored on
    the server. Tool calls go back as ``function_call`` items and results as
    ``function_call_output`` items, both matched by ``call_id``.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            items.append({"role": "system", "content": _response_content(message.content, "system")})
        elif isinstance(message, HumanMessage):
            items.append({"role": "user", "content": _response_content(message.content, "user")})
        elif isinstance(message, AIMessage):
            text = _response_content(message.content, "assistant") if message.content else ""
            if text:
                items.append({"role": "assistant", "content": text})
            for call in list(message.tool_calls) + list(message.invalid_tool_calls):
                args = call.get("args")
                # Valid calls hold parsed arguments; invalid ones keep the raw text, which is sent as written.
                arguments = args if isinstance(args, str) else json.dumps(args or {})
                items.append({"type": "function_call", "call_id": call.get("id") or "",
                              "name": call.get("name") or "", "arguments": arguments})
        elif isinstance(message, ToolMessage):
            items.append({"type": "function_call_output", "call_id": message.tool_call_id,
                          "output": _response_content(message.content, "tool") if message.content else ""})
        else:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")
    return items


def _parse_arguments_for_response(raw: Any) -> tuple[bool, Any]:
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return True, {}
    if isinstance(raw, dict):
        return True, raw
    return _parse_arguments(raw)


def _summary_text(summary: Any) -> str:
    return "".join(_field(part, "text", "") or "" for part in summary or [])


def _response_usage(usage: Any) -> UsageMetadata | None:
    if usage is None:
        return None
    metadata = UsageMetadata(
        input_tokens=_field(usage, "input_tokens", 0) or 0,
        output_tokens=_field(usage, "output_tokens", 0) or 0,
        total_tokens=_field(usage, "total_tokens", 0) or 0,
    )
    cached = getattr(usage, "cached_tokens", 0) or 0
    if cached:
        metadata["input_token_details"] = {"cache_read": cached}
    reasoning = getattr(usage, "reasoning_tokens", 0) or 0
    if reasoning:
        metadata["output_token_details"] = {"reasoning": reasoning}
    return metadata


def response_to_ai_message(response: Response) -> AIMessage:
    """Convert a Responses API response to an ``AIMessage``.

    Text comes from message items, function calls become tool calls (or invalid tool calls
    when their arguments are not a JSON object), reasoning summaries go to
    ``additional_kwargs["reasoning_content"]``, and any other output item (for example a
    hosted tool call) is kept unchanged in ``additional_kwargs["responses_output_items"]``.
    """
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    invalid: list[InvalidToolCall] = []
    reasoning: list[str] = []
    other_items: list[dict[str, Any]] = []
    for item in response.output:
        kind = _field(item, "type")
        if kind == "message":
            text = "".join(
                _field(part, "text", "") or "" for part in _field(item, "content", None) or []
                if _field(part, "type") == "output_text"
            )
            if text:
                texts.append(text)
        elif kind == "function_call":
            call_id = _field(item, "call_id") or _field(item, "id") or ""
            name = _field(item, "name") or ""
            ok, value = _parse_arguments_for_response(_field(item, "arguments"))
            if ok:
                tool_calls.append(ToolCall(name=name, args=value, id=call_id))
            else:
                invalid.append(InvalidToolCall(type="invalid_tool_call", name=name,
                                               args=_field(item, "arguments"), id=call_id, error=value))
        elif kind == "reasoning":
            summary = _summary_text(_field(item, "summary", None))
            if summary:
                reasoning.append(summary)
        else:
            other_items.append(item.model_dump(exclude_none=True) if hasattr(item, "model_dump")
                               else dict(item))
    additional: dict[str, Any] = {}
    if reasoning:
        additional["reasoning_content"] = reasoning
    if other_items:
        additional["responses_output_items"] = other_items
    metadata: dict[str, Any] = {"model_name": response.model, "id": response.id, "status": response.status}
    if response.incomplete_details is not None:
        metadata["incomplete_reason"] = response.incomplete_details.reason
    return AIMessage(
        content="".join(texts),
        tool_calls=tool_calls,
        invalid_tool_calls=invalid,
        usage_metadata=_response_usage(response.usage),
        additional_kwargs=additional,
        response_metadata=metadata,
    )


def response_event_to_chunk(event: ResponseStreamEvent) -> AIMessageChunk | None:
    """Convert one Responses stream event to a chunk, or ``None`` for events with nothing to show.

    Text and reasoning deltas, function-call starts and argument deltas, and the terminal
    event (which carries usage) are converted. A failed response raises ``ResponsesError``
    with the provider's message, so a failure is never mistaken for an empty reply.
    """
    kind = event.type
    if kind == "response.output_text.delta":
        return AIMessageChunk(content=event.delta or "")
    if kind == "response.reasoning_summary_text.delta":
        return AIMessageChunk(content="", additional_kwargs={"reasoning_delta": event.delta or ""})
    if kind == "response.output_item.added":
        item = event.item
        if _field(item, "type") == "function_call":
            return AIMessageChunk(content="", tool_call_chunks=[ToolCallChunk(
                name=_field(item, "name"), args="", id=_field(item, "call_id") or _field(item, "id"),
                index=event.output_index or 0)])
        return None
    if kind == "response.function_call_arguments.delta":
        return AIMessageChunk(content="", tool_call_chunks=[ToolCallChunk(
            name=None, args=event.delta or "", id=None, index=event.output_index or 0)])
    if kind in ("response.completed", "response.incomplete"):
        response = event.response
        metadata: dict[str, Any] = {"id": getattr(response, "id", None), "status": getattr(response, "status", None)}
        return AIMessageChunk(content="", usage_metadata=_response_usage(event.usage),
                              response_metadata=metadata)
    if kind == "response.failed":
        error = event.response.error if event.response is not None else None
        message = (getattr(error, "message", None) if error is not None else None) or "The response failed."
        raise ResponsesError(message)
    return None

