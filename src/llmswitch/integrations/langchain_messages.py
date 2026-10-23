"""
llmswitch.integrations.langchain_messages

Pure conversions between LangChain messages and llmswitch's chat shapes. Nothing
here makes a network call, so each function can be tested on its own.

- LangChain messages -> chat message dicts (``to_chat_messages``)
- ``ChatCompletion`` -> ``AIMessage`` (``to_ai_message``)
- ``ChatCompletionChunk`` -> ``AIMessageChunk`` (``to_ai_chunk``)
"""
from __future__ import annotations

import json
from collections.abc import Sequence
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

from ..models import ChatCompletion, ChatCompletionChunk

_TEXT_BLOCK = "text"


def _image_url(block: dict[str, Any]) -> str:
    """Return a URL for an image block, from either the OpenAI or the LangChain standard shape."""
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


def _content_parts(content: Any) -> list[dict[str, Any]]:
    """Normalize message content into chat content parts."""
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
        else:
            kind = block.get("type") if isinstance(block, dict) else type(block).__name__
            raise ValueError(f"Unsupported message content block: {kind}")
    return parts


def _text_only(parts: list[dict[str, Any]]) -> str:
    return "".join(part["text"] for part in parts)


def _content_for_role(content: Any, role: str) -> Any:
    """Keep images as parts for user messages; other roles only take text."""
    parts = _content_parts(content)
    if role == "user" and any(part["type"] != _TEXT_BLOCK for part in parts):
        return parts
    return _text_only(parts)


def _tool_call_dict(call_id: str | None, name: str | None, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _assistant_dict(message: AIMessage) -> dict[str, Any]:
    text = _text_only(_content_parts(message.content)) if message.content else ""
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
    return result


def to_chat_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to the chat message dicts that llmswitch sends."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            converted.append({"role": "system", "content": _content_for_role(message.content, "system")})
        elif isinstance(message, HumanMessage):
            converted.append({"role": "user", "content": _content_for_role(message.content, "user")})
        elif isinstance(message, AIMessage):
            converted.append(_assistant_dict(message))
        elif isinstance(message, ToolMessage):
            converted.append({"role": "tool", "tool_call_id": message.tool_call_id,
                              "content": _content_for_role(message.content, "tool")})
        else:
            raise ValueError(f"Unsupported LangChain message type: {type(message).__name__}")
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
    LangChain can join them across chunks, as it does for other providers.
    """
    content = ""
    fragments: list[ToolCallChunk] = []
    finish_reason = None
    for choice in chunk.choices:
        delta = choice.delta
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
        usage_metadata=_usage_metadata(chunk.usage),
        response_metadata={"model_name": chunk.model, "finish_reason": finish_reason, "id": chunk.id},
    )
