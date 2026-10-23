"""
providers._responses_emulation

The Responses API for providers that only have chat completions (Anthropic and
Vertex AI). A Responses request is translated into a chat request, sent through
the provider's own chat provider, and the result is translated back into a
unified ``Response`` (or a stream of ``ResponseStreamEvent``).

Only what chat completions can express is emulated: text and image input, function
tools, structured output, sampling parameters and streaming. Server-side state
(``previous_response_id``, ``conversation``, ``store``, ``background``), stored
objects (retrieve, delete, cancel, input items, conversations, compaction) and hosted
tools are rejected before a request is sent; see ``core.responses_capabilities``.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel

from ..base import BaseResponsesProvider
from ..errors import ChatCompletionError, InvalidResponseRequestError, ResponsesError, UnsupportedCapabilityError
from ..models import (
    ProviderExtensions,
    Response,
    ResponseIncompleteDetails,
    ResponseInputTokensDetails,
    ResponseOutputTokensDetails,
    ResponseStreamEvent,
    ResponseUsage,
    response_output_item_from_data,
    response_stream_event_from_data,
)
from ..structured import populate_response_parsed, pydantic_to_text_format
from ._responses_base import _extract_text_format

# Request fields that are forwarded to the chat provider unchanged.
_PASSTHROUGH = ("temperature", "top_p", "max_output_tokens", "parallel_tool_calls")

# Request fields that are accepted and have no effect on a chat completion.
_IGNORED = frozenset({
    "metadata", "user", "safety_identifier", "service_tier", "prompt_cache_key", "prompt_cache_options",
    "prompt_cache_retention", "truncation", "include", "top_logprobs", "max_tool_calls", "context",
    "stream_options", "reasoning", "extra_headers", "extra_query", "extra_body", "timeout", "store",
})

_HANDLED = frozenset({"model", "instructions", "tools", "tool_choice", "text", "text_format", "response_format"})

_ROLES = {"user": "user", "assistant": "assistant", "system": "system", "developer": "system"}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _invalid(provider: str, message: str) -> InvalidResponseRequestError:
    return InvalidResponseRequestError(message, provider=provider)


def _as_dict(value: Any) -> Any:
    return value.model_dump(exclude_none=True) if isinstance(value, BaseModel) else value


def _content(content: Any, role: str, provider: str) -> Any:
    """Translate Responses message content into chat message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        part = _as_dict(part)
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": part.get("text", "")})
        elif kind == "refusal":
            parts.append({"type": "text", "text": part.get("refusal", "")})
        elif kind == "input_image":
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if not url:
                raise _invalid(provider, "input_image requires an image_url; file_id is not supported")
            image: dict[str, Any] = {"url": url}
            if part.get("detail"):
                image["detail"] = part["detail"]
            parts.append({"type": "image_url", "image_url": image})
        else:
            raise _invalid(provider, f"unsupported input content type '{kind}'")
    if role != "user":
        return "".join(p["text"] for p in parts if p["type"] == "text")
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def input_to_messages(input: Any, instructions: Any, provider: str) -> list[dict[str, Any]]:  # noqa: A002
    """Translate Responses ``input`` and ``instructions`` into chat messages."""
    messages: list[dict[str, Any]] = []
    if instructions:
        if not isinstance(instructions, str):
            raise _invalid(provider, "instructions must be a string")
        messages.append({"role": "system", "content": instructions})
    if isinstance(input, str):
        messages.append({"role": "user", "content": input})
        return messages
    if not input:
        raise _invalid(provider, "input is required")

    pending_calls: list[dict[str, Any]] = []

    def flush_calls() -> None:
        if not pending_calls:
            return
        if messages and messages[-1]["role"] == "assistant" and "tool_calls" not in messages[-1]:
            messages[-1]["tool_calls"] = list(pending_calls)
        else:
            messages.append({"role": "assistant", "content": None, "tool_calls": list(pending_calls)})
        pending_calls.clear()

    for raw in input:
        item = _as_dict(raw)
        kind = item.get("type")
        if kind == "function_call":
            pending_calls.append({"id": item["call_id"], "type": "function",
                                  "function": {"name": item["name"], "arguments": item.get("arguments", "")}})
            continue
        flush_calls()
        if kind == "function_call_output":
            output = item.get("output", "")
            messages.append({"role": "tool", "tool_call_id": item["call_id"],
                             "content": output if isinstance(output, str) else json.dumps(output)})
        elif kind == "reasoning":
            continue
        elif kind in (None, "message"):
            role = _ROLES.get(item.get("role", ""))
            if role is None:
                raise _invalid(provider, f"unsupported message role '{item.get('role')}'")
            messages.append({"role": role, "content": _content(item.get("content"), role, provider)})
        else:
            raise _invalid(provider, f"unsupported input item type '{kind}'")
    flush_calls()
    return messages


def tools_to_chat(tools: list[Any], provider: str) -> list[dict[str, Any]]:
    """Translate Responses function tools into chat function tools."""
    chat_tools = []
    for tool in tools:
        tool = _as_dict(tool)
        if tool.get("type") != "function":
            raise UnsupportedCapabilityError(provider=provider, feature=f"tool.{tool.get('type')}")
        if "function" in tool:  # already in chat shape
            chat_tools.append(tool)
            continue
        function = {"name": tool["name"], "parameters": tool.get("parameters") or {"type": "object", "properties": {}}}
        for key in ("description", "strict"):
            if tool.get(key) is not None:
                function[key] = tool[key]
        chat_tools.append({"type": "function", "function": function})
    return chat_tools


def tool_choice_to_chat(tool_choice: Any, provider: str) -> Any:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    tool_choice = _as_dict(tool_choice)
    if tool_choice.get("type") == "function":
        name = tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")
        return {"type": "function", "function": {"name": name}}
    raise _invalid(provider, f"unsupported tool_choice type '{tool_choice.get('type')}'")


def text_to_response_format(text: Any, provider: str) -> Optional[dict[str, Any]]:
    """Translate the Responses ``text.format`` block into a chat ``response_format``."""
    fmt = _as_dict(text).get("format") if text else None
    if not fmt or fmt.get("type") == "text":
        return None
    if fmt["type"] == "json_object":
        return {"type": "json_object"}
    if fmt["type"] == "json_schema":
        schema = {key: fmt[key] for key in ("name", "schema", "strict") if fmt.get(key) is not None}
        return {"type": "json_schema", "json_schema": schema}
    raise _invalid(provider, f"unsupported text format '{fmt['type']}'")


@dataclass
class _RequestInfo:
    """What a Response echoes back from its request."""

    model: Optional[str]
    instructions: Optional[str] = None
    echo: dict[str, Any] = field(default_factory=dict)


def to_chat_request(provider: str, default_model: Optional[str], input: Any,  # noqa: A002
                    kwargs: dict[str, Any], text_format: Optional[type],
                    stream: bool) -> tuple[list[dict[str, Any]], dict[str, Any], _RequestInfo]:
    """Build ``(messages, chat kwargs, request info)`` from a Responses request."""
    unknown = sorted(set(kwargs) - _HANDLED - _IGNORED - set(_PASSTHROUGH))
    if unknown:
        raise _invalid(provider, f"unsupported Responses fields for {provider}: {', '.join(unknown)}")
    model = kwargs.get("model") or default_model
    messages = input_to_messages(input, kwargs.get("instructions"), provider)
    chat: dict[str, Any] = {key: kwargs[key] for key in _PASSTHROUGH if kwargs.get(key) is not None}
    if model:
        chat["model"] = model
    echo: dict[str, Any] = {key: kwargs[key] for key in _PASSTHROUGH if kwargs.get(key) is not None}
    if kwargs.get("tools"):
        chat["tools"] = tools_to_chat(kwargs["tools"], provider)
        echo["tools"] = [_as_dict(tool) for tool in kwargs["tools"]]
    if kwargs.get("tool_choice") is not None:
        chat["tool_choice"] = tool_choice_to_chat(kwargs["tool_choice"], provider)
        echo["tool_choice"] = kwargs["tool_choice"]
    text = kwargs.get("text")
    if text_format is not None and stream:
        text = pydantic_to_text_format(text_format)
    response_format = text_to_response_format(text, provider)
    if response_format is not None:
        echo["text"] = _as_dict(text)
        chat["response_format"] = response_format
    if text_format is not None and not stream:
        chat["response_format"] = text_format
    return messages, chat, _RequestInfo(model=model, instructions=kwargs.get("instructions"), echo=echo)


def _usage(usage: Any) -> Optional[ResponseUsage]:
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(usage, "thinking_tokens", 0) or getattr(details, "reasoning_tokens", 0) or 0
    return ResponseUsage(
        input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, total_tokens=usage.total_tokens,
        input_tokens_details=ResponseInputTokensDetails(cached_tokens=usage.cached_tokens),
        output_tokens_details=ResponseOutputTokensDetails(reasoning_tokens=reasoning),
    )


def _status(finish_reason: Optional[str]) -> tuple[str, Optional[ResponseIncompleteDetails]]:
    if finish_reason in ("length", "max_completion_tokens"):
        return "incomplete", ResponseIncompleteDetails(reason="max_output_tokens")
    if finish_reason == "content_filter":
        return "incomplete", ResponseIncompleteDetails(reason="content_filter")
    return "completed", None


def _message_item(text: str, status: str = "completed", item_id: Optional[str] = None) -> dict[str, Any]:
    content = [{"type": "output_text", "text": text, "annotations": []}] if text else []
    return {"type": "message", "id": item_id or _new_id("msg"), "status": status, "role": "assistant",
            "content": content}


def _call_item(call_id: str, name: str, arguments: str, status: str = "completed",
               item_id: Optional[str] = None) -> dict[str, Any]:
    return {"type": "function_call", "id": item_id or _new_id("fc"), "status": status, "call_id": call_id,
            "name": name, "arguments": arguments}


def _build_response(provider: str, info: _RequestInfo, *, response_id: str, created: float, model: Optional[str],
                    status: str, output: list[dict[str, Any]], usage: Any = None,
                    incomplete: Optional[ResponseIncompleteDetails] = None,
                    finish_reason: Optional[str] = None) -> Response:
    return Response(
        id=response_id, created_at=created, completed_at=time.time() if status != "in_progress" else None,
        model=model or info.model, status=status,
        output=[response_output_item_from_data(item) for item in output],
        usage=_usage(usage), incomplete_details=incomplete, instructions=info.instructions,
        extensions=ProviderExtensions(provider=provider, fields={"emulated_via": "chat.completions",
                                                                 "finish_reason": finish_reason}),
        **info.echo,
    )


def completion_to_response(completion: Any, info: _RequestInfo, provider: str) -> Response:
    """Translate a unified ``ChatCompletion`` into a unified ``Response``."""
    choice = completion.choices[0]
    message = choice.message
    output: list[dict[str, Any]] = []
    if message.content:
        output.append(_message_item(message.content))
    for call in message.tool_calls or []:
        output.append(_call_item(call.id, call.function.name, call.function.arguments))
    status, incomplete = _status(choice.finish_reason)
    response = _build_response(provider, info, response_id=_new_id("resp"), created=completion.created,
                               model=completion.model, status=status, output=output, usage=completion.usage,
                               incomplete=incomplete, finish_reason=choice.finish_reason)
    if getattr(message, "parsed", None) is not None:
        for item in response.output:
            if item.type == "message":
                item.parsed = message.parsed
    return response


class StreamTranslator:
    """Turns chat completion chunks into Responses stream events."""

    def __init__(self, info: _RequestInfo, provider: str) -> None:
        self.info = info
        self.provider = provider
        self.response_id = _new_id("resp")
        self.created = time.time()
        self.model = info.model
        self.finish_reason: Optional[str] = None
        self.usage: Any = None
        self._sequence = 0
        self._items: list[dict[str, Any]] = []        # output items in output_index order
        self._message_index: Optional[int] = None
        self._call_index: dict[int, int] = {}         # chat tool-call index -> output index
        self._generated_call_ids: set[int] = set()    # output indexes whose call_id we made up

    def _event(self, type: str, **fields: Any) -> ResponseStreamEvent:  # noqa: A002
        event = response_stream_event_from_data({"type": type, "sequence_number": self._sequence, **fields})
        self._sequence += 1
        return event

    def _response(self, status: str, output: list[dict[str, Any]], **extra: Any) -> Response:
        return _build_response(self.provider, self.info, response_id=self.response_id, created=self.created,
                               model=self.model, status=status, output=output, finish_reason=self.finish_reason,
                               **extra)

    def start(self) -> list[ResponseStreamEvent]:
        response = self._response("in_progress", [])
        return [self._event("response.created", response=response),
                self._event("response.in_progress", response=response)]

    def feed(self, chunk: Any) -> list[ResponseStreamEvent]:
        events: list[ResponseStreamEvent] = []
        self.model = chunk.model or self.model
        if chunk.usage is not None:
            self.usage = chunk.usage
        for choice in chunk.choices:
            if choice.index != 0:
                continue
            delta = choice.delta
            if delta.content:
                events += self._text(delta.content)
            for call in delta.tool_calls or []:
                events += self._tool_call(call)
            if choice.finish_reason is not None:
                self.finish_reason = choice.finish_reason
        return events

    def _text(self, text: str) -> list[ResponseStreamEvent]:
        events = []
        if self._message_index is None:
            self._message_index = len(self._items)
            item = _message_item("", "in_progress")
            item["text"] = ""
            self._items.append(item)
            events.append(self._event("response.output_item.added", output_index=self._message_index,
                                      item=_message_item("", "in_progress", item["id"])))
            events.append(self._event("response.content_part.added", item_id=item["id"],
                                      output_index=self._message_index, content_index=0,
                                      part={"type": "output_text", "text": "", "annotations": []}))
        item = self._items[self._message_index]
        item["text"] += text
        events.append(self._event("response.output_text.delta", item_id=item["id"],
                                  output_index=self._message_index, content_index=0, delta=text))
        return events

    def _tool_call(self, call: Any) -> list[ResponseStreamEvent]:
        events = []
        if call.index not in self._call_index:
            self._call_index[call.index] = len(self._items)
            item = _call_item(call.id or _new_id("call"), "", "", "in_progress")
            if not call.id:
                self._generated_call_ids.add(len(self._items))
            self._items.append(item)
            started_name = (call.function.name if call.function else None) or ""
            item["name"] = started_name
            events.append(self._event("response.output_item.added", output_index=self._call_index[call.index],
                                      item=dict(item)))
        index = self._call_index[call.index]
        item = self._items[index]
        if call.id and index in self._generated_call_ids:
            item["call_id"] = call.id
            self._generated_call_ids.discard(index)
        if call.function is not None:
            if call.function.name and not item["name"]:
                item["name"] = call.function.name
            if call.function.arguments:
                item["arguments"] += call.function.arguments
                events.append(self._event("response.function_call_arguments.delta", item_id=item["id"],
                                          output_index=index, delta=call.function.arguments))
        return events

    def finish(self) -> list[ResponseStreamEvent]:
        events: list[ResponseStreamEvent] = []
        output = []
        for index, item in enumerate(self._items):
            if item["type"] == "message":
                text = item.pop("text")
                part = {"type": "output_text", "text": text, "annotations": []}
                done = _message_item(text, "completed", item["id"])
                events += [
                    self._event("response.output_text.done", item_id=item["id"], output_index=index,
                                content_index=0, text=text),
                    self._event("response.content_part.done", item_id=item["id"], output_index=index,
                                content_index=0, part=part),
                    self._event("response.output_item.done", output_index=index, item=done),
                ]
            else:
                done = {**item, "status": "completed"}
                events += [
                    self._event("response.function_call_arguments.done", item_id=item["id"], output_index=index,
                                name=item["name"], arguments=item["arguments"]),
                    self._event("response.output_item.done", output_index=index, item=done),
                ]
            output.append(done)
        status, incomplete = _status(self.finish_reason)
        terminal = "response.completed" if status == "completed" else "response.incomplete"
        events.append(self._event(terminal, response=self._response(status, output, usage=self.usage,
                                                                    incomplete=incomplete)))
        return events


def _responses_error(exc: ChatCompletionError, provider: str) -> ResponsesError:
    return ResponsesError(exc.message, status_code=exc.status_code, provider=provider, operation="create")


class _EmulatedResponsesBase(BaseResponsesProvider):
    """Wraps a chat provider; the unsupported management verbs come from ``BaseResponsesProvider``."""

    chat_cls: type
    provider_name: str

    def __init__(self, **kwargs: Any) -> None:
        self._chat = self.chat_cls(**kwargs)
        self.model: Optional[str] = self._chat.model
        self.client = self._chat.client
        self.endpoint: Optional[str] = kwargs.get("endpoint")
        self.timeout: Optional[float] = kwargs.get("timeout")

    def _prepare(self, input: Any, stream: bool,  # noqa: A002
                 kwargs: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], _RequestInfo, Optional[type]]:
        kwargs = dict(kwargs)
        text_format = _extract_text_format(kwargs)
        messages, chat_kwargs, info = to_chat_request(self.provider_name, self.model, input, kwargs, text_format,
                                                      stream)
        return messages, chat_kwargs, info, text_format

    def _finish(self, completion: Any, info: _RequestInfo, text_format: Optional[type]) -> Response:
        response = completion_to_response(completion, info, self.provider_name)
        if text_format is not None:
            populate_response_parsed(response, text_format)
        return response


class SyncEmulatedResponsesProvider(_EmulatedResponsesBase):
    def create(self, input: Any = None, stream: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        messages, chat_kwargs, info, text_format = self._prepare(input, stream, kwargs)
        if stream:
            return self._streaming_create(messages, chat_kwargs, info)
        try:
            completion = self._chat.create(messages, stream=False, **chat_kwargs)
        except ChatCompletionError as exc:
            raise _responses_error(exc, self.provider_name) from exc
        return self._finish(completion, info, text_format)

    def _streaming_create(self, messages: list[dict[str, Any]], chat_kwargs: dict[str, Any],
                          info: _RequestInfo) -> Generator[ResponseStreamEvent, None, None]:
        translator = StreamTranslator(info, self.provider_name)
        try:
            yield from translator.start()
            for chunk in self._chat.create(messages, stream=True, **chat_kwargs):
                yield from translator.feed(chunk)
            yield from translator.finish()
        except ChatCompletionError as exc:
            raise _responses_error(exc, self.provider_name) from exc


class AsyncEmulatedResponsesProvider(_EmulatedResponsesBase):
    async def create(self, input: Any = None, stream: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        messages, chat_kwargs, info, text_format = self._prepare(input, stream, kwargs)
        if stream:
            return self._streaming_create(messages, chat_kwargs, info)
        try:
            completion = await self._chat.create(messages, stream=False, **chat_kwargs)
        except ChatCompletionError as exc:
            raise _responses_error(exc, self.provider_name) from exc
        return self._finish(completion, info, text_format)

    async def _streaming_create(self, messages: list[dict[str, Any]], chat_kwargs: dict[str, Any],
                                info: _RequestInfo) -> AsyncGenerator[ResponseStreamEvent, None]:
        translator = StreamTranslator(info, self.provider_name)
        try:
            for event in translator.start():
                yield event
            async for chunk in await self._chat.create(messages, stream=True, **chat_kwargs):
                for event in translator.feed(chunk):
                    yield event
            for event in translator.finish():
                yield event
        except ChatCompletionError as exc:
            raise _responses_error(exc, self.provider_name) from exc

    async def aclose(self) -> None:
        close = getattr(self._chat, "aclose", None)
        if close is not None:
            await close()
