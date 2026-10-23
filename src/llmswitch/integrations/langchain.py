"""
llmswitch.integrations.langchain

``ChatLLMSwitch``, a LangChain chat model backed by ``LLMSwitch`` and ``AsyncLLMSwitch``.
Every provider that supports chat completions works, and rate limits, retries,
circuit breaking, failover and ``on_event`` apply to calls made through it.

It needs ``langchain-core``, installed by the ``langchain`` extra. Importing this
module without it raises ``ImportError`` with the install command.

    from llmswitch.integrations.langchain import ChatLLMSwitch

    llm = ChatLLMSwitch(provider="anthropic", api_key="...", model="claude-opus-4-5", temperature=0.2)
    llm.invoke("Hello")
"""
from __future__ import annotations

import inspect
import json
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import Any

try:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import Field, PrivateAttr

    from .langchain_messages import to_ai_chunk, to_ai_message, to_chat_messages
except ImportError as exc:  # pragma: no cover - exercised by tests that hide langchain_core
    raise ImportError(
        "The LangChain integration needs langchain-core. Install it with: "
        'pip install "llmswitch-sdk[langchain]"'
    ) from exc

from ..core import AsyncLLMSwitch, LLMSwitch
from ..errors import ToolLoopLimitError

# LangChain passes some internal keyword arguments (for example ``ls_structured_output_format``).
# They describe tracing, not the request, so they are not forwarded.
_INTERNAL_PREFIX = "ls_"

def _tool_choice(value: Any) -> Any:
    """Translate LangChain's tool_choice spellings into chat completions form."""
    if value is None:
        return None
    if value is False:
        return "none"
    if value is True or value == "any":
        return "required"
    if isinstance(value, str):
        if value in ("auto", "none", "required"):
            return value
        return {"type": "function", "function": {"name": value}}
    return value


class ChatLLMSwitch(BaseChatModel):
    """
    LangChain chat model over llmswitch.

    Options that the model itself understands are fields: ``temperature``, ``max_tokens``,
    ``stop``, and ``call_options`` (extra keyword arguments sent with every request).
    Every other keyword argument is a switch option, passed to ``LLMSwitch`` (for example
    ``provider``, ``api_key``, ``model``, ``endpoint``, ``backup``, ``on_event``).
    """

    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    call_options: dict[str, Any] = Field(default_factory=dict)

    _switch_options: dict[str, Any] = PrivateAttr(default_factory=dict)
    _switch: Any = PrivateAttr(default=None)
    _async_switch: Any = PrivateAttr(default=None)
    _async_lock: Any = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, **kwargs: Any) -> None:
        own_fields = set(type(self).model_fields)
        switch_options = {key: kwargs.pop(key) for key in list(kwargs) if key not in own_fields}
        super().__init__(**kwargs)
        self._switch_options = switch_options
        self._switch = LLMSwitch(**switch_options)

    @property
    def _llm_type(self) -> str:
        return "llmswitch"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"provider": self._switch_options.get("provider"), "model": self._switch_options.get("model")}

    def close(self) -> None:
        """Close the sync client and its provider connections."""
        self._switch.close()

    async def aclose(self) -> None:
        """Close the async client, if one was created, and the sync client."""
        if self._async_switch is not None:
            await self._async_switch.aclose()
        self._switch.close()

    def _async(self) -> Any:
        """Build the async client on first use, so sync-only users never create one.

        Locked, so concurrent first calls from several threads share one client.
        """
        with self._async_lock:
            if self._async_switch is None:
                self._async_switch = AsyncLLMSwitch(**self._switch_options)
            return self._async_switch

    def _request(self, stop: list[str] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
        request: dict[str, Any] = dict(self.call_options)
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        stop_sequences = stop or self.stop
        if stop_sequences:
            request["stop"] = list(stop_sequences)
        for key, value in kwargs.items():
            if key.startswith(_INTERNAL_PREFIX):
                continue
            request[key] = _tool_choice(value) if key == "tool_choice" else value
        return request

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        completion = self._switch.chat.completions.create(
            messages=to_chat_messages(messages), **self._request(stop, kwargs))
        return ChatResult(generations=[ChatGeneration(message=to_ai_message(completion))])

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        completion = await self._async().chat.completions.create(
            messages=to_chat_messages(messages), **self._request(stop, kwargs))
        return ChatResult(generations=[ChatGeneration(message=to_ai_message(completion))])

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        stream = self._switch.chat.completions.create(
            messages=to_chat_messages(messages), stream=True, **self._request(stop, kwargs))
        for chunk in stream:
            message = to_ai_chunk(chunk)
            if run_manager is not None and message.content:
                run_manager.on_llm_new_token(message.content, chunk=ChatGenerationChunk(message=message))
            yield ChatGenerationChunk(message=message)

    async def _astream(self, messages: list[BaseMessage], stop: list[str] | None = None,
                       run_manager: AsyncCallbackManagerForLLMRun | None = None,
                       **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        stream = await self._async().chat.completions.create(
            messages=to_chat_messages(messages), stream=True, **self._request(stop, kwargs))
        async for chunk in stream:
            message = to_ai_chunk(chunk)
            if run_manager is not None and message.content:
                await run_manager.on_llm_new_token(message.content, chunk=ChatGenerationChunk(message=message))
            yield ChatGenerationChunk(message=message)

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Return a copy of the model that sends ``tools`` with every request.

        ``tools`` may be LangChain tools, Pydantic classes, plain functions or OpenAI-style
        dicts. ``tool_choice`` accepts ``"auto"``, ``"none"``, ``"required"`` (or ``"any"``),
        or the name of one tool to force.
        """
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted, **kwargs)


Tool = BaseTool | Callable[..., Any]


def _check_steps(max_steps: int) -> None:
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")


def _tool_map(tools: Sequence[Tool]) -> dict[str, Tool]:
    mapping: dict[str, Tool] = {}
    for tool in tools:
        name: str | None = tool.name if isinstance(tool, BaseTool) else getattr(tool, "__name__", None)
        if not name:
            raise ValueError("Plain callable tools need a __name__; use a def function or a LangChain tool, "
                             "not functools.partial.")
        if name in mapping:
            raise ValueError(f"Duplicate tool name: {name}")
        mapping[name] = tool
    return mapping


def _as_messages(messages: str | Sequence[BaseMessage]) -> list[BaseMessage]:
    if isinstance(messages, str):
        return [HumanMessage(content=messages)]
    return list(messages)


def _content(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


def _tool_error(call_id: str, text: str) -> ToolMessage:
    return ToolMessage(content=f"Error: {text}", tool_call_id=call_id, status="error")


def _call_id(call: Mapping[str, Any]) -> str:
    return call.get("id") or f"call_{uuid.uuid4().hex[:24]}"


def _unknown_tool(name: str, mapping: dict[str, Tool], call_id: str) -> ToolMessage:
    available = ", ".join(sorted(mapping)) or "none"
    return _tool_error(call_id, f"unknown tool '{name}'. Available tools: {available}.")


def _invalid_call(call: Mapping[str, Any]) -> ToolMessage:
    return _tool_error(_call_id(call),
                       f"the arguments for '{call.get('name')}' are not valid JSON: {call.get('error')}")


def _success(call_id: str, result: Any) -> ToolMessage:
    return ToolMessage(content=_content(result), tool_call_id=call_id, status="success")


def _run_sync(tool: Tool, args: dict[str, Any]) -> Any:
    if isinstance(tool, BaseTool):
        return tool.invoke(args)
    return tool(**args)


async def _run_async(tool: Tool, args: dict[str, Any]) -> Any:
    if isinstance(tool, BaseTool):
        return await tool.ainvoke(args)
    result = tool(**args)
    if inspect.isawaitable(result):
        result = await result
    return result


def run_tool_loop(
    model: BaseChatModel,
    messages: str | Sequence[BaseMessage],
    tools: Sequence[Tool],
    *,
    max_steps: int = 5,
) -> list[BaseMessage]:
    """Run a bounded tool loop and return the whole conversation.

    The model is bound to ``tools``. Each turn that requests tools runs them in order, and
    their results go back to the model as tool messages. A failed tool, an unknown tool
    name or malformed arguments become error tool messages, so the model can recover.

    ``max_steps`` is the most model calls allowed. If the model still asks for tools on
    the last one, :class:`ToolLoopLimitError` is raised. The returned list ends with the
    model's final answer.
    """
    _check_steps(max_steps)
    mapping = _tool_map(tools)
    bound = model.bind_tools(list(tools))
    history = _as_messages(messages)
    for step in range(1, max_steps + 1):
        ai = bound.invoke(history)
        history.append(ai)
        if not ai.tool_calls and not ai.invalid_tool_calls:
            return history
        if step == max_steps:
            break
        history.extend(_sync_results(ai, mapping))
    raise ToolLoopLimitError(max_steps)


async def arun_tool_loop(
    model: BaseChatModel,
    messages: str | Sequence[BaseMessage],
    tools: Sequence[Tool],
    *,
    max_steps: int = 5,
) -> list[BaseMessage]:
    """Async :func:`run_tool_loop`. Tools run one after another, so their order is kept."""
    _check_steps(max_steps)
    mapping = _tool_map(tools)
    bound = model.bind_tools(list(tools))
    history = _as_messages(messages)
    for step in range(1, max_steps + 1):
        ai = await bound.ainvoke(history)
        history.append(ai)
        if not ai.tool_calls and not ai.invalid_tool_calls:
            return history
        if step == max_steps:
            break
        history.extend(await _async_results(ai, mapping))
    raise ToolLoopLimitError(max_steps)


def _sync_results(ai: AIMessage, mapping: dict[str, Tool]) -> list[ToolMessage]:
    results: list[ToolMessage] = []
    for call in ai.tool_calls:
        call_id = _call_id(call)
        tool = mapping.get(call["name"])
        if tool is None:
            results.append(_unknown_tool(call["name"], mapping, call_id))
            continue
        try:
            results.append(_success(call_id, _run_sync(tool, call["args"])))
        except Exception as exc:  # noqa: BLE001 - any tool failure goes back to the model
            results.append(_tool_error(call_id, f"{type(exc).__name__}: {exc}"))
    results.extend(_invalid_call(call) for call in ai.invalid_tool_calls)
    return results


async def _async_results(ai: AIMessage, mapping: dict[str, Tool]) -> list[ToolMessage]:
    results: list[ToolMessage] = []
    for call in ai.tool_calls:
        call_id = _call_id(call)
        tool = mapping.get(call["name"])
        if tool is None:
            results.append(_unknown_tool(call["name"], mapping, call_id))
            continue
        try:
            results.append(_success(call_id, await _run_async(tool, call["args"])))
        except Exception as exc:  # noqa: BLE001 - any tool failure goes back to the model
            results.append(_tool_error(call_id, f"{type(exc).__name__}: {exc}"))
    results.extend(_invalid_call(call) for call in ai.invalid_tool_calls)
    return results
