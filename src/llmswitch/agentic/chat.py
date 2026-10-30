"""
llmswitch.agentic.chat

``ChatLLMSwitch``, a chat model backed by ``LLMSwitch`` and ``AsyncLLMSwitch``.
Every provider that supports chat completions works, and rate limits, retries,
circuit breaking, failover and ``on_event`` apply to calls made through it.

It needs the ``agentic`` extra. Importing this module without it raises ``ImportError``
with the install command.

    from llmswitch.agentic import ChatLLMSwitch

    llm = ChatLLMSwitch(provider="anthropic", api_key="...", model="claude-opus-4-5", temperature=0.2)
    llm.invoke("Hello")
"""
from __future__ import annotations

import inspect
import json
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import Any, Literal

try:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import Field, PrivateAttr

    from .messages import (
        response_event_to_chunk,
        response_to_ai_message,
        to_ai_chunk,
        to_ai_message,
        to_chat_messages,
        to_response_input,
    )
except ImportError as exc:  # pragma: no cover - exercised by tests that hide the optional dependency
    raise ImportError(
        "The agentic features need the agentic extra. Install it with: "
        'pip install "llmswitch-sdk[agentic]"'
    ) from exc

from ..core import AsyncLLMSwitch, LLMSwitch
from ..errors import ToolLoopLimitError

# The framework passes some internal keyword arguments (for example ``ls_structured_output_format``).
# They describe tracing, not the request, so they are not forwarded.
_INTERNAL_PREFIX = "ls_"

# The Responses API is available from these providers only.
_RESPONSES_PROVIDERS = frozenset({"openai", "azure_openai"})

def _tool_choice(value: Any) -> Any:
    """Translate the tool_choice spellings used by the agentic model into chat completions form."""
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


# Guards the one-time creation of each model's async client. A module-level lock keeps the
# model itself free of lock objects, so it can still be copied and pickled.
_ASYNC_CLIENT_LOCK = threading.Lock()


class ChatLLMSwitch(BaseChatModel):
    """A chat model for agents, built on ``LLMSwitch`` and ``AsyncLLMSwitch``.

    It is a LangChain chat model, so it works with ``invoke``, ``ainvoke``, ``stream``, ``astream``, ``bind_tools``
    and the rest of the chat model interface. Rate limits, retries, the circuit breaker, failover and ``on_event``
    apply to its calls, the same as for ``LLMSwitch``. Use it with ``Agent``, ``run_tool_loop`` or on its own.

    Options that the model itself understands are fields. Every other keyword argument is a switch option, passed to
    ``LLMSwitch``: ``provider``, ``api_key``, ``model``, ``endpoint``, ``backup``, ``on_event`` and so on.

    Attributes:
        temperature (float, optional): Sampling temperature. Not sent when None.
        max_tokens (int, optional): The most output tokens per call. Not sent when None.
        stop (list[str], optional): Stop sequences. Not supported with ``api="responses"``.
        call_options (dict): Extra keyword arguments sent with every request, for example
            ``{"thinking": {"type": "enabled", "budget_tokens": 1024}}`` for Anthropic.
        api (str): ``"chat"`` (default) for chat completions, or ``"responses"`` for the Responses API. The Responses
            API is available for ``openai`` and ``azure_openai`` only.

    Example:
        llm = ChatLLMSwitch(provider="openai", api_key="sk-...", model="gpt-4o-mini", max_tokens=300)
        print(llm.invoke("Say hello in French.").content)
    """

    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    call_options: dict[str, Any] = Field(default_factory=dict)
    api: Literal["chat", "responses"] = "chat"

    _switch_options: dict[str, Any] = PrivateAttr(default_factory=dict)
    _switch: Any = PrivateAttr(default=None)
    _async_switch: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        """Create a chat model.

        Options that the model itself understands are set as fields (``temperature``, ``max_tokens``, ``stop``,
        ``call_options`` and ``api``). Every other keyword argument is a switch option, passed to ``LLMSwitch``, so the
        same options apply: ``provider``, ``api_key``, ``model``, ``endpoint``, ``backup``, ``on_event``, the rate
            limits and
        the retry settings.

        Args:
            **kwargs: The model fields and the switch options described in the class docstring.

        Raises:
            ConfigurationError: A switch option is not defined by any provider.
        """
        own_fields = set(type(self).model_fields)
        switch_options = {key: kwargs.pop(key) for key in list(kwargs) if key not in own_fields}
        super().__init__(**kwargs)
        if self.api == "responses":
            provider = switch_options.get("provider")
            if provider not in _RESPONSES_PROVIDERS:
                raise ValueError("api='responses' is supported only for provider='openai' or 'azure_openai'.")
            if self.stop:
                raise ValueError("stop sequences are not supported with api='responses'.")
        self._switch_options = switch_options
        self._switch = LLMSwitch(**switch_options)

    @property
    def _llm_type(self) -> str:
        return "llmswitch"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        # The core client's order: a configured deployment name wins, then a per-call model, then the model.
        model = (self._switch_options.get("deployment_name") or self.call_options.get("model")
                 or self._switch_options.get("model"))
        return {"provider": self._switch_options.get("provider"), "model": model, "api": self.api}

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
        with _ASYNC_CLIENT_LOCK:
            if self._async_switch is None:
                self._async_switch = AsyncLLMSwitch(**self._switch_options)
            return self._async_switch

    def _request(self, stop: list[str] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the keyword arguments for one request. ``None`` means "not set", so it is never sent."""
        request: dict[str, Any] = dict(self.call_options)
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        stop_sequences = stop or self.stop
        if stop_sequences:
            request["stop"] = list(stop_sequences)
        for key, value in kwargs.items():
            if key.startswith(_INTERNAL_PREFIX) or value is None:
                continue
            request[key] = _tool_choice(value) if key == "tool_choice" else value
        return {key: value for key, value in request.items() if value is not None}

    def _responses_request(self, stop: list[str] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
        """The request for the Responses API: its token limit name and flat tool shapes differ from chat."""
        if stop or self.stop:
            raise ValueError("stop sequences are not supported with api='responses'.")
        request = self._request(None, kwargs)
        for chat_name in ("max_tokens", "max_completion_tokens"):
            if chat_name in request:
                request.setdefault("max_output_tokens", request.pop(chat_name))
                request.pop(chat_name, None)
        if "tools" in request:
            request["tools"] = [_responses_tool(tool) for tool in request["tools"]]
        choice = request.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            request["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
        return request

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        if self.api == "responses":
            response = self._switch.responses.create(
                input=to_response_input(messages), **self._responses_request(stop, kwargs))
            return ChatResult(generations=[ChatGeneration(message=response_to_ai_message(response))])
        completion = self._switch.chat.completions.create(
            messages=to_chat_messages(messages, self._switch_options.get("provider")), **self._request(stop, kwargs))
        return ChatResult(generations=[ChatGeneration(message=to_ai_message(completion))])

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        if self.api == "responses":
            response = await self._async().responses.create(
                input=to_response_input(messages), **self._responses_request(stop, kwargs))
            return ChatResult(generations=[ChatGeneration(message=response_to_ai_message(response))])
        completion = await self._async().chat.completions.create(
            messages=to_chat_messages(messages, self._switch_options.get("provider")), **self._request(stop, kwargs))
        return ChatResult(generations=[ChatGeneration(message=to_ai_message(completion))])

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        if self.api == "responses":
            events = self._switch.responses.create(
                input=to_response_input(messages), stream=True, **self._responses_request(stop, kwargs))
            for event in events:
                message = response_event_to_chunk(event)
                if message is None:
                    continue
                if run_manager is not None and message.content:
                    run_manager.on_llm_new_token(message.content, chunk=ChatGenerationChunk(message=message))
                yield ChatGenerationChunk(message=message)
            return
        stream = self._switch.chat.completions.create(
            messages=to_chat_messages(messages, self._switch_options.get("provider")), stream=True, **self._request(stop, kwargs))
        for chunk in stream:
            message = to_ai_chunk(chunk)
            if run_manager is not None and message.content:
                run_manager.on_llm_new_token(message.content, chunk=ChatGenerationChunk(message=message))
            yield ChatGenerationChunk(message=message)

    async def _astream(self, messages: list[BaseMessage], stop: list[str] | None = None,
                       run_manager: AsyncCallbackManagerForLLMRun | None = None,
                       **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        if self.api == "responses":
            events = await self._async().responses.create(
                input=to_response_input(messages), stream=True, **self._responses_request(stop, kwargs))
            async for event in events:
                message = response_event_to_chunk(event)
                if message is None:
                    continue
                if run_manager is not None and message.content:
                    await run_manager.on_llm_new_token(message.content, chunk=ChatGenerationChunk(message=message))
                yield ChatGenerationChunk(message=message)
            return
        stream = await self._async().chat.completions.create(
            messages=to_chat_messages(messages, self._switch_options.get("provider")), stream=True, **self._request(stop, kwargs))
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
        """Return a copy of the model that sends these tools with every request.

        Args:
            tools (list): Tool objects, plain functions with docstrings, Pydantic classes, or OpenAI-style dicts.
            tool_choice (str or bool, optional): ``"auto"`` lets the model decide. ``"none"`` (or False) forbids tools.
                ``"required"`` or ``"any"`` forces a tool call. A tool name forces that tool. Defaults to None, which
                leaves the choice to the provider.
            **kwargs: Passed to the request, as for ``invoke``.

        Returns:
            ChatLLMSwitch: A model with the tools bound. Its replies can contain ``tool_calls``.
        """
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted, **kwargs)


def _responses_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a chat-style function tool into the Responses shape (name and schema at the top level)."""
    if tool.get("type") != "function" or "function" not in tool:
        return dict(tool)
    return {"type": "function", **tool["function"]}


Tool = BaseTool | Callable[..., Any]


def _check_steps(max_steps: int) -> None:
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")


def _tool_map(tools: Sequence[Tool]) -> dict[str, Tool]:
    mapping: dict[str, Tool] = {}
    for tool in tools:
        name: str | None = tool.name if isinstance(tool, BaseTool) else getattr(tool, "__name__", None)
        if not name:
            raise ValueError("Plain callable tools need a __name__; use a def function or a tool object, "
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
    """Run a model with tools until it answers without asking for one, and return the conversation.

    Each turn that asks for tools runs them in order, and the results go back to the model as tool messages. A failed
    tool, an unknown tool name or malformed arguments become error messages, so the model can recover.

    Args:
        model (ChatLLMSwitch): The chat model to run. It must support ``bind_tools``.
        messages (list): The conversation so far, as LangChain messages. Add the user's question as the last message.
        tools (list): The tools the model may call. Plain functions need a docstring and a name.
        max_steps (int): The most model calls allowed. Defaults to 5.

    Returns:
        list: The full conversation, from the input messages to the model's final answer, with every tool call and
        result in order.

    Raises:
        ToolLoopLimitError: The model still asked for tools after ``max_steps`` model calls.
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
    """Async :func:`run_tool_loop`. Tools run one after another, in the same order.

    Args:
        model (ChatLLMSwitch): The chat model to run. It must support ``bind_tools``.
        messages (list): The conversation so far, as LangChain messages.
        tools (list): The tools the model may call.
        max_steps (int): The most model calls allowed. Defaults to 5.

    Returns:
        An awaitable that gives the full conversation, as :func:`run_tool_loop` does.

    Raises:
        ToolLoopLimitError: The model still asked for tools after ``max_steps`` model calls.
    """
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
