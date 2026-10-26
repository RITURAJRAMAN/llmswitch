"""
llmswitch.agentic.agent

The agent loop. Each step asks the model for an answer. A final answer (no tool calls) goes
through an optional reflection check, which can send it back for a revision. Tool calls are run
and their results are added to the conversation before the next step.

The loop stops with a status in the result:

- ``completed``: the answer passed reflection (or reflection is off).
- ``completed_with_reflection_limit``: the answer was still rejected after ``reflection_max_retries``
  revisions, so the last answer is returned.
- ``stopped`` with a ``reason``: ``max_iterations_reached``, ``max_runtime_exceeded`` or
  ``max_input_tokens_per_step_exceeded``.

Every model call is a step record, so the caller can see what happened on each turn.
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
except ImportError as exc:  # pragma: no cover - exercised by tests that hide the optional dependency
    raise ImportError(
        "The agentic features need the agentic extra. Install it with: "
        'pip install "llmswitch-sdk[agentic]"'
    ) from exc

from ..ratelimit.tokenizer import count_tokens
from .chat import Tool, _as_messages, _async_results, _sync_results, _tool_map

REFLECTION_PROMPT = (
    "Evaluate if the assistant answer is complete and correct for the user request. "
    'Respond ONLY with JSON: {"is_sufficient": boolean, "feedback": string}.'
)

# The verdict is read from the reply text. The check is asked again when the reply is not a readable
# verdict, and only repeated failure raises. No tool is bound for this call: a tool-call verdict was
# unreliable on Vertex AI, where the model sometimes called a tool that was not declared.
REFLECTION_ATTEMPTS = 3

# Some models (Gemini 2.5 among them) spend output tokens on internal reasoning, so a small limit
# can cut the verdict off. The verdict is short, so the check gets at least this many output tokens.
REFLECTION_MIN_OUTPUT_TOKENS = 2048


@dataclass(frozen=True)
class AgentConfig:
    """Limits and behaviour for an :class:`Agent`.

    ``max_iterations`` is the most model calls one run may make. ``max_runtime_seconds`` is the
    wall-clock limit for one run. ``reflection_max_retries`` is how many revisions a rejected answer
    may get before the last answer is returned. ``max_input_tokens_per_step`` stops a run before a
    call whose input would exceed the limit, using the library's own token count (an estimate).
    """

    max_iterations: int = 12
    max_runtime_seconds: float = 180.0
    tool_choice: str | None = "auto"
    max_input_tokens_per_step: int | None = None
    reflection_enabled: bool = True
    reflection_max_retries: int = 2

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if self.max_input_tokens_per_step is not None and self.max_input_tokens_per_step < 1:
            raise ValueError("max_input_tokens_per_step must be at least 1 when set")
        if self.reflection_max_retries < 0:
            raise ValueError("reflection_max_retries must not be negative")


@dataclass
class AgentStep:
    """One model call in a run, and the tool results it caused."""

    step: int
    assistant_content: str
    tool_calls: list[Any] = field(default_factory=list)
    model_used: str | None = None
    tool_results: list[ToolMessage] = field(default_factory=list)


@dataclass
class _Budget:
    """Iteration, runtime and per-step input limits for one run."""

    max_iterations: int
    max_runtime_seconds: float
    max_input_tokens_per_step: int | None
    started_at: float = field(default_factory=time.monotonic)

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_runtime_seconds - (time.monotonic() - self.started_at))

    def runtime_exceeded(self) -> bool:
        return self.remaining_seconds() <= 0

    def input_exceeded(self, tokens: int) -> bool:
        return self.max_input_tokens_per_step is not None and tokens > self.max_input_tokens_per_step


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not content:
        return ""
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def _transcript(messages: Sequence[BaseMessage]) -> str:
    """The conversation as plain text, with tool calls and results written out as text.

    The reflection check reads this instead of the raw message list. Raw tool-call history can
    lead some models to call tools the check does not declare, so the verdict is requested
    without any tool history attached.
    """
    lines: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            lines.append(f"Tool result: {_text(message.content)}")
            continue
        if isinstance(message, AIMessage):
            if message.content:
                lines.append(f"Assistant: {_text(message.content)}")
            for call in message.tool_calls or []:
                lines.append(f"Assistant called tool {call.get('name')} with {call.get('args')}")
            continue
        role = "User" if isinstance(message, HumanMessage) else "System"
        lines.append(f"{role}: {_text(message.content)}")
    return "\n".join(lines)


def _reflection_request(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    return [HumanMessage(content=f"Conversation so far:\n{_transcript(messages)}\n\n{REFLECTION_PROMPT}")]


def _verdict(reply: Any) -> tuple[bool, str] | None:
    """Read the verdict JSON from a reply, or ``None`` when the reply does not contain one."""
    text = _text(getattr(reply, "content", reply)).strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("is_sufficient"), bool):
        return None
    return value["is_sufficient"], value.get("feedback") or "Improve answer quality and correctness."


def estimate_input_tokens(messages: Sequence[BaseMessage], model_name: str | None = None) -> int:
    """Estimate the input size of a request with the library's own token counter."""
    total = 0
    for message in messages:
        total += count_tokens(_text(message.content), model_name) + 4  # a few tokens per message of overhead
        for call in getattr(message, "tool_calls", None) or []:
            total += count_tokens(str(call.get("name", "")) + str(call.get("args", "")), model_name)
    return total


class Agent:
    """A think-act-observe agent over a chat model.

    ``model`` is a :class:`~llmswitch.agentic.ChatLLMSwitch` (or any chat model that supports
    ``bind_tools``). ``tools`` are tool objects or plain functions.

    Usage::

        agent = Agent(model=llm, tools=[get_weather], config=AgentConfig(max_iterations=6))
        result = agent.invoke("Weather in Paris, then a short summary.")
        print(result["status"], result["output"])
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[Tool] | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.model = model
        self.tools = list(tools or [])
        self.config = config or AgentConfig()
        self._tool_map = _tool_map(self.tools) if self.tools else {}

    # --- public API -------------------------------------------------------------------------

    def invoke(self, input: str | list[BaseMessage], *, max_runtime_seconds: float | None = None) -> dict[str, Any]:
        """Run the loop synchronously and return the result dictionary."""
        budget = self._budget(max_runtime_seconds)
        bound = self._bound_model()
        messages = _as_messages(input)
        steps: list[AgentStep] = []
        reflection_retries = 0
        for step_number in range(1, budget.max_iterations + 1):
            if budget.runtime_exceeded():
                return self._stopped("max_runtime_exceeded", messages, steps)
            tokens = estimate_input_tokens(messages, self._model_name())
            if budget.input_exceeded(tokens):
                return self._stopped("max_input_tokens_per_step_exceeded", messages, steps,
                                     input_tokens=tokens, limit=budget.max_input_tokens_per_step)
            answer = bound.invoke(messages)
            record = self._record(step_number, answer)
            messages.append(answer)
            if budget.runtime_exceeded():
                return self._stopped("max_runtime_exceeded", messages, [*steps, record])
            if not answer.tool_calls:
                if not self.config.reflection_enabled:
                    return self._completed("completed", answer, messages, [*steps, record])
                ok, feedback = self._reflect(messages)
                if budget.runtime_exceeded():
                    return self._stopped("max_runtime_exceeded", messages, [*steps, record])
                if ok:
                    return self._completed("completed", answer, messages, [*steps, record])
                if reflection_retries >= self.config.reflection_max_retries:
                    return self._completed("completed_with_reflection_limit", answer, messages, [*steps, record])
                reflection_retries += 1
                messages.append(HumanMessage(content=f"Please revise your answer. Feedback: {feedback}"))
                steps.append(record)
                continue
            record.tool_results = _sync_results(answer, self._tool_map)
            messages.extend(record.tool_results)
            steps.append(record)
        return self._stopped("max_iterations_reached", messages, steps)

    async def ainvoke(self, input: str | list[BaseMessage], *, max_runtime_seconds: float | None = None) -> dict[str, Any]:
        """Run the loop asynchronously and return the result dictionary."""
        budget = self._budget(max_runtime_seconds)
        bound = self._bound_model()
        messages = _as_messages(input)
        steps: list[AgentStep] = []
        reflection_retries = 0
        for step_number in range(1, budget.max_iterations + 1):
            if budget.runtime_exceeded():
                return self._stopped("max_runtime_exceeded", messages, steps)
            tokens = estimate_input_tokens(messages, self._model_name())
            if budget.input_exceeded(tokens):
                return self._stopped("max_input_tokens_per_step_exceeded", messages, steps,
                                     input_tokens=tokens, limit=budget.max_input_tokens_per_step)
            answer = await bound.ainvoke(messages)
            record = self._record(step_number, answer)
            messages.append(answer)
            if budget.runtime_exceeded():
                return self._stopped("max_runtime_exceeded", messages, [*steps, record])
            if not answer.tool_calls:
                if not self.config.reflection_enabled:
                    return self._completed("completed", answer, messages, [*steps, record])
                ok, feedback = await self._areflect(messages)
                if budget.runtime_exceeded():
                    return self._stopped("max_runtime_exceeded", messages, [*steps, record])
                if ok:
                    return self._completed("completed", answer, messages, [*steps, record])
                if reflection_retries >= self.config.reflection_max_retries:
                    return self._completed("completed_with_reflection_limit", answer, messages, [*steps, record])
                reflection_retries += 1
                messages.append(HumanMessage(content=f"Please revise your answer. Feedback: {feedback}"))
                steps.append(record)
                continue
            record.tool_results = await _async_results(answer, self._tool_map)
            messages.extend(record.tool_results)
            steps.append(record)
        return self._stopped("max_iterations_reached", messages, steps)

    # --- helpers ----------------------------------------------------------------------------

    def _budget(self, max_runtime_seconds: float | None) -> _Budget:
        runtime = self.config.max_runtime_seconds
        if max_runtime_seconds is not None:
            if max_runtime_seconds < 0:
                raise ValueError("max_runtime_seconds must not be negative")
            runtime = min(runtime, max_runtime_seconds)
        return _Budget(self.config.max_iterations, runtime, self.config.max_input_tokens_per_step)

    def _bound_model(self) -> Any:
        if not self.tools:
            return self.model
        return self.model.bind_tools(self.tools, tool_choice=self.config.tool_choice)

    def _model_name(self) -> str | None:
        params = getattr(self.model, "_identifying_params", None) or {}
        return params.get("model")

    def _record(self, step_number: int, answer: AIMessage) -> AgentStep:
        return AgentStep(
            step=step_number,
            assistant_content=_text(answer.content),
            tool_calls=list(answer.tool_calls or []),
            model_used=self._model_name(),
        )

    def _reflection_model(self) -> Any:
        limit = getattr(self.model, "max_tokens", None)
        if limit is not None and limit < REFLECTION_MIN_OUTPUT_TOKENS:
            return self.model.bind(max_tokens=REFLECTION_MIN_OUTPUT_TOKENS)
        return self.model

    def _reflect(self, messages: list[BaseMessage]) -> tuple[bool, str]:
        checker = self._reflection_model()
        for _ in range(REFLECTION_ATTEMPTS):
            verdict = _verdict(checker.invoke(_reflection_request(messages)))
            if verdict is not None:
                return verdict
        raise ValueError(f"The reflection check did not return a verdict in {REFLECTION_ATTEMPTS} attempts.")

    async def _areflect(self, messages: list[BaseMessage]) -> tuple[bool, str]:
        checker = self._reflection_model()
        for _ in range(REFLECTION_ATTEMPTS):
            verdict = _verdict(await checker.ainvoke(_reflection_request(messages)))
            if verdict is not None:
                return verdict
        raise ValueError(f"The reflection check did not return a verdict in {REFLECTION_ATTEMPTS} attempts.")

    @staticmethod
    def _completed(status: str, answer: AIMessage, messages: list[BaseMessage], steps: list[AgentStep]) -> dict[str, Any]:
        return {"status": status, "output": answer.content, "messages": messages, "steps": steps}

    @staticmethod
    def _stopped(reason: str, messages: list[BaseMessage], steps: list[AgentStep], **extra: Any) -> dict[str, Any]:
        return {"status": "stopped", "reason": reason, "messages": messages, "steps": steps, **extra}
