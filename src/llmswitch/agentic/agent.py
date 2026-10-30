"""
llmswitch.agentic.agent

The agent loop. Each step asks the model for an answer. A final answer (no tool calls) goes
through an optional reflection check, which can send it back for a revision. Tool calls are run
and their results are added to the conversation before the next step.

The loop stops with a status in the result:

- ``completed``: the answer passed reflection (or reflection is off).
- ``completed_with_reflection_limit``: the answer was still rejected after ``reflection_max_retries``
  revisions, so the last answer is returned.
- ``interrupted`` with a ``reason`` and a ``checkpoint_id``: ``tool_error`` (a tool failed and
  ``interrupt_on_tool_error`` is set) or ``uncertain_tool_execution`` (a tool may have started before
  the run was paused, and it is not safe to run it again). :meth:`Agent.resume` continues from there.
- ``stopped`` with a ``reason``: ``max_iterations_reached``, ``max_runtime_exceeded``,
  ``max_input_tokens_per_step_exceeded``, ``tool_retry_exhausted`` (a failure remained after the retries),
  ``tool_failed`` (a failure with no retries configured) or ``reflection_verdict_unreadable``.

Every model call is a step record, so the caller can see what happened on each turn. The agent writes a
checkpoint at each phase of a step (before reflection, before tools, after tools, after a rejected
answer). A checkpoint records each tool's progress as it runs, so a resumed run never repeats a
finished tool, and never repeats an unfinished non-idempotent tool.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, messages_from_dict, messages_to_dict
except ImportError as exc:  # pragma: no cover - exercised by tests that hide the optional dependency
    raise ImportError(
        "The agentic features need the agentic extra. Install it with: "
        'pip install "llmswitch-sdk[agentic]"'
    ) from exc

from ..ratelimit.tokenizer import count_tokens
from .chat import Tool, _as_messages, _tool_map
from .checkpoints import SCHEMA_VERSION, CheckpointStore, InMemoryCheckpointStore
from .events import AgentEventHandler, notify
from .execution import (
    Isolation,
    ToolExecutionResult,
    aexecute_tool_calls,
    aretry_failed_tool_calls,
    execute_tool_calls,
    is_idempotent,
    process_isolation_available,
    retry_failed_tool_calls,
)
from .routing import BaseRoutingPolicy

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

_PHASES = ("awaiting_reflection", "awaiting_tools", "tools_done", "reflection_feedback")


@dataclass(frozen=True)
class AgentConfig:
    """Limits and behaviour of an ``Agent`` run. Every field has a default.

    Budgets:
        max_iterations (int): The most model calls one run may make to answer or to request tools. Reflection checks
            are not counted. Defaults to 12.
        max_runtime_seconds (float): The wall-clock limit for one run. Defaults to 180.
        max_input_tokens_per_step (int, optional): Stops a run before a call whose input would be larger, counted by
            the library's own estimate. None means no limit.

    Reflection:
        reflection_enabled (bool): Check each answer with a second model call that returns a JSON verdict. Defaults to
            True.
        reflection_max_retries (int): How many revisions a rejected answer may get before the last answer is returned.
            Defaults to 2.

    Tools:
        tool_choice (str, optional): The ``tool_choice`` sent with each step. Defaults to ``"auto"``.
        tool_timeout_seconds (float): The most time one tool call may take. Defaults to 30.
        tool_isolation (str): ``"thread"`` (default) runs a sync tool on a thread. ``"process"`` runs it in a forked
            child process that is killed at the timeout, so its work stops. POSIX only.
        parallel_tools (bool): Run the tool calls of one step at the same time. Defaults to True.
        max_concurrency (int): The most tool calls running at once. Defaults to 4.
        tool_retry_attempts (int): Extra rounds for failures that are safe to retry. Defaults to 0.
        tool_retry_backoff_seconds (float): The pause between retry rounds. Defaults to 0.
        retry_on_tool_not_found (bool): Retry a call to a tool name that does not exist. Defaults to False.
        retry_non_idempotent_tools (bool): Retry a failed tool that is not marked ``idempotent``. Defaults to False.
        interrupt_on_tool_error (bool): Pause the run after a failed tool, so it can be resumed. Defaults to False.
        escalate_on_retry_exhausted (bool): Stop the run when a tool failure remains after its retries. Defaults to
            False.

    Raises:
        ValueError: A value is out of range, for example a negative budget or an unknown ``tool_isolation``.
    """

    max_iterations: int = 12
    max_runtime_seconds: float = 180.0
    tool_choice: str | None = "auto"
    max_input_tokens_per_step: int | None = None
    reflection_enabled: bool = True
    reflection_max_retries: int = 2
    tool_timeout_seconds: float = 30.0
    parallel_tools: bool = True
    max_concurrency: int = 4
    tool_isolation: Isolation = "thread"
    tool_retry_attempts: int = 0
    tool_retry_backoff_seconds: float = 0.0
    retry_on_tool_not_found: bool = False
    retry_non_idempotent_tools: bool = False
    interrupt_on_tool_error: bool = False
    escalate_on_retry_exhausted: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if self.max_input_tokens_per_step is not None and self.max_input_tokens_per_step < 1:
            raise ValueError("max_input_tokens_per_step must be at least 1 when set")
        if self.reflection_max_retries < 0:
            raise ValueError("reflection_max_retries must not be negative")
        if self.tool_timeout_seconds <= 0:
            raise ValueError("tool_timeout_seconds must be positive")
        if self.tool_isolation not in ("thread", "process"):
            raise ValueError("tool_isolation must be 'thread' or 'process'")
        if self.tool_isolation == "process" and not process_isolation_available():
            raise ValueError("tool_isolation='process' needs a platform with fork (POSIX); use 'thread' here")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.tool_retry_attempts < 0:
            raise ValueError("tool_retry_attempts must not be negative")
        if self.tool_retry_backoff_seconds < 0:
            raise ValueError("tool_retry_backoff_seconds must not be negative")


@dataclass
class AgentStep:
    """One model call in a run, and the tool results it caused.

    Attributes:
        step (int): The step number, counted from 1.
        assistant_content (str): The text of the model's answer for this step.
        tool_calls (list): The tool calls the model asked for in this step.
        model_used (str, optional): The model that answered this step. Shows which model a routing policy chose.
        tool_results (list[ToolExecutionResult]): The result of each tool call, in order.
    """

    step: int
    assistant_content: str
    tool_calls: list[Any] = field(default_factory=list)
    model_used: str | None = None
    tool_results: list[ToolExecutionResult] = field(default_factory=list)


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


@dataclass
class _Pending:
    """A step that has started but not finished: its kind, its record, and tool progress to continue from."""

    kind: Literal["reflect", "tools"]
    record: AgentStep
    prior: list[ToolExecutionResult | None] | None = None
    journal: _Journal | None = None
    llm: Any = None  # the model chosen for this step; None when the step is resumed and must be routed again


class _Journal:
    """Writes each tool's progress into its checkpoint, so a resumed run knows what already happened.

    Entries are ``not_started``, ``started`` (the tool was about to run) or ``completed`` (with its result).
    Calls can come from several threads, so writes are serialised.
    """

    def __init__(self, store: CheckpointStore, checkpoint_id: str, state: dict[str, Any], count: int) -> None:
        self._store = store
        self._checkpoint_id = checkpoint_id
        self._state = state
        self._lock = threading.Lock()
        if not isinstance(state.get("progress"), list) or len(state["progress"]) != count:
            state["progress"] = [{"status": "not_started"} for _ in range(count)]

    @property
    def checkpoint_id(self) -> str:
        return self._checkpoint_id

    def start(self, index: int) -> None:
        self._write(index, {"status": "started"})

    def result(self, index: int, result: ToolExecutionResult) -> None:
        self._write(index, {"status": "completed", "result": asdict(result)})

    def _write(self, index: int, entry: dict[str, Any]) -> None:
        with self._lock:
            self._state["progress"][index] = entry
            self._store.save(self._checkpoint_id, self._state)


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
    from langchain_core.messages import ToolMessage

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


def _step_to_dict(step: AgentStep) -> dict[str, Any]:
    return asdict(step)


def _step_from_dict(data: dict[str, Any]) -> AgentStep:
    return AgentStep(
        step=int(data["step"]),
        assistant_content=str(data.get("assistant_content", "")),
        tool_calls=list(data.get("tool_calls") or []),
        model_used=data.get("model_used"),
        tool_results=[ToolExecutionResult(**item) for item in data.get("tool_results") or []],
    )


def _result_from_entry(entry: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(**entry["result"])


class Agent:
    """A loop of model calls and tool calls, with budgets, reflection, checkpoints and routing.

    Each step calls the model. If the model asks for tools, they run and their results go back to the model. If it
    answers, the answer is checked by reflection, and a rejected answer is revised. A run ends with a result
    dictionary whose ``status`` says how it ended. A run that is ``interrupted`` can be continued with ``resume``.

    Example:
        agent = Agent(model=llm, tools=[get_weather], config=AgentConfig(max_iterations=6))
        result = agent.invoke("Weather in Paris, then a short summary.")
        if result["status"] == "interrupted":
            result = agent.resume(result["checkpoint_id"])
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[Tool] | None = None,
        config: AgentConfig | None = None,
        event_handler: AgentEventHandler | None = None,
        checkpoint_store: CheckpointStore | None = None,
        fallback_model: BaseChatModel | None = None,
        routing_policy: BaseRoutingPolicy | None = None,
        router: Callable[[list[BaseMessage], int], BaseChatModel] | None = None,
    ) -> None:
        """Create an agent around a chat model.

        Args:
            model (ChatLLMSwitch): The chat model that makes the decisions. It must support ``bind_tools``.
            tools (list, optional): Tool objects or plain functions with docstrings. Defaults to no tools.
            config (AgentConfig, optional): Limits and behaviour. Defaults to ``AgentConfig()``.
            event_handler (AgentEventHandler, optional): Receives the events of each run.
            checkpoint_store (CheckpointStore, optional): Where checkpoints are kept. Defaults to an in-memory store.
                Use ``FileCheckpointStore`` to resume a run in another process.
            fallback_model (ChatLLMSwitch, optional): A second model for routing. Used by the routing policy or by
                ``router``.
            routing_policy (BaseRoutingPolicy, optional): Chooses between ``model`` and ``fallback_model`` for each
                step.
            router (callable, optional): A function ``router(messages, step)`` that returns the model for a step. It
                takes
                precedence over ``routing_policy``.
        """
        self.model = model
        self.fallback_model = fallback_model
        self.routing_policy = routing_policy
        self.router = router
        self.tools = list(tools or [])
        self.config = config or AgentConfig()
        self.event_handler = event_handler
        self.checkpoint_store: CheckpointStore = checkpoint_store or InMemoryCheckpointStore()
        self._tool_map = _tool_map(self.tools) if self.tools else {}
        # Routing state, such as a round-robin position, carries over from one run to the next.
        self._routing_state: dict[str, Any] = {}

    # --- public API -------------------------------------------------------------------------

    def invoke(self, input: str | list[BaseMessage], *, max_runtime_seconds: float | None = None) -> dict[str, Any]:
        """Run the agent on a task and return the result when it ends.

        Args:
            input (str or list): The task as text, or the conversation as LangChain messages.
            max_runtime_seconds (float, optional): A shorter wall-clock limit for this run. It cannot lengthen the limit
                in
                the config.

        Returns:
            dict: The result. ``status`` is one of ``completed``, ``completed_with_reflection_limit``, ``interrupted``
                or
            ``stopped``. ``output`` holds the final answer when there is one. ``reason`` says why a run was interrupted
                or
            stopped, and ``checkpoint_id`` is set for an interrupted run. ``messages`` and ``steps`` give the full
                history.
        """
        routing = dict(self._routing_state)
        result = self._drive(
            messages=_as_messages(input), steps=[], step=1, budget=self._budget(max_runtime_seconds),
            reflection_retries=0, had_tool_error=False, pending=None, routing=routing,
        )
        self._routing_state.update(routing)
        return result

    async def ainvoke(self, input: str | list[BaseMessage], *, max_runtime_seconds: float | None = None) -> dict[str, Any]:
        """Async :meth:`invoke`. Tool timeouts cancel a tool that is a coroutine.

        Args:
            input (str or list): The task as text, or the conversation as LangChain messages.
            max_runtime_seconds (float, optional): A shorter wall-clock limit for this run.

        Returns:
            An awaitable that gives the result dictionary described in :meth:`invoke`.
        """
        routing = dict(self._routing_state)
        result = await self._adrive(
            messages=_as_messages(input), steps=[], step=1, budget=self._budget(max_runtime_seconds),
            reflection_retries=0, had_tool_error=False, pending=None, routing=routing,
        )
        self._routing_state.update(routing)
        return result

    def resume(self, checkpoint_id: str, user_feedback: str | None = None) -> dict[str, Any]:
        """Continue an interrupted run from its checkpoint.

        Args:
            checkpoint_id (str): The ``checkpoint_id`` from the interrupted result.
            user_feedback (str, optional): Text added to the conversation. When the checkpoint is at a model answer, or
                at
                tools that have not started, the feedback replaces that answer and the step runs again.

        Returns:
            dict: The result of the continued run, in the same form as :meth:`invoke`. A checkpoint whose tools may
                already
            have run is returned as ``interrupted`` with reason ``uncertain_tool_execution``, and is not continued.
        """
        plan = self._restore(checkpoint_id, user_feedback)
        if "status" in plan:  # already a final result, such as an uncertain tool execution
            return plan
        result = self._drive(**plan)
        return result

    async def aresume(self, checkpoint_id: str, user_feedback: str | None = None) -> dict[str, Any]:
        """Async :meth:`resume`.

        Args:
            checkpoint_id (str): The ``checkpoint_id`` from the interrupted result.
            user_feedback (str, optional): Text added to the conversation, as in :meth:`resume`.

        Returns:
            An awaitable that gives the result dictionary described in :meth:`invoke`.
        """
        plan = self._restore(checkpoint_id, user_feedback)
        if "status" in plan:
            return plan
        return await self._adrive(**plan)

    # --- the loop ---------------------------------------------------------------------------

    def _drive(self, *, messages: list[BaseMessage], steps: list[AgentStep], step: int, budget: _Budget,
               reflection_retries: int, had_tool_error: bool, pending: _Pending | None,
               routing: dict[str, Any]) -> dict[str, Any]:
        while step <= budget.max_iterations:
            if pending is None:
                if budget.runtime_exceeded():
                    return self._stopped("max_runtime_exceeded", messages, steps)
                llm = self._choose(messages, step, had_tool_error, routing)
                tokens = estimate_input_tokens(messages, self._model_name(llm))
                if budget.input_exceeded(tokens):
                    return self._stopped("max_input_tokens_per_step_exceeded", messages, steps,
                                         input_tokens=tokens, limit=budget.max_input_tokens_per_step)
                notify(self.event_handler, "on_step_start", step, {"messages": len(messages)})
                answer = self._bound_model(llm).invoke(messages)
                record = self._record(step, answer, llm)
                messages.append(answer)
                if budget.runtime_exceeded():
                    return self._stopped("max_runtime_exceeded", messages, [*steps, record])
                pending = _Pending("tools" if answer.tool_calls else "reflect", record, llm=llm)
            if pending.kind == "reflect":
                llm = pending.llm or self._choose(messages, step, had_tool_error, routing)
                if not self.config.reflection_enabled:
                    return self._completed("completed", messages, [*steps, pending.record], step)
                self._save(self._state_for(routing, "awaiting_reflection", messages, steps, step, budget,
                                       reflection_retries, had_tool_error, pending=pending.record))
                verdict = self._reflect(llm, messages)
                if verdict is None:
                    return self._stopped("reflection_verdict_unreadable", messages, [*steps, pending.record])
                ok, feedback = verdict
                if budget.runtime_exceeded():
                    return self._stopped("max_runtime_exceeded", messages, [*steps, pending.record])
                if ok:
                    return self._completed("completed", messages, [*steps, pending.record], step)
                if reflection_retries >= self.config.reflection_max_retries:
                    return self._completed("completed_with_reflection_limit", messages,
                                           [*steps, pending.record], step)
                reflection_retries += 1
                messages.append(HumanMessage(content=f"Please revise your answer. Feedback: {feedback}"))
                steps.append(pending.record)
                self._save(self._state_for(routing, "reflection_feedback", messages, steps, step, budget,
                                       reflection_retries, had_tool_error))
                notify(self.event_handler, "on_step_end", step, {"finished": False})
                step, pending = step + 1, None
                continue
            # tools: either a new step, or a step resumed from a checkpoint
            record = pending.record
            if pending.journal is None:
                state = self._state_for(routing, "awaiting_tools", messages, steps, step, budget, reflection_retries,
                                    had_tool_error, pending=record)
                checkpoint_id = str(uuid.uuid4())
                pending.journal = _Journal(self.checkpoint_store, checkpoint_id, state, len(record.tool_calls))
                self.checkpoint_store.save(checkpoint_id, state)
            record.tool_results = self._run_tools(step, record, pending.journal, pending.prior)
            messages.extend(result.as_tool_message() for result in record.tool_results)
            steps.append(record)
            had_tool_error = any(not result.ok for result in record.tool_results)
            done = self._save(self._state_for(routing, "tools_done", messages, steps, step, budget, reflection_retries,
                                          had_tool_error))
            outcome = self._after_tools(record.tool_results, messages, steps, done)
            if outcome is not None:
                return outcome
            notify(self.event_handler, "on_step_end", step, {"finished": False})
            step, pending = step + 1, None
        return self._stopped("max_iterations_reached", messages, steps)

    async def _adrive(self, *, messages: list[BaseMessage], steps: list[AgentStep], step: int, budget: _Budget,
                      reflection_retries: int, had_tool_error: bool, pending: _Pending | None,
               routing: dict[str, Any]) -> dict[str, Any]:
        while step <= budget.max_iterations:
            if pending is None:
                if budget.runtime_exceeded():
                    return self._stopped("max_runtime_exceeded", messages, steps)
                llm = self._choose(messages, step, had_tool_error, routing)
                tokens = estimate_input_tokens(messages, self._model_name(llm))
                if budget.input_exceeded(tokens):
                    return self._stopped("max_input_tokens_per_step_exceeded", messages, steps,
                                         input_tokens=tokens, limit=budget.max_input_tokens_per_step)
                notify(self.event_handler, "on_step_start", step, {"messages": len(messages)})
                answer = await self._bound_model(llm).ainvoke(messages)
                record = self._record(step, answer, llm)
                messages.append(answer)
                if budget.runtime_exceeded():
                    return self._stopped("max_runtime_exceeded", messages, [*steps, record])
                pending = _Pending("tools" if answer.tool_calls else "reflect", record, llm=llm)
            if pending.kind == "reflect":
                llm = pending.llm or self._choose(messages, step, had_tool_error, routing)
                if not self.config.reflection_enabled:
                    return self._completed("completed", messages, [*steps, pending.record], step)
                self._save(self._state_for(routing, "awaiting_reflection", messages, steps, step, budget,
                                       reflection_retries, had_tool_error, pending=pending.record))
                verdict = await self._areflect(llm, messages)
                if verdict is None:
                    return self._stopped("reflection_verdict_unreadable", messages, [*steps, pending.record])
                ok, feedback = verdict
                if budget.runtime_exceeded():
                    return self._stopped("max_runtime_exceeded", messages, [*steps, pending.record])
                if ok:
                    return self._completed("completed", messages, [*steps, pending.record], step)
                if reflection_retries >= self.config.reflection_max_retries:
                    return self._completed("completed_with_reflection_limit", messages,
                                           [*steps, pending.record], step)
                reflection_retries += 1
                messages.append(HumanMessage(content=f"Please revise your answer. Feedback: {feedback}"))
                steps.append(pending.record)
                self._save(self._state_for(routing, "reflection_feedback", messages, steps, step, budget,
                                       reflection_retries, had_tool_error))
                notify(self.event_handler, "on_step_end", step, {"finished": False})
                step, pending = step + 1, None
                continue
            record = pending.record
            if pending.journal is None:
                state = self._state_for(routing, "awaiting_tools", messages, steps, step, budget, reflection_retries,
                                    had_tool_error, pending=record)
                checkpoint_id = str(uuid.uuid4())
                pending.journal = _Journal(self.checkpoint_store, checkpoint_id, state, len(record.tool_calls))
                self.checkpoint_store.save(checkpoint_id, state)
            record.tool_results = await self._arun_tools(step, record, pending.journal, pending.prior)
            messages.extend(result.as_tool_message() for result in record.tool_results)
            steps.append(record)
            had_tool_error = any(not result.ok for result in record.tool_results)
            done = self._save(self._state_for(routing, "tools_done", messages, steps, step, budget, reflection_retries,
                                          had_tool_error))
            outcome = self._after_tools(record.tool_results, messages, steps, done)
            if outcome is not None:
                return outcome
            notify(self.event_handler, "on_step_end", step, {"finished": False})
            step, pending = step + 1, None
        return self._stopped("max_iterations_reached", messages, steps)

    # --- resuming -----------------------------------------------------------------------------

    def _restore(self, checkpoint_id: str, user_feedback: str | None) -> dict[str, Any]:
        """Rebuild the run from a checkpoint. Returns the arguments for ``_drive``, or a final result."""
        state = self.checkpoint_store.load(checkpoint_id)
        if state is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema version.")
        phase = state.get("phase")
        if phase not in _PHASES:
            raise ValueError(f"Unknown checkpoint phase: {phase}")
        messages = list(messages_from_dict(state["messages"]))
        steps = [_step_from_dict(item) for item in state.get("steps") or []]
        budget = self._budget(float(state["remaining_runtime_seconds"]))
        reflection_retries = int(state.get("reflection_retries", 0))
        had_tool_error = bool(state.get("had_tool_error", False))
        step = int(state["step"])
        pending: _Pending | None = None

        if phase == "awaiting_reflection":
            record = _step_from_dict(state["pending"])
            if user_feedback:
                self._replace_last_answer(messages, user_feedback)
            else:
                pending = _Pending("reflect", record)
        elif phase == "awaiting_tools":
            record = _step_from_dict(state["pending"])
            progress = state["progress"]
            if user_feedback:
                if any(entry.get("status") != "not_started" for entry in progress):
                    return self._interrupted("uncertain_tool_execution", messages, steps, checkpoint_id)
                self._replace_last_answer(messages, user_feedback)
            else:
                prior: list[ToolExecutionResult | None] = [None] * len(record.tool_calls)
                for index, entry in enumerate(progress):
                    status = entry.get("status")
                    if status == "completed":
                        prior[index] = _result_from_entry(entry)
                    elif status == "started" and not is_idempotent(self._tool_map.get(
                            record.tool_calls[index].get("name") or "")):
                        return self._interrupted("uncertain_tool_execution", messages, steps, checkpoint_id)
                journal = _Journal(self.checkpoint_store, checkpoint_id, state, len(record.tool_calls))
                pending = _Pending("tools", record, prior=prior, journal=journal)
        else:
            step += 1
            if user_feedback:
                messages.append(HumanMessage(content=user_feedback))
        return {"messages": messages, "steps": steps, "step": step, "budget": budget,
                "reflection_retries": reflection_retries, "had_tool_error": had_tool_error, "pending": pending,
                "routing": dict(state.get("routing") or {})}

    @staticmethod
    def _replace_last_answer(messages: list[BaseMessage], feedback: str) -> None:
        if not messages or not isinstance(messages[-1], AIMessage):
            raise ValueError("The checkpoint has no pending assistant answer to replace.")
        messages.pop()
        messages.append(HumanMessage(content=feedback))

    # --- helpers ----------------------------------------------------------------------------

    def _budget(self, max_runtime_seconds: float | None) -> _Budget:
        runtime = self.config.max_runtime_seconds
        if max_runtime_seconds is not None:
            if max_runtime_seconds < 0:
                raise ValueError("max_runtime_seconds must not be negative")
            runtime = min(runtime, max_runtime_seconds)
        return _Budget(self.config.max_iterations, runtime, self.config.max_input_tokens_per_step)

    def _choose(self, messages: list[BaseMessage], step: int, had_tool_error: bool,
                routing: dict[str, Any]) -> BaseChatModel:
        """The model for a step: a custom router, a routing policy, or the escalation default."""
        if self.router is not None:
            return self.router(messages, step)
        if self.routing_policy is not None:
            decision = self.routing_policy.select_llm(
                messages=messages, step=step, had_tool_error=had_tool_error,
                primary_llm=self.model, fallback_llm=self.fallback_model, state=routing,
            )
            notify(self.event_handler, "on_routing_decision", step, {
                "policy": decision.policy_name, "reason": decision.reason,
                "metadata": decision.metadata, "model": self._model_name(decision.llm),
            })
            return decision.llm
        if had_tool_error and self.fallback_model is not None:
            return self.fallback_model
        return self.model

    def _bound_model(self, llm: BaseChatModel) -> Any:
        if not self.tools:
            return llm
        return llm.bind_tools(self.tools, tool_choice=self.config.tool_choice)

    @staticmethod
    def _model_name(llm: BaseChatModel) -> str | None:
        params = getattr(llm, "_identifying_params", None) or {}
        return params.get("model")

    def _record(self, step_number: int, answer: AIMessage, llm: BaseChatModel) -> AgentStep:
        return AgentStep(
            step=step_number,
            assistant_content=_text(answer.content),
            tool_calls=list(answer.tool_calls or []),
            model_used=self._model_name(llm),
        )

    def _state_for(self, routing: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """A checkpoint that also records the routing state, so a resumed run keeps the same rotation."""
        return {**self._state(*args, **kwargs), "routing": dict(routing)}

    def _state(self, phase: str, messages: list[BaseMessage], steps: list[AgentStep], step: int,
               budget: _Budget, reflection_retries: int, had_tool_error: bool,
               pending: AgentStep | None = None) -> dict[str, Any]:
        """A checkpoint in JSON-compatible form."""
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "step": step,
            "messages": messages_to_dict(messages),
            "steps": [_step_to_dict(item) for item in steps],
            "pending": _step_to_dict(pending) if pending is not None else None,
            "progress": None,
            "remaining_runtime_seconds": budget.remaining_seconds(),
            "reflection_retries": reflection_retries,
            "had_tool_error": had_tool_error,
        }

    def _save(self, state: dict[str, Any]) -> str:
        checkpoint_id = str(uuid.uuid4())
        self.checkpoint_store.save(checkpoint_id, state)
        return checkpoint_id

    @staticmethod
    def _reflection_model(llm: BaseChatModel) -> Any:
        limit = getattr(llm, "max_tokens", None)
        if limit is not None and limit < REFLECTION_MIN_OUTPUT_TOKENS:
            return llm.bind(max_tokens=REFLECTION_MIN_OUTPUT_TOKENS)
        return llm

    def _reflect(self, llm: BaseChatModel, messages: list[BaseMessage]) -> tuple[bool, str] | None:
        """The verdict, or ``None`` when the check gave no readable verdict in ``REFLECTION_ATTEMPTS`` tries."""
        checker = self._reflection_model(llm)
        for _ in range(REFLECTION_ATTEMPTS):
            verdict = _verdict(checker.invoke(_reflection_request(messages)))
            if verdict is not None:
                return verdict
        return None

    async def _areflect(self, llm: BaseChatModel, messages: list[BaseMessage]) -> tuple[bool, str] | None:
        """Async :meth:`_reflect`."""
        checker = self._reflection_model(llm)
        for _ in range(REFLECTION_ATTEMPTS):
            verdict = _verdict(await checker.ainvoke(_reflection_request(messages)))
            if verdict is not None:
                return verdict
        return None

    def _run_tools(self, step_number: int, record: AgentStep, journal: _Journal,
                   prior: list[ToolExecutionResult | None] | None) -> list[ToolExecutionResult]:
        """Run the tools that have not finished yet, then retry the retryable failures among them."""
        cfg = self.config
        calls = list(record.tool_calls)
        indices = [i for i in range(len(calls)) if prior is None or prior[i] is None]
        results: list[ToolExecutionResult | None] = list(prior) if prior is not None else [None] * len(calls)
        if indices:
            subset = [calls[i] for i in indices]
            fresh = execute_tool_calls(
                step=step_number, tool_calls=subset, tool_map=self._tool_map, timeout_s=cfg.tool_timeout_seconds, isolation=cfg.tool_isolation,
                parallel=cfg.parallel_tools, max_concurrency=cfg.max_concurrency, handler=self.event_handler,
                on_start=lambda local: journal.start(indices[local]),
                on_result=lambda local, result: journal.result(indices[local], result),
            )
            if cfg.tool_retry_attempts and any(not result.ok for result in fresh):
                fresh = retry_failed_tool_calls(
                    step=step_number, tool_calls=subset, results=fresh, tool_map=self._tool_map,
                    timeout_s=cfg.tool_timeout_seconds, isolation=cfg.tool_isolation, attempts=cfg.tool_retry_attempts,
                    backoff_seconds=cfg.tool_retry_backoff_seconds,
                    retry_on_tool_not_found=cfg.retry_on_tool_not_found,
                    retry_non_idempotent=cfg.retry_non_idempotent_tools,
                    parallel=cfg.parallel_tools, max_concurrency=cfg.max_concurrency, handler=self.event_handler,
                    on_start=lambda local: journal.start(indices[local]),
                    on_result=lambda local, result: journal.result(indices[local], result),
                )
            for index, result in zip(indices, fresh, strict=True):
                results[index] = result
        return [result for result in results if result is not None]

    async def _arun_tools(self, step_number: int, record: AgentStep, journal: _Journal,
                          prior: list[ToolExecutionResult | None] | None) -> list[ToolExecutionResult]:
        """Async :meth:`_run_tools`."""
        cfg = self.config
        calls = list(record.tool_calls)
        indices = [i for i in range(len(calls)) if prior is None or prior[i] is None]
        results: list[ToolExecutionResult | None] = list(prior) if prior is not None else [None] * len(calls)
        if indices:
            subset = [calls[i] for i in indices]
            fresh = await aexecute_tool_calls(
                step=step_number, tool_calls=subset, tool_map=self._tool_map, timeout_s=cfg.tool_timeout_seconds, isolation=cfg.tool_isolation,
                parallel=cfg.parallel_tools, max_concurrency=cfg.max_concurrency, handler=self.event_handler,
                on_start=lambda local: journal.start(indices[local]),
                on_result=lambda local, result: journal.result(indices[local], result),
            )
            if cfg.tool_retry_attempts and any(not result.ok for result in fresh):
                fresh = await aretry_failed_tool_calls(
                    step=step_number, tool_calls=subset, results=fresh, tool_map=self._tool_map,
                    timeout_s=cfg.tool_timeout_seconds, isolation=cfg.tool_isolation, attempts=cfg.tool_retry_attempts,
                    backoff_seconds=cfg.tool_retry_backoff_seconds,
                    retry_on_tool_not_found=cfg.retry_on_tool_not_found,
                    retry_non_idempotent=cfg.retry_non_idempotent_tools,
                    parallel=cfg.parallel_tools, max_concurrency=cfg.max_concurrency, handler=self.event_handler,
                    on_start=lambda local: journal.start(indices[local]),
                    on_result=lambda local, result: journal.result(indices[local], result),
                )
            for index, result in zip(indices, fresh, strict=True):
                results[index] = result
        return [result for result in results if result is not None]

    def _after_tools(self, results: list[ToolExecutionResult], messages: list[BaseMessage],
                     steps: list[AgentStep], checkpoint_id: str) -> dict[str, Any] | None:
        """Decide whether a step with failed tools ends the run. Returns the result, or ``None`` to go on."""
        failed = [result for result in results if not result.ok]
        if not failed:
            return None
        if self.config.escalate_on_retry_exhausted:
            # "Exhausted" only when a failure was actually retried; a failure that was never retried is "failed".
            retried = any(result.attempts > 1 for result in failed)
            reason = "tool_retry_exhausted" if retried else "tool_failed"
            return {
                "status": "stopped", "reason": reason,
                "failed_tools": [{"tool_name": r.tool_name, "error": r.error, "attempts": r.attempts} for r in failed],
                "messages": messages, "steps": steps,
            }
        if self.config.interrupt_on_tool_error:
            return self._interrupted("tool_error", messages, steps, checkpoint_id)
        return None

    def _interrupted(self, reason: str, messages: list[BaseMessage], steps: list[AgentStep],
                     checkpoint_id: str) -> dict[str, Any]:
        return {"status": "interrupted", "reason": reason, "checkpoint_id": checkpoint_id,
                "messages": messages, "steps": steps}

    def _completed(self, status: str, messages: list[BaseMessage], steps: list[AgentStep], step: int) -> dict[str, Any]:
        notify(self.event_handler, "on_step_end", step, {"finished": True})
        answer = messages[-1]
        return {"status": status, "output": answer.content, "messages": messages, "steps": steps}

    @staticmethod
    def _stopped(reason: str, messages: list[BaseMessage], steps: list[AgentStep], **extra: Any) -> dict[str, Any]:
        return {"status": "stopped", "reason": reason, "messages": messages, "steps": steps, **extra}
