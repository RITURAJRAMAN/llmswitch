"""
llmswitch.agentic.events

Hooks that the agent calls as a run progresses: steps start and end, and each tool starts,
finishes or fails. Implement :class:`AgentEventHandler` (any object with these methods works),
or use one of the two handlers here.

- :class:`CollectingEventHandler` keeps every event in memory, for tests and debugging.
- :class:`LoggingEventHandler` logs one line per event. It logs only the names and types of
  arguments and results, never their values, so credentials and user content stay out of logs.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.redaction import redact_sensitive

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentEventHandler(Protocol):
    """Callback interface for agent events. Every method is called on the thread that runs the agent."""

    def on_step_start(self, step: int, context: dict[str, Any]) -> None: ...

    def on_tool_start(self, step: int, tool_name: str, args: dict[str, Any]) -> None: ...

    def on_tool_end(self, step: int, tool_name: str, result: Any, duration_s: float) -> None: ...

    def on_tool_error(self, step: int, tool_name: str, error: str, duration_s: float) -> None: ...

    def on_step_end(self, step: int, context: dict[str, Any]) -> None: ...


@dataclass
class AgentEvent:
    """One recorded event: its name, the step it belongs to, and its payload."""

    event: str
    step: int
    payload: dict[str, Any]
    ts: float = field(default_factory=time.time)


class CollectingEventHandler:
    """Keeps every event in ``events`` in the order they happened."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def on_step_start(self, step: int, context: dict[str, Any]) -> None:
        self.events.append(AgentEvent("step_start", step, context))

    def on_tool_start(self, step: int, tool_name: str, args: dict[str, Any]) -> None:
        self.events.append(AgentEvent("tool_start", step, {"tool_name": tool_name, "args": args}))

    def on_tool_end(self, step: int, tool_name: str, result: Any, duration_s: float) -> None:
        self.events.append(AgentEvent("tool_end", step, {"tool_name": tool_name, "duration_s": duration_s,
                                                         "result": result}))

    def on_tool_error(self, step: int, tool_name: str, error: str, duration_s: float) -> None:
        self.events.append(AgentEvent("tool_error", step, {"tool_name": tool_name, "duration_s": duration_s,
                                                           "error": error}))

    def on_step_end(self, step: int, context: dict[str, Any]) -> None:
        self.events.append(AgentEvent("step_end", step, context))


def notify(handler: AgentEventHandler | None, name: str, *args: Any) -> None:
    """Call one handler method. A handler that raises is logged, and the run continues."""
    method = getattr(handler, name, None)
    if method is None:
        return
    try:
        method(*args)
    except Exception:  # noqa: BLE001 - observing a run must never break it
        logger.exception("Agent event handler %s raised", name)


def _shape(value: Any) -> dict[str, Any]:
    """A summary of a value that holds no content: key names for mappings, type and size otherwise."""
    redacted = redact_sensitive(value)
    if isinstance(redacted, dict):
        return {"keys": sorted(str(key) for key in redacted)}
    if isinstance(redacted, (list, tuple)):
        return {"type": type(redacted).__name__, "length": len(redacted)}
    return {"type": type(redacted).__name__}


class LoggingEventHandler:
    """Writes one log line per event, with names and types only."""

    def __init__(self, logger_name: str = "llmswitch.agentic") -> None:
        self.logger = logging.getLogger(logger_name)

    def on_step_start(self, step: int, context: dict[str, Any]) -> None:
        self.logger.info("step_start | step=%s | context=%s", step, _shape(context))

    def on_tool_start(self, step: int, tool_name: str, args: dict[str, Any]) -> None:
        self.logger.info("tool_start | step=%s | tool=%s | args=%s", step, tool_name, _shape(args))

    def on_tool_end(self, step: int, tool_name: str, result: Any, duration_s: float) -> None:
        self.logger.info("tool_end | step=%s | tool=%s | duration_s=%.3f | result=%s",
                         step, tool_name, duration_s, _shape(result))

    def on_tool_error(self, step: int, tool_name: str, error: str, duration_s: float) -> None:
        # Errors start with the exception's class name, so only that prefix is logged, not the message.
        self.logger.warning("tool_error | step=%s | tool=%s | duration_s=%.3f | error_type=%s",
                            step, tool_name, duration_s, error.split(":", 1)[0][:80])

    def on_step_end(self, step: int, context: dict[str, Any]) -> None:
        self.logger.info("step_end | step=%s | context=%s", step, _shape(context))
