"""
core.events

Observability hook. A client built with ``on_event=callback`` calls the callback
with a dict for each of these events:

* ``request``  - a call is about to be sent
* ``response`` - the call returned (for a stream, once it is established)
* ``error``    - the call failed
* ``retry``    - a transient error is about to be retried
* ``failover`` - the primary provider failed and the backup is being used

Every event carries ``event``, ``operation``, ``provider`` and ``model``. The
callback runs on the calling thread; exceptions it raises are logged and ignored.
"""
from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]

_active: ContextVar[Optional[tuple[EventCallback, dict[str, Any]]]] = ContextVar("llmswitch_events", default=None)


def _deliver(callback: EventCallback, meta: dict[str, Any], event: str, fields: dict[str, Any]) -> None:
    try:
        callback({"event": event, **meta, **fields})
    except Exception:  # noqa: BLE001
        logger.exception("on_event callback raised for event '%s'", event)


def emit(event: str, **fields: Any) -> None:
    """Send an event to the callback of the call in progress, if any."""
    active = _active.get()
    if active is not None:
        _deliver(active[0], active[1], event, fields)


def current_emitter() -> Callable[..., None]:
    """Capture the callback in effect now, for code that runs after the call returns (such as a stream)."""
    active = _active.get()
    if active is None:
        return lambda event, **fields: None
    callback, meta = active
    return lambda event, **fields: _deliver(callback, meta, event, fields)


@contextmanager
def event_scope(callback: Optional[EventCallback], **meta: Any):
    """Make ``callback`` the active event sink while the block runs."""
    if callback is None:
        yield
        return
    token = _active.set((callback, meta))
    try:
        yield
    finally:
        _active.reset(token)


def _usage_fields(result: Any) -> dict[str, Any]:
    usage = getattr(result, "usage", None)
    if usage is None:
        return {}
    counts = {name: getattr(usage, name) for name in
              ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
              if isinstance(getattr(usage, name, None), int)}
    return {"usage": counts} if counts else {}


def _meta(owner: Any, operation: str, extra: dict[str, Any]) -> dict[str, Any]:
    config = owner.config
    model = getattr(config, "deployment_name", None) or getattr(config, "model", None)
    return {"operation": operation, "provider": config.provider, "model": model, **extra}


def traced(operation: str, detail: Optional[Callable[[tuple, dict], dict[str, Any]]] = None):
    """Decorate a client method so it reports request, response and error events.

    ``detail(args, kwargs)`` may return extra fields to add to every event of the call.
    """
    def decorator(method):
        def prepare(owner, args, kwargs):
            callback = getattr(owner, "_on_event", None)
            extra = detail(args, kwargs) if (callback is not None and detail is not None) else {}
            return callback, _meta(owner, operation, extra) if callback is not None else {}

        if inspect.iscoroutinefunction(method):
            @functools.wraps(method)
            async def async_wrapper(self, *args, **kwargs):
                callback, meta = prepare(self, args, kwargs)
                if callback is None:
                    return await method(self, *args, **kwargs)
                with event_scope(callback, **meta):
                    emit("request", stream=bool(kwargs.get("stream", False)))
                    started = time.monotonic()
                    try:
                        result = await method(self, *args, **kwargs)
                    except Exception as exc:
                        emit("error", error=type(exc).__name__, status_code=getattr(exc, "status_code", None),
                             latency=round(time.monotonic() - started, 4))
                        raise
                    emit("response", stream=bool(kwargs.get("stream", False)),
                         latency=round(time.monotonic() - started, 4), **_usage_fields(result))
                    return result
            return async_wrapper

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            callback, meta = prepare(self, args, kwargs)
            if callback is None:
                return method(self, *args, **kwargs)
            with event_scope(callback, **meta):
                emit("request", stream=bool(kwargs.get("stream", False)))
                started = time.monotonic()
                try:
                    result = method(self, *args, **kwargs)
                except Exception as exc:
                    emit("error", error=type(exc).__name__, status_code=getattr(exc, "status_code", None),
                         latency=round(time.monotonic() - started, 4))
                    raise
                emit("response", stream=bool(kwargs.get("stream", False)),
                     latency=round(time.monotonic() - started, 4), **_usage_fields(result))
                return result
        return wrapper
    return decorator
