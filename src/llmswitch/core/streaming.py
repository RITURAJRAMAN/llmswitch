"""Lazy stream-session helpers with pre-output establishment semantics."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from .circuit_breaker import is_health_failure
from .events import current_emitter
from .execution import ExecutionOutcome, ExecutionStatus
from .retry import _is_retryable, _retry_delay, _within_deadline


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def chat_chunk_has_visible_output(chunk: Any) -> bool:
    for choice in _field(chunk, "choices", ()) or ():
        delta = _field(choice, "delta")
        if delta is None:
            continue
        if _field(delta, "content") not in (None, ""):
            return True
        if _field(delta, "refusal") not in (None, ""):
            return True
        if _field(delta, "thinking") not in (None, ""):
            return True
        if _field(delta, "thinking_signature") not in (None, ""):
            return True
        if _field(delta, "function_call") is not None:
            return True
        if _field(delta, "tool_calls"):
            return True
        if _field(delta, "citations"):
            return True
    return False


def response_event_has_visible_output(event: Any) -> bool:
    if _field(event, "text") not in (None, ""):
        return True
    delta = _field(event, "delta")
    if isinstance(delta, str):
        return delta != ""
    if delta not in (None, (), [], {}, ""):
        return True
    item = _field(event, "item")
    if item is not None:
        return _field(item, "type") is not None
    response = _field(event, "response")
    if response is not None and getattr(response, "output_text", None):
        return True
    return False


def response_event_terminal_status(event: Any) -> ExecutionStatus | None:
    event_type = _field(event, "type")
    if event_type == "response.completed":
        return ExecutionStatus.COMPLETED
    if event_type == "response.failed":
        return ExecutionStatus.FAILED_TERMINAL
    if event_type == "response.incomplete":
        return ExecutionStatus.INCOMPLETE
    return None


def _additional_stream_tokens(usage: Any, prompt_tokens: int | None, estimated_prompt_tokens: int) -> int:
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
    return max(0, total_tokens - (prompt_tokens if prompt_tokens is not None else estimated_prompt_tokens))


@dataclass
class StreamHandle:
    stream: Any
    breaker: Any = None
    limiter_scope: str | None = None
    iterator: Any = None
    prompt_tokens: int | None = None


class SyncStreamSession:
    def __init__(
        self,
        *,
        primary_factory: Callable[[], StreamHandle],
        backup_factory: Callable[[], StreamHandle] | None = None,
        is_output_visible: Callable[[Any], bool] | None = None,
        terminal_status: Callable[[Any], ExecutionStatus | None] | None = None,
        on_complete: Callable[..., Any] | None = None,
        max_retries: int = 0,
        base_delay: float = 0.0,
        max_delay: float = 60.0,
        deadline: float | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._primary_factory = primary_factory
        self._backup_factory = backup_factory
        self._is_output_visible = is_output_visible or (lambda _item: True)
        self._terminal_status_fn = terminal_status or (lambda _item: None)
        self._on_complete = on_complete
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._deadline = deadline
        self._emit = current_emitter()
        self._started = False
        self._finished = False
        self._closed = False
        self._active: StreamHandle | None = None
        self._buffer: deque[Any] = deque()
        self._output_visible = False
        self._last_usage = None
        self._terminal_status: ExecutionStatus | None = None
        self.outcome = ExecutionOutcome(status=ExecutionStatus.NOT_STARTED)

    def __iter__(self) -> "SyncStreamSession":
        return self

    def _close_handle(self, handle: StreamHandle | None) -> None:
        if handle is None:
            return
        close = getattr(handle.stream, "close", None)
        if callable(close):
            close()

    def _abandon_handle(self, handle: StreamHandle | None) -> None:
        if handle is not None and handle.breaker is not None:
            handle.breaker.abandon_probe()
        self._close_handle(handle)

    def _record_failure(self, handle: StreamHandle | None, exc: BaseException) -> None:
        if handle is not None and handle.breaker is not None:
            if isinstance(exc, Exception):
                handle.breaker.record(ok=not is_health_failure(exc))
            else:
                handle.breaker.abandon_probe()
        status = ExecutionStatus.FAILED_RETRYABLE if isinstance(exc, Exception) and _is_retryable(exc) else ExecutionStatus.FAILED_TERMINAL
        self.outcome = ExecutionOutcome(
            status=status,
            provider_dispatched=handle is not None,
            output_visible=self._output_visible,
            retryable=isinstance(exc, Exception) and _is_retryable(exc),
            error=exc if isinstance(exc, Exception) else None,
        )
        self._close_handle(handle)

    def _observe_item(self, item: Any) -> None:
        usage = getattr(item, "usage", None)
        if usage is not None:
            self._last_usage = usage
        terminal_status = self._terminal_status_fn(item)
        if terminal_status is not None:
            self._terminal_status = terminal_status
        if self._is_output_visible(item):
            self._output_visible = True
            self.outcome = ExecutionOutcome(
                status=ExecutionStatus.STREAMING,
                provider_dispatched=True,
                output_visible=True,
            )

    def _release_buffer(self) -> bool:
        return self._output_visible or self._terminal_status is not None

    def _finish_active(self) -> None:
        if self._finished:
            return
        status = self._terminal_status or ExecutionStatus.COMPLETED
        handle = self._active
        try:
            if handle is not None and handle.breaker is not None:
                handle.breaker.record(ok=status == ExecutionStatus.COMPLETED)
            if self._on_complete is not None:
                self._on_complete(
                    self._last_usage,
                    handle.limiter_scope if handle is not None else None,
                    handle.prompt_tokens if handle is not None else None,
                )
            self.outcome = ExecutionOutcome(
                status=status,
                provider_dispatched=handle is not None,
                output_visible=self._output_visible,
            )
        except BaseException as exc:
            self._buffer.clear()
            self.outcome = ExecutionOutcome(
                status=ExecutionStatus.FAILED_TERMINAL,
                provider_dispatched=handle is not None,
                output_visible=self._output_visible,
                error=exc if isinstance(exc, Exception) else None,
            )
            raise
        finally:
            try:
                self._close_handle(handle)
            finally:
                self._active = None
                self._finished = True

    def _set_active(self, handle: StreamHandle) -> None:
        self._active = handle
        handle.iterator = iter(handle.stream)
        self.outcome = ExecutionOutcome(status=ExecutionStatus.DISPATCHED, provider_dispatched=True)

    def _establish_with(self, factory: Callable[[], StreamHandle]) -> None:
        handle = factory()
        self._set_active(handle)
        self._buffer.clear()
        while True:
            try:
                item = next(handle.iterator)
            except StopIteration:
                self._finish_active()
                return
            self._observe_item(item)
            self._buffer.append(item)
            if self._release_buffer():
                return

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        primary_exc: Exception | None = None
        try:
            self._establish_with_retries(self._primary_factory)
            return
        except Exception as exc:
            if self._finished:
                raise
            primary_exc = exc
            if self._backup_factory is None or not _is_retryable(exc):
                raise
        self._emit("failover", error=type(primary_exc).__name__, status_code=getattr(primary_exc, "status_code", None))
        try:
            self._establish_with_retries(self._backup_factory)
        except Exception as backup_exc:
            if self._finished:
                raise
            raise primary_exc from backup_exc

    def _establish_with_retries(self, factory: Callable[[], StreamHandle]) -> None:
        started = time.monotonic()
        for attempt in range(self._max_retries + 1):
            try:
                self._establish_with(factory)
                return
            except Exception as exc:
                if self._finished:
                    raise
                self._record_failure(self._active, exc)
                self._active = None
                self._buffer.clear()
                self._output_visible = False
                self._terminal_status = None
                self._last_usage = None
                if not _is_retryable(exc) or attempt == self._max_retries:
                    raise
                delay = _retry_delay(exc, attempt, self._base_delay, self._max_delay)
                if not _within_deadline(started, delay, self._deadline):
                    raise
                self._emit("retry", attempt=attempt + 1, max_retries=self._max_retries, delay=round(delay, 3),
                           error=type(exc).__name__, status_code=getattr(exc, "status_code", None))
                time.sleep(delay)

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        self._ensure_started()
        if self._buffer:
            return self._buffer.popleft()
        if self._finished:
            raise StopIteration
        if self._active is None:
            raise StopIteration
        try:
            item = next(self._active.iterator)
        except StopIteration:
            self._finish_active()
            raise
        except BaseException as exc:
            self._record_failure(self._active, exc)
            self._active = None
            raise
        self._observe_item(item)
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer.clear()
        if not self._started:
            self.outcome = ExecutionOutcome(status=ExecutionStatus.CANCELLED)
            return
        if not self._finished:
            if self.outcome.status not in (ExecutionStatus.FAILED_RETRYABLE, ExecutionStatus.FAILED_TERMINAL):
                self.outcome = ExecutionOutcome(
                    status=ExecutionStatus.CANCELLED,
                    provider_dispatched=self._active is not None,
                    output_visible=self._output_visible,
                )
            self._abandon_handle(self._active)
            self._active = None
            self._finished = True


class AsyncStreamSession:
    def __init__(
        self,
        *,
        primary_factory: Callable[[], Any],
        backup_factory: Callable[[], Any] | None = None,
        is_output_visible: Callable[[Any], bool] | None = None,
        terminal_status: Callable[[Any], ExecutionStatus | None] | None = None,
        on_complete: Callable[..., Any] | None = None,
        max_retries: int = 0,
        base_delay: float = 0.0,
        max_delay: float = 60.0,
        deadline: float | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._primary_factory = primary_factory
        self._backup_factory = backup_factory
        self._is_output_visible = is_output_visible or (lambda _item: True)
        self._terminal_status_fn = terminal_status or (lambda _item: None)
        self._on_complete = on_complete
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._deadline = deadline
        self._emit = current_emitter()
        self._started = False
        self._finished = False
        self._closed = False
        self._active: StreamHandle | None = None
        self._buffer: deque[Any] = deque()
        self._output_visible = False
        self._last_usage = None
        self._terminal_status: ExecutionStatus | None = None
        self.outcome = ExecutionOutcome(status=ExecutionStatus.NOT_STARTED)

    def __aiter__(self) -> "AsyncStreamSession":
        return self

    async def _close_handle(self, handle: StreamHandle | None) -> None:
        if handle is None:
            return
        aclose = getattr(handle.stream, "aclose", None)
        close = getattr(handle.stream, "close", None)
        if callable(aclose):
            result = aclose()
            if result is not None:
                await result
            return
        if callable(close):
            close()

    async def _abandon_handle(self, handle: StreamHandle | None) -> None:
        if handle is not None and handle.breaker is not None:
            handle.breaker.abandon_probe()
        await self._close_handle(handle)

    async def _record_failure(self, handle: StreamHandle | None, exc: BaseException) -> None:
        if handle is not None and handle.breaker is not None:
            if isinstance(exc, Exception):
                handle.breaker.record(ok=not is_health_failure(exc))
            else:
                handle.breaker.abandon_probe()
        status = (
            ExecutionStatus.CANCELLED if isinstance(exc, asyncio.CancelledError)
            else ExecutionStatus.FAILED_RETRYABLE if isinstance(exc, Exception) and _is_retryable(exc)
            else ExecutionStatus.FAILED_TERMINAL
        )
        self.outcome = ExecutionOutcome(
            status=status,
            provider_dispatched=handle is not None,
            output_visible=self._output_visible,
            retryable=isinstance(exc, Exception) and _is_retryable(exc),
            error=exc if isinstance(exc, Exception) else None,
        )
        await self._close_handle(handle)

    def _observe_item(self, item: Any) -> None:
        usage = getattr(item, "usage", None)
        if usage is not None:
            self._last_usage = usage
        terminal_status = self._terminal_status_fn(item)
        if terminal_status is not None:
            self._terminal_status = terminal_status
        if self._is_output_visible(item):
            self._output_visible = True
            self.outcome = ExecutionOutcome(
                status=ExecutionStatus.STREAMING,
                provider_dispatched=True,
                output_visible=True,
            )

    def _release_buffer(self) -> bool:
        return self._output_visible or self._terminal_status is not None

    async def _finish_active(self) -> None:
        if self._finished:
            return
        status = self._terminal_status or ExecutionStatus.COMPLETED
        handle = self._active
        try:
            if handle is not None and handle.breaker is not None:
                handle.breaker.record(ok=status == ExecutionStatus.COMPLETED)
            if self._on_complete is not None:
                result = self._on_complete(
                    self._last_usage,
                    handle.limiter_scope if handle is not None else None,
                    handle.prompt_tokens if handle is not None else None,
                )
                if result is not None:
                    await result
            self.outcome = ExecutionOutcome(
                status=status,
                provider_dispatched=handle is not None,
                output_visible=self._output_visible,
            )
        except BaseException as exc:
            self._buffer.clear()
            self.outcome = ExecutionOutcome(
                status=ExecutionStatus.CANCELLED if isinstance(exc, asyncio.CancelledError)
                else ExecutionStatus.FAILED_TERMINAL,
                provider_dispatched=handle is not None,
                output_visible=self._output_visible,
                error=exc if isinstance(exc, Exception) else None,
            )
            raise
        finally:
            try:
                await self._close_handle(handle)
            finally:
                self._active = None
                self._finished = True

    def _set_active(self, handle: StreamHandle) -> None:
        self._active = handle
        handle.iterator = aiter(handle.stream)
        self.outcome = ExecutionOutcome(status=ExecutionStatus.DISPATCHED, provider_dispatched=True)

    async def _establish_with(self, factory: Callable[[], Any]) -> None:
        handle = await factory()
        self._set_active(handle)
        self._buffer.clear()
        while True:
            try:
                item = await anext(handle.iterator)
            except StopAsyncIteration:
                await self._finish_active()
                return
            self._observe_item(item)
            self._buffer.append(item)
            if self._release_buffer():
                return

    async def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        primary_exc: Exception | None = None
        try:
            await self._establish_with_retries(self._primary_factory)
            return
        except asyncio.CancelledError:
            if self._finished:
                raise
            await self._abandon_handle(self._active)
            self._active = None
            self._buffer.clear()
            self._output_visible = False
            self._terminal_status = None
            self.outcome = ExecutionOutcome(status=ExecutionStatus.CANCELLED)
            raise
        except Exception as exc:
            if self._finished:
                raise
            primary_exc = exc
            if self._backup_factory is None or not _is_retryable(exc):
                raise
        self._emit("failover", error=type(primary_exc).__name__, status_code=getattr(primary_exc, "status_code", None))
        try:
            await self._establish_with_retries(self._backup_factory)
        except asyncio.CancelledError:
            if self._finished:
                raise
            await self._abandon_handle(self._active)
            self._active = None
            self._buffer.clear()
            self.outcome = ExecutionOutcome(status=ExecutionStatus.CANCELLED)
            raise
        except Exception as backup_exc:
            if self._finished:
                raise
            raise primary_exc from backup_exc

    async def _establish_with_retries(self, factory: Callable[[], Any]) -> None:
        started = time.monotonic()
        for attempt in range(self._max_retries + 1):
            try:
                await self._establish_with(factory)
                return
            except Exception as exc:
                if self._finished:
                    raise
                await self._record_failure(self._active, exc)
                self._active = None
                self._buffer.clear()
                self._output_visible = False
                self._terminal_status = None
                self._last_usage = None
                if not _is_retryable(exc) or attempt == self._max_retries:
                    raise
                delay = _retry_delay(exc, attempt, self._base_delay, self._max_delay)
                if not _within_deadline(started, delay, self._deadline):
                    raise
                self._emit("retry", attempt=attempt + 1, max_retries=self._max_retries, delay=round(delay, 3),
                           error=type(exc).__name__, status_code=getattr(exc, "status_code", None))
                await asyncio.sleep(delay)

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        await self._ensure_started()
        if self._buffer:
            return self._buffer.popleft()
        if self._finished:
            raise StopAsyncIteration
        if self._active is None:
            raise StopAsyncIteration
        try:
            item = await anext(self._active.iterator)
        except StopAsyncIteration:
            await self._finish_active()
            raise
        except BaseException as exc:
            await self._record_failure(self._active, exc)
            self._active = None
            raise
        self._observe_item(item)
        return item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer.clear()
        if not self._started:
            self.outcome = ExecutionOutcome(status=ExecutionStatus.CANCELLED)
            return
        if not self._finished:
            if self.outcome.status not in (ExecutionStatus.FAILED_RETRYABLE, ExecutionStatus.FAILED_TERMINAL):
                self.outcome = ExecutionOutcome(
                    status=ExecutionStatus.CANCELLED,
                    provider_dispatched=self._active is not None,
                    output_visible=self._output_visible,
                )
            await self._abandon_handle(self._active)
            self._active = None
            self._finished = True
