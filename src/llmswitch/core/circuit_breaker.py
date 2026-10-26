"""
core.circuit_breaker

A minimal thread-safe circuit breaker plus a process-wide registry. Used opt-in
(``LLMSwitch(..., circuit_breaker=True)``) to fail fast when a provider is
persistently unhealthy — instead of driving every request through the full retry
loop against a provider that is down.

State machine: ``closed`` → (consecutive failures ≥ threshold) → ``open`` →
(after ``reset_timeout`` seconds) → ``half_open`` → (success) → ``closed`` or
(failure) → ``open``.

While half-open, one trial call (the *probe*) is admitted at a time. Each admitted
call gets a probe token when it is the probe, and hands that token back when it
finishes. Only the holder of the current token can resolve the probe, so a stale
call, or a stream that was dropped late, cannot release a probe it does not own.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from ..errors import APIConnectionError, TimeOutError
from .retry import _is_retryable

_DEFAULT_FAILURE_THRESHOLD = 5
_DEFAULT_RESET_TIMEOUT = 30.0


class CircuitBreaker:
    """Thread-safe circuit breaker for one provider target."""

    def __init__(self, failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
                 reset_timeout: float = _DEFAULT_RESET_TIMEOUT) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0
        self._probe: object | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def acquire(self) -> tuple[bool, object | None]:
        """Admit a call. Returns ``(admitted, probe)``; ``probe`` is set only for the trial call.

        An open breaker refuses calls until ``reset_timeout`` has passed. Then the next
        call becomes the half-open probe, and other calls are refused until it resolves.
        """
        with self._lock:
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self.reset_timeout:
                    self._state = "half_open"
                    self._probe = object()
                    return True, self._probe
                return False, None
            if self._state == "half_open":
                if self._probe is not None:
                    return False, None  # a probe is already in flight
                self._probe = object()
                return True, self._probe
            return True, None

    def record(self, *, ok: bool, probe: object | None = None) -> None:
        """Record a call outcome: success closes the breaker; failure may open it.

        A ``probe`` that is no longer the current probe is ignored: the breaker has
        already moved on, so the outcome is stale.
        """
        with self._lock:
            if probe is not None:
                if probe is not self._probe:
                    return
                self._probe = None
            if ok:
                self._failures = 0
                self._state = "closed"
                self._probe = None
                return
            self._failures += 1
            if self._state == "half_open" or self._failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.monotonic()
                self._probe = None

    def abandon(self, probe: object | None) -> None:
        """Reopen an unresolved probe without counting a new failure.

        Only the current probe can be abandoned. Any other token is ignored.
        """
        with self._lock:
            if probe is not None and probe is self._probe and self._state == "half_open":
                self._state = "open"
                self._opened_at = time.monotonic()
                self._probe = None


_breakers: dict = {}
_breakers_lock = threading.Lock()


def get_breaker(key, failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
                reset_timeout: float = _DEFAULT_RESET_TIMEOUT) -> CircuitBreaker:
    """Return the process-wide breaker for a target-specific key."""
    with _breakers_lock:
        breaker = _breakers.get(key)
        if breaker is None:
            breaker = CircuitBreaker(failure_threshold, reset_timeout)
            _breakers[key] = breaker
        return breaker


def is_health_failure(exc: Exception) -> bool:
    """True for failures that say the provider is unhealthy: timeouts, connection errors and 5xx.

    4xx and 429 mean a healthy but throttled or rejected request, so they do not count.
    A timeout counts even though it is not retried, because a provider that keeps timing
    out is exactly what the breaker should stop calling.
    """
    if isinstance(exc, TimeOutError):
        return True
    return _is_retryable(exc) and getattr(exc, "status_code", 500) >= 500


@contextmanager
def provider_call(
    provider: str,
    model: Any,
    *,
    breaker: Optional[CircuitBreaker] = None,
    record_success: bool = True,
) -> Generator[object | None, None, None]:
    """
    Guard a provider call with an optional circuit breaker.

    - If ``breaker`` refuses the call, raises ``APIConnectionError`` without calling the
      provider. The switch treats that as a primary failure and fails over to the backup.
    - Records health failures after retries. 4xx and successful calls do not open the
      breaker. Lazy streams pass ``record_success=False`` and record the outcome when the
      stream finishes.
    - Yields the probe token (``None`` unless this call is the half-open probe). A stream
      keeps the token and resolves the probe with it.

    With ``breaker=None`` this is a transparent pass-through.
    """
    admitted, probe = breaker.acquire() if breaker is not None else (True, None)
    if not admitted:
        raise APIConnectionError(str(model or provider), provider=provider)
    try:
        yield probe
    except Exception as exc:
        if breaker is not None:
            breaker.record(ok=not is_health_failure(exc), probe=probe)
        raise
    except BaseException:
        if breaker is not None:
            breaker.abandon(probe)
        raise
    else:
        if breaker is not None and record_success:
            breaker.record(ok=True, probe=probe)
