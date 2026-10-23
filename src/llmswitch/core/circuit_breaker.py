"""
core.circuit_breaker

A minimal thread-safe circuit breaker plus a process-wide registry. Used opt-in
(``LLMSwitch(..., circuit_breaker=True)``) to fail fast when a provider is
persistently unhealthy — instead of driving every request through the full retry
loop against a provider that is down.

State machine: ``closed`` → (consecutive failures ≥ threshold) → ``open`` →
(after ``reset_timeout`` seconds) → ``half_open`` → (success) → ``closed`` or
(failure) → ``open``.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from ..errors import APIConnectionError
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
        self._half_open_probe = False
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        """Whether a request may proceed; trips open→half-open once the timeout elapses.

        In the half-open state only a single trial request is admitted at a time
        (concurrent callers are rejected) so a recovering provider is not hit by a
        thundering herd of probes.
        """
        with self._lock:
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self.reset_timeout:
                    self._state = "half_open"
                    self._half_open_probe = True
                    return True
                return False
            if self._state == "half_open":
                if self._half_open_probe:
                    return False  # a probe is already in flight
                self._half_open_probe = True
                return True
            return True

    def record(self, *, ok: bool) -> None:
        """Record a call outcome: success closes the breaker; failure may open it."""
        with self._lock:
            self._half_open_probe = False
            if ok:
                self._failures = 0
                self._state = "closed"
                return
            self._failures += 1
            if self._state == "half_open" or self._failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.monotonic()

    def abandon_probe(self) -> None:
        """Reopen an unresolved half-open probe without counting a new failure."""
        with self._lock:
            if self._state == "half_open" and self._half_open_probe:
                self._state = "open"
                self._opened_at = time.monotonic()
                self._half_open_probe = False


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
    """True for transient provider-health failures (retryable 5xx / connection) that
    should count against the breaker; 4xx and 429 (healthy-but-throttled) do not."""
    return _is_retryable(exc) and getattr(exc, "status_code", 500) >= 500


@contextmanager
def provider_call(
    provider: str,
    model: Any,
    *,
    breaker: Optional[CircuitBreaker] = None,
    record_success: bool = True,
) -> Generator[None, None, None]:
    """
    Guard a provider call with an optional circuit breaker.

    - If ``breaker`` is open, raises ``APIConnectionError`` immediately (the switch
      treats this as a primary failure → backup failover) without calling the provider.
        - Records transient/5xx failures after retries; 4xx and successful calls do not
            open the breaker. Lazy streams defer successful completion until exhaustion.

    With ``breaker=None`` this is a transparent pass-through.
    """
    if breaker is not None and not breaker.allow():
        raise APIConnectionError(str(model or provider), provider=provider)
    try:
        yield
    except Exception as exc:
        if breaker is not None:
            breaker.record(ok=not is_health_failure(exc))
        raise
    except BaseException:
        if breaker is not None:
            breaker.abandon_probe()
        raise
    else:
        if breaker is not None and record_success:
            breaker.record(ok=True)
