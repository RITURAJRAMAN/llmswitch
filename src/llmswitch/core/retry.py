"""
core.retry

Transient-error classification and exponential back-off (full-jitter) retry
drivers shared by the sync and async switches.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time

from ..errors import (
    APIConnectionError,
    ChatCompletionError,
    EmbeddingGenerationError,
    RateLimitExceededError,
    TimeOutError,
)
from .events import emit

logger = logging.getLogger(__name__)


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors that are safe to retry.

    A timeout is not retried: the provider may already have processed the request,
    so a retry could bill it twice. Connection errors and 5xx responses are retried.
    """
    if isinstance(exc, TimeOutError):
        return False
    if isinstance(exc, (RateLimitExceededError, APIConnectionError)):
        return True
    if isinstance(exc, (ChatCompletionError, EmbeddingGenerationError)):
        return getattr(exc, "status_code", 500) >= 500
    # Also handle raw provider exceptions not yet wrapped (status_code >= 500)
    status = getattr(exc, "status_code", None)
    return status is not None and status >= 500


def _retry_delay(exc: Exception, attempt: int, base: float, cap: float) -> float:
    """Full-jitter exponential backoff, honouring retry_after when available."""
    if isinstance(exc, RateLimitExceededError):
        retry_after = exc.context.get("retry_after", 0)
        if retry_after and retry_after > 0:
            return float(retry_after)
    # Full jitter: uniform(0, min(cap, base * 2^attempt))
    ceiling = min(cap, base * (2 ** attempt))
    return random.uniform(0.0, ceiling)


def _within_deadline(started: float, delay: float, deadline: float | None) -> bool:
    """True when sleeping ``delay`` seconds still leaves the call inside its retry deadline."""
    return deadline is None or (time.monotonic() - started) + delay <= deadline


def _retry_call(fn, *args, max_retries: int, base_delay: float, max_delay: float,  # noqa: RET503
                deadline: float | None = None, **kwargs):
    """Synchronous retry with exponential back-off for transient errors.

    Attempts ``fn(*args, **kwargs)`` up to ``max_retries + 1`` times total.
    Non-retryable errors (4xx, config errors) are re-raised immediately, and so is the
    last error once waiting for the next attempt would pass ``deadline`` seconds.
    """
    started = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            delay = _retry_delay(exc, attempt, base_delay, max_delay)
            if not _within_deadline(started, delay, deadline):
                raise
            emit("retry", attempt=attempt + 1, max_retries=max_retries, delay=round(delay, 3),
                 error=type(exc).__name__, status_code=getattr(exc, "status_code", None))
            logger.warning(
                "Retryable error on attempt %d/%d (%s). Retrying in %.2fs.",
                attempt + 1, max_retries, type(exc).__name__, delay,
            )
            time.sleep(delay)


async def _async_retry_call(fn, *args, max_retries: int, base_delay: float, max_delay: float,  # noqa: RET503
                            deadline: float | None = None, **kwargs):
    """Asynchronous retry with exponential back-off for transient errors.

    Attempts ``await fn(*args, **kwargs)`` up to ``max_retries + 1`` times total.
    Non-retryable errors (4xx, config errors) are re-raised immediately, and so is the
    last error once waiting for the next attempt would pass ``deadline`` seconds.
    """
    started = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            delay = _retry_delay(exc, attempt, base_delay, max_delay)
            if not _within_deadline(started, delay, deadline):
                raise
            emit("retry", attempt=attempt + 1, max_retries=max_retries, delay=round(delay, 3),
                 error=type(exc).__name__, status_code=getattr(exc, "status_code", None))
            logger.warning(
                "Retryable error on attempt %d/%d (%s). Retrying in %.2fs.",
                attempt + 1, max_retries, type(exc).__name__, delay,
            )
            await asyncio.sleep(delay)
