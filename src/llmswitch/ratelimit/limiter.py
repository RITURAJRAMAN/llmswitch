"""
ratelimit.limiter

Redis-backed fixed-window rate limiter (RPM + TPM) with distributed,
atomic-pipeline state, plus rate-limit metrics.

Connection pooling
------------------
Redis ``ConnectionPool`` objects are cached per ``redis_url`` at module scope so
that many ``RedisRateLimiter`` instances (e.g. one per switch, created
per-request in a server) share a single pool instead of each opening their own.
Because the pool is shared, ``close()`` releases this instance's client but does
*not* disconnect the shared pool.

Token counting / truncation are delegated to :mod:`llmswitch.ratelimit.tokenizer`;
the methods remain on this class to preserve the public API
(``rate_limiter.count_tokens(...)`` / ``truncate_message(...)``).
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Callable, Literal, Optional

from ..errors import LLMSwitchError, RateLimitBackendUnavailableError
from . import tokenizer as _tokenizer
from .logging import log_structured

# The 'redis' library is optional; when absent the limiter transparently bypasses.
try:
    import redis
    from redis.exceptions import (
        ConnectionError as RedisConnectionError,
    )
    from redis.exceptions import RedisError
except ImportError:
    redis = None  # type: ignore[assignment]


    class RedisError(Exception):  # type: ignore[no-redef]
        pass


    class RedisConnectionError(Exception):  # type: ignore[no-redef]
        pass

logger = logging.getLogger(__name__)

# Shared connection pools, keyed by redis_url, so switches reuse connections.
_POOL_CACHE: dict[str, Any] = {}
_POOL_CACHE_LOCK = threading.Lock()
_MEMORY_WINDOWS: dict[tuple[str, str, str, str, int, int], list[int]] = {}
_MEMORY_WINDOWS_LOCK = threading.Lock()
_MEMORY_WINDOW_CAP = 10_000
_MEMORY_NEXT_EXPIRY: int | None = None
_MEMORY_EXPIRY_SOURCE = _MEMORY_WINDOWS
_DISABLED_REDIS_URL = "redis://server_disabled"


def _get_pool(redis_url: str, timeout: float):
    """Return a process-shared ConnectionPool for ``redis_url`` (created once)."""
    cache_key = f"{redis_url}|{timeout}"
    pool = _POOL_CACHE.get(cache_key)
    if pool is None:
        with _POOL_CACHE_LOCK:
            pool = _POOL_CACHE.get(cache_key)
            if pool is None:
                pool = redis.ConnectionPool.from_url(
                    redis_url,
                    socket_connect_timeout=timeout,
                    socket_timeout=timeout,
                )
                _POOL_CACHE[cache_key] = pool
    return pool


def rate_limit_scope(provider: str | None, endpoint: str | None,
                     deployment_name: str | None, model: str | None) -> str:
    """Return a stable, opaque quota namespace for a provider target."""
    raw = "\x1f".join((provider or "", endpoint or "", deployment_name or "", model or ""))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class RateLimitMetrics:
    """Tracks rate-limit hits (requests a limit denied) and misses (allowed)."""

    hits: int = 0
    misses: int = 0
    _lock = threading.Lock()

    @classmethod
    def record_hit(cls) -> None:
        with cls._lock:
            cls.hits += 1

    @classmethod
    def record_miss(cls) -> None:
        with cls._lock:
            cls.misses += 1

    @classmethod
    def to_dict(cls) -> dict[str, int]:
        with cls._lock:
            return {"rate_limit_hits": cls.hits, "rate_limit_misses": cls.misses}


class RedisRateLimiter:
    """
    A robust and efficient rate limiter.

    Handles both requests per minute (RPM) and tokens per minute (TPM) using a
    fixed-window algorithm. ``redis_url=None`` uses shared in-process counters;
    ``redis://server_disabled`` explicitly disables enforcement. A configured
    Redis backend can fail open, fail closed, or fall back to in-process state.
    """

    def __init__(
        self,
        redis_url: str | None,
        key_prefix: str = "rate_limit:",
        token_counter: Optional[Callable[[str, Optional[str]], int]] = None,
        *,
        failure_policy: Literal["memory_fallback", "fail_open", "fail_closed"] = "memory_fallback",
        scope: str = "default",
        connect_timeout: float = 1.0,
        on_limit: Literal["wait", "raise"] = "wait",
        max_wait: Optional[float] = None,
    ) -> None:
        """
        Args:
            redis_url (str | None): The connection URL for Redis. ``None`` uses
                in-process limits; ``redis://server_disabled`` disables limits.
            key_prefix (str): A prefix for all Redis keys to avoid conflicts.
            token_counter (Callable[[str, str], int] | None): Optional custom
                ``(text, model_name) -> int`` counter that overrides the default
                tiktoken-based counting used for rate-limit accounting — useful for
                accurate counts on non-OpenAI providers (e.g. an Anthropic/Gemini
                counter). Does not affect truncation, which still slices via tiktoken.
            on_limit ("wait" | "raise"): What a request does when its limit is reached.
                ``"wait"`` sleeps until the window reopens; ``"raise"`` fails immediately.
            max_wait (float | None): Longest wait, in seconds. A longer wait raises
                instead of sleeping. ``None`` means no limit.
        """
        self.redis: Optional["redis.Redis"] = None
        self.key_prefix: str = key_prefix
        self.scope = scope
        self.failure_policy = failure_policy
        self.on_limit = on_limit
        self.max_wait = max_wait
        self.mode = "memory" if redis_url is None else "redis"
        self.redis_error_logged = False
        self._token_counter = token_counter

        if redis_url == _DISABLED_REDIS_URL:
            self.mode = "disabled"
            return
        if redis_url is None:
            return

        if not redis:
            self._handle_backend_failure("redis package is not installed")
            return

        try:
            client = redis.Redis(connection_pool=_get_pool(redis_url, connect_timeout))
            client.ping()  # Validate connectivity before trusting the client.
            self.redis = client
        except (RedisConnectionError, RedisError) as e:
            self._handle_backend_failure(str(e))

    def _handle_backend_failure(self, _error: str) -> None:
        """Switch away from a failed Redis backend according to configured policy."""
        self.redis = None
        if self.failure_policy == "memory_fallback":
            self.mode = "memory"
        elif self.failure_policy == "fail_open":
            self.mode = "fail_open"
        else:
            self.mode = "fail_closed"
        if not self.redis_error_logged:
            self.redis_error_logged = True
            logger.warning("Redis rate-limit backend unavailable; applying policy '%s'.", self.failure_policy)

    def _backend_unavailable(self) -> tuple[bool, str]:
        if self.mode == "disabled":
            return True, "disabled"
        if self.mode == "fail_open":
            return True, "redis_error"
        if self.mode == "fail_closed":
            raise RateLimitBackendUnavailableError()
        raise RuntimeError("No fallback policy available for rate limiter")

    def _get_window_keys(
        self,
        key: str,
        window_seconds: int,
        request_type: str = "completion",
        scope: str | None = None,
    ) -> tuple[str, str]:
        """Generate the RPM and TPM Redis keys for the current time window."""
        suffix = f":{request_type}:{window_seconds}:{int(time.time() // window_seconds)}"
        prefix = f"{self.key_prefix}{scope or self.scope}:{key}"
        return f"{prefix}:rpm{suffix}", f"{prefix}:tpm{suffix}"

    def _memory_key(
        self,
        key: str,
        window_seconds: int,
        request_type: str,
        scope: str | None = None,
    ) -> tuple[str, str, str, str, int, int]:
        return (
            self.key_prefix, scope or self.scope, key, request_type,
            window_seconds, int(time.time() // window_seconds),
        )

    @staticmethod
    def _ensure_memory_capacity(current: tuple[str, str, str, str, int, int]) -> None:
        global _MEMORY_NEXT_EXPIRY, _MEMORY_EXPIRY_SOURCE
        now = time.time()
        if _MEMORY_EXPIRY_SOURCE is not _MEMORY_WINDOWS:
            _MEMORY_NEXT_EXPIRY = None
            _MEMORY_EXPIRY_SOURCE = _MEMORY_WINDOWS
        if _MEMORY_NEXT_EXPIRY is None or _MEMORY_NEXT_EXPIRY <= now:
            _MEMORY_NEXT_EXPIRY = None
            for old_key in list(_MEMORY_WINDOWS):
                expiry = (old_key[5] + 1) * old_key[4]
                if expiry <= now:
                    _MEMORY_WINDOWS.pop(old_key, None)
                elif _MEMORY_NEXT_EXPIRY is None or expiry < _MEMORY_NEXT_EXPIRY:
                    _MEMORY_NEXT_EXPIRY = expiry
        if current not in _MEMORY_WINDOWS:
            if len(_MEMORY_WINDOWS) >= _MEMORY_WINDOW_CAP:
                raise RateLimitBackendUnavailableError("memory rate-limit capacity")
            expiry = (current[5] + 1) * current[4]
            if _MEMORY_NEXT_EXPIRY is None or expiry < _MEMORY_NEXT_EXPIRY:
                _MEMORY_NEXT_EXPIRY = expiry

    def _incr_memory_window(self, key: str, request_type: str, tokens: int,
                            window_seconds: int, scope: str | None = None) -> tuple[int, int]:
        current = self._memory_key(key, window_seconds, request_type, scope=scope)
        with _MEMORY_WINDOWS_LOCK:
            self._ensure_memory_capacity(current)
            counts = _MEMORY_WINDOWS.setdefault(current, [0, 0])
            counts[0] += 1
            counts[1] += tokens
            return counts[0], counts[1]

    def _incr_window(self, rpm_key: str, tpm_key: str, tokens: int, window_seconds: int) -> tuple[int, int]:
        """
        Increment the window's RPM/TPM counters and return ``(rpm_count, tpm_count)``.

        Uses a transactional pipeline (MULTI/EXEC) so the two increments + TTL reads
        execute atomically; each key's TTL is set once on first write so the fixed
        window does not slide.
        """
        assert self.redis is not None  # mode is 'redis'
        with self.redis.pipeline() as pipe:
            pipe.incr(rpm_key)
            pipe.incrby(tpm_key, tokens)
            pipe.expire(rpm_key, window_seconds, nx=True)
            pipe.expire(tpm_key, window_seconds, nx=True)
            results = pipe.execute()
        rpm_count, tpm_count = int(results[0]), int(results[1])
        return rpm_count, tpm_count

    def is_allowed(
            self, key: str, max_requests: int, max_tokens: int, window_seconds: int, tokens: int = 1,
            request_type: str = "completion", suppress_log: bool = False, scope: str | None = None
    ) -> tuple[bool, str]:
        """Fixed-window RPM/TPM check.

        Increments the window counters as a *reservation* before evaluating the
        limit, so a denied request still counts toward its window. Returns
        ``(allowed, reason)``. Note: fixed windows permit up to ~2x burst across
        a window boundary.
        """
        if self.mode in {"disabled", "fail_open", "fail_closed"}:
            return self._backend_unavailable()

        if tokens > max_tokens:
            raise LLMSwitchError(f"Total tokens in messages ({tokens}) exceed the model's token limit ({max_tokens}).")

        try:
            if self.mode == "memory":
                rpm_count, tpm_count = self._incr_memory_window(
                    key, request_type, tokens, window_seconds, scope=scope
                )
            elif self.redis is not None:
                rpm_key, tpm_key = self._get_window_keys(key, window_seconds, request_type, scope=scope)
                rpm_count, tpm_count = self._incr_window(rpm_key, tpm_key, tokens, window_seconds)
            else:
                return self._backend_unavailable()

            rpm_over = rpm_count > max_requests
            tpm_over = tpm_count > max_tokens
            if rpm_over and tpm_over:
                reason, allowed, level, msg = "both_exceeded", False, "WARNING", "Both RPM and TPM limits exceeded"
            elif rpm_over:
                reason, allowed, level, msg = "rpm_exceeded", False, "WARNING", "Rate limit exceeded"
            elif tpm_over:
                reason, allowed, level, msg = "tpm_exceeded", False, "WARNING", "Token limit exceeded"
            else:
                reason, allowed, level, msg = "allowed", True, "DEBUG", "Rate limit allowed"

            RateLimitMetrics.record_miss() if allowed else RateLimitMetrics.record_hit()
            if not suppress_log:
                log_structured(level, msg, user_id=key, request_type=request_type,
                               rpm=f"{rpm_count}/{max_requests}", tpm=f"{tpm_count}/{max_tokens}")
            return allowed, reason

        except RedisError as e:
            self._handle_backend_failure(str(e))
            if self.mode == "memory":
                return self.is_allowed(
                    key, max_requests, max_tokens, window_seconds, tokens,
                    request_type=request_type, suppress_log=suppress_log, scope=scope,
                )
            return self._backend_unavailable()

    def get_remaining(self, key: str, max_requests: int, max_tokens: int, window_seconds: int,
                      request_type: str = "completion", scope: str | None = None) -> dict:
        """
        Return the remaining requests, tokens, and time in the current window.

        Note: not atomic; intended for informational purposes only.
        """
        if self.mode in {"disabled", "fail_open"}:
            return {"remaining_requests": None, "remaining_tokens": None, "remaining_time": None}

        if self.mode == "fail_closed":
            raise RateLimitBackendUnavailableError()

        if self.mode == "memory":
            current = self._memory_key(key, window_seconds, request_type, scope=scope)
            with _MEMORY_WINDOWS_LOCK:
                rpm_count, tpm_count = _MEMORY_WINDOWS.get(current, [0, 0])
            now = int(time.time())
            remaining_time = max(0, ((now // window_seconds) + 1) * window_seconds - now)
            return {
                "remaining_requests": max_requests - rpm_count,
                "remaining_tokens": max_tokens - tpm_count,
                "remaining_time": remaining_time,
            }

        rpm_key, tpm_key = self._get_window_keys(key, window_seconds, request_type, scope=scope)
        now = int(time.time())
        window_start = (now // window_seconds) * window_seconds

        try:
            assert self.redis is not None  # mode is 'redis'
            with self.redis.pipeline() as pipe:
                pipe.get(rpm_key)
                pipe.get(tpm_key)
                pipe.ttl(rpm_key)
                rpm_count_bytes, tpm_count_bytes, remaining_time = pipe.execute()

            rpm_count = int(rpm_count_bytes.decode('utf-8')) if rpm_count_bytes else 0
            tpm_count = int(tpm_count_bytes.decode('utf-8')) if tpm_count_bytes else 0

            # If ttl is -1, key has no expiry, so calculate remaining time
            if remaining_time == -1:
                remaining_time = max(0, window_start + window_seconds - now)

            return {
                "remaining_requests": max_requests - rpm_count,
                "remaining_tokens": max_tokens - tpm_count,
                "remaining_time": remaining_time,
            }
        except (RedisError, UnicodeDecodeError, ValueError) as e:
            self._handle_backend_failure(str(e))
            if self.mode == "memory":
                return self.get_remaining(key, max_requests, max_tokens, window_seconds, request_type, scope=scope)
            if self.mode == "fail_closed":
                raise RateLimitBackendUnavailableError() from e
            return {"remaining_requests": None, "remaining_tokens": None, "remaining_time": None}

    # --- Token counting / truncation (delegated to ratelimit.tokenizer) ---

    def fallback_count_tokens(self, text: str) -> int:
        """Estimate token count by word count * 1.3 (used when tiktoken is unavailable)."""
        return _tokenizer.fallback_count_tokens(text)

    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """Count tokens with a custom counter when configured, else tiktoken (cached encoding)."""
        if self._token_counter is not None:
            return self._token_counter(text, model_name)
        return _tokenizer.count_tokens(text, model_name=model_name)

    def truncate_message(self, message, max_tokens: int, truncate: bool = False,
                         model_name: Optional[str] = None) -> Optional[str]:
        """Truncate a message to fit within ``max_tokens`` when ``truncate`` is True."""
        return _tokenizer.truncate_message(message, max_tokens, truncate=truncate,
                                           model_name=model_name)

    def update_token_usage(self, key: str, tokens_used: int, window_seconds: int,
                           request_type: str = "completion", scope: str | None = None) -> None:
        """Add ``tokens_used`` to the current TPM window without incrementing the request count."""
        if self.mode == "disabled":
            return

        if self.mode == "memory":
            current = self._memory_key(key, window_seconds, request_type, scope=scope)
            with _MEMORY_WINDOWS_LOCK:
                self._ensure_memory_capacity(current)
                counts = _MEMORY_WINDOWS.setdefault(current, [0, 0])
                counts[1] += tokens_used
            return
        if self.mode == "fail_open":
            return
        if self.mode == "fail_closed":
            raise RateLimitBackendUnavailableError()

        _, tpm_key = self._get_window_keys(key, window_seconds, request_type, scope=scope)

        try:
            assert self.redis is not None  # mode is 'redis'
            with self.redis.pipeline() as pipe:
                pipe.incrby(tpm_key, tokens_used)
                pipe.expire(tpm_key, window_seconds, nx=True)
                pipe.execute()
        except RedisError as e:
            self._handle_backend_failure(str(e))
            if self.mode == "memory":
                self.update_token_usage(key, tokens_used, window_seconds, request_type, scope=scope)
            elif self.mode == "fail_closed":
                raise RateLimitBackendUnavailableError() from e

    def close(self) -> None:
        """
        Release this instance's Redis client.

        The connection pool is shared process-wide (see module docstring), so it
        is intentionally *not* disconnected here — doing so would drop connections
        still in use by other switches bound to the same ``redis_url``.
        """
        if not self.redis:
            return
        try:
            close_fn = getattr(self.redis, "close", None)
            if callable(close_fn):
                close_fn()
        except RedisError as e:
            logger.debug("Ignoring Redis close error: %s", e)
