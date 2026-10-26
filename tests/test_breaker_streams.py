"""Circuit breaker probes: an abandoned stream frees only its own probe, and timeouts count as failures."""
import gc

import pytest

from llmswitch import TimeOutError
from llmswitch.core.circuit_breaker import CircuitBreaker, is_health_failure, provider_call
from llmswitch.core.streaming import AsyncStreamSession, StreamHandle, SyncStreamSession


def half_open_breaker():
    """Return a breaker in half-open state and the probe token the first trial call holds."""
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0)
    breaker.record(ok=False)  # open; reset_timeout 0 means the next call becomes the probe
    admitted, probe = breaker.acquire()
    assert admitted is True and probe is not None
    assert breaker.state == "half_open"
    return breaker, probe


def endless_chunks():
    for index in range(100):
        yield {"choices": [{"delta": {"content": str(index)}}]}


async def async_chunks():
    for index in range(100):
        yield {"choices": [{"delta": {"content": str(index)}}]}


def test_an_abandoned_sync_stream_releases_its_own_probe():
    breaker, probe = half_open_breaker()
    session = SyncStreamSession(
        primary_factory=lambda: StreamHandle(endless_chunks(), breaker=breaker, probe=probe),
        is_output_visible=lambda item: True)
    assert next(session)["choices"][0]["delta"]["content"] == "0"
    del session
    gc.collect()
    assert breaker.acquire()[0] is True  # the probe was released, so a new trial is admitted


def test_a_closed_sync_stream_also_releases_the_probe():
    breaker, probe = half_open_breaker()
    session = SyncStreamSession(
        primary_factory=lambda: StreamHandle(endless_chunks(), breaker=breaker, probe=probe),
        is_output_visible=lambda item: True)
    next(session)
    session.close()
    assert breaker.state == "open"


async def test_an_abandoned_async_stream_releases_its_own_probe():
    breaker, probe = half_open_breaker()

    async def factory():
        return StreamHandle(async_chunks(), breaker=breaker, probe=probe)

    session = AsyncStreamSession(primary_factory=factory, is_output_visible=lambda item: True)
    await session.__anext__()
    del session
    gc.collect()
    assert breaker.acquire()[0] is True


def test_a_finished_stream_does_not_touch_the_breaker_again():
    breaker, probe = half_open_breaker()
    session = SyncStreamSession(
        primary_factory=lambda: StreamHandle(iter([]), breaker=breaker, probe=probe),
        is_output_visible=lambda item: True)
    assert list(session) == []
    del session
    gc.collect()
    assert breaker.state == "closed"


def test_a_stale_probe_cannot_release_the_current_one():
    breaker, current = half_open_breaker()
    breaker.abandon(object())  # a token that was never the probe
    assert breaker.state == "half_open"
    assert breaker.acquire() == (False, None)  # the current probe still holds the breaker
    breaker.abandon(current)
    assert breaker.state == "open"


def test_a_stale_success_cannot_resolve_the_current_probe():
    breaker, current = half_open_breaker()
    breaker.record(ok=True, probe=object())
    assert breaker.state == "half_open"
    breaker.record(ok=True, probe=current)
    assert breaker.state == "closed"


def test_a_closed_call_failure_during_a_probe_does_not_clear_the_probe():
    breaker, current = half_open_breaker()
    breaker.record(ok=True)  # a call that started before the trip finishes successfully
    assert breaker.state == "closed"
    breaker.record(ok=False, probe=current)  # the old probe's late outcome is stale and ignored
    assert breaker.state == "closed"


def test_timeouts_count_as_health_failures():
    assert is_health_failure(TimeOutError(5)) is True


def test_repeated_timeouts_open_the_breaker():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=60)
    for _ in range(2):
        with pytest.raises(TimeOutError):
            with provider_call("openai", "gpt-4o", breaker=breaker):
                raise TimeOutError(5)
    assert breaker.state == "open"


def test_a_refused_call_raises_connection_error_without_calling_the_provider():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60)
    breaker.record(ok=False)
    called = []
    from llmswitch import APIConnectionError
    with pytest.raises(APIConnectionError):
        with provider_call("openai", "gpt-4o", breaker=breaker):
            called.append(1)
    assert called == []
