"""A stream the caller abandons must not leave the circuit breaker stuck half-open."""
import gc

import pytest

from llmswitch.core.circuit_breaker import CircuitBreaker
from llmswitch.core.streaming import AsyncStreamSession, StreamHandle, SyncStreamSession


def half_open_breaker():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0)
    breaker.record(ok=False)  # open; reset_timeout 0 means the next allow() goes half-open
    assert breaker.allow() is True  # takes the single half-open probe
    assert breaker.state == "half_open"
    return breaker


def endless_chunks():
    for index in range(100):
        yield {"choices": [{"delta": {"content": str(index)}}]}


def test_an_abandoned_sync_stream_releases_the_probe():
    breaker = half_open_breaker()
    session = SyncStreamSession(primary_factory=lambda: StreamHandle(endless_chunks(), breaker=breaker),
                                is_output_visible=lambda item: True)
    assert next(session)["choices"][0]["delta"]["content"] == "0"
    del session
    gc.collect()
    assert breaker.allow() is True  # the probe was released, so a new trial is admitted


def test_a_closed_sync_stream_also_releases_the_probe():
    breaker = half_open_breaker()
    session = SyncStreamSession(primary_factory=lambda: StreamHandle(endless_chunks(), breaker=breaker),
                                is_output_visible=lambda item: True)
    next(session)
    session.close()
    assert breaker.state == "open"


async def async_chunks():
    for index in range(100):
        yield {"choices": [{"delta": {"content": str(index)}}]}


async def test_an_abandoned_async_stream_releases_the_probe():
    breaker = half_open_breaker()

    async def factory():
        return StreamHandle(async_chunks(), breaker=breaker)

    session = AsyncStreamSession(primary_factory=factory, is_output_visible=lambda item: True)
    await session.__anext__()
    del session
    gc.collect()
    assert breaker.allow() is True


@pytest.mark.parametrize("finished", [True])
def test_a_finished_stream_does_not_touch_the_breaker_again(finished):
    breaker = half_open_breaker()
    session = SyncStreamSession(primary_factory=lambda: StreamHandle(iter([]), breaker=breaker),
                                is_output_visible=lambda item: True)
    assert list(session) == []
    del session
    gc.collect()
    assert breaker.state == "closed"
