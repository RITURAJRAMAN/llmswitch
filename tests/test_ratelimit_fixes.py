"""In-process rate limits: a full table evicts instead of failing, and denied requests give back their budget."""
import uuid

import pytest

from llmswitch.ratelimit import limiter as limiter_module
from llmswitch.ratelimit.limiter import RedisRateLimiter


@pytest.fixture
def memory_limiter():
    limiter_module._MEMORY_WINDOWS.clear()
    return RedisRateLimiter(None)


def key():
    return f"rl-{uuid.uuid4().hex}"


def test_a_full_table_evicts_the_oldest_entry_instead_of_failing_everyone(monkeypatch, memory_limiter):
    monkeypatch.setattr(limiter_module, "_MEMORY_WINDOW_CAP", 3)
    monkeypatch.setattr(limiter_module, "_capacity_warned", False)
    users = [key() for _ in range(5)]
    for user in users:
        allowed, _ = memory_limiter.is_allowed(user, 10, 10_000, 60, 1)
        assert allowed is True
    assert len(limiter_module._MEMORY_WINDOWS) <= 3
    newest = key()
    assert memory_limiter.is_allowed(newest, 10, 10_000, 60, 1)[0] is True


def test_the_eviction_is_logged_once(monkeypatch, memory_limiter, caplog):
    monkeypatch.setattr(limiter_module, "_MEMORY_WINDOW_CAP", 1)
    monkeypatch.setattr(limiter_module, "_capacity_warned", False)
    with caplog.at_level("WARNING", logger="llmswitch.ratelimit.limiter"):
        for _ in range(4):
            memory_limiter.is_allowed(key(), 10, 10_000, 60, 1)
    assert sum("evicting the oldest" in r.getMessage() for r in caplog.records) == 1


def test_a_denied_request_does_not_use_up_the_token_budget(memory_limiter):
    user = key()
    assert memory_limiter.is_allowed(user, 100, 10_000, 60, 9_000)[0] is True
    allowed, reason = memory_limiter.is_allowed(user, 100, 10_000, 60, 2_000)
    assert (allowed, reason) == (False, "tpm_exceeded")
    # 9,000 + 500 fits in 10,000, so this must be allowed even though the 2,000 request was denied.
    assert memory_limiter.is_allowed(user, 100, 10_000, 60, 500)[0] is True


def test_a_denied_request_does_not_use_up_the_request_budget(memory_limiter):
    user = key()
    assert memory_limiter.is_allowed(user, 1, 10_000, 60, 1)[0] is True
    assert memory_limiter.is_allowed(user, 1, 10_000, 60, 1)[0] is False
    assert memory_limiter.is_allowed(user, 1, 10_000, 60, 1)[0] is False
    counts = [v for k, v in limiter_module._MEMORY_WINDOWS.items() if k[2] == user]
    assert counts == [[1, 1]]


class RecordingPipeline:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def decrby(self, name, amount):
        self.log.append(("decrby", name, amount))

    def expire(self, name, seconds, nx=False):
        self.log.append(("expire", name, seconds))

    def execute(self):
        self.log.append(("execute",))


class RecordingRedis:
    def __init__(self):
        self.log = []

    def pipeline(self):
        return RecordingPipeline(self.log)


def test_redis_rollback_gives_back_exactly_the_reserved_amounts_and_keeps_a_ttl(memory_limiter):
    memory_limiter.redis = RecordingRedis()
    memory_limiter._release_window("rpm-key", "tpm-key", 1500, 60)
    assert memory_limiter.redis.log == [
        ("decrby", "rpm-key", 1), ("decrby", "tpm-key", 1500),
        ("expire", "rpm-key", 60), ("expire", "tpm-key", 60), ("execute",),
    ]


def test_a_failed_redis_rollback_does_not_turn_a_denial_into_an_error(memory_limiter, caplog):
    class Broken(RecordingRedis):
        def pipeline(self):
            raise limiter_module.RedisError("connection lost")

    memory_limiter.redis = Broken()
    with caplog.at_level("WARNING", logger="llmswitch.ratelimit.limiter"):
        memory_limiter._release_window("rpm-key", "tpm-key", 1, 60)
    assert any("Could not release" in r.getMessage() for r in caplog.records)


def test_a_memory_rollback_never_goes_below_zero(memory_limiter):
    user = key()
    window = memory_limiter._memory_key(user, 60, "completion")
    memory_limiter._release_memory_window(window, 50)
    assert window not in limiter_module._MEMORY_WINDOWS


def test_a_rollback_uses_the_reservation_window_not_the_current_clock(monkeypatch, memory_limiter):
    user = key()
    clock = [100.1]
    monkeypatch.setattr(limiter_module.time, "time", lambda: clock[0])
    original = memory_limiter._memory_key(user, 1, "completion")
    assert memory_limiter._incr_memory_window(original, 10) == (1, 10)
    clock[0] = 101.5  # the window has rolled over before the denied request is released
    memory_limiter._release_memory_window(original, 10)
    assert limiter_module._MEMORY_WINDOWS[original] == [0, 0]
    assert not any(k != original and k[2] == user and v[0] for k, v in limiter_module._MEMORY_WINDOWS.items())
