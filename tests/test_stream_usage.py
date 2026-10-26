"""Streams are counted against the token limit even when the provider sends no usage."""
from types import SimpleNamespace

from llmswitch.core.streaming import AsyncStreamSession, StreamHandle, SyncStreamSession, stream_completion_tokens
from llmswitch.errors import APIConnectionError
from llmswitch.ratelimit.tokenizer import count_tokens


def text_chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def test_a_stream_without_usage_is_counted_from_its_text():
    text = "The population of Lyon is about two million people."
    assert stream_completion_tokens(None, None, 100, text, "gpt-4o-mini") == count_tokens(text, "gpt-4o-mini")
    assert stream_completion_tokens(None, None, 100, text, "gpt-4o-mini") > 0


def test_a_stream_with_no_text_and_no_usage_adds_nothing():
    assert stream_completion_tokens(None, None, 100, "", "gpt-4o-mini") == 0


def test_provider_usage_stays_the_exact_count_when_it_is_sent():
    usage = SimpleNamespace(total_tokens=130)

    assert stream_completion_tokens(usage, 120, 999, "ignored text", "gpt-4o-mini") == 10
    assert stream_completion_tokens(usage, None, 120, "ignored text", "gpt-4o-mini") == 10


def test_a_zero_usage_from_the_provider_falls_back_to_the_text():
    text = "Two words"

    assert stream_completion_tokens(SimpleNamespace(total_tokens=0), 5, 5, text, None) == count_tokens(text, None)


def test_a_sync_stream_reports_its_text_to_the_completion_callback_when_usage_is_missing():
    seen = []

    def on_complete(last_usage, scope, prompt_tokens, output_text):
        seen.append((last_usage, output_text))

    session = SyncStreamSession(
        primary_factory=lambda: StreamHandle(iter([text_chunk("Hello "), text_chunk("world")])),
        is_output_visible=lambda item: True, on_complete=on_complete)
    assert [chunk["choices"][0]["delta"]["content"] for chunk in session] == ["Hello ", "world"]
    assert seen == [(None, "Hello world")]


def test_a_sync_stream_keeps_thinking_text_in_the_count():
    seen = []
    chunks = [{"choices": [{"delta": {"thinking": "Let me think. "}}]}, text_chunk("Done.")]
    session = SyncStreamSession(
        primary_factory=lambda: StreamHandle(iter(chunks)),
        is_output_visible=lambda item: True,
        on_complete=lambda usage, scope, prompt, output_text: seen.append(output_text))
    list(session)
    assert seen == ["Let me think. Done."]


def test_a_retried_stream_counts_only_the_text_of_the_attempt_that_finished():
    attempts = {"n": 0}

    def primary():
        attempts["n"] += 1
        if attempts["n"] == 1:
            def broken():
                yield text_chunk("partial that is dropped")
                raise APIConnectionError("dropped")
            return StreamHandle(broken())
        return StreamHandle(iter([text_chunk("final answer")]))

    seen = []
    session = SyncStreamSession(
        primary_factory=primary, is_output_visible=lambda item: False, max_retries=1, base_delay=0.0,
        on_complete=lambda usage, scope, prompt, output_text: seen.append(output_text))
    list(session)
    assert seen == ["final answer"]


async def test_an_async_stream_reports_its_text_when_usage_is_missing():
    seen = []

    async def chunks():
        yield text_chunk("Async ")
        yield text_chunk("text")

    async def on_complete(last_usage, scope, prompt_tokens, output_text):
        seen.append(output_text)

    async def factory():
        return StreamHandle(chunks())

    session = AsyncStreamSession(primary_factory=factory, is_output_visible=lambda item: True, on_complete=on_complete)
    collected = [chunk async for chunk in session]
    assert len(collected) == 2
    assert seen == ["Async text"]


def test_streamed_tool_call_arguments_are_counted_when_usage_is_missing():
    seen = []
    chunks = [{"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '{"city": '}}]}}]},
              {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '"Lyon"}'}}]}}]}]
    session = SyncStreamSession(
        primary_factory=lambda: StreamHandle(iter(chunks)),
        is_output_visible=lambda item: True,
        on_complete=lambda usage, scope, prompt, output_text: seen.append(output_text))
    list(session)
    assert seen == ['{"city": "Lyon"}']
