"""Signed thinking blocks: kept from an Anthropic response and sent back unchanged in the next request."""
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from llmswitch.agentic.messages import to_ai_message, to_chat_messages  # noqa: E402
from llmswitch.models import ChatCompletion, ChatCompletionMessage, Choice, ProviderExtensions  # noqa: E402
from llmswitch.providers.anthropic_provider import (  # noqa: E402
    _openai_messages_to_anthropic,
    _replayed_thinking_blocks,
    _thinking_blocks,
)

BLOCKS = [
    {"type": "thinking", "thinking": "Need the population first.", "signature": "sig-1"},
    {"type": "redacted_thinking", "data": "opaque-data"},
]


def anthropic_content():
    return [
        SimpleNamespace(type="thinking", thinking="Need the population first.", signature="sig-1"),
        SimpleNamespace(type="redacted_thinking", data="opaque-data"),
        SimpleNamespace(type="tool_use", id="t1", name="lookup", input={"city": "Lyon"}),
    ]


def test_the_thinking_blocks_of_a_response_are_kept_in_the_form_the_next_request_needs():
    assert _thinking_blocks(anthropic_content()) == BLOCKS


def test_an_assistant_turn_with_thinking_sends_the_blocks_first_then_the_tool_use():
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
           "extensions": {"provider": "anthropic", "fields": {"thinking_blocks": BLOCKS}}}
    _, messages = _openai_messages_to_anthropic([{"role": "user", "content": "hi"}, msg])

    assistant = messages[-1]["content"]
    assert [part["type"] for part in assistant] == ["thinking", "redacted_thinking", "tool_use"]
    assert assistant[0] == BLOCKS[0] and assistant[1] == BLOCKS[1]


def test_thinking_blocks_for_another_provider_are_not_sent_to_anthropic():
    msg = {"role": "assistant", "content": "answer",
           "extensions": {"provider": "openai", "fields": {"thinking_blocks": BLOCKS}}}

    assert _replayed_thinking_blocks(msg) == []


def test_thinking_blocks_held_as_provider_extensions_are_read_too():
    msg = {"role": "assistant", "content": "answer",
           "extensions": ProviderExtensions(provider="anthropic", fields={"thinking_blocks": BLOCKS})}

    assert _replayed_thinking_blocks(msg) == BLOCKS


def completion_with_thinking(blocks):
    message = ChatCompletionMessage(
        role="assistant", content="",
        tool_calls=None,
        extensions=ProviderExtensions(provider="anthropic", fields={"thinking_blocks": blocks}) if blocks else None,
    )
    return ChatCompletion(id="c1", object="chat.completion", created=0, model="m",
                          choices=[Choice(index=0, message=message, finish_reason="stop")])


def test_an_agentic_answer_keeps_the_thinking_blocks_in_additional_kwargs():
    answer = to_ai_message(completion_with_thinking(BLOCKS))

    assert answer.additional_kwargs["provider_data"] == {"provider": "anthropic", "fields": {"thinking_blocks": BLOCKS}}
    assert answer.content == ""


def test_an_answer_without_thinking_has_no_thinking_blocks():
    assert to_ai_message(completion_with_thinking(None)).additional_kwargs == {}


def test_the_agentic_history_sends_the_thinking_blocks_back_to_anthropic_only():
    turn = AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"city": "Lyon"}, "id": "t1"}],
                     additional_kwargs={"provider_data": {"provider": "anthropic",
                                                          "fields": {"thinking_blocks": BLOCKS}}})
    history = [HumanMessage(content="Lyon?"), turn, ToolMessage(content="2.1 million", tool_call_id="t1")]

    anthropic = to_chat_messages(history, "anthropic")[1]
    openai = to_chat_messages(history, "openai")[1]

    assert anthropic["extensions"] == {"provider": "anthropic", "fields": {"thinking_blocks": BLOCKS}}
    assert "extensions" not in openai


def test_a_turn_without_thinking_is_converted_as_before():
    turn = AIMessage(content="plain answer")

    assert to_chat_messages([turn], "anthropic") == [{"role": "assistant", "content": "plain answer"}]
