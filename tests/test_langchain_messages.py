"""LangChain message conversion: pure functions, no network. Skipped when langchain-core is not installed."""
import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

from llmswitch.integrations.langchain_messages import to_ai_chunk, to_ai_message, to_chat_messages  # noqa: E402
from llmswitch.models import (  # noqa: E402
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkDelta,
    CompletionUsage,
    Function,
)


def completion(content=None, tool_calls=None, finish="stop", usage=None):
    return ChatCompletion(
        id="c-1", created=1, model="m-1", usage=usage,
        choices=[Choice(index=0, finish_reason=finish,
                        message=ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls))])


def call(call_id, name, arguments):
    return ChatCompletionMessageToolCall(id=call_id, type="function", function=Function(name=name, arguments=arguments))


# --- LangChain messages -> chat dicts -------------------------------------------------------------

def test_plain_text_messages_keep_their_roles():
    converted = to_chat_messages([SystemMessage(content="Be brief."), HumanMessage(content="Hi")])
    assert converted == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


def test_a_user_message_with_an_image_keeps_its_parts():
    message = HumanMessage(content=[
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://x/cat.png", "detail": "low"}},
    ])
    assert to_chat_messages([message])[0]["content"] == [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://x/cat.png"}},
    ]


def test_the_langchain_standard_image_shapes_are_accepted():
    url = HumanMessage(content=[{"type": "image", "url": "https://x/a.png"}])
    data = HumanMessage(content=[{"type": "image", "base64": "QUJD", "mime_type": "image/jpeg"}])
    assert to_chat_messages([url])[0]["content"][0]["image_url"]["url"] == "https://x/a.png"
    assert to_chat_messages([data])[0]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_text_only_block_lists_are_joined_for_non_user_roles():
    message = SystemMessage(content=[{"type": "text", "text": "Rule one. "}, {"type": "text", "text": "Rule two."}])
    assert to_chat_messages([message]) == [{"role": "system", "content": "Rule one. Rule two."}]


def test_an_ai_message_with_tool_calls_becomes_a_function_call_message():
    message = AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}])
    assert to_chat_messages([message]) == [{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}],
    }]


def test_an_invalid_tool_call_is_sent_back_as_it_was_written():
    message = AIMessage(content="", invalid_tool_calls=[{"name": "f", "args": '{"x":', "id": "c9", "error": "bad"}])
    assert to_chat_messages([message])[0]["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"x":'}


def test_a_tool_message_carries_its_call_id():
    converted = to_chat_messages([ToolMessage(content="Sunny", tool_call_id="call_1")])
    assert converted == [{"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}]


def test_a_full_tool_round_trip_keeps_the_order():
    messages = [
        HumanMessage(content="Weather?"),
        AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}]),
        ToolMessage(content="Sunny", tool_call_id="c1"),
    ]
    assert [m["role"] for m in to_chat_messages(messages)] == ["user", "assistant", "tool"]


def test_an_unsupported_block_is_rejected_with_its_type():
    with pytest.raises(ValueError, match="video"):
        to_chat_messages([HumanMessage(content=[{"type": "video", "url": "x"}])])


def test_an_unsupported_image_block_is_rejected():
    with pytest.raises(ValueError, match="Unsupported image"):
        to_chat_messages([HumanMessage(content=[{"type": "image"}])])


def test_an_unsupported_message_type_is_rejected():
    from langchain_core.messages import FunctionMessage
    with pytest.raises(ValueError, match="FunctionMessage"):
        to_chat_messages([FunctionMessage(content="x", name="f")])


# --- completion -> AIMessage ----------------------------------------------------------------------

def test_a_text_completion_becomes_an_ai_message_with_usage_and_metadata():
    usage = CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    result = to_ai_message(completion(content="hello", usage=usage))
    assert isinstance(result, AIMessage)
    assert result.content == "hello"
    assert result.usage_metadata == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
    assert result.response_metadata == {"model_name": "m-1", "finish_reason": "stop", "id": "c-1"}


def test_cached_and_reasoning_tokens_go_into_the_detail_fields():
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14, thinking_tokens=2,
                            prompt_tokens_details={"cached_tokens": 6})
    metadata = to_ai_message(completion(content="x", usage=usage)).usage_metadata
    assert metadata["input_token_details"] == {"cache_read": 6}
    assert metadata["output_token_details"] == {"reasoning": 2}


def test_a_missing_usage_gives_no_usage_metadata():
    assert to_ai_message(completion(content="x")).usage_metadata is None


def test_tool_calls_are_parsed_into_args():
    result = to_ai_message(completion(tool_calls=[call("c1", "get_weather", '{"city": "Paris"}')], finish="tool_calls"))
    assert result.content == ""
    assert result.tool_calls == [{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1", "type": "tool_call"}]
    assert result.invalid_tool_calls == []


def test_malformed_arguments_become_invalid_tool_calls_with_the_error():
    result = to_ai_message(completion(tool_calls=[call("c2", "get_weather", '{"city":')]))
    assert result.tool_calls == []
    invalid = result.invalid_tool_calls[0]
    assert (invalid["name"], invalid["id"], invalid["args"]) == ("get_weather", "c2", '{"city":')
    assert invalid["error"]


def test_arguments_that_are_not_an_object_are_invalid():
    result = to_ai_message(completion(tool_calls=[call("c3", "f", "[1, 2]")]))
    assert result.invalid_tool_calls[0]["error"] == "tool arguments must be a JSON object"


def test_empty_arguments_mean_no_arguments():
    result = to_ai_message(completion(tool_calls=[call("c4", "ping", "")]))
    assert result.tool_calls[0]["args"] == {}


# --- chunk -> AIMessageChunk -----------------------------------------------------------------------

def chunk(content=None, tool_calls=None, finish=None, usage=None, choices=True):
    delta = ChoiceDelta(role="assistant", content=content, tool_calls=tool_calls)
    return ChatCompletionChunk(id="c-1", created=1, model="m-1", usage=usage,
                               choices=[ChunkDelta(index=0, delta=delta, finish_reason=finish)] if choices else [])


def test_a_text_chunk_becomes_an_ai_message_chunk():
    result = to_ai_chunk(chunk(content="Hel"))
    assert isinstance(result, AIMessageChunk)
    assert result.content == "Hel"


def test_tool_call_fragments_become_tool_call_chunks_that_join_up():
    first = ChoiceDeltaToolCall(index=0, id="c1", type="function",
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"city"'))
    second = ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments=': "Paris"}'))
    joined = to_ai_chunk(chunk(tool_calls=[first])) + to_ai_chunk(chunk(tool_calls=[second]))
    assert joined.tool_calls == [{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1", "type": "tool_call"}]


def test_the_closing_usage_chunk_carries_usage_only():
    usage = CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    result = to_ai_chunk(chunk(usage=usage, choices=False))
    assert result.content == ""
    assert result.usage_metadata["total_tokens"] == 8


def test_the_finish_reason_is_kept_on_the_final_chunk():
    result = to_ai_chunk(chunk(content="", finish="stop"))
    assert result.response_metadata["finish_reason"] == "stop"


def test_an_image_in_a_system_message_is_a_clear_error_not_a_crash():
    with pytest.raises(ValueError, match="only supported in user messages, not system"):
        to_chat_messages([SystemMessage(content=[{"type": "image_url", "image_url": {"url": "https://x/a.png"}}])])


def test_an_image_in_an_assistant_message_is_a_clear_error():
    with pytest.raises(ValueError, match="not assistant messages"):
        to_chat_messages([AIMessage(content=[{"type": "image", "url": "https://x/a.png"}])])


def test_provider_reasoning_blocks_are_not_replayed():
    message = AIMessage(content=[
        {"type": "thinking", "thinking": "let me think", "signature": "sig"},
        {"type": "text", "text": "Answer."},
        {"type": "tool_use", "id": "c1", "name": "f", "input": {}},
    ], tool_calls=[{"name": "f", "args": {}, "id": "c1", "type": "tool_call"}])
    converted = to_chat_messages([message])[0]
    assert converted["content"] == "Answer."
    assert [c["id"] for c in converted["tool_calls"]] == ["c1"]


def test_a_redacted_thinking_block_is_skipped_too():
    message = AIMessage(content=[{"type": "redacted_thinking", "data": "x"}, {"type": "text", "text": "Hi"}])
    assert to_chat_messages([message])[0]["content"] == "Hi"
