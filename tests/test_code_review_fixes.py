"""Regression tests for the security, functional, logical, and latency fixes across src/llmswitch."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from llmswitch.agentic.aggregation import JsonMergeAggregator
from llmswitch.agentic.checkpoints import FileCheckpointStore
from llmswitch.agentic.contracts import SharedState
from llmswitch.core.circuit_breaker import CircuitBreaker, provider_call
from llmswitch.core.redaction import redact_sensitive
from llmswitch.core.responses_capabilities import ResponseCreateRequest, _validate_mcp_policy
from llmswitch.core.retry import _is_retryable
from llmswitch.errors import (
    InvalidResponseError,
    InvalidResponseRequestError,
    RateLimitExceededError,
    ResponsesError,
    ToolLoopLimitError,
    UnsupportedCapabilityError,
)
from llmswitch.models import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Choice,
    CompletionUsage,
    Function,
    ResponseStreamEvent,
)
from llmswitch.ratelimit.batching import BatchUtility
from llmswitch.ratelimit.tokenizer import count_tokens, fallback_count_tokens
from llmswitch.structured import (
    deserialize_to_model,
    pydantic_to_response_format,
    pydantic_to_text_format,
)


class _SampleItem(BaseModel):
    name: str
    score: int


def test_tool_loop_limit_error_round_trip() -> None:
    err = ToolLoopLimitError(7)
    data = err.to_dict()
    assert data["context"]["max_steps"] == 7
    restored = ToolLoopLimitError.from_dict(data)
    assert isinstance(restored, ToolLoopLimitError)
    assert restored.max_steps == 7


def test_structured_schema_cache_isolation_and_markdown_extraction() -> None:
    rf1 = pydantic_to_response_format(_SampleItem)
    rf1["json_schema"]["schema"]["mutated"] = True
    rf2 = pydantic_to_response_format(_SampleItem)
    assert "mutated" not in rf2["json_schema"]["schema"]

    tf1 = pydantic_to_text_format(_SampleItem)
    tf1["format"]["schema"]["mutated"] = True
    tf2 = pydantic_to_text_format(_SampleItem)
    assert "mutated" not in tf2["format"]["schema"]

    raw_fenced = "Here is the output:\n```json\n{\"name\": \"alpha\", \"score\": 42}\n```\nDone!"
    parsed = deserialize_to_model(_SampleItem, raw_fenced)
    assert parsed is not None
    assert parsed.name == "alpha"
    assert parsed.score == 42


def test_tokenizer_special_tokens_and_cjk_fallback() -> None:
    # Must not raise ValueError on special tokens like <|endoftext|>
    tokens = count_tokens("Hello <|endoftext|> world", "gpt-4o")
    assert tokens > 0
    # Continuous CJK string must scale with length in fallback_count_tokens
    cjk = "你好世界" * 100
    assert fallback_count_tokens(cjk) >= 100


def test_batching_zero_requests_and_zero_progress() -> None:
    util = BatchUtility()
    assert util.plan_batches([], 10, 100, batching=False) == []
    assert util.plan_batches([], 10, 100, batching=True) == []
    util.print_progress_bar(0, 0)  # must not raise ZeroDivisionError


def test_circuit_breaker_half_open_probe_not_closed_by_429() -> None:
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.0)
    admitted, probe = breaker.acquire()
    assert admitted is True
    breaker.record(ok=False, probe=probe)
    assert breaker.state == "open"

    # A 429 inside provider_call during a half-open probe must abandon the probe (reopen), NOT close the breaker
    with pytest.raises(RateLimitExceededError):
        with provider_call("openai", "gpt-4o", breaker=breaker):
            raise RateLimitExceededError(1)
    assert breaker.state == "open"


def test_mcp_ssrf_protection_and_reserved_extra_body() -> None:
    from types import SimpleNamespace

    config = SimpleNamespace(
        provider="openai",
        mcp_policy={
            "enabled": True,
            "allowed_origins": ["localhost", "127.0.0.1", "169.254.169.254", "https://mcp.example.com:8443/path"],
        },
    )
    for bad_url in (
        "https://localhost/mcp",
        "https://127.0.0.1/mcp",
        "https://169.254.169.254/latest/meta-data",
        "http://mcp.example.com/mcp",
        "https://user:pass@mcp.example.com/mcp",
    ):
        with pytest.raises(InvalidResponseRequestError):
            _validate_mcp_policy(config, {"type": "mcp", "server_label": "s", "server_url": bad_url})

    _validate_mcp_policy(config, {"type": "mcp", "server_label": "s", "server_url": "https://mcp.example.com:8443/mcp"})

    with pytest.raises(ValueError, match="extra_body cannot override"):
        ResponseCreateRequest(input="hi", extra_body={"tools": []})


def test_redaction_covers_pydantic_models_and_server_url() -> None:
    req = ResponseCreateRequest(
        input="hi",
        tools=[{"type": "mcp", "server_label": "s", "server_url": "https://secret.example.com/token"}],
    )
    redacted = redact_sensitive(req)
    assert redacted["tools"][0]["server_url"] == "*REDACTED*"


def test_retry_excludes_non_retryable_responses_and_capability_errors() -> None:
    assert _is_retryable(UnsupportedCapabilityError(feature="mcp", provider="anthropic")) is False
    assert _is_retryable(InvalidResponseError("bad")) is False
    assert _is_retryable(ResponsesError("not implemented", status_code=501)) is False
    assert _is_retryable(ResponsesError("bad gateway", status_code=502)) is True
    assert _is_retryable(RateLimitExceededError(1)) is True


def test_agentic_messages_missing_id_and_dict_event_response() -> None:
    pytest.importorskip("langchain_core")
    from llmswitch.agentic.messages import response_event_to_chunk, to_ai_message

    comp = ChatCompletion(
        id="c1",
        object="chat.completion",
        created=1,
        model="m",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="",
                            type="function",
                            function=Function(name="lookup", arguments='{"q": "x"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    ai = to_ai_message(comp)
    assert ai.tool_calls[0]["id"].startswith("call_")

    ev = ResponseStreamEvent(
        type="response.completed",
        response={"id": "resp_1", "status": "completed", "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}},
    )
    chunk = response_event_to_chunk(ev)
    assert chunk is not None
    assert chunk.response_metadata == {"id": "resp_1", "status": "completed"}
    assert chunk.usage_metadata["total_tokens"] == 7


def test_agent_handles_invalid_tool_calls_without_skipping_to_reflection() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, InvalidToolCall
    from langchain_core.outputs import ChatGeneration, ChatResult

    from llmswitch.agentic.agent import Agent, AgentConfig

    class _DummyModel(BaseChatModel):
        turn: int = 0

        @property
        def _llm_type(self) -> str:
            return "dummy"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.turn += 1
            if self.turn == 1:
                msg = AIMessage(
                    content="",
                    tool_calls=[],
                    invalid_tool_calls=[
                        InvalidToolCall(
                            type="invalid_tool_call",
                            id="call_bad",
                            name="my_tool",
                            args="{not json",
                            error="Expecting property name",
                        )
                    ],
                )
            else:
                msg = AIMessage(content="Recovered after invalid tool call.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    def my_tool(x: int) -> int:
        """Return x."""
        return x

    agent = Agent(model=_DummyModel(), tools=[my_tool], config=AgentConfig(reflection_enabled=False))
    res = agent.invoke("test")
    assert res["status"] == "completed"
    assert res["output"] == "Recovered after invalid tool call."
    assert len(res["steps"]) == 2
    assert res["steps"][0].tool_results[0].error_kind == "parse"


def test_shared_state_o_n_projection_and_json_merge_conflicts(tmp_path) -> None:
    state = SharedState({"a": "x" * 50, "b": "y" * 50, "c": "z" * 50})
    proj = state.project_for_prompt(max_bytes=80)
    assert "a" in proj
    assert len(json.dumps(proj).encode("utf-8")) <= 80

    agg = JsonMergeAggregator(conflict_policy="first_writer_wins")
    res = agg.aggregate(
        [
            {"specialist": "s1", "status": "completed", "output": {"k": 1}},
            {"specialist": "s2", "status": "completed", "output": {"k": 2}},
            {"specialist": "s3", "status": "completed", "output": {"k": 3}},
        ]
    )
    assert res.consensus == {"k": 1}
    assert res.metadata["conflicts"]["k"] == [1, 2, 3]

    # FileCheckpointStore handles non-primitive objects via default=str and corrupt JSON via ValueError
    store = FileCheckpointStore(tmp_path)
    store.save("cp_1", {"phase": "tools_done", "custom": {1, 2}})
    loaded = store.load("cp_1")
    assert loaded is not None
    (tmp_path / "cp_bad.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid checkpoint JSON"):
        store.load("cp_bad")
