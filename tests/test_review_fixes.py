"""Regression tests for review findings: retries, redaction, error exports and logged drops."""
import logging

import pytest

from llmswitch import APIConnectionError, LLMSwitchError, TimeOutError, ToolLoopLimitError
from llmswitch.core.redaction import redact_sensitive
from llmswitch.core.retry import _is_retryable, _retry_call
from llmswitch.providers.anthropic_provider import _content_to_anthropic_parts


def test_a_timeout_is_not_retried():
    assert _is_retryable(TimeOutError(5)) is False


def test_a_connection_error_is_still_retried():
    assert _is_retryable(APIConnectionError("https://x", provider="openai")) is True


def test_a_timeout_is_raised_after_one_attempt(monkeypatch):
    calls = []

    def flaky():
        calls.append(1)
        raise TimeOutError(5)

    monkeypatch.setattr("llmswitch.core.retry.time.sleep", lambda seconds: None)
    with pytest.raises(TimeOutError):
        _retry_call(flaky, max_retries=3, base_delay=0, max_delay=0, deadline=None)
    assert len(calls) == 1


@pytest.mark.parametrize("header", ["x-api-key", "X-Goog-Api-Key", "proxy-authorization", "cookie"])
def test_credential_headers_are_redacted(header):
    assert redact_sensitive({"extra_headers": {header: "secret-value"}})["extra_headers"][header] == "*REDACTED*"


def test_tool_loop_limit_error_is_a_library_error_exported_from_the_package():
    error = ToolLoopLimitError(3)
    assert isinstance(error, LLMSwitchError)
    assert error.status_code == 400 and error.max_steps == 3
    import llmswitch.errors as errors
    assert errors.ToolLoopLimitError is ToolLoopLimitError


def test_the_langchain_module_re_exports_the_same_error():
    pytest.importorskip("langchain_core")
    from llmswitch.integrations.langchain import ToolLoopLimitError as from_integration
    assert from_integration is ToolLoopLimitError


def test_unsupported_anthropic_parts_are_dropped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="llmswitch.providers.anthropic_provider"):
        parts = _content_to_anthropic_parts([{"type": "text", "text": "hi"},
                                             {"type": "file", "file": {"file_id": "f1"}}])
    assert parts == [{"type": "text", "text": "hi"}]
    assert any("dropping unsupported content part of type file" in r.getMessage() for r in caplog.records)
