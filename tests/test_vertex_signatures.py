"""Gemini function-call thought signatures: read from the response and sent back with the function call."""
import base64
from types import SimpleNamespace

from llmswitch.models import ProviderExtensions
from llmswitch.providers.vertexai_provider import _replayed_thought_signatures, _thought_signatures


def candidate(*parts):
    return SimpleNamespace(content=SimpleNamespace(parts=list(parts)))


def test_each_function_call_signature_is_read_as_base64_in_order():
    raw = b"\x01\x02signature"
    found = _thought_signatures(candidate(
        SimpleNamespace(text="thinking out loud", function_call=None, thought_signature=None),
        SimpleNamespace(function_call=SimpleNamespace(name="a"), thought_signature=raw),
        SimpleNamespace(function_call=SimpleNamespace(name="b"), thought_signature=None),
    ))

    assert found == [base64.b64encode(raw).decode("ascii"), None]


def test_a_candidate_without_signatures_reads_as_none_per_call():
    assert _thought_signatures(candidate(SimpleNamespace(function_call=SimpleNamespace(name="a"),
                                                         thought_signature=None))) == [None]


def test_only_vertex_extensions_are_replayed():
    fields = {"thought_signatures": {"call_0": "c2ln"}}

    assert _replayed_thought_signatures({"extensions": ProviderExtensions(provider="vertexai", fields=fields)}) == {
        "call_0": "c2ln"}
    assert _replayed_thought_signatures({"extensions": {"provider": "vertexai", "fields": fields}}) == {"call_0": "c2ln"}
    assert _replayed_thought_signatures({"extensions": {"provider": "anthropic", "fields": fields}}) == {}
    assert _replayed_thought_signatures({}) == {}
