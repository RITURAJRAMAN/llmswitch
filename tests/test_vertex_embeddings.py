"""Vertex AI embeddings: model, option and output handling, with the Google client replaced by a fake."""
import asyncio
from types import SimpleNamespace

import pytest

from llmswitch import AsyncLLMSwitch, LLMSwitch, LLMSwitchError
from llmswitch.providers import vertexai_provider


class FakeGenaiClient:
    created = []
    embed_calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeGenaiClient.created.append(kwargs)
        self.models = SimpleNamespace(embed_content=self.embed_content)

    @staticmethod
    def embed_content(*, model, contents, config=None):
        FakeGenaiClient.embed_calls.append({"model": model, "contents": list(contents), "config": config})
        vectors = [SimpleNamespace(values=[float(index), 0.5], statistics=SimpleNamespace(token_count=3))
                   for index, _ in enumerate(contents)]
        return SimpleNamespace(embeddings=vectors)


@pytest.fixture(autouse=True)
def fake_google(monkeypatch):
    FakeGenaiClient.created = []
    FakeGenaiClient.embed_calls = []
    monkeypatch.setattr(vertexai_provider, "load_credentials_from_file", lambda path, scopes: ("creds", "my-project"))
    monkeypatch.setattr(vertexai_provider.genai, "Client", FakeGenaiClient)


def vertex(**overrides):
    options = dict(provider="vertexai", location="global", credentials_path="service-account.json",
                   model="gemini-2.5-pro")
    options.update(overrides)
    return LLMSwitch(**options)


def test_embeddings_use_the_configured_location_and_the_requested_model():
    with vertex() as client:
        reply = client.embeddings.create(["alpha", "beta"], model="gemini-embedding-001")

    assert FakeGenaiClient.created[-1]["location"] == "global"
    assert FakeGenaiClient.created[-1]["project"] == "my-project"
    assert FakeGenaiClient.embed_calls[0]["model"] == "gemini-embedding-001"
    assert FakeGenaiClient.embed_calls[0]["contents"] == ["alpha", "beta"]
    assert [item.index for item in reply.data] == [0, 1]
    assert reply.usage.prompt_tokens == 6 and reply.usage.total_tokens == 6
    assert reply.model == "gemini-embedding-001"


def test_output_dimensionality_is_sent_in_the_embedding_config():
    with vertex() as client:
        client.embeddings.create(["alpha"], model="gemini-embedding-001", output_dimensionality=256)

    assert FakeGenaiClient.embed_calls[0]["config"].output_dimensionality == 256


def test_an_unsupported_embedding_option_is_refused_before_any_call():
    with vertex() as client:
        with pytest.raises(LLMSwitchError, match="Unsupported embedding options for Vertex AI: dimensions"):
            client.embeddings.create(["alpha"], model="gemini-embedding-001", dimensions=3)

    assert FakeGenaiClient.embed_calls == []


def test_embeddings_need_a_model():
    with vertex(model=None) as client:
        with pytest.raises(LLMSwitchError, match="need a model"):
            client.embeddings.create(["alpha"])


def test_the_async_client_returns_the_same_vectors():
    async def go():
        async with AsyncLLMSwitch(provider="vertexai", location="global",
                                  credentials_path="service-account.json") as client:
            return await client.embeddings.create(["alpha", "beta"], model="gemini-embedding-001")

    reply = asyncio.run(go())
    assert len(reply.data) == 2 and reply.data[1].embedding == [1.0, 0.5]
