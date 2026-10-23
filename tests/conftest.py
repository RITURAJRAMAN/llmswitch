import pytest

from llmswitch.core.registry import (
    ASYNC_CHAT_PROVIDER_REGISTRY,
    CHAT_PROVIDER_REGISTRY,
    EMBEDDING_PROVIDER_REGISTRY,
)
from llmswitch.models import CreateEmbeddingResponse, Embedding, Usage


class StubChat:
    """Records the last call and returns a canned value. Used in place of a real SDK provider."""

    calls: list = []
    error: Exception | None = None
    closed: bool = False

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def create(self, messages, stream=False, **kwargs):
        StubChat.calls.append({"messages": messages, "stream": stream, **kwargs})
        if StubChat.error is not None:
            raise StubChat.error
        if stream:
            return iter([{"ok": True, "stream": True}])
        return {"ok": True, "stream": stream}

    def parse(self, messages, response_format, **kwargs):
        return self.create(messages, stream=False, response_format=response_format, **kwargs)

    def close(self):
        StubChat.closed = True


class StubAsyncChat(StubChat):
    async def create(self, messages, stream=False, **kwargs):
        StubChat.calls.append({"messages": messages, "stream": stream, **kwargs})
        if StubChat.error is not None:
            raise StubChat.error
        if stream:
            return _agen()
        return {"ok": True, "stream": False}

    async def aclose(self):
        StubChat.closed = True


class StubEmbeddings:
    calls: list = []
    error: Exception | None = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def create(self, input, **kwargs):  # noqa: A002
        StubEmbeddings.calls.append({"input": list(input), **kwargs})
        if StubEmbeddings.error is not None:
            raise StubEmbeddings.error
        return CreateEmbeddingResponse(
            data=[Embedding(embedding=[float(len(text))], index=i, object="embedding") for i, text in enumerate(input)],
            model=kwargs.get("model") or "stub",
            object="list",
            usage=Usage(prompt_tokens=0, total_tokens=0),
            created=0,
        )


async def _agen():
    yield {"chunk": 1}
    yield {"chunk": 2}


@pytest.fixture
def stubs(monkeypatch):
    """Route every provider name to the stub classes and reset recorded state."""
    StubChat.calls, StubChat.error, StubChat.closed = [], None, False
    StubEmbeddings.calls, StubEmbeddings.error = [], None
    for name in ("openai", "azure_openai", "azure_foundry", "anthropic"):
        monkeypatch.setitem(ASYNC_CHAT_PROVIDER_REGISTRY, name, StubAsyncChat)
    for name in ("openai", "azure_openai", "azure_foundry", "anthropic", "vertexai"):
        monkeypatch.setitem(CHAT_PROVIDER_REGISTRY, name, StubChat)
    for name in ("openai", "azure_openai", "azure_foundry"):
        monkeypatch.setitem(EMBEDDING_PROVIDER_REGISTRY, name, StubEmbeddings)
    return StubChat, StubEmbeddings


def provider_for(client, provider_cls):
    """Return the cached provider instance for ``provider_cls`` from the namespace that uses it."""
    name = provider_cls.__name__
    if "Responses" in name:
        namespace = client.responses
    elif "Embedding" in name:
        namespace = client.embeddings
    else:
        namespace = client.chat.completions
    return namespace._instantiate_provider(client.config, provider_cls)
