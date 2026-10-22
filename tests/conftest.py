import pytest

from llmswitch.core.registry import (
    ASYNC_CHAT_PROVIDER_REGISTRY,
    CHAT_PROVIDER_REGISTRY,
    EMBEDDING_PROVIDER_REGISTRY,
)


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
        return {"data": [[float(len(text))] for text in input], "model": kwargs.get("model")}


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
