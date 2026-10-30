"""Typing-only interfaces for the endpoints of the clients.

``LLMSwitch`` exposes its endpoints as nested attributes, such as ``client.chat.completions.create``. The objects
behind those attributes are private classes, with helpers the user never calls. The protocols here describe only
their public members, so an editor shows ``create`` and ``parse`` under ``completions``, and nothing private.

Nothing in this module runs at import time apart from the class definitions. The clients do not check these
protocols at runtime; mypy checks that the implementations match them.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

if TYPE_CHECKING:
    from ..models import (
        ChatCompletion,
        ChatCompletionChunk,
        CompactedResponse,
        Conversation,
        ConversationDeleted,
        CreateEmbeddingResponse,
        InputTokenCountResponse,
        Response,
        ResponseDeleted,
        ResponseItem,
        ResponseItemList,
        ResponseStreamEvent,
    )
    from .responses_capabilities import ResponseCapabilities


# --- Synchronous client ------------------------------------------------------------------------------------


class ChatCompletionsAPI(Protocol):
    """Chat completions: ``client.chat.completions``."""

    @overload
    def create(self, messages: list[dict[str, Any]], stream: Literal[False] = False, truncate: bool = False,
               **kwargs: Any) -> ChatCompletion: ...

    @overload
    def create(self, messages: list[dict[str, Any]], stream: Literal[True], truncate: bool = False,
               **kwargs: Any) -> Iterator[ChatCompletionChunk]: ...

    def create(self, messages: list[dict[str, Any]], stream: bool = False, truncate: bool = False,
               **kwargs: Any) -> Any: ...

    def parse(self, messages: list[dict[str, Any]], response_format: Any, truncate: bool = False,
              **kwargs: Any) -> ChatCompletion: ...


class ChatAPI(Protocol):
    """Chat namespace: ``client.chat``."""

    @property
    def completions(self) -> ChatCompletionsAPI: ...


class EmbeddingsAPI(Protocol):
    """Embeddings: ``client.embeddings``."""

    def create(self, input: str | list[str], batching: bool = False, batchsize: int = 32,
               **kwargs: Any) -> CreateEmbeddingResponse: ...


class ResponseInputItemsAPI(Protocol):
    """The input items of a stored response: ``client.responses.input_items``."""

    def list(self, response_id: str, **kwargs: Any) -> ResponseItemList: ...


class ResponseInputTokensAPI(Protocol):
    """Token counting for Responses requests: ``client.responses.input_tokens``."""

    def count(self, **kwargs: Any) -> InputTokenCountResponse: ...


class ConversationsAPI(Protocol):
    """Stored conversations: ``client.responses.conversations``."""

    def create(self, items: Any = None, **kwargs: Any) -> Conversation: ...

    def retrieve(self, conversation_id: str, **kwargs: Any) -> Conversation: ...

    def update(self, conversation_id: str, **kwargs: Any) -> Conversation: ...

    def delete(self, conversation_id: str, **kwargs: Any) -> ConversationDeleted: ...

    def list_items(self, conversation_id: str, **kwargs: Any) -> ResponseItemList: ...

    def retrieve_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> ResponseItem: ...

    def delete_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any: ...


class ResponsesAPI(Protocol):
    """The Responses API: ``client.responses``."""

    @property
    def input_items(self) -> ResponseInputItemsAPI: ...

    @property
    def input_tokens(self) -> ResponseInputTokensAPI: ...

    @property
    def conversations(self) -> ConversationsAPI: ...

    def capabilities(self) -> ResponseCapabilities: ...

    @overload
    def create(self, input: Any = None, stream: Literal[False] = False, truncate: bool = False,
               **kwargs: Any) -> Response: ...

    @overload
    def create(self, input: Any = None, *, stream: Literal[True], truncate: bool = False,
               **kwargs: Any) -> Iterator[ResponseStreamEvent]: ...

    def create(self, input: Any = None, stream: bool = False, truncate: bool = False, **kwargs: Any) -> Any: ...

    def stream(self, input: Any = None, truncate: bool = False, **kwargs: Any) -> Iterator[ResponseStreamEvent]: ...

    def create_background(self, input: Any = None, truncate: bool = False, **kwargs: Any) -> Response: ...

    def submit_mcp_approval(self, previous_response_id: str, approval: Any, **kwargs: Any) -> Response: ...

    def wait(self, response_id: str, *, timeout: float = 600.0, poll_interval: float = 1.0,
             max_poll_interval: float = 10.0, cancel_on_timeout: bool = False,
             on_poll: Callable[..., Any] | None = None, **kwargs: Any) -> Response: ...

    def parse(self, input: Any = None, text_format: Any = None, truncate: bool = False,
              **kwargs: Any) -> Response: ...

    def retrieve(self, response_id: str, **kwargs: Any) -> Response: ...

    def delete(self, response_id: str, **kwargs: Any) -> ResponseDeleted: ...

    def cancel(self, response_id: str, **kwargs: Any) -> Response: ...

    def compact(self, **kwargs: Any) -> CompactedResponse: ...


# --- Asynchronous client -----------------------------------------------------------------------------------
# The same members as above. Each call is a coroutine, so ``await`` it to get the result.


class AsyncChatCompletionsAPI(Protocol):
    """Async chat completions: ``client.chat.completions``. Use ``await`` on each call."""

    @overload
    async def create(self, messages: list[dict[str, Any]], stream: Literal[False] = False,
                     truncate: bool = False, **kwargs: Any) -> ChatCompletion: ...

    @overload
    async def create(self, messages: list[dict[str, Any]], stream: Literal[True], truncate: bool = False,
                     **kwargs: Any) -> AsyncIterator[ChatCompletionChunk]: ...

    async def create(self, messages: list[dict[str, Any]], stream: bool = False, truncate: bool = False,
                     **kwargs: Any) -> Any: ...

    async def parse(self, messages: list[dict[str, Any]], response_format: Any, truncate: bool = False,
                    **kwargs: Any) -> ChatCompletion: ...


class AsyncChatAPI(Protocol):
    """Async chat namespace: ``client.chat``."""

    @property
    def completions(self) -> AsyncChatCompletionsAPI: ...


class AsyncEmbeddingsAPI(Protocol):
    """Async embeddings: ``client.embeddings``."""

    async def create(self, input: str | list[str], batching: bool = False, batchsize: int = 32,
                     **kwargs: Any) -> CreateEmbeddingResponse: ...


class AsyncResponseInputItemsAPI(Protocol):
    """The input items of a stored response, async: ``client.responses.input_items``."""

    async def list(self, response_id: str, **kwargs: Any) -> ResponseItemList: ...


class AsyncResponseInputTokensAPI(Protocol):
    """Token counting for Responses requests, async: ``client.responses.input_tokens``."""

    async def count(self, **kwargs: Any) -> InputTokenCountResponse: ...


class AsyncConversationsAPI(Protocol):
    """Stored conversations, async: ``client.responses.conversations``."""

    async def create(self, items: Any = None, **kwargs: Any) -> Conversation: ...

    async def retrieve(self, conversation_id: str, **kwargs: Any) -> Conversation: ...

    async def update(self, conversation_id: str, **kwargs: Any) -> Conversation: ...

    async def delete(self, conversation_id: str, **kwargs: Any) -> ConversationDeleted: ...

    async def list_items(self, conversation_id: str, **kwargs: Any) -> ResponseItemList: ...

    async def retrieve_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> ResponseItem: ...

    async def delete_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any: ...


class AsyncResponsesAPI(Protocol):
    """The Responses API, async: ``client.responses``."""

    @property
    def input_items(self) -> AsyncResponseInputItemsAPI: ...

    @property
    def input_tokens(self) -> AsyncResponseInputTokensAPI: ...

    @property
    def conversations(self) -> AsyncConversationsAPI: ...

    def capabilities(self) -> ResponseCapabilities: ...

    @overload
    async def create(self, input: Any = None, stream: Literal[False] = False, truncate: bool = False,
                     **kwargs: Any) -> Response: ...

    @overload
    async def create(self, input: Any = None, *, stream: Literal[True], truncate: bool = False,
                     **kwargs: Any) -> AsyncIterator[ResponseStreamEvent]: ...

    async def create(self, input: Any = None, stream: bool = False, truncate: bool = False,
                     **kwargs: Any) -> Any: ...

    async def stream(self, input: Any = None, truncate: bool = False,
                     **kwargs: Any) -> AsyncIterator[ResponseStreamEvent]: ...

    async def create_background(self, input: Any = None, truncate: bool = False, **kwargs: Any) -> Response: ...

    async def submit_mcp_approval(self, previous_response_id: str, approval: Any, **kwargs: Any) -> Response: ...

    async def wait(self, response_id: str, *, timeout: float = 600.0, poll_interval: float = 1.0,
                   max_poll_interval: float = 10.0, cancel_on_timeout: bool = False,
                   on_poll: Callable[..., Any] | None = None, **kwargs: Any) -> Response: ...

    async def parse(self, input: Any = None, text_format: Any = None, truncate: bool = False,
                    **kwargs: Any) -> Response: ...

    async def retrieve(self, response_id: str, **kwargs: Any) -> Response: ...

    async def delete(self, response_id: str, **kwargs: Any) -> ResponseDeleted: ...

    async def cancel(self, response_id: str, **kwargs: Any) -> Response: ...

    async def compact(self, **kwargs: Any) -> CompactedResponse: ...
