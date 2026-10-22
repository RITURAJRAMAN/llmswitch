"""
providers._responses_base

Shared Responses API provider implementations for the OpenAI and Azure OpenAI
SDKs, which expose an identical ``client.responses.*`` surface. Concrete
providers subclass these mixins and only supply the constructed client
(``self.client``), a default model, and an endpoint for error reporting.

Covered operations: ``create`` (streaming and non-streaming), ``parse``,
``retrieve``, ``delete``, ``cancel``, ``compact``, ``input_items.list`` and
``input_tokens.count``.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any, Optional

from ..base import BaseResponsesProvider
from ..errors import ResponsesError
from ..models import (
    CompactedResponse,
    Conversation,
    ConversationDeleted,
    InputTokenCountResponse,
    Response,
    ResponseDeleted,
    ResponseItemList,
    ResponseStreamEvent,
)
from ..structured import (
    is_pydantic_model_class,
    populate_response_parsed,
    pydantic_to_text_format,
)
from ._shared import (
    close_async_stream,
    close_stream,
    map_compacted_response,
    map_conversation,
    map_conversation_deleted,
    map_input_token_count,
    map_response,
    map_response_deleted,
    map_response_item,
    map_response_item_page,
    map_response_stream_event,
    translate_openai_error,
)

logger = logging.getLogger(__name__)


def _extract_text_format(kwargs: dict[str, Any]) -> Optional[type]:
    """
    Pop a Pydantic model class from ``text_format`` / ``response_format`` / ``text``.

    ``text_format`` is the native Responses spelling; the other two are accepted
    so callers can reuse their Chat Completions structured-output code unchanged.
    """
    for key in ("text_format", "response_format", "text"):
        value = kwargs.get(key)
        if is_pydantic_model_class(value):
            kwargs.pop(key)
            return value
    return None


class _ResponsesProviderBase(BaseResponsesProvider):
    """State and request-shaping shared by the sync and async Responses providers."""

    client: Any
    model: Optional[str]
    endpoint: Optional[str]
    timeout: Optional[float]
    provider_name: str = "openai"

    def _build_kwargs(self, input: Any, kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: A002
        """Assemble SDK kwargs, defaulting the model and omitting an absent input."""
        payload = dict(kwargs)
        payload["model"] = payload.pop("model", None) or self.model
        if input is not None:
            payload["input"] = input
        return payload

    def _fail(self, exc: Exception, operation: str, resource_id: Optional[str] = None) -> Exception:
        logger.error("%s responses.%s error | Exception: %s", self.provider_name, operation, exc)
        return translate_openai_error(exc, self.endpoint, self.provider_name, ResponsesError,
                                      operation=operation, timeout_seconds=getattr(self, "timeout", None),
                                      resource_id=resource_id)


class SyncResponsesProviderMixin(_ResponsesProviderBase):
    """Synchronous ``client.responses.*`` implementation."""

    def create(self, input: Any = None, stream: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        text_format = _extract_text_format(kwargs)
        if stream:
            # Streaming has no native .parse() helper — send the schema instead.
            if text_format is not None:
                kwargs["text"] = pydantic_to_text_format(text_format)
            return self._streaming_create(input, **kwargs)
        return self._non_streaming_create(input, text_format, **kwargs)

    def _non_streaming_create(self, input: Any, text_format: Optional[type], **kwargs: Any) -> Response:  # noqa: A002
        try:
            payload = self._build_kwargs(input, kwargs)
            if text_format is not None:
                raw = self.client.responses.parse(text_format=text_format, **payload)
            else:
                raw = self.client.responses.create(stream=False, **payload)
            mapped = map_response(raw, provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "create") from e
        if text_format is not None:
            populate_response_parsed(mapped, text_format)
        return mapped

    def _streaming_create(self, input: Any, **kwargs: Any) -> Generator[ResponseStreamEvent, None, None]:  # noqa: A002
        stream = None
        try:
            stream = self.client.responses.create(stream=True, **self._build_kwargs(input, kwargs))
            for event in stream:
                yield map_response_stream_event(event, provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "create") from e
        finally:
            if stream is not None:
                close_stream(stream)

    def retrieve(self, response_id: str, **kwargs: Any) -> Response:
        try:
            return map_response(self.client.responses.retrieve(response_id, **kwargs), provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "retrieve", response_id) from e

    def delete(self, response_id: str, **kwargs: Any) -> ResponseDeleted:
        try:
            return map_response_deleted(
                self.client.responses.delete(response_id, **kwargs),
                response_id,
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "delete", response_id) from e

    def cancel(self, response_id: str, **kwargs: Any) -> Response:
        try:
            return map_response(self.client.responses.cancel(response_id, **kwargs), provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "cancel", response_id) from e

    def compact(self, **kwargs: Any) -> CompactedResponse:
        try:
            return map_compacted_response(
                self.client.responses.compact(**self._build_kwargs(None, kwargs)),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "compact") from e

    def list_input_items(self, response_id: str, **kwargs: Any) -> ResponseItemList:
        try:
            return map_response_item_page(
                self.client.responses.input_items.list(response_id, **kwargs),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "input_items.list", response_id) from e

    def count_input_tokens(self, **kwargs: Any) -> InputTokenCountResponse:
        try:
            return map_input_token_count(
                self.client.responses.input_tokens.count(**self._build_kwargs(None, kwargs)),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "input_tokens.count") from e

    def create_conversation(self, items: Any = None, **kwargs: Any) -> Conversation:
        try:
            payload = dict(kwargs)
            if items is not None:
                payload["items"] = items
            return map_conversation(self.client.conversations.create(**payload), provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "conversations.create") from e

    def retrieve_conversation(self, conversation_id: str, **kwargs: Any) -> Conversation:
        try:
            return map_conversation(
                self.client.conversations.retrieve(conversation_id, **kwargs),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "conversations.retrieve", conversation_id) from e

    def update_conversation(self, conversation_id: str, **kwargs: Any) -> Conversation:
        try:
            return map_conversation(
                self.client.conversations.update(conversation_id, **kwargs),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "conversations.update", conversation_id) from e

    def delete_conversation(self, conversation_id: str, **kwargs: Any) -> ConversationDeleted:
        try:
            return map_conversation_deleted(
                self.client.conversations.delete(conversation_id, **kwargs),
                conversation_id,
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "conversations.delete", conversation_id) from e

    def list_conversation_items(self, conversation_id: str, **kwargs: Any) -> ResponseItemList:
        try:
            return map_response_item_page(
                self.client.conversations.items.list(conversation_id, **kwargs),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "conversations.items.list", conversation_id) from e

    def retrieve_conversation_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        try:
            return map_response_item(
                self.client.conversations.items.retrieve(item_id, conversation_id=conversation_id, **kwargs),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "conversations.items.retrieve", conversation_id) from e

    def delete_conversation_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Conversation:
        try:
            return map_conversation(
                self.client.conversations.items.delete(item_id, conversation_id=conversation_id, **kwargs),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "conversations.items.delete", conversation_id) from e


class AsyncResponsesProviderMixin(_ResponsesProviderBase):
    """Asynchronous ``client.responses.*`` implementation."""

    async def create(self, input: Any = None, stream: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        text_format = _extract_text_format(kwargs)
        if stream:
            if text_format is not None:
                kwargs["text"] = pydantic_to_text_format(text_format)
            return self._streaming_create(input, **kwargs)
        return await self._non_streaming_create(input, text_format, **kwargs)

    async def _non_streaming_create(self, input: Any, text_format: Optional[type],  # noqa: A002
                                    **kwargs: Any) -> Response:
        try:
            payload = self._build_kwargs(input, kwargs)
            if text_format is not None:
                raw = await self.client.responses.parse(text_format=text_format, **payload)
            else:
                raw = await self.client.responses.create(stream=False, **payload)
            mapped = map_response(raw, provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "create") from e
        if text_format is not None:
            populate_response_parsed(mapped, text_format)
        return mapped

    async def _streaming_create(self, input: Any, **kwargs: Any) -> AsyncGenerator[  # noqa: A002
            ResponseStreamEvent, None]:
        stream = None
        try:
            stream = await self.client.responses.create(stream=True, **self._build_kwargs(input, kwargs))
            async for event in stream:
                yield map_response_stream_event(event, provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "create") from e
        finally:
            if stream is not None:
                await close_async_stream(stream)

    async def retrieve(self, response_id: str, **kwargs: Any) -> Response:
        try:
            return map_response(await self.client.responses.retrieve(response_id, **kwargs), provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "retrieve", response_id) from e

    async def delete(self, response_id: str, **kwargs: Any) -> ResponseDeleted:
        try:
            return map_response_deleted(
                await self.client.responses.delete(response_id, **kwargs),
                response_id,
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "delete", response_id) from e

    async def cancel(self, response_id: str, **kwargs: Any) -> Response:
        try:
            return map_response(await self.client.responses.cancel(response_id, **kwargs), provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "cancel", response_id) from e

    async def compact(self, **kwargs: Any) -> CompactedResponse:
        try:
            return map_compacted_response(
                await self.client.responses.compact(**self._build_kwargs(None, kwargs)),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "compact") from e

    async def list_input_items(self, response_id: str, **kwargs: Any) -> ResponseItemList:
        try:
            return map_response_item_page(
                await self.client.responses.input_items.list(response_id, **kwargs),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "input_items.list", response_id) from e

    async def count_input_tokens(self, **kwargs: Any) -> InputTokenCountResponse:
        try:
            page = await self.client.responses.input_tokens.count(**self._build_kwargs(None, kwargs))
            return map_input_token_count(page, provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "input_tokens.count") from e

    async def create_conversation(self, items: Any = None, **kwargs: Any) -> Conversation:
        try:
            payload = dict(kwargs)
            if items is not None:
                payload["items"] = items
            return map_conversation(await self.client.conversations.create(**payload), provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "conversations.create") from e

    async def retrieve_conversation(self, conversation_id: str, **kwargs: Any) -> Conversation:
        try:
            return map_conversation(
                await self.client.conversations.retrieve(conversation_id, **kwargs),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "conversations.retrieve", conversation_id) from e

    async def update_conversation(self, conversation_id: str, **kwargs: Any) -> Conversation:
        try:
            return map_conversation(
                await self.client.conversations.update(conversation_id, **kwargs),
                provider=self.provider_name,
            )
        except Exception as e:
            raise self._fail(e, "conversations.update", conversation_id) from e

    async def delete_conversation(self, conversation_id: str, **kwargs: Any) -> ConversationDeleted:
        try:
            deleted = await self.client.conversations.delete(conversation_id, **kwargs)
            return map_conversation_deleted(deleted, conversation_id, provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "conversations.delete", conversation_id) from e

    async def list_conversation_items(self, conversation_id: str, **kwargs: Any) -> ResponseItemList:
        try:
            page = await self.client.conversations.items.list(conversation_id, **kwargs)
            return map_response_item_page(page, provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "conversations.items.list", conversation_id) from e

    async def retrieve_conversation_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        try:
            item = await self.client.conversations.items.retrieve(item_id, conversation_id=conversation_id, **kwargs)
            return map_response_item(item, provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "conversations.items.retrieve", conversation_id) from e

    async def delete_conversation_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Conversation:
        try:
            conversation = await self.client.conversations.items.delete(item_id, conversation_id=conversation_id, **kwargs)
            return map_conversation(conversation, provider=self.provider_name)
        except Exception as e:
            raise self._fail(e, "conversations.items.delete", conversation_id) from e
