from abc import ABC, abstractmethod


class BaseChatProvider(ABC):
    """
    Abstract base class for chat providers.
    """

    @abstractmethod
    def create(self, messages, **kwargs):
        """
        Create a chat completion.

        Args:
            messages: List of message dicts or objects.
            **kwargs: Additional parameters for provider.

        Returns:
            ChatCompletion or generator of ChatCompletionChunk.
        """


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """

    @abstractmethod
    def create(self, inputs, **kwargs):
        """
        Create embeddings for the given inputs.

        Args:
            inputs: List of input strings or objects.
            **kwargs: Additional parameters for provider.

        Returns:
            Embedding response object.
        """


class BaseResponsesProvider(ABC):
    """
    Abstract base class for Responses API providers.

    Only ``create`` is required; the remaining verbs raise
    :class:`~llmswitch.errors.ResponsesError` by default so a provider can
    implement a partial surface without breaking the client.
    """

    @abstractmethod
    def create(self, input, **kwargs):  # noqa: A002
        """
        Create a model response.

        Args:
            input: A string or list of input items.
            **kwargs: Additional parameters for provider (``stream``, ``tools``, …).

        Returns:
            Response, or a generator of ResponseStreamEvent when streaming.
        """

    def _unsupported(self, operation: str):
        from .errors import ResponsesError
        raise ResponsesError(
            f"{type(self).__name__} does not support responses.{operation}",
            status_code=501,
        )

    def retrieve(self, response_id, **kwargs):
        """Retrieve a previously created response by id."""
        return self._unsupported("retrieve")

    def delete(self, response_id, **kwargs):
        """Delete a stored response by id."""
        return self._unsupported("delete")

    def cancel(self, response_id, **kwargs):
        """Cancel an in-progress background response."""
        return self._unsupported("cancel")

    def compact(self, **kwargs):
        """Compact a response's conversation history."""
        return self._unsupported("compact")

    def list_input_items(self, response_id, **kwargs):
        """List the input items of a response."""
        return self._unsupported("input_items.list")

    def count_input_tokens(self, **kwargs):
        """Count the input tokens a request would consume."""
        return self._unsupported("input_tokens.count")

    def create_conversation(self, items=None, **kwargs):
        return self._unsupported("conversations.create")

    def retrieve_conversation(self, conversation_id, **kwargs):
        return self._unsupported("conversations.retrieve")

    def update_conversation(self, conversation_id, **kwargs):
        return self._unsupported("conversations.update")

    def delete_conversation(self, conversation_id, **kwargs):
        return self._unsupported("conversations.delete")

    def list_conversation_items(self, conversation_id, **kwargs):
        return self._unsupported("conversations.items.list")

    def retrieve_conversation_item(self, conversation_id, item_id, **kwargs):
        return self._unsupported("conversations.items.retrieve")

    def delete_conversation_item(self, conversation_id, item_id, **kwargs):
        return self._unsupported("conversations.items.delete")
