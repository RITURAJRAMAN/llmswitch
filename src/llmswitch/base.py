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
