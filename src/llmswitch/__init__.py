"""
llmswitch

One client interface for chat completions and embeddings across OpenAI, Azure
OpenAI, Azure AI Foundry, Anthropic and Vertex AI.

    from llmswitch import LLMSwitch
"""
import logging

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .core import AsyncLLMSwitch, LLMSwitch  # noqa: E402
from .errors import (  # noqa: E402
    APIConnectionError,
    AuthenticationError,
    ChatCompletionError,
    ConfigurationError,
    ContentFilterError,
    EmbeddingGenerationError,
    InvalidResponseError,
    LLMSwitchError,
    ModelNotFoundError,
    RateLimitExceededError,
    TimeOutError,
)
from .models import ProviderExtensions  # noqa: E402

__all__ = [
    "__version__",
    "LLMSwitch",
    "AsyncLLMSwitch",
    "LLMSwitchError",
    "ConfigurationError",
    "ModelNotFoundError",
    "APIConnectionError",
    "AuthenticationError",
    "RateLimitExceededError",
    "ChatCompletionError",
    "EmbeddingGenerationError",
    "InvalidResponseError",
    "ContentFilterError",
    "TimeOutError",
    "ProviderExtensions",
]
