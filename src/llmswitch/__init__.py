"""
llmswitch

One client interface for chat completions, embeddings and the Responses API across
OpenAI, Azure OpenAI, Azure AI Foundry, Anthropic and Vertex AI.

    from llmswitch import LLMSwitch
"""
import logging

__version__ = "0.1.3"

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .core import AsyncLLMSwitch, LLMSwitch  # noqa: E402
from .core.responses_capabilities import (  # noqa: E402
    CapabilityStatus,
    ResponseCapabilities,
    ResponseCreateRequest,
)
from .errors import (  # noqa: E402
    APIConnectionError,
    AuthenticationError,
    ChatCompletionError,
    ConfigurationError,
    ContentFilterError,
    ContextLengthExceededError,
    EmbeddingGenerationError,
    InvalidResponseError,
    InvalidResponseRequestError,
    LLMSwitchError,
    ModelNotFoundError,
    RateLimitBackendUnavailableError,
    RateLimitExceededError,
    ResponseNotFoundError,
    ResponsesError,
    TimeOutError,
    UnsupportedCapabilityError,
)
from .models import (  # noqa: E402
    CompactedResponse,
    Conversation,
    ConversationDeleted,
    InputTokenCountResponse,
    ProviderExtensions,
    Response,
    ResponseCompletedEvent,
    ResponseDeleted,
    ResponseFunctionCallItem,
    ResponseItem,
    ResponseItemList,
    ResponseMessageItem,
    ResponseOutputItem,
    ResponseOutputTextDeltaEvent,
    ResponseReasoningItem,
    ResponseStreamEvent,
    ResponseUsage,
)
from .responses_tools import (  # noqa: E402
    CodeInterpreterTool,
    ComputerTool,
    ConnectorTool,
    FileSearchTool,
    ImageGenerationTool,
    McpApprovalDecision,
    McpTool,
    WebSearchTool,
)

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
    "ContextLengthExceededError",
    "RateLimitBackendUnavailableError",
    "TimeOutError",
    "ResponsesError",
    "ResponseNotFoundError",
    "InvalidResponseRequestError",
    "UnsupportedCapabilityError",
    "CapabilityStatus",
    "ResponseCapabilities",
    "ResponseCreateRequest",
    "Response",
    "ResponseOutputItem",
    "ResponseMessageItem",
    "ResponseFunctionCallItem",
    "ResponseReasoningItem",
    "ResponseItem",
    "ResponseItemList",
    "ResponseUsage",
    "ResponseStreamEvent",
    "ResponseCompletedEvent",
    "ResponseOutputTextDeltaEvent",
    "ResponseDeleted",
    "CompactedResponse",
    "Conversation",
    "ConversationDeleted",
    "InputTokenCountResponse",
    "WebSearchTool",
    "FileSearchTool",
    "CodeInterpreterTool",
    "ImageGenerationTool",
    "ComputerTool",
    "McpTool",
    "ConnectorTool",
    "McpApprovalDecision",
    "ProviderExtensions",
]
