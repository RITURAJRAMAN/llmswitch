"""
errors.py

This module defines custom exception classes for the LLM Switch system. It provides a base error class (LLMSwitchError) with serialization, context, and automatic registration, as well as specific error subclasses for common failure scenarios (model not found, invalid response, rate limit, etc.).

Classes:
    LLMSwitchError: Base exception for all LLM Switch errors.
    ModelNotFoundError: Raised when a requested model is not found.
    InvalidResponseError: Raised for invalid responses from external services.
    APIConnectionError: Raised when failing to connect to an API endpoint.
    RateLimitExceededError: Raised when a rate limit is exceeded.
    AuthenticationError: Raised for authentication failures.
    TimeOutError: Raised when a request times out.
    EmbeddingGenerationError: Raised for embedding generation errors.
    ChatCompletionError: Raised for chat completion errors.
    ResponsesError: Raised for Responses API errors.
    ResponseNotFoundError: Raised when a Responses API object is not found by id.

Constants:
    _error_registry: Maps error class names to their classes for lookup.
"""

import json
from typing import Any, Optional

# A registry to automatically map error names to their classes for deserialization
_error_registry: dict[str, type["LLMSwitchError"]] = {}


class LLMSwitchError(Exception):
    """
    Base exception for all LLM Switch errors.
    Provides serialization, registration, and context handling for error objects.
    """
    status_code: int = 500
    error_code: str = "E_UNKNOWN"

    def __init__(self, message: str, status_code: Optional[int] = None,
                 error_code: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize the error with message, error type, status code, error code, and extra context.
        Args:
            message (str): Human-readable error message.
            status_code (Optional[int]): HTTP status code (defaults to 500).
            error_code (Optional[str]): Application-specific error code.
            **kwargs: Additional context for the error.
        """
        super().__init__(message)
        self.message: str = message
        self.status_code: int = status_code if status_code is not None else self.status_code
        self.error_code: str = error_code if error_code is not None else self.error_code
        self.context: dict[str, Any] = kwargs  # Extra context for debugging

    @property
    def provider(self) -> Optional[str]:
        """The provider that raised the error, when known."""
        return self.context.get("provider")

    @property
    def user_id(self) -> Optional[str]:
        """The ``user_id`` the request was made for, when known."""
        return self.context.get("user_id")

    @property
    def retry_after(self) -> Optional[int]:
        """Seconds to wait before retrying, for rate-limit errors."""
        return self.context.get("retry_after")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Automatically register subclasses in the error registry for lookup by name.
        """
        super().__init_subclass__(**kwargs)
        _error_registry[cls.__name__] = cls

    def __str__(self) -> str:
        """
        String representation of the error, including error code and message.
        """
        return f"{self.__class__.__name__}: {self.message} (code: {self.error_code})"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the error into a serializable dict for logging or API responses.
        Returns:
            dict: Dictionary with error details and context.
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
            "error_code": self.error_code,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMSwitchError":
        """
        Reconstruct an error object from its dict representation.
        Args:
            data (dict): Dictionary with error details.
        Returns:
            LLMSwitchError: The reconstructed error object.
        """
        error_class = _error_registry.get(data.get("error", ""), cls)
        # Reconstruct without invoking subclass __init__ (whose signatures differ
        # and do not accept status_code/error_code/context), so every subclass
        # round-trips losslessly from its to_dict() form.
        message = data.get("message", "")
        obj = error_class.__new__(error_class)
        Exception.__init__(obj, message)
        obj.message = message
        obj.status_code = data.get("status_code", error_class.status_code)
        obj.error_code = data.get("error_code", error_class.error_code)
        obj.context = dict(data.get("context") or {})
        return obj


# --- Specific Error Classes ---

class ConfigurationError(LLMSwitchError):
    """Raised when switch or failover configuration is invalid."""
    status_code = 400
    error_code = "E_CONFIGURATION"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message=message, **context)


class InvalidResponseRequestError(LLMSwitchError):
    """Raised when a Responses create request violates the switch contract."""
    status_code = 400
    error_code = "E_INVALID_RESPONSE_REQUEST"

    def __init__(self, message: str, provider: Optional[str] = None) -> None:
        super().__init__(message=message, provider=provider)


class UnsupportedCapabilityError(LLMSwitchError):
    """Raised when an explicitly unsupported provider capability is requested."""
    status_code = 501
    error_code = "E_UNSUPPORTED_CAPABILITY"

    def __init__(self, *, provider: str, feature: str, model: Optional[str] = None,
                 api_version: Optional[str] = None, detail: Optional[str] = None) -> None:
        target = f" for model/deployment '{model}'" if model else ""
        extra = f" {detail}" if detail else ""
        super().__init__(
            message=f"Provider '{provider}' does not support Responses capability '{feature}'{target}.{extra}",
            provider=provider,
            feature=feature,
            model=model,
            api_version=api_version,
        )


class ContextLengthExceededError(LLMSwitchError):
    """Raised when a request cannot fit within its configured context budget."""
    status_code = 400
    error_code = "E_CONTEXT_LENGTH_EXCEEDED"

    def __init__(self, *, input_tokens: int, input_budget: int,
                 context_window: int, reserved_output_tokens: int,
                 overhead_tokens: int) -> None:
        super().__init__(
            message=(
                f"Request input requires {input_tokens} tokens but only {input_budget} are available "
                f"within context window {context_window} after reserving {reserved_output_tokens} output "
                f"and {overhead_tokens} overhead tokens."
            ),
            input_tokens=input_tokens,
            input_budget=input_budget,
            context_window=context_window,
            reserved_output_tokens=reserved_output_tokens,
            overhead_tokens=overhead_tokens,
        )


class RateLimitBackendUnavailableError(LLMSwitchError):
    """Raised when fail-closed rate limiting cannot contact its configured backend."""
    status_code = 503
    error_code = "E_RATE_LIMIT_BACKEND_UNAVAILABLE"

    def __init__(self, backend: str = "redis") -> None:
        super().__init__(message=f"Rate-limit backend '{backend}' is unavailable.", backend=backend)

class ModelNotFoundError(LLMSwitchError):
    """Raised when a requested model is not found in the system."""
    status_code = 404
    error_code = "E_MODEL_NOT_FOUND"

    def __init__(self, model_name: str, provider: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """
        Args:
            model_name (str): Name of the missing model.
            provider (Optional[str]): Provider name.
            user_id (Optional[str]): User identifier.
        """
        super().__init__(
            message=f"Model '{model_name}' not found.",
            model_name=model_name,
            provider=provider,
            user_id=user_id
        )


class InvalidResponseError(LLMSwitchError):
    """Raised when an invalid response is received from an external service or model."""
    status_code = 502
    error_code = "E_INVALID_RESPONSE"

    def __init__(self, message: str, provider: Optional[str] = None) -> None:
        """
        Args:
            message (str): The invalid response message.
        """
        super().__init__(
            message=f"Invalid response received: {message}",
            response=message,
            provider=provider
        )


class APIConnectionError(LLMSwitchError):
    """Raised when the system fails to connect to an external API endpoint."""
    status_code = 503
    error_code = "E_API_CONNECTION"

    def __init__(self, api_url: str, provider: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """
        Args:
            api_url (str): The API endpoint URL.
            provider (Optional[str]): Provider name.
            user_id (Optional[str]): User identifier.
        """
        super().__init__(
            message=f"Failed to connect to API at {api_url}",
            api_url=api_url,
            provider=provider,
            user_id=user_id
        )


class RateLimitExceededError(LLMSwitchError):
    """Raised when a rate limit is exceeded and the client must wait before retrying."""
    status_code = 429
    error_code = "E_RATE_LIMIT_EXCEEDED"

    def __init__(self, retry_after: int) -> None:
        """
        Args:
            retry_after (int): Number of seconds to wait before retrying.
        """
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            retry_after=retry_after
        )


class AuthenticationError(LLMSwitchError):
    """Raised when authentication fails due to invalid credentials or API key."""
    status_code = 401
    error_code = "E_AUTHENTICATION_FAILED"

    def __init__(self, message: str = "Authentication failed. Please check your API key.",
                 provider: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """
        Args:
            message (str): Error message.
            provider (Optional[str]): Provider name.
            user_id (Optional[str]): User identifier.
        """
        super().__init__(message=message, provider=provider, user_id=user_id)


class TimeOutError(LLMSwitchError):
    """Raised when a request times out before completion."""
    status_code = 504
    error_code = "E_TIMEOUT"

    def __init__(self, timeout_duration: int) -> None:
        """
        Args:
            timeout_duration (int): Number of seconds before timeout.
        """
        super().__init__(
            message=f"Request timed out after {timeout_duration} seconds.",
            timeout_duration=timeout_duration
        )


class EmbeddingGenerationError(LLMSwitchError):
    """Raised when a generic error occurs during embedding generation."""
    status_code = 500
    error_code = "E_EMBEDDING_GENERATION"

    def __init__(self, message: str, status_code: Optional[int] = None,
                 provider: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """
        Args:
            message (str): Error message.
            status_code (Optional[int]): HTTP status code.
            provider (Optional[str]): Provider
            user_id (Optional[str]): User Identifier
        """
        super().__init__(
            message=f"{message}",
            status_code=status_code,
            provider=provider,
            user_id=user_id
        )


class ChatCompletionError(LLMSwitchError):
    """Raised when a generic error occurs during chat completion."""
    status_code = 500
    error_code = "E_CHAT_COMPLETION"

    def __init__(self, message: str, status_code: Optional[int] = None,
                 provider: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """
        Args:
            message (str): Error message.
            status_code (Optional[int]): HTTP status code.
            provider (Optional[str]): Provider name.
            user_id (Optional[str]): User identifier.
        """
        super().__init__(
            message=f"{message}",
            status_code=status_code,
            provider=provider,
            user_id=user_id
        )


class ResponsesError(LLMSwitchError):
    """Raised when a generic error occurs during a Responses API operation."""
    status_code = 500
    error_code = "E_RESPONSES"

    def __init__(self, message: str, status_code: Optional[int] = None,
                 provider: Optional[str] = None, user_id: Optional[str] = None,
                 operation: Optional[str] = None) -> None:
        """
        Args:
            message (str): Error message.
            status_code (Optional[int]): HTTP status code.
            provider (Optional[str]): Provider name.
            user_id (Optional[str]): User identifier.
            operation (Optional[str]): Responses operation that failed (create, retrieve, …).
        """
        super().__init__(
            message=f"{message}",
            status_code=status_code,
            provider=provider,
            user_id=user_id,
            operation=operation,
        )


class ResponseNotFoundError(LLMSwitchError):
    """Raised when a Responses API object cannot be found by id."""
    status_code = 404
    error_code = "E_RESPONSE_NOT_FOUND"

    def __init__(self, response_id: str, provider: Optional[str] = None,
                 user_id: Optional[str] = None) -> None:
        """
        Args:
            response_id (str): The response identifier that was not found.
            provider (Optional[str]): Provider name.
            user_id (Optional[str]): User identifier.
        """
        super().__init__(
            message=f"Response '{response_id}' not found.",
            response_id=response_id,
            provider=provider,
            user_id=user_id,
        )


class ContentFilterError(LLMSwitchError):
    """Raised when a response is blocked by content filtering (Azure OpenAI / Azure Foundry)."""
    status_code = 400
    error_code = "E_CONTENT_FILTER"

    def __init__(self, message: str = "Content was filtered by the provider's content policy.",
                 provider: Optional[str] = None, user_id: Optional[str] = None,
                 filter_results: Optional[Any] = None) -> None:
        """
        Args:
            message (str): Error message.
            provider (Optional[str]): Provider name.
            user_id (Optional[str]): User identifier.
            filter_results (Optional[Any]): Raw content filter results from the provider.
        """
        super().__init__(
            message=message,
            provider=provider,
            user_id=user_id,
            filter_results=filter_results,
        )


def _extract_filter_results(data: Any) -> Any:
    """Extract provider-specific filter payloads when present."""
    if not isinstance(data, dict):
        return None

    candidates = [data]
    error = data.get("error")
    if isinstance(error, dict):
        candidates.append(error)

    for candidate in list(candidates):
        inner = candidate.get("innererror") or candidate.get("inner_error")
        if isinstance(inner, dict):
            candidates.append(inner)

    for candidate in candidates:
        for key in ("content_filter_result", "content_filter_results", "filter_results"):
            if key in candidate:
                return candidate[key]
    return None


def detect_content_filter_error(exc: Exception, provider: Optional[str] = None,
                                user_id: Optional[str] = None) -> Optional[ContentFilterError]:
    """Return a typed content-filter error when the provider payload indicates filtering."""
    body = getattr(exc, "body", None)
    if body is None:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                body = response.json()
            except Exception:
                body = getattr(response, "text", None)

    body_text = json.dumps(body, default=str) if isinstance(body, (dict, list)) else str(body or "")
    message = str(exc)
    haystack = f"{message} {body_text}".lower()

    if any(marker in haystack for marker in (
            "content_filter",
            "content policy",
            "responsible ai policy",
            "safety system",
    )):
        return ContentFilterError(
            message=message or "Content was filtered by the provider's content policy.",
            provider=provider,
            user_id=user_id,
            filter_results=_extract_filter_results(body),
        )
    return None
