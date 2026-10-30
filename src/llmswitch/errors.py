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
    """Base class for every error raised by llmswitch.

    Each error carries a message and a stable ``error_code``, so you can branch on it instead of parsing text. When
    the provider's status code is known, ``status_code`` holds it. ``to_dict`` and ``from_dict`` convert an error to
    and from a plain dict, for logs and API responses.

    Attributes:
        message (str): The human-readable message.
        status_code (int or None): The HTTP status code from the provider, when there is one.
        error_code (str): A stable code such as ``E_RATE_LIMIT_EXCEEDED``.
        provider (str or None): The provider that raised the error, when known.
        user_id (str or None): The ``user_id`` the request was made for, when known.
        retry_after (float or None): Seconds to wait before retrying, for rate limit errors.
    """
    status_code: int = 500
    error_code: str = "E_UNKNOWN"

    def __init__(self, message: str, status_code: Optional[int] = None,
                 error_code: Optional[str] = None, **kwargs: Any) -> None:
        """Create the error.

        Args:
            message (str): The human-readable message.
            status_code (int, optional): The HTTP status code from the provider.
            error_code (str, optional): A stable code for the error. Each subclass sets its own default.
            **kwargs: Extra context, stored with the error and returned by ``to_dict``.
        """
        super().__init__(message)
        self.message: str = message
        self.status_code: int = status_code if status_code is not None else self.status_code
        self.error_code: str = error_code if error_code is not None else self.error_code
        self.context: dict[str, Any] = kwargs  # Extra context for debugging

    @property
    def provider(self) -> Optional[str]:
        """The provider that raised the error, when known.

        Returns:
            str or None: The provider name, such as ``openai``.
        """
        return self.context.get("provider")

    @property
    def user_id(self) -> Optional[str]:
        """The ``user_id`` the request was made for, when known.

        Returns:
            str or None: The user identifier that rate limits were counted against.
        """
        return self.context.get("user_id")

    @property
    def retry_after(self) -> Optional[int]:
        """Seconds to wait before retrying, for rate limit errors.

        Returns:
            float or None: The suggested wait in seconds, or None when the error does not suggest one.
        """
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
        """Convert the error into a plain dict, for logs or API responses.

        Returns:
            dict: The error class name, message, status code, error code and any extra context.
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
        """Rebuild an error from the dict that ``to_dict`` produced.

        Args:
            data (dict): A dict as returned by ``to_dict``.

        Returns:
            LLMSwitchError: An error of the class named in the dict, or the base class when the name is unknown.
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
        """Create a configuration error.

        Args:
            message (str): What is wrong with the configuration.
            **context: Extra details, such as the option name, stored with the error.
        """
        super().__init__(message=message, **context)


class InvalidResponseRequestError(LLMSwitchError):
    """Raised when a Responses create request violates the switch contract."""
    status_code = 400
    error_code = "E_INVALID_RESPONSE_REQUEST"

    def __init__(self, message: str, provider: Optional[str] = None) -> None:
        """Create an error for a Responses request the API contract does not accept.

        Args:
            message (str): What is wrong with the request.
            provider (str, optional): The provider that rejected it.
        """
        super().__init__(message=message, provider=provider)


class UnsupportedCapabilityError(LLMSwitchError):
    """Raised when an explicitly unsupported provider capability is requested."""
    status_code = 501
    error_code = "E_UNSUPPORTED_CAPABILITY"

    def __init__(self, *, provider: str, feature: str, model: Optional[str] = None,
                 api_version: Optional[str] = None, detail: Optional[str] = None) -> None:
        """Create an error for a feature the provider or model does not support.

        Args:
            provider (str): The provider that lacks the feature.
            feature (str): The feature that was requested.
            model (str, optional): The model, when the limit depends on it.
            api_version (str, optional): The API version, when the limit depends on it.
            detail (str, optional): More detail about the limit.
        """
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
        """Create an error for a request that cannot fit its context budget.

        Args:
            input_tokens (int): The number of input tokens in the request.
            input_budget (int): The number of input tokens the budget allows.
            context_window (int): The configured context window, in tokens.
            reserved_output_tokens (int): Output tokens kept free for the reply.
            overhead_tokens (int): Protocol tokens counted on top of the input.
        """
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
        """Create an error for a rate limit backend that cannot be reached.

        Args:
            backend (str): The backend that is unavailable, for example ``redis``.
        """
        super().__init__(message=f"Rate-limit backend '{backend}' is unavailable.", backend=backend)

class ModelNotFoundError(LLMSwitchError):
    """Raised when a requested model is not found in the system."""
    status_code = 404
    error_code = "E_MODEL_NOT_FOUND"

    def __init__(self, model_name: str, provider: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """Create an error for a model that is not known to the provider.

        Args:
            model_name (str): The model that was requested.
            provider (str, optional): The provider that was asked.
            user_id (str, optional): The user the request was made for.
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
        """Create an error for a reply that the provider or model returned in an invalid form.

        Args:
            message (str): What was invalid about the reply.
            provider (str, optional): The provider that returned it.
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
        """Create an error for a connection that could not be made.

        Args:
            api_url (str): The URL that could not be reached.
            provider (str, optional): The provider.
            user_id (str, optional): The user the request was made for.
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
        """Create an error for a rate limit that stayed full.

        Args:
            retry_after (float, optional): Seconds to wait before retrying, when known.
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
        """Create an error for credentials the provider rejected.

        Args:
            message (str): The message from the provider or the library.
            provider (str, optional): The provider that rejected the credentials.
            user_id (str, optional): The user the request was made for.
        """
        super().__init__(message=message, provider=provider, user_id=user_id)


class TimeOutError(LLMSwitchError):
    """Raised when a request times out before completion."""
    status_code = 504
    error_code = "E_TIMEOUT"

    def __init__(self, timeout_duration: int) -> None:
        """Create an error for a request that ran out of time.

        Args:
            timeout_duration (float): The time limit that passed, in seconds.
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
        """Create an error for a failed embedding request.

        Args:
            message (str): The message from the provider or the library.
            status_code (int, optional): The HTTP status code from the provider.
            provider (str, optional): The provider.
            user_id (str, optional): The user the request was made for.
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
        """Create an error for a failed chat completion request.

        Args:
            message (str): The message from the provider or the library.
            status_code (int, optional): The HTTP status code from the provider. Server errors (5xx) are retried.
            provider (str, optional): The provider.
            user_id (str, optional): The user the request was made for.
        """
        super().__init__(
            message=f"{message}",
            status_code=status_code,
            provider=provider,
            user_id=user_id
        )


class ResponsesError(LLMSwitchError):
    """Raised when a Responses API operation fails.

    Carries the same attributes as ``LLMSwitchError``, plus ``operation``, the name of the call that failed, such as
    ``create`` or ``retrieve``.
    """
    status_code = 500
    error_code = "E_RESPONSES"

    def __init__(self, message: str, status_code: Optional[int] = None,
                 provider: Optional[str] = None, user_id: Optional[str] = None,
                 operation: Optional[str] = None) -> None:
        """Create an error for a failed Responses API operation.

        Args:
            message (str): The message from the provider or the library.
            status_code (int, optional): The HTTP status code from the provider.
            provider (str, optional): The provider.
            user_id (str, optional): The user the request was made for.
            operation (str, optional): The call that failed, such as ``create`` or ``retrieve``.
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
        """Create an error for a stored response that does not exist.

        Args:
            response_id (str): The id that was not found.
            provider (str, optional): The provider.
            user_id (str, optional): The user the request was made for.
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
        """Create an error for a prompt or reply that the content filter blocked.

        Args:
            message (str): The message from the provider.
            provider (str, optional): The provider.
            user_id (str, optional): The user the request was made for.
            filter_results (dict, optional): The provider's filter results, which say which category blocked the
                content.
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


class ToolLoopLimitError(LLMSwitchError):
    """Raised when the model still asks for tools after ``max_steps`` model calls."""
    status_code = 400
    error_code = "E_TOOL_LOOP_LIMIT"

    def __init__(self, max_steps: int) -> None:
        """Create an error for a tool loop that reached its step limit.

        Args:
            max_steps (int): The step limit that was reached.
        """
        super().__init__(f"The model still requested tools after {max_steps} model calls.")
        self.max_steps = max_steps
