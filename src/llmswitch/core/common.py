"""
core.common

Helpers shared by the synchronous and asynchronous clients: configuration
loading, input validation, provider lookup and the Responses provider checks.
"""
from __future__ import annotations

import difflib
import logging
from collections.abc import Mapping
from typing import Any, NoReturn

from ..config import CONFIG_REGISTRY
from ..errors import (
    APIConnectionError,
    AuthenticationError,
    ChatCompletionError,
    ConfigurationError,
    EmbeddingGenerationError,
    LLMSwitchError,
    ResponsesError,
)
from .registry import UNAVAILABLE_PROVIDERS

logger = logging.getLogger(__name__)


def _known_option_names() -> set[str]:
    """Every option that some provider defines. One constructor call works for any provider."""
    names: set[str] = set()
    for config_cls in CONFIG_REGISTRY.values():
        names.update(config_cls.model_fields)
    return names


def _reject_unknown_options(kwargs: dict[str, Any]) -> None:
    """Fail on options that no provider defines, so a typo is not silently ignored."""
    known = _known_option_names()
    unknown = sorted(name for name in kwargs if name not in known)
    if not unknown:
        return
    parts = []
    for name in unknown:
        close = difflib.get_close_matches(name, known, n=1)
        parts.append(f"'{name}' (did you mean '{close[0]}'?)" if close else f"'{name}'")
    raise ConfigurationError(
        f"Unknown option{'s' if len(unknown) > 1 else ''}: {', '.join(parts)}. "
        "Native SDK arguments belong in client_options.",
        details={"unknown": unknown},
    )


def build_config(kwargs: dict[str, Any]) -> Any:
    """Validate constructor kwargs against the provider's configuration model."""
    provider = kwargs.get("provider")
    if not isinstance(provider, str):
        raise LLMSwitchError(f"Unsupported provider config: {provider}")
    config_cls = CONFIG_REGISTRY.get(provider)
    if config_cls is None:
        raise LLMSwitchError(f"Unsupported provider config: {provider}")
    _reject_unknown_options(kwargs)
    try:
        return config_cls(**kwargs)
    except ValueError as exc:
        if hasattr(exc, "errors"):
            errors = exc.errors()
            reasons = "; ".join(str(error["msg"]).removeprefix("Value error, ") for error in errors)
            raise ConfigurationError(f"Invalid switch configuration: {reasons}", details=errors) from exc
        raise ConfigurationError(f"Invalid switch configuration: {exc}", details=str(exc)) from exc


def validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of message dicts")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("every message must be a dict")


def normalize_embedding_input(input: Any) -> list[str]:  # noqa: A002
    inputs = [input] if isinstance(input, str) else list(input or [])
    if not inputs:
        raise ValueError("Embedding input must contain at least one string")
    if not all(isinstance(text, str) and text for text in inputs):
        raise ValueError("Embedding input must be a non-empty string or a list of non-empty strings")
    return inputs


def sdk_unavailable_message(provider: str, reason: str) -> str:
    return (f"Provider '{provider}' could not load its SDK ({reason}). "
            "Reinstall the package with: pip install --upgrade llmswitch-sdk")


def provider_class(provider: str, registry: Mapping[str, type], operation: str) -> type:
    """Return the provider class for ``operation``, or raise a message that says why it is missing."""
    provider_cls = registry.get(provider)
    if provider_cls is not None:
        return provider_cls
    if provider in UNAVAILABLE_PROVIDERS:
        raise LLMSwitchError(sdk_unavailable_message(provider, UNAVAILABLE_PROVIDERS[provider]), provider=provider)
    raise LLMSwitchError(f"Unsupported provider for {operation}: {provider}", provider=provider)


def responses_provider_class(config: Any, registry: Mapping[str, type]) -> type:
    provider_cls = registry.get(config.provider)
    if provider_cls is not None:
        return provider_cls
    if config.provider in UNAVAILABLE_PROVIDERS:
        reason = UNAVAILABLE_PROVIDERS[config.provider]
        raise ResponsesError(sdk_unavailable_message(config.provider, reason), status_code=501, provider=config.provider)
    raise ResponsesError(f"Provider '{config.provider}' does not support the Responses API",
                         status_code=501, provider=config.provider)


def raise_chat_error(config: Any, exc: Exception, user_key: str) -> NoReturn:
    """Re-raise a provider exception as the library's typed error; typed errors pass through unchanged."""
    if isinstance(exc, LLMSwitchError):
        raise exc
    if isinstance(exc, ValueError):
        error: LLMSwitchError = ChatCompletionError(str(exc), status_code=400, provider=config.provider,
                                                    user_id=user_key)
    elif isinstance(exc, ConnectionError):
        error = APIConnectionError(config.endpoint, provider=config.provider, user_id=user_key)
    elif isinstance(exc, PermissionError):
        error = AuthenticationError(provider=config.provider, user_id=user_key)
    else:
        error = ChatCompletionError(str(exc), status_code=getattr(exc, "status_code", 500),
                                    provider=config.provider, user_id=user_key)
    raise error from exc


def raise_embedding_error(config: Any, exc: Exception) -> NoReturn:
    """Embedding counterpart of ``raise_chat_error``."""
    if isinstance(exc, LLMSwitchError):
        raise exc
    if isinstance(exc, ValueError):
        error: LLMSwitchError = EmbeddingGenerationError(str(exc), status_code=400, provider=config.provider)
    elif isinstance(exc, ConnectionError):
        error = APIConnectionError(config.endpoint, provider=config.provider)
    elif isinstance(exc, PermissionError):
        error = AuthenticationError(provider=config.provider)
    else:
        error = EmbeddingGenerationError(str(exc), status_code=getattr(exc, "status_code", 500),
                                         provider=config.provider)
    raise error from exc
