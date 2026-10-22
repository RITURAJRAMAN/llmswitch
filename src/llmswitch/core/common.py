"""
core.common

Helpers shared by the synchronous and asynchronous clients: configuration
loading, model resolution, input validation, provider construction and error
translation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import CONFIG_REGISTRY
from ..constants import DEFAULT_ANTHROPIC_MODEL
from ..errors import (
    APIConnectionError,
    AuthenticationError,
    ChatCompletionError,
    ConfigurationError,
    EmbeddingGenerationError,
    LLMSwitchError,
)

logger = logging.getLogger(__name__)


def build_config(kwargs: dict[str, Any]) -> Any:
    """Validate constructor kwargs against the provider's configuration model."""
    provider = kwargs.get("provider")
    if not isinstance(provider, str):
        raise LLMSwitchError(f"Unsupported provider config: {provider}")
    config_cls = CONFIG_REGISTRY.get(provider)
    if config_cls is None:
        raise LLMSwitchError(f"Unsupported provider config: {provider}")
    try:
        return config_cls(**kwargs)
    except ValueError as exc:
        if hasattr(exc, "errors"):
            errors = exc.errors()
            reasons = "; ".join(str(error["msg"]).removeprefix("Value error, ") for error in errors)
            raise ConfigurationError(f"Invalid switch configuration: {reasons}", details=errors) from exc
        raise ConfigurationError(f"Invalid switch configuration: {exc}", details=str(exc)) from exc


def resolve_model(config: Any, kwargs: dict[str, Any]) -> Optional[str]:
    """Pick the effective model and write it into ``kwargs``.

    A configured deployment wins over everything. Otherwise a per-call ``model``
    overrides the configured model. Anthropic falls back to a default model.
    """
    deployment_name = getattr(config, "deployment_name", None)
    model = deployment_name or kwargs.get("model") or getattr(config, "model", None)
    if model is None and config.provider == "anthropic":
        model = DEFAULT_ANTHROPIC_MODEL
    if model is not None:
        kwargs["model"] = model
    return model


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


def provider_kwargs(config: Any) -> dict[str, Any]:
    """Union of constructor kwargs; each provider reads the ones it understands."""
    endpoint = getattr(config, "endpoint", None)
    deployment_name = getattr(config, "deployment_name", None)
    return {
        "api_key": getattr(config, "api_key", None),
        "endpoint": endpoint,
        "azure_endpoint": endpoint,
        "api_version": getattr(config, "api_version", None),
        "deployment_name": deployment_name,
        "azure_deployment": deployment_name,
        "model": getattr(config, "model", None),
        "location": getattr(config, "location", None),
        "credentials_path": getattr(config, "credentials_path", None),
        "scopes": getattr(config, "scopes", None),
        "organization": getattr(config, "organization", None),
        "project": getattr(config, "project", None),
        "timeout": getattr(config, "timeout", None),
        "enable_prompt_caching": getattr(config, "enable_prompt_caching", None),
        "client_options": getattr(config, "client_options", None),
    }


def map_chat_error(exc: Exception, config: Any) -> LLMSwitchError:
    """Translate a provider exception into the public error hierarchy."""
    if isinstance(exc, LLMSwitchError):
        return exc
    if isinstance(exc, ValueError):
        return ChatCompletionError(str(exc), status_code=400, provider=config.provider)
    if isinstance(exc, ConnectionError):
        return APIConnectionError(config.endpoint, provider=config.provider)
    if isinstance(exc, PermissionError):
        return AuthenticationError(provider=config.provider)
    return ChatCompletionError(str(exc), status_code=getattr(exc, "status_code", 500), provider=config.provider)


def map_embedding_error(exc: Exception, config: Any) -> LLMSwitchError:
    if isinstance(exc, LLMSwitchError):
        return exc
    if isinstance(exc, ValueError):
        return EmbeddingGenerationError(str(exc), status_code=400, provider=config.provider)
    if isinstance(exc, ConnectionError):
        return APIConnectionError(config.endpoint, provider=config.provider)
    if isinstance(exc, PermissionError):
        return AuthenticationError(provider=config.provider)
    return EmbeddingGenerationError(str(exc), status_code=getattr(exc, "status_code", 500),
                                    provider=config.provider)
