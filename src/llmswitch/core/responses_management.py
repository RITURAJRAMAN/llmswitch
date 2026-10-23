"""Shared preparation and error translation for Responses management calls."""
from __future__ import annotations

from typing import Any

from ..constants import DEFAULT_MAX_RETRIES
from ..errors import APIConnectionError, AuthenticationError, LLMSwitchError, ResponsesError
from .execution import build_execution_plan, provider_request_with_payload
from .responses_capabilities import validate_response_operation

_FEATURES = {
    "list_input_items": "input_items.list",
    "count_input_tokens": "input_tokens.count",
    "create_conversation": "conversations.create",
    "retrieve_conversation": "conversations.retrieve",
    "update_conversation": "conversations.update",
    "delete_conversation": "conversations.delete",
    "list_conversation_items": "conversations.items.list",
    "retrieve_conversation_item": "conversations.items.retrieve",
    "delete_conversation_item": "conversations.items.delete",
}

_MUTATING_OPERATIONS = {
    "delete", "cancel", "create_conversation", "update_conversation",
    "delete_conversation", "delete_conversation_item",
}


def prepare_management_call(
    config: Any,
    operation: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any], int, Any]:
    """Validate and plan a management request without dispatching it."""
    validate_response_operation(config, _FEATURES.get(operation, operation))
    default_retries = 0 if operation in _MUTATING_OPERATIONS else getattr(
        config, "max_retries", DEFAULT_MAX_RETRIES
    )
    max_retries: int = kwargs.pop("max_retries", default_retries)
    live_logs = kwargs.pop("live_logs", False)
    normalized_request = dict(kwargs)
    payload_field = None
    if operation in {"count_input_tokens", "compact"} and "input" in kwargs:
        normalized_request["input"] = kwargs.get("input")
        payload_field = "input"
    elif operation == "create_conversation":
        normalized_request["input"] = args[0] if args else None
        payload_field = "items"
    plan = build_execution_plan(
        config, operation=f"responses.{operation}", normalized_request=normalized_request,
    )
    provider_kwargs = dict(provider_request_with_payload(plan, input_key=payload_field))
    if operation != "count_input_tokens":
        provider_kwargs.pop("model", None)
    call_args = args if operation != "create_conversation" else ()
    return call_args, provider_kwargs, max_retries, live_logs


def translate_response_error(config: Any, exc: Exception, user_key: str, operation: str) -> Exception:
    """Map a non-typed provider exception onto the switch's error hierarchy."""
    if isinstance(exc, LLMSwitchError):
        return exc
    if isinstance(exc, ValueError):
        return ResponsesError(str(exc), status_code=400, provider=config.provider,
                              user_id=user_key, operation=operation)
    if isinstance(exc, ConnectionError):
        return APIConnectionError(getattr(config, "endpoint", None) or "<unknown>",
                                  provider=config.provider, user_id=user_key)
    if isinstance(exc, PermissionError):
        return AuthenticationError(provider=config.provider, user_id=user_key)
    return ResponsesError(str(exc), status_code=getattr(exc, "status_code", 500),
                          provider=config.provider, user_id=user_key, operation=operation)


def raise_response_error(config: Any, exc: Exception, user_key: str, operation: str) -> None:
    """Preserve typed provider errors without creating a self-referential cause."""
    error = translate_response_error(config, exc, user_key, operation)
    if error is exc:
        raise exc
    raise error from exc
