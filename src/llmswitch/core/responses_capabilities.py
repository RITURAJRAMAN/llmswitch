"""Capability contracts and request validation for the Responses API."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Mapping
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..errors import InvalidResponseRequestError, UnsupportedCapabilityError
from ..structured import is_pydantic_model_class


class CapabilityStatus(str, Enum):
    """Support level for a Responses feature on a configured target."""

    SUPPORTED = "supported"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class ResponseCapabilities(BaseModel):
    """Resolved Responses capabilities for one configured provider target."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["openai", "azure_openai"]
    model: str | None = None
    api_version: str | None = None
    features: Mapping[str, CapabilityStatus]

    def status(self, feature: str) -> CapabilityStatus:
        """Return a feature's support level, defaulting to unknown."""
        return self.features.get(feature, CapabilityStatus.UNKNOWN)

    def supports(self, feature: str) -> bool | None:
        """Return True, False, or None when support is deployment-dependent."""
        status = self.status(feature)
        if status is CapabilityStatus.SUPPORTED:
            return True
        if status is CapabilityStatus.UNSUPPORTED:
            return False
        return None


class ResponseCreateRequest(BaseModel):
    """Strict, provider-neutral shape for a Responses ``create`` request.

    This model validates the stable adapter contract. ``extra_body`` remains an
    intentional escape hatch for newly released provider fields until they are
    represented in a later capability-matrix revision.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model: str | None = None
    input: str | list[Any] | None = None
    instructions: str | list[Any] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, gt=0)
    reasoning: Any = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    text: Any = None
    text_format: Any = None
    response_format: Any = None
    previous_response_id: str | None = None
    conversation: str | dict[str, Any] | None = None
    store: bool | None = None
    background: bool | None = None
    stream_options: dict[str, bool] | None = None
    metadata: dict[str, str] | None = None
    include: list[str] | None = None
    prompt: Any = None
    prompt_cache_key: str | None = None
    prompt_cache_options: Any = None
    prompt_cache_retention: str | None = None
    truncation: str | None = None
    safety_identifier: str | None = None
    service_tier: str | None = None
    user: str | None = None
    context: Any = None
    extra_headers: Mapping[str, str] | None = None
    extra_query: Mapping[str, str] | None = None
    extra_body: Mapping[str, Any] | None = None
    timeout: float | None = Field(default=None, gt=0)

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if tools is None:
            return None
        for tool in tools:
            if not isinstance(tool.get("type"), str) or not tool["type"]:
                raise ValueError("every Responses tool requires a non-empty string 'type'")
        return tools

    @field_validator("include")
    @classmethod
    def validate_include(cls, include: list[str] | None) -> list[str] | None:
        allowed = {
            "web_search_call.action.sources",
            "code_interpreter_call.outputs",
            "computer_call_output.output.image_url",
            "file_search_call.results",
            "message.input_image.image_url",
            "message.output_text.logprobs",
            "reasoning.encrypted_content",
        }
        if include is not None:
            unknown = sorted(set(include) - allowed)
            if unknown:
                raise ValueError(f"unsupported include values: {', '.join(unknown)}")
        return include

    @model_validator(mode="after")
    def validate_state_selection(self) -> "ResponseCreateRequest":
        if self.previous_response_id is not None and self.conversation is not None:
            raise ValueError("previous_response_id and conversation cannot be supplied together")
        for name in ("text_format", "response_format"):
            value = getattr(self, name)
            if value is not None and not is_pydantic_model_class(value):
                raise ValueError(f"{name} must be a Pydantic model class")
        return self


_KNOWN_TOOL_FEATURES = {
    "function": "tool.function",
    "web_search": "tool.web_search",
    "web_search_preview": "tool.web_search",
    "file_search": "tool.file_search",
    "code_interpreter": "tool.code_interpreter",
    "image_generation": "tool.image_generation",
    "computer": "tool.computer",
    "mcp": "tool.mcp",
    "tool_search": "tool.tool_search",
    "namespace": "tool.tool_search",
}

_OPENAI_FEATURES = {
    "create": CapabilityStatus.SUPPORTED,
    "retrieve": CapabilityStatus.SUPPORTED,
    "delete": CapabilityStatus.SUPPORTED,
    "cancel": CapabilityStatus.SUPPORTED,
    "compact": CapabilityStatus.SUPPORTED,
    "input_items.list": CapabilityStatus.SUPPORTED,
    "input_tokens.count": CapabilityStatus.SUPPORTED,
    "background": CapabilityStatus.SUPPORTED,
    "conversation": CapabilityStatus.SUPPORTED,
    "conversations.create": CapabilityStatus.SUPPORTED,
    "conversations.retrieve": CapabilityStatus.SUPPORTED,
    "conversations.update": CapabilityStatus.SUPPORTED,
    "conversations.delete": CapabilityStatus.SUPPORTED,
    "conversations.items.list": CapabilityStatus.SUPPORTED,
    "conversations.items.retrieve": CapabilityStatus.SUPPORTED,
    "conversations.items.delete": CapabilityStatus.SUPPORTED,
    "tool.function": CapabilityStatus.SUPPORTED,
    "tool.web_search": CapabilityStatus.SUPPORTED,
    "tool.file_search": CapabilityStatus.SUPPORTED,
    "tool.code_interpreter": CapabilityStatus.SUPPORTED,
    "tool.image_generation": CapabilityStatus.SUPPORTED,
    "tool.computer": CapabilityStatus.SUPPORTED,
    "tool.mcp": CapabilityStatus.SUPPORTED,
    "tool.tool_search": CapabilityStatus.SUPPORTED,
}

_AZURE_FEATURES = {
    "create": CapabilityStatus.SUPPORTED,
    "retrieve": CapabilityStatus.SUPPORTED,
    "delete": CapabilityStatus.SUPPORTED,
    "cancel": CapabilityStatus.SUPPORTED,
    "compact": CapabilityStatus.UNKNOWN,
    "input_items.list": CapabilityStatus.SUPPORTED,
    "input_tokens.count": CapabilityStatus.UNKNOWN,
    "background": CapabilityStatus.UNKNOWN,
    "conversation": CapabilityStatus.UNKNOWN,
    "conversations.create": CapabilityStatus.UNKNOWN,
    "conversations.retrieve": CapabilityStatus.UNKNOWN,
    "conversations.update": CapabilityStatus.UNKNOWN,
    "conversations.delete": CapabilityStatus.UNKNOWN,
    "conversations.items.list": CapabilityStatus.UNKNOWN,
    "conversations.items.retrieve": CapabilityStatus.UNKNOWN,
    "conversations.items.delete": CapabilityStatus.UNKNOWN,
    "tool.function": CapabilityStatus.SUPPORTED,
    "tool.web_search": CapabilityStatus.UNKNOWN,
    "tool.file_search": CapabilityStatus.UNKNOWN,
    "tool.code_interpreter": CapabilityStatus.UNKNOWN,
    "tool.image_generation": CapabilityStatus.UNKNOWN,
    "tool.computer": CapabilityStatus.UNKNOWN,
    "tool.mcp": CapabilityStatus.UNKNOWN,
    "tool.tool_search": CapabilityStatus.UNKNOWN,
}


def resolve_response_capabilities(config: Any) -> ResponseCapabilities:
    """Resolve the static and user-overridden contract for one adapter target."""
    provider = getattr(config, "provider", None)
    if provider == "openai":
        features = dict(_OPENAI_FEATURES)
        if getattr(config, "endpoint", None):
            features = {feature: CapabilityStatus.UNKNOWN for feature in features}
        model = getattr(config, "model", None)
        api_version = None
    elif provider == "azure_openai":
        features = dict(_AZURE_FEATURES)
        model = getattr(config, "deployment_name", None) or getattr(config, "model", None)
        api_version = getattr(config, "responses_api_version", None) or getattr(config, "api_version", None)
    else:
        raise UnsupportedCapabilityError(provider=str(provider), feature="responses")

    for feature, enabled in (getattr(config, "responses_capabilities", None) or {}).items():
        features[feature] = CapabilityStatus.SUPPORTED if enabled else CapabilityStatus.UNSUPPORTED
    return ResponseCapabilities(provider=provider, model=model, api_version=api_version, features=features)


def validate_response_operation(config: Any, operation: str) -> ResponseCapabilities:
    """Reject an operation only when the resolved capability is explicitly unsupported."""
    capabilities = resolve_response_capabilities(config)
    if capabilities.status(operation) is CapabilityStatus.UNSUPPORTED:
        raise UnsupportedCapabilityError(
            provider=capabilities.provider,
            feature=operation,
            model=capabilities.model,
            api_version=capabilities.api_version,
        )
    return capabilities


def validate_responses_create_request(
        config: Any,
        input: Any,
        kwargs: dict[str, Any],
) -> ResponseCapabilities:
    """Validate a normalized create request before rate limiting or SDK dispatch."""
    try:
        request = ResponseCreateRequest(input=input, **kwargs)
    except Exception as exc:
        raise InvalidResponseRequestError(str(exc), provider=getattr(config, "provider", None)) from exc

    capabilities = validate_response_operation(config, "create")
    for tool in request.tools or []:
        tool_type = tool["type"]
        feature = _KNOWN_TOOL_FEATURES.get(tool_type)
        if feature is None:
            raise InvalidResponseRequestError(
                f"unsupported Responses tool type '{tool_type}'", provider=capabilities.provider
            )
        if capabilities.status(feature) is CapabilityStatus.UNSUPPORTED:
            raise UnsupportedCapabilityError(
                provider=capabilities.provider,
                feature=feature,
                model=capabilities.model,
                api_version=capabilities.api_version,
            )
    if request.background:
        validate_response_operation(config, "background")
    if request.conversation is not None:
        validate_response_operation(config, "conversation")
    for tool in request.tools or []:
        if tool["type"] == "mcp":
            _validate_mcp_policy(config, tool)
    return capabilities


def _validate_mcp_policy(config: Any, tool: dict[str, Any]) -> None:
    """Require explicit local allowlisting before exposing remote MCP/connectors."""
    policy = getattr(config, "mcp_policy", None) or {}
    if not policy.get("enabled", False):
        raise InvalidResponseRequestError("remote MCP and connectors require mcp_policy={'enabled': True}", provider=config.provider)
    labels = set(policy.get("allowed_server_labels") or [])
    connector_ids = set(policy.get("allowed_connector_ids") or [])
    origins = set(policy.get("allowed_origins") or [])
    if not labels and not connector_ids and not origins:
        raise InvalidResponseRequestError("mcp_policy must declare an allowed server label, origin, or connector ID", provider=config.provider)
    if labels and tool.get("server_label") not in labels:
        raise InvalidResponseRequestError("MCP server label is not allowed by mcp_policy", provider=config.provider)
    if tool.get("connector_id") and tool["connector_id"] not in connector_ids:
        raise InvalidResponseRequestError("MCP connector ID is not allowed by mcp_policy", provider=config.provider)
    url = tool.get("server_url")
    if origins and url and urlparse(url).netloc not in origins:
        raise InvalidResponseRequestError("MCP server origin is not allowed by mcp_policy", provider=config.provider)


def normalize_response_tool_definitions(kwargs: dict[str, Any]) -> None:
    """Convert typed tool helper models into the dictionaries expected by SDKs."""
    tools = kwargs.get("tools")
    if not isinstance(tools, list):
        return
    kwargs["tools"] = [
        tool.model_dump(exclude_none=True) if isinstance(tool, BaseModel) else tool
        for tool in tools
    ]
