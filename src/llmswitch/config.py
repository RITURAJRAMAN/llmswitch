"""
config.py

This module defines configuration classes for different LLM providers using Pydantic models. These classes validate and store provider-specific settings such as API keys, endpoints, rate limits, and deployment information. The CONFIG_REGISTRY dictionary maps provider names to their configuration classes for easy lookup.

Classes:
    BaseLLMConfig: Base configuration for all providers.
    AzureOpenAIConfig, AzureFoundryConfig, VertexAIConfig, OpenAIConfig, AnthropicConfig:
        Provider-specific configuration models.

Constants:
    CONFIG_REGISTRY: Maps provider names to their config classes.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .constants import (
    DEFAULT_COMPLETION_CONTEXT_WINDOW,
    DEFAULT_EMBEDDING_RPM,
    DEFAULT_EMBEDDING_TPM,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RPM,
    DEFAULT_TPM,
)

# Native-SDK client arguments that the unified constructor sets itself. They cannot
# be overridden through ``client_options``; pass them as the unified arguments instead.
RESERVED_CLIENT_OPTIONS = frozenset({
    "api_key", "api_version", "azure_endpoint", "azure_deployment", "base_url", "endpoint",
    "organization", "project", "timeout", "credential", "credentials", "location",
    "vertexai", "http_options", "responses_api_version",
})


class BaseLLMConfig(BaseModel):
    """
    Base configuration for LLM providers, including model, rate limits, and Redis settings.
    All provider-specific configs should inherit from this class.
    """
    # A unified switch accepts the union of provider-specific constructor
    # signatures. Providers consume their own fields and safely ignore the rest.
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    provider: str = Field(..., min_length=1, description="Provider name (openai, azure_openai, azure_foundry, etc.)")
    model: Optional[str] = Field(default=None, description="Model name or ID")
    deployment_name: Optional[str] = Field(
        default=None,
        description="Provider deployment/model alias; when supplied it is the default effective model",
    )

    @property
    def effective_model(self) -> Optional[str]:
        """Return the unified deployment-first default model identifier."""
        return self.deployment_name or self.model
    rpm: int = Field(default=DEFAULT_RPM, ge=0, description="Requests per minute limit (chat)")
    tpm: int = Field(default=DEFAULT_TPM, ge=0, description="Tokens per minute limit (chat)")
    embedding_rpm: int = Field(default=DEFAULT_EMBEDDING_RPM, ge=0,
                               description="Requests per minute limit (embeddings)")
    embedding_tpm: int = Field(default=DEFAULT_EMBEDDING_TPM, ge=0,
                               description="Tokens per minute limit (embeddings)")
    completion_context_window: int = Field(default=DEFAULT_COMPLETION_CONTEXT_WINDOW, gt=0,
                                           description="Context window size for completion models")
    context_output_reserve_tokens: int = Field(default=0, ge=0,
                                                description="Output-token capacity reserved for every request")
    context_overhead_tokens: int = Field(default=0, ge=0,
                                         description="Additional caller-known protocol overhead tokens")
    responses_capabilities: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-target Responses capability overrides; False rejects a feature before dispatch",
    )
    mcp_policy: Optional[dict[str, Any]] = Field(
        default=None, description="Explicit policy required before remote MCP or connector tools are sent"
    )
    redis_url: Optional[str] = Field(default=None,
                                     description="Redis URL for distributed rate limiting; None uses process-local limits")
    rate_limit_failure_policy: Literal["memory_fallback", "fail_open", "fail_closed"] = Field(
        default="memory_fallback",
        description="Behavior when a configured Redis rate-limit backend is unavailable",
    )
    redis_connect_timeout: float = Field(default=1.0, gt=0, allow_inf_nan=False,
                                         description="Redis connect/read timeout in seconds")
    user_id: str = Field(default="global", min_length=1, description="User identifier for rate limiting")
    backup: Optional[dict[str, Any]] = Field(default=None, description="Backup provider config for failover")
    timeout: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False,
                                     description="Request timeout in seconds (applies to all providers)")
    client_options: Optional[dict[str, Any]] = Field(
        default=None,
        description="Extra keyword arguments passed to the native SDK client constructor, "
                    "for example default_headers or max_retries",
    )

    @field_validator("client_options")
    @classmethod
    def reject_reserved_client_options(cls, value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if value is None:
            return None
        reserved = sorted(RESERVED_CLIENT_OPTIONS.intersection(value))
        if reserved:
            raise ValueError(
                f"client_options cannot set {', '.join(reserved)}; use the unified constructor arguments instead"
            )
        return value

    max_retries: int = Field(default=DEFAULT_MAX_RETRIES, ge=0,
                             description="Number of retry attempts for transient errors (429, 5xx, connection). 0 = no retry.")
    retry_base_delay: float = Field(default=DEFAULT_RETRY_BASE_DELAY, ge=0, allow_inf_nan=False,
                                    description="Base delay in seconds for exponential back-off between retries.")
    retry_deadline: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False,
                                            description="Maximum seconds one provider call keeps retrying; None = no limit.")
    on_rate_limit: Literal["wait", "raise"] = Field(
        default="wait",
        description="When a rate limit is reached: wait for the window to reopen, or raise immediately.")
    rate_limit_max_wait: Optional[float] = Field(
        default=None, gt=0, allow_inf_nan=False,
        description="Longest wait in seconds for the rate-limit window; a longer wait raises instead. None = no limit.")

    @field_validator("provider", "user_id", mode="before")
    @classmethod
    def required_string_fields(cls, v, info: ValidationInfo):
        """
        Validate required string fields to ensure they are not empty.
        """
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError(f"{info.field_name} is required and cannot be empty")
        return v

    @field_validator("backup")
    @classmethod
    def validate_backup_shape(cls, value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if value is None:
            return None
        if not value.get("provider") or not isinstance(value["provider"], str):
            raise ValueError("backup.provider is required and must be a non-empty string")
        return value


class AzureOpenAIConfig(BaseLLMConfig):
    """
    Configuration for Azure OpenAI provider, including API key, endpoint, and deployment info.
    Inherits from BaseLLMConfig.
    """
    api_key: str = Field(..., min_length=1, description="API key for authentication")
    endpoint: str = Field(..., min_length=1, description="Custom endpoint or base URL")
    api_version: str = Field(..., min_length=1, description="API version (e.g., for Azure OpenAI)")
    responses_api_version: Optional[str] = Field(
        default=None, description="API version override for responses.* calls (Azure OpenAI)")
    deployment_name: str = Field(..., min_length=1, description="Deployment name for Azure OpenAI")
    region: Optional[str] = Field(default=None, description="Region for Azure services")


class AzureFoundryConfig(BaseLLMConfig):
    """
    Configuration for Azure Foundry provider, including API key, endpoint, and deployment info.
    Inherits from BaseLLMConfig.
    """
    api_key: str = Field(..., min_length=1, description="API key for authentication")
    endpoint: str = Field(..., min_length=1, description="Custom endpoint or base URL")
    deployment_name: str = Field(..., min_length=1, description="Deployment Name or Model name for Azure Foundry")
    api_version: str = Field(..., min_length=1, description="API version (e.g., for Azure Foundry)")


class VertexAIConfig(BaseLLMConfig):
    """
    Configuration for VertexAI provider, including location, scopes, and model info.
    Inherits from BaseLLMConfig.
    """
    location: str = Field(..., min_length=1, description="GCP location (e.g., us-central1)")
    credentials_path: str = Field(..., min_length=1, description="Path to GCP credentials JSON file")
    scopes: list[str] = Field(default_factory=lambda: ["https://www.googleapis.com/auth/cloud-platform"],
                              description="OAuth scopes for GCP authentication")


class OpenAIConfig(BaseLLMConfig):
    """
    Configuration for native OpenAI provider, including API key and optional base URL.
    Inherits from BaseLLMConfig.
    """
    api_key: str = Field(..., min_length=1, description="API key for OpenAI authentication")
    endpoint: Optional[str] = Field(default=None, description="Custom base URL (e.g., for proxies or compatible APIs)")
    organization: Optional[str] = Field(default=None, description="OpenAI organization ID")
    project: Optional[str] = Field(default=None, description="OpenAI project ID to scope requests")


class AnthropicConfig(BaseLLMConfig):
    """
    Configuration for the Anthropic (Claude) provider.

    Prompt caching is enabled by default (``enable_prompt_caching=True``).
    When enabled, the provider automatically injects ``cache_control`` on
    system messages and tool definitions so repeated calls reuse the cached
    prefix instead of re-charging full input-token cost.

    Inherits from BaseLLMConfig.
    """
    api_key: str = Field(..., min_length=1, description="Anthropic API key")
    deployment_name: Optional[str] = Field(
        default=None,
        description=(
            "Preferred deployment/model identifier for Anthropic-compatible endpoints. "
            "If provided with model, deployment_name takes precedence."
        ),
    )
    endpoint: Optional[str] = Field(
        default=None,
        description="Custom base URL (e.g., for an Anthropic-compatible proxy or gateway)",
    )
    enable_prompt_caching: bool = Field(
        default=True,
        description=(
            "Auto-inject cache_control on system messages and tool definitions. "
            "Requires ≥2048 cacheable tokens for Claude Sonnet/Opus, ≥1024 for Haiku."
        ),
    )


# Register in config registry
CONFIG_REGISTRY: dict[str, type[BaseLLMConfig]] = {
    "azure_openai": AzureOpenAIConfig,
    "azure_foundry": AzureFoundryConfig,
    "vertexai": VertexAIConfig,
    "openai": OpenAIConfig,
    "anthropic": AnthropicConfig,
}
