"""
config.py

Pydantic configuration models, one per supported provider. ``CONFIG_REGISTRY``
maps each provider name to its configuration class.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

# Native-SDK client arguments that the unified constructor sets itself. They cannot
# be overridden through ``client_options``; pass them as the unified arguments instead.
RESERVED_CLIENT_OPTIONS = frozenset({
    "api_key", "api_version", "azure_endpoint", "azure_deployment", "base_url", "endpoint",
    "organization", "project", "timeout", "credential", "credentials", "location",
    "vertexai", "http_options",
})


class BaseLLMConfig(BaseModel):
    """Fields shared by every provider. Provider configs inherit from this class."""

    # The unified constructor accepts the union of provider-specific fields;
    # each provider reads what it needs and ignores the rest.
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    provider: str = Field(..., min_length=1, description="Provider name (openai, azure_openai, ...)")
    model: Optional[str] = Field(default=None, description="Model name or ID")
    deployment_name: Optional[str] = Field(
        default=None,
        description="Deployment or model alias; when set it takes precedence over the per-call model",
    )
    timeout: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False,
                                     description="Request timeout in seconds")
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

    @field_validator("provider", mode="before")
    @classmethod
    def required_string_fields(cls, v, info: ValidationInfo):
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError(f"{info.field_name} is required and cannot be empty")
        return v


class AzureOpenAIConfig(BaseLLMConfig):
    """Azure OpenAI: API key, endpoint, API version and deployment name are required."""

    api_key: str = Field(..., min_length=1, description="API key for authentication")
    endpoint: str = Field(..., min_length=1, description="Azure OpenAI resource endpoint")
    api_version: str = Field(..., min_length=1, description="Azure OpenAI API version")
    deployment_name: str = Field(..., min_length=1, description="Azure OpenAI deployment name")


class AzureFoundryConfig(BaseLLMConfig):
    """Azure AI Foundry: API key, endpoint, API version and deployment name are required."""

    api_key: str = Field(..., min_length=1, description="API key for authentication")
    endpoint: str = Field(..., min_length=1, description="Azure AI Foundry endpoint")
    deployment_name: str = Field(..., min_length=1, description="Deployment or model name")
    api_version: str = Field(..., min_length=1, description="Azure AI Foundry API version")


class VertexAIConfig(BaseLLMConfig):
    """Vertex AI: location and service-account credentials are required."""

    location: str = Field(..., min_length=1, description="GCP location (e.g. us-central1)")
    credentials_path: str = Field(..., min_length=1, description="Path to GCP credentials JSON file")
    scopes: list[str] = Field(default_factory=lambda: ["https://www.googleapis.com/auth/cloud-platform"],
                              description="OAuth scopes for GCP authentication")


class OpenAIConfig(BaseLLMConfig):
    """Native OpenAI: API key is required; endpoint, organization and project are optional."""

    api_key: str = Field(..., min_length=1, description="OpenAI API key")
    endpoint: Optional[str] = Field(default=None, description="Custom base URL for compatible APIs")
    organization: Optional[str] = Field(default=None, description="OpenAI organization ID")
    project: Optional[str] = Field(default=None, description="OpenAI project ID")


class AnthropicConfig(BaseLLMConfig):
    """Anthropic (Claude): API key is required. Prompt caching is on by default."""

    api_key: str = Field(..., min_length=1, description="Anthropic API key")
    endpoint: Optional[str] = Field(default=None, description="Custom base URL for an Anthropic-compatible gateway")
    enable_prompt_caching: bool = Field(
        default=True,
        description="Inject cache_control on system messages and tool definitions",
    )


CONFIG_REGISTRY = {
    "azure_openai": AzureOpenAIConfig,
    "azure_foundry": AzureFoundryConfig,
    "vertexai": VertexAIConfig,
    "openai": OpenAIConfig,
    "anthropic": AnthropicConfig,
}
