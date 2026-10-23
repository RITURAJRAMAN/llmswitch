"""Provider-wide capability manifests used by execution planning."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from ..errors import LLMSwitchError


class CapabilitySupport(str, Enum):
    """Whether a target supports a capability, does not, or requires target validation."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderCapabilityManifest:
    """Static provider contract; deployment-specific overrides are applied later."""

    provider: str
    operations: Mapping[str, CapabilitySupport] = field(default_factory=dict)
    features: Mapping[str, CapabilitySupport] = field(default_factory=dict)

    def operation(self, name: str) -> CapabilitySupport:
        return self.operations.get(name, CapabilitySupport.UNSUPPORTED)

    def feature(self, name: str) -> CapabilitySupport:
        return self.features.get(name, CapabilitySupport.UNKNOWN)


_MANIFESTS: dict[str, ProviderCapabilityManifest] = {
    "openai": ProviderCapabilityManifest(
        provider="openai",
        operations=MappingProxyType({
            "chat.sync": CapabilitySupport.SUPPORTED,
            "chat.async": CapabilitySupport.SUPPORTED,
            "embeddings": CapabilitySupport.SUPPORTED,
            "responses": CapabilitySupport.SUPPORTED,
        }),
        features=MappingProxyType({
            "streaming": CapabilitySupport.SUPPORTED,
            "function_calling": CapabilitySupport.SUPPORTED,
            "structured_output": CapabilitySupport.SUPPORTED,
            "multimodal": CapabilitySupport.SUPPORTED,
            "background": CapabilitySupport.SUPPORTED,
            "conversations": CapabilitySupport.SUPPORTED,
        }),
    ),
    "azure_openai": ProviderCapabilityManifest(
        provider="azure_openai",
        operations=MappingProxyType({
            "chat.sync": CapabilitySupport.SUPPORTED,
            "chat.async": CapabilitySupport.SUPPORTED,
            "embeddings": CapabilitySupport.SUPPORTED,
            "responses": CapabilitySupport.UNKNOWN,
        }),
        features=MappingProxyType({
            "streaming": CapabilitySupport.SUPPORTED,
            "function_calling": CapabilitySupport.UNKNOWN,
            "structured_output": CapabilitySupport.UNKNOWN,
            "multimodal": CapabilitySupport.UNKNOWN,
            "background": CapabilitySupport.UNKNOWN,
            "conversations": CapabilitySupport.UNKNOWN,
        }),
    ),
    "azure_foundry": ProviderCapabilityManifest(
        provider="azure_foundry",
        operations=MappingProxyType({
            "chat.sync": CapabilitySupport.SUPPORTED,
            "chat.async": CapabilitySupport.SUPPORTED,
            "embeddings": CapabilitySupport.SUPPORTED,
            "responses": CapabilitySupport.UNSUPPORTED,
        }),
        features=MappingProxyType({
            "streaming": CapabilitySupport.SUPPORTED,
            "function_calling": CapabilitySupport.UNKNOWN,
            "structured_output": CapabilitySupport.UNKNOWN,
            "multimodal": CapabilitySupport.UNKNOWN,
        }),
    ),
    "anthropic": ProviderCapabilityManifest(
        provider="anthropic",
        operations=MappingProxyType({
            "chat.sync": CapabilitySupport.SUPPORTED,
            "chat.async": CapabilitySupport.SUPPORTED,
            "embeddings": CapabilitySupport.UNSUPPORTED,
            "responses": CapabilitySupport.SUPPORTED,
        }),
        features=MappingProxyType({
            "streaming": CapabilitySupport.SUPPORTED,
            "function_calling": CapabilitySupport.SUPPORTED,
            "structured_output": CapabilitySupport.SUPPORTED,
            "multimodal": CapabilitySupport.SUPPORTED,
            "prompt_caching": CapabilitySupport.SUPPORTED,
        }),
    ),
    "vertexai": ProviderCapabilityManifest(
        provider="vertexai",
        operations=MappingProxyType({
            "chat.sync": CapabilitySupport.SUPPORTED,
            "chat.async": CapabilitySupport.SUPPORTED,
            "embeddings": CapabilitySupport.UNSUPPORTED,
            "responses": CapabilitySupport.SUPPORTED,
        }),
        features=MappingProxyType({
            "streaming": CapabilitySupport.SUPPORTED,
            "function_calling": CapabilitySupport.UNKNOWN,
            "structured_output": CapabilitySupport.SUPPORTED,
            "multimodal": CapabilitySupport.SUPPORTED,
        }),
    ),
}


def provider_capabilities(provider: str) -> ProviderCapabilityManifest:
    """Return the static manifest for a configured provider."""
    try:
        return _MANIFESTS[provider]
    except KeyError as exc:
        raise ValueError(f"No capability manifest registered for provider: {provider}") from exc


def require_operation(provider: str, operation: str) -> ProviderCapabilityManifest:
    """Reject an operation only when the provider contract explicitly forbids it."""
    manifest = provider_capabilities(provider)
    if manifest.operation(operation) is CapabilitySupport.UNSUPPORTED:
        raise LLMSwitchError(f"Provider '{provider}' does not support operation '{operation}'", provider=provider)
    return manifest
