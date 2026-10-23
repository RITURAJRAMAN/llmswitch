"""
core.failover

Backup/failover helpers shared by the sync chat, async chat, and embedding
handlers. Centralizes building the backup provider from ``config.backup`` so the
three call sites no longer duplicate the config-merge + provider-resolution
logic.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from ..config import CONFIG_REGISTRY
from ..errors import LLMSwitchError
from .execution import RequestExecutionPlan, build_execution_plan

# Operational defaults inherited from the primary config when the backup config
# leaves them unset.
_INHERITED_KEYS = (
    "redis_url",
    "timeout",
    "max_retries",
    "retry_base_delay",
    "completion_context_window",
    "rpm",
    "tpm",
    "embedding_rpm",
    "embedding_tpm",
    "user_id",
    "enable_prompt_caching",
)


def merge_backup_config(primary_config: Any, backup_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge backup config with operational defaults from the primary config."""
    merged = dict(backup_cfg)
    for key in _INHERITED_KEYS:
        if merged.get(key) is None:
            value = getattr(primary_config, key, None)
            if value is not None:
                merged[key] = value
    return merged


def build_backup_config(primary_config: Any) -> Any | None:
    """Validate and return the backup's provider-specific configuration."""
    backup_cfg = getattr(primary_config, "backup", None)
    if not backup_cfg:
        return None
    if not isinstance(backup_cfg, dict):
        raise ValueError("backup must be a provider configuration dictionary")
    provider = backup_cfg.get("provider")
    if not isinstance(provider, str):
        raise ValueError("backup.provider is required and must be a string")
    backup_config_cls = CONFIG_REGISTRY.get(provider)
    if not backup_config_cls:
        raise ValueError(f"Backup provider config not found: {provider}")
    return backup_config_cls(**merge_backup_config(primary_config, backup_cfg))


def validate_backup_config(primary_config: Any) -> None:
    """Eagerly validate backup settings during switch construction."""
    build_backup_config(primary_config)


def validate_backup_for_registry(primary_config: Any, provider_registry: Mapping[str, type]) -> None:
    """Ensure a configured backup supports the operation before the primary is called."""
    backup_config = build_backup_config(primary_config)
    if backup_config is not None and backup_config.provider not in provider_registry:
        raise LLMSwitchError(
            f"Backup provider '{backup_config.provider}' does not support this switch operation."
        )


def prepare_backup_provider(
        config: Any,
        provider_registry: Mapping[str, type],
        instantiate_provider: Callable[[Any, type], Any],
) -> Any:
    """
    Build the backup provider declared in ``config.backup``.

    Returns ``None`` when no backup is configured. Raises ``LLMSwitchError``
    when the backup provider is configured but its config/provider class cannot
    be resolved (callers wrap this so the *primary* error is surfaced).
    """
    backup_config = build_backup_config(config)
    if backup_config is None:
        return None
    backup_provider_cls = provider_registry.get(backup_config.provider)
    if not backup_provider_cls:
        raise LLMSwitchError(f"Backup provider class not found: {backup_config.provider}")
    return instantiate_provider(backup_config, backup_provider_cls)


def build_backup_execution_plan(
    primary_config: Any,
    *,
    operation: str,
    normalized_request: dict[str, Any],
) -> tuple[Any, RequestExecutionPlan] | None:
    """Build a target-specific backup plan without leaking the primary request model."""
    backup_config = build_backup_config(primary_config)
    if backup_config is None:
        return None
    request = dict(normalized_request)
    request.pop("model", None)
    return backup_config, build_execution_plan(
        backup_config,
        operation=operation,
        normalized_request=request,
    )
