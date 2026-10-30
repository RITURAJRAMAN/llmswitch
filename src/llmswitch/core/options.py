"""The options accepted by ``LLMSwitch`` and ``AsyncLLMSwitch``, as a typed dictionary.

The constructors take ``**kwargs: Unpack[LLMSwitchOptions]``. Editors then list the option names as you type, and
mypy reports an option that is misspelled or has the wrong type. This module has no runtime effect: the constructors
still validate the options exactly as before, and a provider-specific option the dictionary does not list is still
accepted at runtime.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Required, TypedDict


class LLMSwitchOptions(TypedDict, total=False):
    """Constructor options for ``LLMSwitch`` and ``AsyncLLMSwitch``. See ``LLMSwitch.__init__`` for the meaning of each.

    ``provider`` is required. Every other option is optional and has the default described in the docstring.
    """

    # Identity and provider
    provider: Required[Literal["openai", "azure_openai", "azure_foundry", "anthropic", "vertexai"]]
    model: str
    deployment_name: str
    api_key: str
    endpoint: str
    api_version: str
    responses_api_version: str
    location: str
    credentials_path: str
    scopes: list[str]
    organization: str
    project: str
    timeout: float
    enable_prompt_caching: bool
    client_options: dict[str, Any]
    user_id: str

    # Rate limits
    rpm: int
    tpm: int
    embedding_rpm: int
    embedding_tpm: int
    on_rate_limit: Literal["wait", "raise"]
    rate_limit_max_wait: float
    redis_url: str
    redis_connect_timeout: float
    rate_limit_failure_policy: Literal["memory_fallback", "fail_open", "fail_closed"]

    # Reliability
    max_retries: int
    retry_base_delay: float
    retry_deadline: float
    circuit_breaker: bool | dict[str, Any]
    backup: dict[str, Any]

    # Context and tokens
    completion_context_window: int
    context_output_reserve_tokens: int
    context_overhead_tokens: int
    token_counter: Callable[[str, str | None], int]

    # Responses
    mcp_policy: dict[str, Any]
    responses_capabilities: dict[str, Any]

    # Other
    on_event: Callable[[dict[str, Any]], None]
    share_clients: bool
