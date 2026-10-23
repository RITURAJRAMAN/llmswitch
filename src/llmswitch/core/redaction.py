"""
core.redaction

Helper for stripping sensitive fields out of structures before they are logged.
"""
from __future__ import annotations

from typing import Any

# Exact (case-insensitive) key names whose values must never be logged. Exact
# match — not substring — so benign keys like ``max_tokens`` are not caught by
# ``token``.
_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "api_key", "apikey", "api-key", "x-api-key", "x-goog-api-key", "openai-api-key",
    "endpoint", "azure_endpoint", "base_url",
    "redis_url",
    "credentials", "credentials_path",
    "authorization", "auth", "proxy-authorization", "cookie", "set-cookie",
    "password", "passwd",
    "secret", "client_secret",
    "token", "access_token", "refresh_token", "id_token",
    "connection_string", "connectionstring",
    "sas_token", "private_key",
})

_REDACTED = "*REDACTED*"


def redact_sensitive(data: Any) -> Any:
    """
    Recursively redact sensitive values from a structure for safe logging.

    Any dict key whose (case-insensitive) name is in :data:`_SENSITIVE_KEYS` has
    its value replaced with ``*REDACTED*``; dicts and lists are traversed so that
    credentials nested inside ``kwargs`` / ``extra_body`` are scrubbed too. The
    input is not mutated — a redacted copy is returned (scalars pass through
    unchanged).
    """
    if isinstance(data, dict):
        return {
            k: (_REDACTED if isinstance(k, str) and k.lower() in _SENSITIVE_KEYS
                else redact_sensitive(v))
            for k, v in data.items()
        }
    if isinstance(data, (list, tuple)):
        return [redact_sensitive(v) for v in data]
    return data
