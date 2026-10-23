"""
ratelimit.tokenizer

Token counting and truncation backed by tiktoken.

The encoding is chosen from the model name alone. tiktoken maps OpenAI model
names (``gpt-4o``, ``o3-mini``, ``text-embedding-3-small``, ...) to their
encodings. Any other name, such as ``claude-opus-4-5`` or a deployment name,
uses ``DEFAULT_ENCODING``, the newest encoding shipped with tiktoken.

Resolved encodings are cached per model name, so the lookup and its fallback
run once per model.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Newest general-purpose encoding in tiktoken. It produces the same token ids as
# o200k_base for ordinary text, and adds special tokens for gpt-oss models.
DEFAULT_ENCODING = "o200k_harmony"

try:
    import tiktoken
except ImportError:
    logger.warning(
        "The 'tiktoken' library is not installed. "
        "Token counting and truncation will use an estimate."
    )
    tiktoken = None  # type: ignore[assignment]


@lru_cache(maxsize=256)
def _resolve_encoding(model_name: Optional[str]):
    """Return the tiktoken encoding for ``model_name``, or ``DEFAULT_ENCODING`` when the model is unknown.

    Returns ``None`` only when tiktoken is not installed.
    """
    if not tiktoken:
        return None
    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("Unexpected error resolving tokenizer for '%s': %s. Using %s.",
                           model_name, e, DEFAULT_ENCODING)
    try:
        return tiktoken.get_encoding(DEFAULT_ENCODING)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to load encoding '%s': %s. Using o200k_base.", DEFAULT_ENCODING, e)
        return tiktoken.get_encoding("o200k_base")


def fallback_count_tokens(text: str) -> int:
    """Estimate token count by word count * 1.3 (used when tiktoken is unavailable)."""
    return int(len(text.split()) * 1.3)


def _encode_length(text: str, model_name: Optional[str]) -> int:
    encoding = _resolve_encoding(model_name)
    if encoding is None:
        return fallback_count_tokens(text)
    try:
        return len(encoding.encode(text))
    except Exception as e:  # noqa: BLE001
        logger.warning("Unexpected error occurred during tokenization %s. Using fallback estimation.", e)
        return fallback_count_tokens(text)


# Chat loops send the same messages again on every call, so counts for ordinary-sized
# texts are cached. Long texts are counted directly to keep the cache small.
_CACHEABLE_CHARS = 8192
_count_cached = lru_cache(maxsize=1024)(_encode_length)


def count_tokens(text: str, model_name: Optional[str] = None) -> int:
    """Count tokens with the encoding for ``model_name``, falling back to estimation."""
    if len(text) <= _CACHEABLE_CHARS:
        return _count_cached(text, model_name)
    return _encode_length(text, model_name)


def truncate_message(message: str, max_tokens: int, truncate: bool = False,
                     model_name: Optional[str] = None) -> Optional[str]:
    """
    Truncate a message to fit within ``max_tokens``.

    Returns the original message untouched when ``truncate`` is False or when the
    message already fits; otherwise returns the truncated string.
    """
    if not truncate:
        return message

    encoding = _resolve_encoding(model_name)
    if encoding is None:
        logger.warning("Tiktoken not available. Cannot truncate message.")
        return message

    try:
        encoded_tokens = encoding.encode(message)
        if len(encoded_tokens) <= max_tokens:
            return message
        truncated_message = encoding.decode(encoded_tokens[:max_tokens])
        logger.info("Message truncated from %s to %s tokens.", len(encoded_tokens), max_tokens)
        return truncated_message
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to truncate message: %s", e)
        return message
