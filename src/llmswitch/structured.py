"""
structured.py

Structured-output helpers shared across providers and the client's ``.parse``
surface, plus JSON normalization for provider SDK objects.

Contents:
    - is_pydantic_model_class: detect ``response_format=MyModel`` (.parse() style).
    - pydantic_to_response_format: build a strict OpenAI-style json_schema dict.
    - pydantic_to_text_format: build the Responses API ``text.format`` block.
    - deserialize_to_model / populate_parsed: best-effort deserialize into a model.
    - populate_response_parsed: same, for a unified Responses ``Response``.
    - to_jsonable: recursively normalize provider SDK objects to JSON-safe values.

"""


from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel

from .models import ProviderExtensions

logger = logging.getLogger(__name__)


def to_jsonable(obj: Any, _depth: int = 0, _seen: "set | None" = None) -> Any:
    """
    Recursively normalize provider SDK objects into JSON-serializable structures.

    Several unified-response fields (``citations``, ``context``, ``citation_metadata``,
    ``grounding_metadata``) pass through native provider objects — Pydantic models,
    Google proto-plus messages, Azure SDK models. Stored raw, they make
    ``ChatCompletion.model_dump_json()`` fail. This converts them to plain
    dicts/lists/scalars while leaving ordinary values untouched, so the unified
    response is always serializable. Best-effort: anything it can't convert is
    stringified rather than left to break serialization downstream.

    Args:
        obj: Any value, possibly a nested SDK/Pydantic object.

    Returns:
        A JSON-serializable equivalent (dict, list, str, int, float, bool, or None).
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Depth + cycle guards so the "never crash" contract holds for self-referential
    # or pathologically nested structures.
    if _depth > 64:
        return str(obj)
    if isinstance(obj, (dict, list, tuple, set)):
        oid = id(obj)
        if _seen is not None and oid in _seen:
            return "<cycle>"
        _seen = (_seen or set()) | {oid}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1, _seen) for k, v in obj.items()}
    if isinstance(obj, set):
        # Sort for deterministic output (sets have no stable order).
        return [to_jsonable(v, _depth + 1, _seen) for v in sorted(obj, key=str)]
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, _depth + 1, _seen) for v in obj]

    # Pydantic v2 models (OpenAI/Azure SDK, Anthropic citations) — prefer JSON mode
    # so leaves like datetimes are already serialized.
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json", warnings=False)
        except TypeError:
            try:
                return to_jsonable(model_dump(), _depth + 1, _seen)
            except Exception:  # noqa: BLE001 - best-effort normalization
                pass
        except Exception:  # noqa: BLE001
            pass

    # Google proto-plus / Pydantic v1 / misc SDK objects
    for attr in ("to_dict", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return to_jsonable(method(), _depth + 1, _seen)
            except Exception:  # noqa: BLE001
                continue

    # Last resort: stringify so serialization can never crash on an unknown object.
    return str(obj)


def unmapped_provider_fields(raw: Any, mapped_fields: set[str]) -> dict[str, Any]:
    """Return only provider payload fields not represented by a normalized model."""
    data = to_jsonable(raw)
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key not in mapped_fields}


def provider_extensions_from_raw(
    provider: str,
    raw: Any,
    mapped_fields: set[str],
) -> ProviderExtensions:
    """Build a versioned extension container from unmapped provider fields."""
    fields = unmapped_provider_fields(raw, mapped_fields)
    return ProviderExtensions(provider=provider, fields=fields)


def provider_extensions_for(provider: str, fields: dict[str, Any] | None = None) -> ProviderExtensions:
    """Build an empty or field-backed provider extension container."""
    return ProviderExtensions(provider=provider, fields=fields or {})


def is_pydantic_model_class(obj: Any) -> bool:
    """
    True when ``obj`` is a Pydantic ``BaseModel`` *subclass* (not an instance).

    This is the signal that a caller passed ``response_format=MyModel`` (the
    class itself, OpenAI ``.parse()`` style) rather than a ``ResponseFormat``
    object or a plain dict. Providers branch on this to deserialize the model
    output into ``message.parsed``.
    """
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _strictify_json_schema(schema: Any) -> Any:
    """
    Recursively make a JSON Schema satisfy OpenAI/Azure strict structured
    outputs: every object gets ``additionalProperties: false`` and lists *all*
    of its properties in ``required``.

    Fallback used only when the OpenAI SDK's own converter is unavailable. It
    intentionally does not strip unsupported keywords, so callers pair it with
    ``strict=False`` to avoid hard 400s on edge-case schemas.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    for defs_key in ("$defs", "definitions"):
        if isinstance(out.get(defs_key), dict):
            out[defs_key] = {k: _strictify_json_schema(v) for k, v in out[defs_key].items()}
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: _strictify_json_schema(v) for k, v in props.items()}
        out["required"] = list(props.keys())
        out["additionalProperties"] = False
    if "items" in out:
        out["items"] = _strictify_json_schema(out["items"])
    if isinstance(out.get("prefixItems"), list):
        out["prefixItems"] = [_strictify_json_schema(s) for s in out["prefixItems"]]
    for combiner in ("anyOf", "oneOf", "allOf"):
        if isinstance(out.get(combiner), list):
            out[combiner] = [_strictify_json_schema(s) for s in out[combiner]]
    return out


@lru_cache(maxsize=256)
def pydantic_to_response_format(model_cls: type[BaseModel]) -> dict[str, Any]:
    """
    Build a strict, OpenAI-style ``json_schema`` ``response_format`` dict from a
    Pydantic model class.

    Used by the streaming path of the OpenAI/Azure providers (where the SDK
    ``.parse()`` helper isn't available) and by Azure Foundry. Strict mode
    requires ``additionalProperties: false`` on every object and every property
    listed in ``required`` — so this delegates to the OpenAI SDK's own
    converter (identical to what ``.parse()`` sends) when available, falling
    back to a self-contained strict transform otherwise.

    Result is cached per model class — schema generation is pure and the same
    model is typically reused across many requests.
    """
    try:
        from openai.lib._parsing._completions import type_to_response_format_param

        response_format = type_to_response_format_param(model_cls)
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            return dict(response_format)
    except Exception:  # noqa: BLE001 - private SDK path; degrade gracefully
        logger.debug("OpenAI strict schema converter unavailable; using fallback transform.")

    # Fallback: best-effort strict schema. strict=False avoids 400s on schemas
    # that contain keywords OpenAI's strict mode does not support.
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_cls.__name__,
            "schema": _strictify_json_schema(model_cls.model_json_schema()),
            "strict": False,
        },
    }


@lru_cache(maxsize=256)
def pydantic_to_text_format(model_cls: type[BaseModel]) -> dict[str, Any]:
    """
    Build the Responses API ``text`` parameter from a Pydantic model class.

    The Responses API takes ``text={"format": {"type": "json_schema", "name": ...,
    "schema": ..., "strict": ...}}`` — a flattened shape, unlike Chat Completions'
    nested ``response_format.json_schema``. This reuses the same strict-schema
    transform and reshapes it.
    """
    response_format = pydantic_to_response_format(model_cls)
    json_schema = response_format.get("json_schema") or {}
    return {
        "format": {
            "type": "json_schema",
            "name": json_schema.get("name", model_cls.__name__),
            "schema": json_schema.get("schema", model_cls.model_json_schema()),
            "strict": json_schema.get("strict", False),
        }
    }


def deserialize_to_model(model_cls: type[BaseModel], content: Any) -> Optional[BaseModel]:
    """
    Best-effort deserialize a JSON string / dict into ``model_cls``.

    Mirrors what OpenAI's ``.parse()`` does for native providers, but for
    providers that only return a raw JSON string. Returns ``None`` (rather than
    raising) when the content is empty or cannot be validated, so a structured
    request never turns a usable text response into a hard error.
    """
    if content is None or content == "":
        return None
    try:
        if isinstance(content, (dict, list)):
            return model_cls.model_validate(content)
        if isinstance(content, (str, bytes, bytearray)):
            return model_cls.model_validate_json(content)
        return model_cls.model_validate(content)
    except Exception:  # noqa: BLE001 - parsing is best-effort
        # Some models wrap JSON in prose or code fences; try to recover a dict.
        if isinstance(content, (str, bytes, bytearray)):
            try:
                return model_cls.model_validate(json.loads(content))
            except Exception:  # noqa: BLE001
                return None
        return None


def populate_parsed(chat_completion: Any, model_cls: type[BaseModel]) -> Any:
    """
    Fill ``choice.message.parsed`` on a unified ``ChatCompletion`` for providers
    without a native ``.parse()``.

    For each choice, if ``parsed`` is not already set, deserialize the message
    ``content`` into ``model_cls``; if there is no content but a tool/function
    call carries the structured payload, deserialize its ``arguments`` instead.
    Mutates and returns the same object.
    """
    for choice in getattr(chat_completion, "choices", None) or []:
        message = getattr(choice, "message", None)
        if message is None or getattr(message, "parsed", None) is not None:
            continue
        parsed = deserialize_to_model(model_cls, getattr(message, "content", None))
        if parsed is None:
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                arguments = getattr(getattr(tool_calls[0], "function", None), "arguments", None)
                parsed = deserialize_to_model(model_cls, arguments)
        if parsed is not None:
            message.parsed = parsed
    return chat_completion


def populate_response_parsed(response: Any, model_cls: type[BaseModel]) -> Any:
    """
    Fill ``item.parsed`` on a unified Responses ``Response`` when the provider
    did not return a natively parsed object.

    Deserializes each message item's concatenated ``output_text`` into
    ``model_cls``, falling back to a function-call item's ``arguments``.
    Mutates and returns the same object.
    """
    for item in getattr(response, "output", None) or []:
        if getattr(item, "parsed", None) is not None:
            continue
        if getattr(item, "type", None) == "message":
            parsed = deserialize_to_model(model_cls, item.text)
        else:
            parsed = deserialize_to_model(model_cls, getattr(item, "arguments", None))
        if parsed is not None:
            item.parsed = parsed
    return response
