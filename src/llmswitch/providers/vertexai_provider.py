"""
vertexai_provider.py

Vertex AI provider for chat completions.
Translates the llmswitch's OpenAI-style surface into Google GenAI requests
while preserving the unified response models.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import mimetypes
import time
from typing import Any

from google import genai
from google.auth import load_credentials_from_file
from google.genai import errors as google_errors
from google.genai import types

from ..base import BaseChatProvider
from ..errors import AuthenticationError, ChatCompletionError, RateLimitExceededError
from ..models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionTokenLogprob,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChoiceLogprobs,
    ChunkDelta,
    CompletionUsage,
    Function,
    TopLogprob,
)
from ..structured import (
    is_pydantic_model_class,
    populate_parsed,
    provider_extensions_for,
    provider_extensions_from_raw,
    to_jsonable,
)
from ._responses_emulation import AsyncEmulatedResponsesProvider, SyncEmulatedResponsesProvider
from ._shared import (
    close_async_stream,
    close_stream,
    get_retry_after_seconds,
    merge_client_options,
    normalize_token_limit,
)

# Name resolves to "llmswitch.providers.vertexai_provider"
logger = logging.getLogger(__name__)

_HTTP_RATE_LIMIT = 429
_HTTP_AUTH_CODES = (401, 403)

GENERATE_CONTENT_CONFIG_FIELDS = set(types.GenerateContentConfig.model_fields.keys())


def _raise_mapped_vertex_error(exc: Exception, context: str) -> None:
    if isinstance(exc, ChatCompletionError):
        raise exc
    if isinstance(exc, google_errors.ClientError):
        if exc.code == _HTTP_RATE_LIMIT:
            raise RateLimitExceededError(get_retry_after_seconds(exc)) from exc
        if exc.code in _HTTP_AUTH_CODES:
            raise AuthenticationError(provider="vertexai") from exc
        raise ChatCompletionError(str(exc), status_code=exc.code, provider="vertexai") from exc
    if isinstance(exc, google_errors.ServerError):
        raise ChatCompletionError(str(exc), status_code=503, provider="vertexai") from exc
    logger.exception(context)
    raise ChatCompletionError(str(exc), status_code=getattr(exc, "status_code", 500), provider="vertexai") from exc


def _created_timestamp(response) -> int:
    """Unix time the response was created, or now when the SDK response has no ``create_time``."""
    create_time = getattr(response, "create_time", None)
    return int(create_time.timestamp()) if create_time is not None else int(time.time())


class _StreamState:
    """Running state while mapping one Vertex AI response stream."""

    def __init__(self, config: Any) -> None:
        self.last_usage: CompletionUsage | None = None
        self.last_response: Any = None
        self.requested_count: int = getattr(config, "candidate_count", None) or 1
        self.seen_candidates: set[int] = set()


class VertexAIChatProvider(BaseChatProvider):
    """
    Gemini ChatCompletion Provider for Vertex AI.
    Adapts Google-hosted Gemini models into a unified openai response format.
    Supports both streaming and non-streaming chat completions.
    """

    def __init__(self, **kwargs):
        """
        Initialize the VertexAIChatProvider.
        Loads credentials, sets up the Vertex AI client, and configures model parameters.
        Args:
            location (str): Vertex AI region.
            model (str): Model name or ID.
            credentials_path (str): Path to service account credentials.
            scopes (list): OAuth scopes for API access.
        Raises:
            ChatCompletionError: If credentials_path is not provided.
        """
        self.location = kwargs.get("location", "us-central1")
        self.model = kwargs.get("deployment_name") or kwargs.get("model", "gemini-2.5-pro")
        self.credentials_path = kwargs.get("credentials_path")
        self.scopes = kwargs.get("scopes", ["https://www.googleapis.com/auth/cloud-platform"])

        if not self.credentials_path:
            raise ChatCompletionError("Valid credentials_path must be provided for Vertex AI.")

        creds, project = load_credentials_from_file(self.credentials_path, scopes=self.scopes)
        self.project = project
        timeout = kwargs.get("timeout")
        http_options = types.HttpOptions(
            api_version="v1",
            timeout=math.ceil(float(timeout) * 1000) if timeout is not None else None,
        )
        self.client = genai.Client(**merge_client_options({
            "vertexai": True,
            "project": self.project,
            "location": self.location,
            "credentials": creds,
            "http_options": http_options,
        }, kwargs.get("client_options")))

    def create(self, messages: list, stream: bool = False, **kwargs):
        """
        Entry point for chat completion requests.
        Dispatches to streaming or non-streaming handler based on 'stream' argument.
        Args:
            messages (list): List of message dicts for prompt construction.
            stream (bool): Whether to use streaming response.
            **kwargs: Additional model/config parameters.
        Returns:
            ChatCompletion or generator of ChatCompletionChunk
        """
        # response_format=MyModel (.parse() style): pass the class straight to
        # google-genai as response_schema. The SDK sanitizes the schema for
        # Gemini (dropping keywords like title/$defs it rejects) and populates
        # response.parsed natively. populate_parsed is a belt-and-suspenders
        # fallback in case .parsed is empty (e.g. SDK version differences).
        parse_model = kwargs.get("response_format")
        if not is_pydantic_model_class(parse_model):
            parse_model = None

        if stream:
            return self._streaming_create(messages, **kwargs)
        result = self._non_streaming_create(messages, **kwargs)
        if parse_model is not None:
            populate_parsed(result, parse_model)
        return result

    def _stringify_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            stringified_parts = []
            for part in content:
                if part.get("type") == "text":
                    stringified_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url", "")
                    if image_url:
                        stringified_parts.append(f"[image: {image_url}]")
            return "\n".join(filter(None, stringified_parts))
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content)

    def _tool_call_id(self, index: int) -> str:
        return f"call_{index}"

    def _normalize_role(self, role: str | None) -> str:
        if not role or role == "model":
            return "assistant"
        return role

    def _normalize_finish_reason(self, finish_reason: Any, has_tool_calls: bool = False) -> str | None:
        if has_tool_calls:
            return "tool_calls"
        if finish_reason is None:
            return None

        raw_value = str(finish_reason).split(".")[-1].lower()
        mapping = {
            "stop": "stop",
            "max_tokens": "length",
            "max_output_tokens": "length",
            "safety": "content_filter",
            "blocklist": "content_filter",
            "prohibited_content": "content_filter",
            "spii": "content_filter",
            "recitation": "content_filter",
            "malformed_function_call": "function_call",
            "tool_calls": "tool_calls",
            "other": "stop",
            "unspecified": "stop",
        }
        return mapping.get(raw_value, "stop")

    def _extract_text_and_function_calls(self, candidate: Any) -> tuple[str, list[Any]]:
        parts = getattr(getattr(candidate, "content", None), "parts", []) or []
        text_parts = []
        function_calls = []
        for part in parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                function_calls.append(function_call)
        return "".join(text_parts), function_calls

    def _build_usage(self, response: Any) -> CompletionUsage:
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        completion_tokens = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        total_tokens = (getattr(usage, "total_token_count", 0) or 0) if usage else 0
        # cached_content_token_count: tokens served from a CachedContent resource
        cache_read = (getattr(usage, "cached_content_token_count", 0) or 0) if usage else 0
        thinking_tokens = (getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0
        return CompletionUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            cache_read_input_tokens=cache_read,
            thinking_tokens=thinking_tokens,
        )

    def _parse_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not arguments:
            return {}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"value": arguments}
        return {"value": arguments}

    def _normalize_tool_response(self, content: Any) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                return parsed if isinstance(parsed, dict) else {"content": parsed}
            except json.JSONDecodeError:
                return {"content": content}
        if content is None:
            return {"content": ""}
        return {"content": content}

    def _image_part_from_url(self, url: str) -> types.Part:
        if url.startswith("data:"):
            header, encoded = url.split(",", 1)
            mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
            return types.Part.from_bytes(data=base64.b64decode(encoded), mime_type=mime_type)
        mime_type = mimetypes.guess_type(url)[0]
        return types.Part.from_uri(file_uri=url, mime_type=mime_type)

    def _convert_content_to_parts(self, content: Any, allow_images: bool = True) -> list[types.Part]:
        if content is None:
            return []
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text":
                    text = part.get("text", "")
                    if text:
                        parts.append(types.Part.from_text(text=text))
                elif allow_images and part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url", "")
                    if image_url:
                        parts.append(self._image_part_from_url(image_url))
            return parts
        return [types.Part.from_text(text=self._stringify_content(content))]

    def _convert_tools(self, tools: Any) -> list[types.Tool]:
        if not tools:
            return []

        declarations = []
        for tool in tools:
            tool_data = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else tool
            if not isinstance(tool_data, dict) or tool_data.get("type") != "function":
                continue

            function_data = tool_data.get("function", {})
            declarations.append(
                types.FunctionDeclaration(
                    name=function_data.get("name"),
                    description=function_data.get("description"),
                    parameters_json_schema=function_data.get("parameters"),
                )
            )

        return [types.Tool(function_declarations=declarations)] if declarations else []

    def _build_tool_config(self, tool_choice: Any, has_tools: bool) -> types.ToolConfig | None:
        if not has_tools:
            return None

        mode = None
        allowed_function_names = None

        if tool_choice in (None, "auto"):
            mode = types.FunctionCallingConfigMode.AUTO
        elif tool_choice == "none":
            mode = types.FunctionCallingConfigMode.NONE
        elif tool_choice == "required":
            mode = types.FunctionCallingConfigMode.ANY
        elif isinstance(tool_choice, str):
            mode = types.FunctionCallingConfigMode.ANY
            allowed_function_names = [tool_choice]
        elif isinstance(tool_choice, dict):
            function_name = tool_choice.get("function", {}).get("name")
            mode = types.FunctionCallingConfigMode.ANY if function_name else types.FunctionCallingConfigMode.AUTO
            allowed_function_names = [function_name] if function_name else None

        if mode is None:
            return None

        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=mode,
                allowed_function_names=allowed_function_names,
            )
        )

    def _convert_response_format(self, response_format: Any) -> dict[str, Any]:
        if not response_format:
            return {}

        # response_format=MyModel: hand the class to the SDK as response_schema.
        # google-genai converts it to a Gemini-compatible Schema and fills
        # response.parsed for us.
        if is_pydantic_model_class(response_format):
            return {
                "response_mime_type": "application/json",
                "response_schema": response_format,
            }

        payload = response_format.model_dump(by_alias=True, exclude_none=True) if hasattr(response_format,
                                                                                          "model_dump") else response_format
        if not isinstance(payload, dict):
            return {}

        response_type = payload.get("type")
        if response_type == "json_object":
            return {"response_mime_type": "application/json"}

        if response_type == "json_schema":
            json_schema = payload.get("json_schema") or {}
            if hasattr(json_schema, "model_dump"):
                json_schema = json_schema.model_dump(by_alias=True, exclude_none=True)
            schema_body = json_schema.get("schema") if isinstance(json_schema, dict) else None
            config_updates = {"response_mime_type": "application/json"}
            if schema_body:
                config_updates["response_json_schema"] = schema_body
            return config_updates

        return {}

    def _map_tool_calls(self, function_calls: list[Any]) -> list[ChatCompletionMessageToolCall] | None:
        if not function_calls:
            return None

        return [
            ChatCompletionMessageToolCall(
                id=self._tool_call_id(index=index),
                type="function",
                function=Function(
                    name=getattr(function_call, "name", ""),
                    arguments=json.dumps(getattr(function_call, "args", {}) or {}),
                ),
            )
            for index, function_call in enumerate(function_calls)
        ]

    def _map_delta_tool_calls(self, function_calls: list[Any]) -> list[ChoiceDeltaToolCall] | None:
        if not function_calls:
            return None

        return [
            ChoiceDeltaToolCall(
                index=index,
                id=self._tool_call_id(index),
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=getattr(function_call, "name", None),
                    arguments=json.dumps(getattr(function_call, "args", {}) or {}),
                ),
            )
            for index, function_call in enumerate(function_calls)
        ]

    def _map_logprobs(self, candidate: Any) -> ChoiceLogprobs | None:
        logprobs_result = getattr(candidate, "logprobs_result", None)
        if not logprobs_result:
            return None
        chosen = getattr(logprobs_result, "chosen_candidates", []) or []
        top_list = getattr(logprobs_result, "top_candidates", []) or []
        if not chosen:
            return None
        content = []
        for i, chosen_token in enumerate(chosen):
            top_logprobs = []
            if i < len(top_list):
                for alt in (getattr(top_list[i], "candidates", []) or []):
                    top_logprobs.append(TopLogprob(
                        token=getattr(alt, "token", ""),
                        bytes=None,
                        logprob=float(getattr(alt, "log_probability", 0.0)),
                    ))
            content.append(ChatCompletionTokenLogprob(
                token=getattr(chosen_token, "token", ""),
                bytes=None,
                logprob=float(getattr(chosen_token, "log_probability", 0.0)),
                top_logprobs=top_logprobs,
            ))
        return ChoiceLogprobs(content=content or None)

    def _convert_messages(self, messages: list, explicit_system_instruction: str | None) -> tuple[
        list[types.Content], str | None]:
        contents = []
        system_parts = [explicit_system_instruction] if explicit_system_instruction else []
        tool_call_name_by_id: dict[str, str] = {}
        pending_tool_response_parts: list[types.Part] = []

        def flush_tool_responses() -> None:
            if pending_tool_response_parts:
                contents.append(types.Content(role="user", parts=list(pending_tool_response_parts)))
                pending_tool_response_parts.clear()

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role in {"system", "developer"}:
                flush_tool_responses()
                text = self._stringify_content(content)
                if text:
                    system_parts.append(text)
                continue

            if role == "user":
                flush_tool_responses()
                parts = self._convert_content_to_parts(content, allow_images=True)
                if parts:
                    contents.append(types.Content(role="user", parts=parts))
                continue

            if role == "assistant":
                flush_tool_responses()
                parts = self._convert_content_to_parts(content, allow_images=False)
                raw_tool_calls = message.get("tool_calls") or []
                for index, tool_call in enumerate(raw_tool_calls):
                    tool_call_data = tool_call.model_dump(exclude_none=True) if hasattr(tool_call,
                                                                                        "model_dump") else tool_call
                    if not isinstance(tool_call_data, dict):
                        continue
                    tool_call_id = tool_call_data.get("id") or self._tool_call_id(index)
                    function_data = tool_call_data.get("function", {})
                    function_name = function_data.get("name", "")
                    if not function_name:
                        continue
                    tool_call_name_by_id[tool_call_id] = function_name
                    parts.append(
                        types.Part.from_function_call(
                            name=function_name,
                            args=self._parse_tool_arguments(function_data.get("arguments")),
                        )
                    )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue

            if role == "tool":
                tool_name = message.get("name") or tool_call_name_by_id.get(message.get("tool_call_id", "")) or "tool"
                pending_tool_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response=self._normalize_tool_response(content),
                    )
                )
                continue

            flush_tool_responses()
            logger.warning("Unknown message role for Vertex AI: %s", role)

        flush_tool_responses()

        if not contents:
            contents = [types.Content(role="user", parts=[types.Part.from_text(text="")])]

        system_instruction = "\n".join(filter(None, system_parts)).strip() or None
        return contents, system_instruction

    def create_cached_content(
            self,
            *,
            contents: list[types.Content] | None = None,
            system_instruction: str | None = None,
            tools: list[types.Tool] | None = None,
            ttl_seconds: int = 3600,
            display_name: str = "llmswitch-cache",
            model: str | None = None,
    ) -> str:
        """
        Create a Vertex AI ``CachedContent`` resource for Gemini context caching.

        The returned ``cache_name`` can be passed as ``cached_content=cache_name``
        to subsequent ``create()`` calls so the static prefix is served from
        the cache instead of being re-billed as full input tokens.

        Args:
            contents: Pre-built ``types.Content`` objects (document chunks, etc.).
            system_instruction: System prompt text to include in the cached context.
            tools: Tool definitions to cache alongside the system prompt.
            ttl_seconds: Cache time-to-live in seconds (default 3600 = 1 hour).
            display_name: Human-readable label for the cached content resource.
            model: Override model name; defaults to ``self.model``.

        Returns:
            The ``CachedContent.name`` resource identifier (e.g.
            ``"projects/.../cachedContents/..."``).  Pass this as
            ``cached_content=<name>`` in subsequent calls.

        Raises:
            ChatCompletionError: If the Vertex AI API returns an error.
        """
        from google.genai.errors import ClientError, ServerError  # local import

        create_config: dict[str, Any] = {
            "display_name": display_name,
            "ttl": f"{ttl_seconds}s",
        }
        if contents:
            create_config["contents"] = contents
        if system_instruction:
            create_config["system_instruction"] = system_instruction
        if tools:
            create_config["tools"] = tools

        try:
            cached = self.client.caches.create(
                model=model or self.model,
                config=types.CreateCachedContentConfig(**create_config),
            )
            logger.info(
                "Vertex AI CachedContent created: name=%s model=%s ttl=%ss",
                cached.name,
                model or self.model,
                ttl_seconds,
            )
            return cached.name
        except (ClientError, ServerError) as e:
            raise ChatCompletionError(
                f"Failed to create CachedContent: {e}",
                status_code=getattr(e, "code", 500),
                provider="vertexai",
            ) from e

    def _build_request(self, messages: list, kwargs: dict[str, Any]) -> tuple[
        list[types.Content], types.GenerateContentConfig]:
        config_kwargs = dict(kwargs)
        config_kwargs.pop("stream_options", None)

        # --- Context caching: pull out cached_content name if provided ---
        cached_content_name: str | None = config_kwargs.pop("cached_content", None)

        explicit_system_instruction = config_kwargs.pop("system_instruction", None)
        contents, system_instruction = self._convert_messages(messages, explicit_system_instruction)

        tools = self._convert_tools(config_kwargs.pop("tools", None))
        tool_choice = config_kwargs.pop("tool_choice", None)
        config_kwargs.pop("parallel_tool_calls", None)

        response_format = self._convert_response_format(config_kwargs.pop("response_format", None))
        config_kwargs.update(response_format)

        normalize_token_limit(config_kwargs, "max_output_tokens")
        if "n" in config_kwargs and "candidate_count" not in config_kwargs:
            config_kwargs["candidate_count"] = config_kwargs.pop("n")
        if "stop" in config_kwargs and "stop_sequences" not in config_kwargs:
            stop = config_kwargs.pop("stop")
            config_kwargs["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        for unsupported_key in [
            "audio",
            "extra_body",
            "logit_bias",
            "metadata",
            "modalities",
            "prediction",
            "reasoning_effort",
            "store",
            "top_logprobs",
            "user",
        ]:
            config_kwargs.pop(unsupported_key, None)

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = tools

        tool_config = self._build_tool_config(tool_choice, bool(tools))
        if tool_config:
            config_kwargs["tool_config"] = tool_config

        # Attach CachedContent resource name when provided
        if cached_content_name:
            config_kwargs["cached_content"] = cached_content_name

        filtered_config_kwargs = {
            key: value
            for key, value in config_kwargs.items()
            if key in GENERATE_CONTENT_CONFIG_FIELDS and value is not None
        }
        return contents, types.GenerateContentConfig(**filtered_config_kwargs)

    def _map_completion(self, response: Any) -> ChatCompletion:
        """Map a ``generate_content`` response to a unified ``ChatCompletion``."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise ChatCompletionError("No candidates returned from Vertex AI.", 404, "vertexai")

        # Native structured outputs: google-genai deserializes the response
        # into the response_schema model and exposes it as response.parsed.
        choices = []
        for index, candidate in enumerate(candidates):
            text, function_calls = self._extract_text_and_function_calls(candidate)
            if len(candidates) == 1 and not function_calls:
                function_calls = list(getattr(response, "function_calls", None) or [])
            mapped_tool_calls = self._map_tool_calls(function_calls)
            finish_reason = self._normalize_finish_reason(
                getattr(candidate, "finish_reason", "stop"),
                has_tool_calls=bool(mapped_tool_calls),
            )
            role = self._normalize_role(getattr(getattr(candidate, "content", None), "role", None))
            choices.append(
                Choice(
                    index=getattr(candidate, "index", None) if getattr(candidate, "index", None) is not None else index,
                    message=ChatCompletionMessage(
                        role=role, content=text or None, tool_calls=mapped_tool_calls,
                        parsed=getattr(response, "parsed", None) if index == 0 else None,
                    ),
                    finish_reason=finish_reason,
                    logprobs=self._map_logprobs(candidate),
                    content_filter_results=getattr(candidate, "safety_ratings", None),
                    citation_metadata=to_jsonable(getattr(candidate, "citation_metadata", None)),
                    grounding_metadata=to_jsonable(getattr(candidate, "grounding_metadata", None)),
                )
            )
        extensions = provider_extensions_from_raw(
            "vertexai",
            response,
            {"response_id", "model_version", "candidates", "usage_metadata", "create_time", "parsed"},
        )
        return ChatCompletion(
            id=getattr(response, "response_id", None) or "vertexai-response",
            object="chat.completion",
            created=_created_timestamp(response),
            model=getattr(response, "model_version", None) or self.model,
            choices=choices,
            usage=self._build_usage(response),
            extensions=extensions,
        )

    def _non_streaming_create(self, messages: list, **kwargs):
        """Non-streaming chat completion; maps API errors to library errors."""
        try:
            contents, config = self._build_request(messages, kwargs)
            response = self.client.models.generate_content(
                model=kwargs.pop("model", self.model),
                contents=contents,
                config=config,
            )
            return self._map_completion(response)
        except Exception as e:
            _raise_mapped_vertex_error(e, "Chat completion error")

    def _streaming_create(self, messages: list, **kwargs):
        """Streaming chat completion; returns a generator of ``ChatCompletionChunk``."""
        try:
            contents, config = self._build_request(messages, kwargs)
            model = kwargs.pop("model", self.model)

            def _wrapped():
                response_stream = None
                try:
                    response_stream = self.client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    state = _StreamState(config)
                    for response in response_stream:
                        chunk = self._map_stream_response(response, state)
                        if chunk is not None:
                            yield chunk
                    final = self._finish_stream(state)
                    if final is not None:
                        yield final
                except Exception as e:
                    _raise_mapped_vertex_error(e, "Streaming chat completion error")
                finally:
                    if response_stream is not None:
                        close_stream(response_stream)

            return _wrapped()

        except Exception as e:
            _raise_mapped_vertex_error(e, "Streaming chat completion error")

    def _map_stream_response(self, response: Any, state: _StreamState) -> ChatCompletionChunk | None:
        """Map one streamed response to a chunk, or ``None`` when it carries nothing to emit."""
        if getattr(response, "usage_metadata", None):
            state.last_usage = self._build_usage(response)

        candidates = getattr(response, "candidates", None) or []
        if not candidates and not getattr(response, "function_calls", None):
            return None
        state.last_response = response

        chunk_choices = []
        for position, candidate in enumerate(candidates or [None]):
            index = getattr(candidate, "index", None)
            if index is None:
                index = position
            if candidate is not None:
                state.seen_candidates.add(index)
            text, function_calls = self._extract_text_and_function_calls(candidate) if candidate else ("", [])
            if len(candidates) <= 1 and not function_calls:
                function_calls = list(getattr(response, "function_calls", None) or [])
            mapped_tool_calls = self._map_delta_tool_calls(function_calls)
            finish_reason = self._normalize_finish_reason(
                getattr(candidate, "finish_reason", None) if candidate else None,
                has_tool_calls=bool(mapped_tool_calls),
            )
            role = self._normalize_role(
                getattr(getattr(candidate, "content", None), "role", None) if candidate else None)
            if not (text or mapped_tool_calls or finish_reason is not None):
                continue
            chunk_choices.append(
                ChunkDelta(
                    index=index,
                    delta=ChoiceDelta(role=role, content=text or None, tool_calls=mapped_tool_calls),
                    finish_reason=finish_reason,
                    content_filter_results=getattr(candidate, "safety_ratings", None) if candidate else None,
                    citation_metadata=to_jsonable(
                        getattr(candidate, "citation_metadata", None)) if candidate else None,
                    grounding_metadata=to_jsonable(
                        getattr(candidate, "grounding_metadata", None)) if candidate else None,
                )
            )
        if not chunk_choices:
            return None
        return ChatCompletionChunk(
            id=getattr(response, "response_id", None) or "vertexai-response",
            object="chat.completion.chunk",
            created=_created_timestamp(response),
            model=getattr(response, "model_version", None) or self.model,
            choices=chunk_choices,
            usage=None,
            extensions=provider_extensions_for("vertexai"),
        )

    def _finish_stream(self, state: _StreamState) -> ChatCompletionChunk | None:
        """Validate the candidate count and build the closing usage chunk, if usage was reported."""
        if state.requested_count > 1 and len(state.seen_candidates) < state.requested_count:
            raise ChatCompletionError(
                f"Vertex AI streaming returned {len(state.seen_candidates)} of {state.requested_count} "
                "requested candidates.",
                provider="vertexai",
            )
        if not state.last_usage:
            return None
        created = int(time.time())
        model = self.model
        response_id = "vertexai-response"
        if state.last_response is not None:
            response_id = getattr(state.last_response, "response_id", None) or response_id
            created = _created_timestamp(state.last_response)
            model = getattr(state.last_response, "model_version", None) or self.model
        return ChatCompletionChunk(
            id=response_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[],
            usage=state.last_usage,
            extensions=provider_extensions_for("vertexai"),
        )


class AsyncVertexAIChatProvider(VertexAIChatProvider):
    """Async Vertex AI chat on the google-genai ``aio`` client; request and response mapping are shared."""

    async def create(self, messages: list, stream: bool = False, **kwargs):  # type: ignore[override]
        parse_model = kwargs.get("response_format")
        if not is_pydantic_model_class(parse_model):
            parse_model = None
        if stream:
            return await self._astreaming_create(messages, **kwargs)
        result = await self._anon_streaming_create(messages, **kwargs)
        if parse_model is not None:
            populate_parsed(result, parse_model)
        return result

    async def _anon_streaming_create(self, messages: list, **kwargs):
        try:
            contents, config = self._build_request(messages, kwargs)
            response = await self.client.aio.models.generate_content(
                model=kwargs.pop("model", self.model),
                contents=contents,
                config=config,
            )
            return self._map_completion(response)
        except Exception as e:
            _raise_mapped_vertex_error(e, "Chat completion error")

    async def _astreaming_create(self, messages: list, **kwargs):
        try:
            contents, config = self._build_request(messages, kwargs)
            model = kwargs.pop("model", self.model)
            response_stream = await self.client.aio.models.generate_content_stream(
                model=model, contents=contents, config=config,
            )
        except Exception as e:
            _raise_mapped_vertex_error(e, "Streaming chat completion error")

        async def _wrapped():
            try:
                state = _StreamState(config)
                async for response in response_stream:
                    chunk = self._map_stream_response(response, state)
                    if chunk is not None:
                        yield chunk
                final = self._finish_stream(state)
                if final is not None:
                    yield final
            except Exception as e:
                _raise_mapped_vertex_error(e, "Streaming chat completion error")
            finally:
                await close_async_stream(response_stream)

        return _wrapped()

    async def aclose(self) -> None:
        await self.client.aio.aclose()


class VertexAIResponsesProvider(SyncEmulatedResponsesProvider):
    """Responses API emulated on top of Vertex AI chat completions."""

    chat_cls = VertexAIChatProvider
    provider_name = "vertexai"


class AsyncVertexAIResponsesProvider(AsyncEmulatedResponsesProvider):
    """Async Responses API emulated on top of Vertex AI chat completions."""

    chat_cls = AsyncVertexAIChatProvider
    provider_name = "vertexai"
