"""
models.py

This module defines Pydantic models for usage statistics, chat completions (streaming and non-streaming), and embeddings. These models are used for type validation, serialization, and documentation of the data structures exchanged between the LLM Switch and its clients or providers.

Sections:
    - Usage statistics models
    - Chat completion models (non-streaming and streaming)
    - Embedding models
"""


from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ImageUrl(BaseModel):
    """
    Multimodal image URL content part.
    """
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class PromptTokensDetails(BaseModel):
    """
    Details about prompt tokens, including audio and cached tokens.
    """
    audio_tokens: Optional[int] = Field(default=0, description="Number of audio tokens")
    cached_tokens: Optional[int] = Field(default=0, description="Number of cached tokens")


class CompletionTokensDetails(BaseModel):
    """
    Details about completion tokens, including accepted, rejected, audio, and reasoning tokens.
    """
    accepted_prediction_tokens: Optional[int] = Field(default=0)
    rejected_prediction_tokens: Optional[int] = Field(default=0)
    audio_tokens: Optional[int] = Field(default=0)
    reasoning_tokens: Optional[int] = Field(default=0)


class CompletionUsage(BaseModel):
    """
    Usage statistics for a completion, including token counts and details.
    """
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails | int | None = None
    prompt_tokens_details: PromptTokensDetails | dict[str, Any] | None = None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    audio_prompt_tokens: int = 0
    latency_checkpoint: Optional[dict[str, Any]] = None
    thinking_tokens: int = 0
    service_tier: Optional[str] = None
    cache_creation_5m_tokens: int = 0
    cache_creation_1h_tokens: int = 0
    web_search_requests: int = 0
    web_fetch_requests: int = 0

    @property
    def cached_tokens(self) -> int:
        """
        Convenience accessor: total prompt tokens served from any server-side cache.

        - OpenAI / Azure OpenAI: reads ``prompt_tokens_details.cached_tokens``
          (automatic prefix caching, ≥1024-token static prefixes).
        - Anthropic: reads ``cache_read_input_tokens``
          (explicit ``cache_control`` breakpoints).
        - Vertex AI Gemini: reads ``cache_read_input_tokens``
          (``CachedContent`` resource).
        """
        if isinstance(self.prompt_tokens_details, PromptTokensDetails):
            openai_cached = self.prompt_tokens_details.cached_tokens or 0
        elif isinstance(self.prompt_tokens_details, dict):
            openai_cached = int(self.prompt_tokens_details.get("cached_tokens", 0) or 0)
        elif hasattr(self.prompt_tokens_details, "cached_tokens"):
            openai_cached = int(getattr(self.prompt_tokens_details, "cached_tokens", 0) or 0)
        else:
            openai_cached = 0
        return openai_cached or (self.cache_read_input_tokens or 0)

    @property
    def cache_savings_pct(self) -> float:
        """
        Percentage of prompt tokens served from cache (0–100).
        Useful for quick cost-saving visibility in logs.
        """
        if not self.prompt_tokens:
            return 0.0
        return round(self.cached_tokens / self.prompt_tokens * 100, 2)


class ProviderExtensions(BaseModel):
    """Versioned provider-specific fields excluded from the portable response contract."""
    provider: str | None = None
    schema_version: int = 1
    fields: dict[str, Any] = Field(default_factory=dict)


class Function(BaseModel):
    """
    Represents a callable function in a chat completion message.
    """
    arguments: str
    name: str
    parsed_arguments: Optional[Any] = None


class ChatCompletionMessageToolCall(BaseModel):
    """
    Represents a tool call in a chat completion message.
    """
    id: str
    function: Function
    type: Literal["function"]


class ChatCompletionAudio(BaseModel):
    """
    Represents audio data in a chat completion message.
    """
    id: str
    data: str
    expires_at: int
    transcript: str


class ChatCompletionMessage(BaseModel):
    """
    Represents a message in a chat completion, including role, content, and tool calls.
    """
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ChatCompletionMessageToolCall]] = None
    function_call: Optional[Function] = None
    refusal: Optional[str] = None
    audio: Optional[ChatCompletionAudio] = None
    annotations: Optional[list[Any]] = None
    reasoning_content: Optional[Any] = None
    parsed: Optional[Any] = None
    thinking: Optional[str] = None
    thinking_signature: Optional[str] = None
    context: Optional[Any] = None
    citations: Optional[list[Any]] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        return self.extensions or ProviderExtensions(fields={})


class TopLogprob(BaseModel):
    """
    Represents a top log probability for a token.
    """
    token: str
    bytes: Optional[list[int]] = None
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    """
    Log probability details for a token in a chat completion.
    """
    token: str
    bytes: Optional[list[int]] = None
    logprob: float
    top_logprobs: list[TopLogprob]


class ChoiceLogprobs(BaseModel):
    """
    Log probability details for choices in a chat completion.
    """
    content: Optional[list[ChatCompletionTokenLogprob]] = None
    refusal: Optional[list[ChatCompletionTokenLogprob]] = None


class Choice(BaseModel):
    """
    Represents a single choice in a chat completion, including message and finish reason.
    """
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[ChoiceLogprobs] = None
    content_filter_results: Optional[Any] = None
    citation_metadata: Optional[Any] = None
    grounding_metadata: Optional[Any] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        return self.extensions or ProviderExtensions(fields={})


class ChatCompletion(BaseModel):
    """
    Represents a full chat completion response, including choices and usage.
    """
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: list[Choice]
    usage: Optional[CompletionUsage] = None
    service_tier: Optional[str] = None
    prompt_filter_results: Optional[list[Any]] = None
    store: Optional[bool] = None
    metadata: Optional[dict[str, str]] = None
    moderation: Optional[Any] = None
    stop_sequence: Optional[str] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """Return provider fields through the versioned extension container."""
        return self.extensions or ProviderExtensions(fields={})

    def __str__(self):
        return repr(self)


class ChoiceDeltaToolCallFunction(BaseModel):
    """
    Represents a function call in a tool call delta for streaming completions.
    """
    arguments: Optional[str] = None
    name: Optional[str] = None


class ChoiceDeltaToolCall(BaseModel):
    """
    Represents a tool call delta in streaming chat completions.
    """
    index: int
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    function: Optional[ChoiceDeltaToolCallFunction] = None


class ChoiceDelta(BaseModel):
    """
    Represents a delta (change) in a choice for streaming chat completions.
    """
    role: Literal["developer", "system", "user", "assistant", "tool", "model"] | None = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ChoiceDeltaToolCall]] = None
    function_call: Optional[ChoiceDeltaToolCallFunction] = None
    refusal: Optional[str] = None
    thinking: Optional[str] = None
    thinking_signature: Optional[str] = None
    citations: Optional[list[Any]] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        return self.extensions or ProviderExtensions(fields={})


class ChunkDelta(BaseModel):
    """
    Represents a chunk delta in streaming chat completions.
    """
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call",
        "finishreason.stop", "max_completion_tokens"]
    ] = None
    logprobs: Optional[ChoiceLogprobs] = None
    content_filter_results: Optional[Any] = None
    citation_metadata: Optional[Any] = None
    grounding_metadata: Optional[Any] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        return self.extensions or ProviderExtensions(fields={})


class ChatCompletionChunk(BaseModel):
    """
    Represents a streaming chat completion chunk, including choices and usage.
    """
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: list[ChunkDelta]
    usage: Optional[CompletionUsage] = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        return self.extensions or ProviderExtensions(fields={})

    def __str__(self):
        return repr(self)


class Usage(BaseModel):
    """
    Usage statistics for embeddings, including prompt and total tokens.
    """
    prompt_tokens: int
    total_tokens: int
    completion_tokens: Optional[int] = 0


class Embedding(BaseModel):
    """
    Represents a single embedding vector and its metadata.
    """
    embedding: list[float]
    index: int
    object: Optional[str] = "embedding"


class CreateEmbeddingResponse(BaseModel):
    """
    Represents the response for an embedding creation request.
    """
    data: list[Embedding]
    model: str | None = None
    object: str
    usage: Usage
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        return self.extensions or ProviderExtensions(fields={})
    created: int

    def __str__(self):
        return repr(self)
