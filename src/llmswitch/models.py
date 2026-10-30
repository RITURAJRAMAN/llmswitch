"""
models.py

This module defines Pydantic models for usage statistics, chat completions (streaming and non-streaming), and embeddings. These models are used for type validation, serialization, and documentation of the data structures exchanged between the LLM Switch and its clients or providers.

Sections:
    - Usage statistics models
    - Chat completion models (non-streaming and streaming)
    - Embedding models
    - Responses API models
"""


from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    """Token counts for one call.

    Attributes:
        prompt_tokens (int): Tokens in the input.
        completion_tokens (int): Tokens in the output.
        total_tokens (int): The sum of the two.
        prompt_tokens_details, completion_tokens_details (optional): Breakdowns, such as cached and reasoning tokens.
        cache_read_input_tokens (int, optional): Input tokens served from the provider's prompt cache.
        thinking_tokens (int): Tokens the model spent thinking, when the provider reports them.
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
        """Input tokens served from a server-side prompt cache, whatever the provider.

        Returns:
            int: The cached prompt tokens. Read from ``prompt_tokens_details`` for OpenAI and Azure OpenAI, and from
            ``cache_read_input_tokens`` for Anthropic and Vertex AI.
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
        """The share of the prompt that came from the cache, for quick cost checks.

        Returns:
            float: A percentage from 0 to 100, rounded to two places. 0.0 when there were no prompt tokens.
        """
        if not self.prompt_tokens:
            return 0.0
        return round(self.cached_tokens / self.prompt_tokens * 100, 2)


class ProviderExtensions(BaseModel):
    """Versioned provider-specific data, kept apart from the portable response fields.

    Use it for values that only one provider sends, such as an Anthropic thinking block or a Gemini thought signature.
    Portable code does not need it.

    Attributes:
        provider (str, optional): The provider that produced the data, such as ``anthropic``.
        schema_version (int): The version of the layout of ``fields``. Defaults to 1.
        fields (dict): The provider-specific values.
    """
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
    """A tool the model asked to run.

    Attributes:
        id (str): The id that the tool result must refer to.
        type (str): Always ``function``.
        function (Function): The tool's ``name`` and its ``arguments``, a JSON string.
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
    """The message of a chat completion reply.

    Attributes:
        role (str): The role, usually ``assistant``.
        content (str or None): The text of the reply. None when the reply is only tool calls.
        tool_calls (list[ChatCompletionMessageToolCall], optional): The tools the model asked to run.
        parsed (object, optional): The instance of the Pydantic class passed to ``parse``.
        thinking (str, optional): The text of the model's thinking, for providers that expose it.
        thinking_signature (str, optional): The signature of the thinking block, which must be sent back unchanged.
        extensions (ProviderExtensions, optional): Provider-specific data, such as signatures to send back.
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
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
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
    """One reply in a ``ChatCompletion``.

    Attributes:
        index (int): The position of the reply in ``choices``.
        message (ChatCompletionMessage): The reply itself.
        finish_reason (str, optional): Why generation stopped, such as ``stop``, ``length`` or ``tool_calls``.
        logprobs (ChoiceLogprobs, optional): Token probabilities, when requested.
        content_filter_results (dict, optional): The provider's content filter results, when it sends them.
        extensions (ProviderExtensions, optional): Provider-specific data.
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
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})


class ChatCompletion(BaseModel):
    """A chat completion: the reply of ``chat.completions.create`` when ``stream`` is False.

    Attributes:
        id (str): The id of the completion.
        object (str): Always ``chat.completion``.
        created (int): The Unix time when the completion was created.
        model (str): The model that produced it.
        choices (list[Choice]): The replies. Usually one; more when ``n`` asks for several.
        usage (CompletionUsage, optional): Token counts for the call.
        system_fingerprint (str, optional): The backend configuration the reply came from, when the provider sends it.
        extensions (ProviderExtensions, optional): Provider-specific data.
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
    """The new content in one streamed chunk.

    Attributes:
        role (str, optional): The role of the message, usually ``assistant``. Sent on the first chunk.
        content (str, optional): The new text.
        tool_calls (list[ChoiceDeltaToolCall], optional): Fragments of tool calls. Join the ``arguments`` text across
            chunks that share an ``index``.
        thinking (str, optional): The new text of a thinking block, when the model thinks.
        thinking_signature (str, optional): The signature that closes a thinking block.
        extensions (ProviderExtensions, optional): Provider-specific data, such as a complete thinking block.
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
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})


class ChunkDelta(BaseModel):
    """The change to one reply in a streamed chunk.

    Attributes:
        index (int): The position of the reply in the completion.
        delta (ChoiceDelta): The new content, tool calls or thinking in this chunk.
        finish_reason (str, optional): Set on the final chunk for this reply.
        extensions (ProviderExtensions, optional): Provider-specific data.
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
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})


class ChatCompletionChunk(BaseModel):
    """One piece of a streamed chat completion.

    A stream is a sequence of chunks. Join the ``delta`` content of each chunk to rebuild the reply. The last chunk
    can carry ``usage`` and a ``finish_reason``.

    Attributes:
        id (str): The id of the completion this chunk belongs to.
        object (str): Always ``chat.completion.chunk``.
        created (int): The Unix time of the chunk.
        model (str): The model that produced it.
        choices (list[ChunkDelta]): The changes to each reply. Each has a ``delta`` and a ``finish_reason``.
        usage (CompletionUsage, optional): Token counts, sent on the final chunk when the provider reports them.
        extensions (ProviderExtensions, optional): Provider-specific data.
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
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
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
    """One embedding vector.

    Attributes:
        embedding (list[float]): The vector.
        index (int): The position of the input text this vector belongs to.
        object (str): Always ``embedding``.
    """
    embedding: list[float]
    index: int
    object: Optional[str] = "embedding"


class CreateEmbeddingResponse(BaseModel):
    """The reply of ``embeddings.create``.

    Attributes:
        data (list[Embedding]): One vector per input text, in the order of the input when ``restore_order`` is set.
        model (str): The model that produced the vectors.
        usage (Usage, optional): Token counts for the call.
        extensions (ProviderExtensions, optional): Provider-specific data.
    """
    data: list[Embedding]
    model: str | None = None
    object: str
    usage: Usage
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})
    created: int

    def __str__(self):
        return repr(self)


_RESPONSES_EXTRA = ConfigDict(extra="allow", populate_by_name=True)


class ResponseInputTokensDetails(BaseModel):
    """Breakdown of input tokens for a Responses API call."""
    model_config = _RESPONSES_EXTRA
    cached_tokens: Optional[int] = 0


class ResponseOutputTokensDetails(BaseModel):
    """Breakdown of output tokens for a Responses API call."""
    model_config = _RESPONSES_EXTRA
    reasoning_tokens: Optional[int] = 0


class ResponseUsage(BaseModel):
    """Token counts for one Responses call.

    Attributes:
        input_tokens (int): Tokens in the input.
        output_tokens (int): Tokens in the output, including reasoning tokens.
        total_tokens (int): The sum of the two.
        input_tokens_details (ResponseInputTokensDetails, optional): Breakdown of the input, such as cached tokens.
        output_tokens_details (ResponseOutputTokensDetails, optional): Breakdown of the output, such as reasoning
            tokens.
    """
    model_config = _RESPONSES_EXTRA
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: Optional[ResponseInputTokensDetails] = None
    output_tokens_details: Optional[ResponseOutputTokensDetails] = None

    @property
    def cached_tokens(self) -> int:
        """Input tokens served from the server-side prompt cache.

        Returns:
            int: The cached input tokens, or 0 when the provider sent no breakdown.
        """
        details = self.input_tokens_details
        return int(getattr(details, "cached_tokens", 0) or 0) if details else 0

    @property
    def reasoning_tokens(self) -> int:
        """Output tokens spent on reasoning, for reasoning models.

        Returns:
            int: The reasoning tokens, or 0 when the provider sent no breakdown.
        """
        details = self.output_tokens_details
        return int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0


class ResponseError(BaseModel):
    """Error block attached to a failed Response."""
    model_config = _RESPONSES_EXTRA
    code: Optional[str] = None
    message: Optional[str] = None


class ResponseIncompleteDetails(BaseModel):
    """Explains why a Response has status ``incomplete``."""
    model_config = _RESPONSES_EXTRA
    reason: Optional[str] = None


class WebSearchSource(BaseModel):
    """A source returned by a built-in web-search tool call."""
    model_config = _RESPONSES_EXTRA
    type: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None


class FileSearchResult(BaseModel):
    """A result returned by a built-in file-search tool call."""
    model_config = _RESPONSES_EXTRA
    file_id: Optional[str] = None
    filename: Optional[str] = None
    score: Optional[float] = None
    text: Optional[str] = None
    attributes: Optional[dict[str, Any]] = None


class CodeInterpreterOutput(BaseModel):
    """A provider-side code-interpreter output or artifact reference."""
    model_config = _RESPONSES_EXTRA
    type: Optional[str] = None
    logs: Optional[str] = None
    files: Optional[list[Any]] = None


class GeneratedImage(BaseModel):
    """A generated image result or provider image reference."""
    model_config = _RESPONSES_EXTRA
    id: Optional[str] = None
    image_url: Optional[str] = None
    b64_json: Optional[str] = None
    result: Optional[str] = None
    revised_prompt: Optional[str] = None


class ComputerAction(BaseModel):
    """A requested computer-use action returned for application-controlled execution."""
    model_config = _RESPONSES_EXTRA
    type: Optional[str] = None


class McpApprovalRequest(BaseModel):
    """A pending MCP approval request returned by a remote tool call."""
    model_config = _RESPONSES_EXTRA
    approval_request_id: str
    server_label: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None


class ResponseOutputItem(BaseModel):
    """One item of a response's ``output``: a message, a tool call or a reasoning block.

    Attributes:
        type (str): The kind of item, such as ``message``, ``function_call`` or ``reasoning``.
        id (str, optional): The id of the item.
        status (str, optional): The status of the item.
        content (list, optional): The parts of a message.
        name (str, optional): The tool name, for a tool call.
        arguments (str, optional): The tool arguments as a JSON string, for a tool call.
        call_id (str, optional): The id a tool result must refer to.
        parsed (object, optional): The parsed structured output, when ``parse`` was used.
    """
    model_config = _RESPONSES_EXTRA
    type: str
    id: Optional[str] = None
    status: Optional[str] = None
    role: Optional[str] = None
    content: Optional[list[Any]] = None
    summary: Optional[list[Any]] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    call_id: Optional[str] = None
    parsed: Optional[Any] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})

    @property
    def text(self) -> str:
        """The text of a message item.

        Returns:
            str: The concatenated ``output_text`` parts of a message item. An empty string for any other type.
        """
        parts: list[str] = []
        for part in self.content or []:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type == "output_text":
                value = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if value:
                    parts.append(value)
        return "".join(parts)


class ResponseMessageItem(ResponseOutputItem):
    """Assistant message output item."""
    type: Literal["message"] = "message"
    role: Optional[str] = "assistant"


class ResponseReasoningItem(ResponseOutputItem):
    """Reasoning output item, including provider-preserved encrypted content."""
    type: Literal["reasoning"] = "reasoning"
    encrypted_content: Optional[str] = None


class ResponseFunctionCallItem(ResponseOutputItem):
    """Model-requested custom function call."""
    type: Literal["function_call"] = "function_call"
    name: str
    call_id: str
    arguments: str = ""


class ResponseWebSearchCallItem(ResponseOutputItem):
    """Built-in web-search tool call."""
    type: Literal["web_search_call"] = "web_search_call"
    action: Optional[Any] = None

    @property
    def sources(self) -> list[WebSearchSource]:
        """Return typed sources when they were requested through ``include``."""
        action = self.action if isinstance(self.action, dict) else {}
        raw_sources = action.get("sources") or []
        return [WebSearchSource.model_validate(source) for source in raw_sources if isinstance(source, dict)]


class ResponseFileSearchCallItem(ResponseOutputItem):
    """Built-in file-search tool call."""
    type: Literal["file_search_call"] = "file_search_call"
    queries: Optional[list[str]] = None
    results: Optional[list[Any]] = None

    @property
    def search_results(self) -> list[FileSearchResult]:
        """Return typed results when they were requested through ``include``."""
        return [FileSearchResult.model_validate(result) for result in self.results or [] if isinstance(result, dict)]


class ResponseCodeInterpreterCallItem(ResponseOutputItem):
    """Built-in code-interpreter tool call."""
    type: Literal["code_interpreter_call"] = "code_interpreter_call"
    container_id: Optional[str] = None
    code: Optional[str] = None
    outputs: Optional[list[Any]] = None

    @property
    def interpreter_outputs(self) -> list[CodeInterpreterOutput]:
        """Return typed provider-side outputs and artifact references."""
        return [CodeInterpreterOutput.model_validate(output) for output in self.outputs or [] if isinstance(output, dict)]


class ResponseImageGenerationCallItem(ResponseOutputItem):
    """Built-in image-generation tool call."""
    type: Literal["image_generation_call"] = "image_generation_call"
    result: Optional[str] = None
    revised_prompt: Optional[str] = None

    @property
    def generated_image(self) -> GeneratedImage:
        """Return the typed generated-image result for this tool call."""
        return GeneratedImage(result=self.result, revised_prompt=self.revised_prompt)


class ResponseComputerCallItem(ResponseOutputItem):
    """Built-in computer-use tool call; execution remains application controlled."""
    type: Literal["computer_call"] = "computer_call"
    action: Optional[Any] = None
    actions: Optional[list[ComputerAction]] = None
    pending_safety_checks: Optional[list[Any]] = None

    @property
    def computer_action(self) -> Optional[ComputerAction]:
        """Return the legacy singular action without executing it."""
        return ComputerAction.model_validate(self.action) if isinstance(self.action, dict) else None

    @property
    def computer_actions(self) -> list[ComputerAction]:
        """Return this call's ordered actions, including legacy singular responses."""
        if self.actions is not None:
            return self.actions
        legacy_action = self.computer_action
        return [legacy_action] if legacy_action is not None else []


class ResponseMcpListToolsItem(ResponseOutputItem):
    """MCP server tool-list output item."""
    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    server_label: Optional[str] = None
    tools: Optional[list[Any]] = None


class ResponseMcpApprovalRequestItem(ResponseOutputItem):
    """Remote MCP tool call awaiting an application approval decision."""
    type: Literal["mcp_approval_request"] = "mcp_approval_request"
    id: str
    server_label: Optional[str] = None

    @property
    def approval_request(self) -> McpApprovalRequest:
        """The approval request this item asks the application to decide on.

        Returns:
            McpApprovalRequest: The pending call, with its approval request id, server label, tool name and arguments.
        """
        return McpApprovalRequest(approval_request_id=self.id, server_label=self.server_label,
                                  name=self.name, arguments=self.arguments)


class ResponseMcpCallItem(ResponseOutputItem):
    """Remote MCP or connector tool call output item."""
    type: Literal["mcp_call"] = "mcp_call"
    server_label: Optional[str] = None
    output: Optional[Any] = None
    error: Optional[Any] = None
    approval_request_id: Optional[str] = None

    @property
    def approval_request(self) -> Optional[McpApprovalRequest]:
        """The approval request behind this call, when the call needed approval.

        Returns:
            McpApprovalRequest or None: The request, or None when the call did not need approval.
        """
        if not self.approval_request_id:
            return None
        return McpApprovalRequest(approval_request_id=self.approval_request_id, server_label=self.server_label,
                                  name=self.name, arguments=self.arguments)


_RESPONSE_OUTPUT_ITEM_TYPES: dict[str, type[ResponseOutputItem]] = {
    "message": ResponseMessageItem,
    "reasoning": ResponseReasoningItem,
    "function_call": ResponseFunctionCallItem,
    "web_search_call": ResponseWebSearchCallItem,
    "file_search_call": ResponseFileSearchCallItem,
    "code_interpreter_call": ResponseCodeInterpreterCallItem,
    "image_generation_call": ResponseImageGenerationCallItem,
    "computer_call": ResponseComputerCallItem,
    "mcp_list_tools": ResponseMcpListToolsItem,
    "mcp_approval_request": ResponseMcpApprovalRequestItem,
    "mcp_call": ResponseMcpCallItem,
}


def response_output_item_from_data(data: dict[str, Any]) -> ResponseOutputItem:
    """Create a typed known output item or preserve an unknown type generically."""
    return _RESPONSE_OUTPUT_ITEM_TYPES.get(data.get("type", ""), ResponseOutputItem).model_validate(data)


class Response(BaseModel):
    """A Responses API response: the reply of ``responses.create``, ``retrieve``, ``cancel`` and ``wait``.

    Attributes:
        id (str): The id of the response. Use it with ``retrieve``, ``wait``, ``cancel`` and ``delete``.
        status (str): ``completed``, ``failed``, ``incomplete``, ``in_progress``, ``queued`` or ``cancelled``.
        output (list[ResponseOutputItem]): The items the model produced: messages, tool calls and reasoning.
        usage (ResponseUsage, optional): Token counts for the call.
        error (ResponseError, optional): Set when ``status`` is ``failed``.
        incomplete_details (ResponseIncompleteDetails, optional): Set when ``status`` is ``incomplete``.
        model (str): The model that produced it.
        extensions (ProviderExtensions, optional): Provider-specific data.
    """
    model_config = _RESPONSES_EXTRA
    id: str
    object: str = "response"
    created_at: Optional[float] = None
    completed_at: Optional[float] = None
    model: Optional[str] = None
    status: Optional[str] = None
    output: list[ResponseOutputItem] = Field(default_factory=list)
    usage: Optional[ResponseUsage] = None
    error: Optional[ResponseError] = None
    incomplete_details: Optional[ResponseIncompleteDetails] = None
    instructions: Optional[Any] = None
    metadata: Optional[dict[str, str]] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[Any] = None
    background: Optional[bool] = None
    parallel_tool_calls: Optional[bool] = None
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None
    text: Optional[Any] = None
    reasoning: Optional[Any] = None
    prompt: Optional[Any] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_options: Optional[Any] = None
    prompt_cache_retention: Optional[str] = None
    safety_identifier: Optional[str] = None
    service_tier: Optional[str] = None
    moderation: Optional[Any] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    truncation: Optional[str] = None
    user: Optional[str] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})

    @property
    def output_text(self) -> str:
        """All the assistant text of the response, joined in order.

        Returns:
            str: The text of every ``message`` item in ``output``. An empty string when there is none.
        """
        return "".join(item.text for item in self.output if item.type == "message")

    @property
    def output_parsed(self) -> Optional[Any]:
        """The structured output of a call to ``responses.parse``.

        Returns:
            object or None: An instance of the Pydantic class passed as ``text_format``, or None when no output was
                parsed.
        """
        for item in self.output:
            if item.parsed is not None:
                return item.parsed
        return None

    @property
    def encrypted_reasoning_content(self) -> list[str | None]:
        """Return encrypted reasoning payloads included for stateless continuation."""
        return [item.encrypted_content for item in self.output if isinstance(item, ResponseReasoningItem)
                and item.encrypted_content]

    @property
    def web_sources(self) -> list[WebSearchSource]:
        """The web sources that the built-in web search tool returned.

        Returns:
            list: The sources of every web search call in ``output``, in order.
        """
        return [source for item in self.output if isinstance(item, ResponseWebSearchCallItem)
                for source in item.sources]

    @property
    def file_search_results(self) -> list[FileSearchResult]:
        """The results that the built-in file search tool returned.

        Returns:
            list: The results of every file search call in ``output``, in order.
        """
        return [result for item in self.output if isinstance(item, ResponseFileSearchCallItem)
                for result in item.search_results]

    @property
    def code_interpreter_outputs(self) -> list[CodeInterpreterOutput]:
        """The outputs and artifact references of the code interpreter tool.

        Returns:
            list: The outputs of every code interpreter call in ``output``, in order.
        """
        return [output for item in self.output if isinstance(item, ResponseCodeInterpreterCallItem)
                for output in item.interpreter_outputs]

    @property
    def generated_images(self) -> list[GeneratedImage]:
        """Return generated-image results from typed image-generation tool calls."""
        return [item.generated_image for item in self.output if isinstance(item, ResponseImageGenerationCallItem)]

    @property
    def computer_actions(self) -> list[ComputerAction]:
        """The computer actions the model requested, for the application to run.

        Returns:
            list: The requested actions of every computer call in ``output``. Use the call items to keep their ids.
        """
        return [action for item in self.output if isinstance(item, ResponseComputerCallItem)
                for action in item.computer_actions]

    @property
    def pending_mcp_approvals(self) -> list[McpApprovalRequest]:
        """Return MCP actions that require an explicit application decision."""
        return [approval for item in self.output
                if isinstance(item, (ResponseMcpApprovalRequestItem, ResponseMcpCallItem))
                if (approval := item.approval_request) is not None]

    def __str__(self):
        return repr(self)


class ResponseItem(BaseModel):
    """An input or output item of a response, as listed by ``input_items.list`` or a conversation.

    Attributes:
        type (str): The kind of item.
        id (str, optional): The id of the item.
        role (str, optional): The role of a message item.
        content (list, optional): The parts of a message item.
        name, arguments, call_id (optional): The tool call fields, for a tool call item.
    """
    model_config = _RESPONSES_EXTRA
    type: Optional[str] = None
    id: Optional[str] = None
    status: Optional[str] = None
    role: Optional[str] = None
    content: Optional[list[Any]] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    call_id: Optional[str] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})


class ResponseItemList(BaseModel):
    """
    Cursor page of :class:`ResponseItem` objects, mirroring the SDK's
    ``SyncCursorPage`` fields.
    """
    model_config = _RESPONSES_EXTRA
    object: str = "list"
    data: list[ResponseItem] = Field(default_factory=list)
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self):
        return repr(self)


class InputTokenCountResponse(BaseModel):
    """Result of ``responses.input_tokens.count`` — a pre-flight token count."""
    model_config = _RESPONSES_EXTRA
    object: str = "response.input_tokens"
    input_tokens: int = 0
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})

    def __str__(self):
        return repr(self)


class CompactedResponse(BaseModel):
    """Result of ``responses.compact`` — a compacted conversation history."""
    model_config = _RESPONSES_EXTRA
    id: str
    object: str = "response.compacted"
    created_at: Optional[float] = None
    output: list[ResponseOutputItem] = Field(default_factory=list)
    usage: Optional[ResponseUsage] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})

    def __str__(self):
        return repr(self)


class ResponseDeleted(BaseModel):
    """Result of ``responses.delete``."""
    model_config = _RESPONSES_EXTRA
    id: str
    object: str = "response"
    deleted: bool = True
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})

    def __str__(self):
        return repr(self)


class Conversation(BaseModel):
    """A conversation stored with the provider.

    Attributes:
        id (str): The id of the conversation.
        object (str): Always ``conversation``.
        created_at (int, optional): The Unix time when it was created.
        metadata (dict, optional): The metadata set when it was created or last updated.
        extensions (ProviderExtensions, optional): Provider-specific data.
    """
    model_config = _RESPONSES_EXTRA
    id: str
    object: str = "conversation"
    created_at: Optional[float] = None
    metadata: Optional[dict[str, str]] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})


class ConversationDeleted(BaseModel):
    """Result of deleting a server-stored conversation."""
    model_config = _RESPONSES_EXTRA
    id: str
    object: str = "conversation.deleted"
    deleted: bool = True
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})


class ResponseStreamEvent(BaseModel):
    """One server-sent event of a streamed Responses call.

    Events arrive in order. Text arrives as ``*.delta`` events, and a terminal event
    (``response.completed``, ``response.failed`` or ``response.incomplete``) carries the final response.

    Attributes:
        type (str): The kind of event, such as ``response.output_text.delta``.
        sequence_number (int, optional): The position of the event in the stream.
        delta (str, optional): The new text or arguments, for delta events.
        item (ResponseOutputItem, optional): The item this event adds or completes.
        response (Response, optional): The response, on lifecycle and terminal events.
    """
    model_config = _RESPONSES_EXTRA
    type: str
    sequence_number: Optional[int] = None
    item_id: Optional[str] = None
    output_index: Optional[int] = None
    content_index: Optional[int] = None
    delta: Optional[Any] = None
    text: Optional[str] = None
    item: Optional[Any] = None
    response: Optional[Response] = None
    extensions: Optional[ProviderExtensions] = None

    @property
    def provider_extensions(self) -> ProviderExtensions:
        """The provider-specific fields of this object, which sit outside the portable response contract.

        Returns:
            ProviderExtensions: The extension block, or an empty one when the provider sent none.
        """
        return self.extensions or ProviderExtensions(fields={})

    @property
    def usage(self) -> Optional[ResponseUsage]:
        """The usage of the response, read from a terminal event, so streams can be metered like chat streams.

        Returns:
            ResponseUsage or None: The usage from ``response``, or None for events that carry no response.
        """
        return self.response.usage if self.response is not None else None

    def __str__(self):
        return repr(self)


class ResponseCreatedEvent(ResponseStreamEvent):
    """Response lifecycle event emitted when a response is created."""
    type: Literal["response.created"] = "response.created"


class ResponseInProgressEvent(ResponseStreamEvent):
    """Response lifecycle event emitted while generation remains active."""
    type: Literal["response.in_progress"] = "response.in_progress"


class ResponseCompletedEvent(ResponseStreamEvent):
    """Terminal event emitted after successful response completion."""
    type: Literal["response.completed"] = "response.completed"


class ResponseFailedEvent(ResponseStreamEvent):
    """Terminal event emitted when the provider reports a response failure."""
    type: Literal["response.failed"] = "response.failed"


class ResponseIncompleteEvent(ResponseStreamEvent):
    """Terminal event emitted when generation ends incomplete."""
    type: Literal["response.incomplete"] = "response.incomplete"


class ResponseOutputTextDeltaEvent(ResponseStreamEvent):
    """Incremental output-text event."""
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    delta: Optional[str] = None


class ResponseFunctionCallArgumentsDeltaEvent(ResponseStreamEvent):
    """Incremental function-call arguments event."""
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"
    delta: Optional[str] = None


class ResponseReasoningSummaryDeltaEvent(ResponseStreamEvent):
    """Incremental reasoning-summary event."""
    type: Literal["response.reasoning_summary_text.delta"] = "response.reasoning_summary_text.delta"
    delta: Optional[str] = None


_RESPONSE_STREAM_EVENT_TYPES: dict[str, type[ResponseStreamEvent]] = {
    "response.created": ResponseCreatedEvent,
    "response.in_progress": ResponseInProgressEvent,
    "response.completed": ResponseCompletedEvent,
    "response.failed": ResponseFailedEvent,
    "response.incomplete": ResponseIncompleteEvent,
    "response.output_text.delta": ResponseOutputTextDeltaEvent,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDeltaEvent,
    "response.reasoning_summary_text.delta": ResponseReasoningSummaryDeltaEvent,
}


def response_stream_event_from_data(data: dict[str, Any]) -> ResponseStreamEvent:
    """Create a typed known stream event or retain future event types generically."""
    return _RESPONSE_STREAM_EVENT_TYPES.get(data.get("type", ""), ResponseStreamEvent).model_validate(data)
