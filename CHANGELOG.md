# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.3] - 2026-10-23

### Added

- LangChain integration in `llmswitch.integrations.langchain`, installed with the `langchain` extra: `pip install "llmswitch-sdk[langchain]"`. Nothing is imported unless you use it, and a missing `langchain-core` gives the install command.
- `ChatLLMSwitch`, a LangChain chat model over `LLMSwitch` and `AsyncLLMSwitch`. It supports `invoke`, `ainvoke`, `stream`, `astream`, `bind_tools` (with `tool_choice`), `close` and `aclose`. Rate limits, retries, failover and `on_event` apply to its calls. It uses chat completions, so every provider that supports chat works, including Anthropic and Vertex AI.
- `run_tool_loop` and `arun_tool_loop`: a bounded tool loop that runs the tools the model asks for and sends the results back. Tool failures, unknown tool names and malformed arguments are returned to the model as error tool messages. `max_steps` limits the model calls, and `ToolLoopLimitError` is raised when the limit is reached.
- Message conversion between LangChain messages and llmswitch's chat format, including images in OpenAI and LangChain standard shapes.
- Examples `examples/langchain_chat.py` and `examples/langchain_tool_loop.py`.
- Fixed in this release, found in review of 0.1.2 behaviour:
  - A stream the caller stops early no longer leaves the circuit breaker stuck half-open. Every later call used to fail until the process restarted.
  - The in-process rate-limit table no longer refuses every new user once it holds 10,000 windows. It drops the oldest window and logs a warning once.
  - A denied request gives back its reservation, so it no longer uses up the window budget.
  - Anthropic honours a per-call `model` when no `deployment_name` is set, as the other providers do. A configured deployment still wins.
  - A prompt that Vertex AI blocks raises `ContentFilterError`, instead of a 404 for non-streaming calls or an empty stream.
  - Timeouts are no longer retried, because the provider may already have processed the request. Connection errors and 5xx responses are still retried.
  - Credential headers `x-api-key`, `x-goog-api-key`, `proxy-authorization` and `cookie` are redacted in `live_logs` output.
  - Anthropic logs a warning when it drops a content part it does not support.
  - `ToolLoopLimitError` is exported from `llmswitch`, and `ChatLLMSwitch` raises a clear error for image content outside user messages.
- A CI job, `test-core-only`, that runs the tests without LangChain installed, so the core package is checked on its own.

## [0.1.2] - 2026-10-23

### Added

- Rate limiting per `user_id`, with `rpm` and `tpm` limits for chat and `embedding_rpm` and `embedding_tpm` for embeddings. A request over the limit waits for the window to reopen and fails with `RateLimitExceededError` if the budget is still exhausted. `on_rate_limit="raise"` fails immediately instead of waiting, and `rate_limit_max_wait` raises when the wait would be longer than the given seconds.
- Optional shared rate limits across processes through Redis: `redis_url` with the `redis` extra. `rate_limit_failure_policy` chooses between in-process fallback, failing open, and failing closed when Redis is unavailable.
- Retries for transient errors (rate limits, connection errors and 5xx responses) with exponential backoff. `max_retries` and `retry_base_delay` control them, and `retry_deadline` caps how long one provider call keeps retrying.
- Circuit breaker for each provider deployment, enabled with `circuit_breaker=True` or a dict with `failure_threshold` and `reset_timeout`.
- Backup provider failover with `backup={"provider": ..., ...}`, used when the primary provider fails with a retryable error.
- Context handling for chat: `truncate=True` trims messages to fit `completion_context_window`. `context_output_reserve_tokens` and `context_overhead_tokens` reserve room for output and protocol overhead.
- Batched embeddings with `batching=True` and `batchsize`.
- Responses API for Anthropic and Vertex AI, emulated on top of chat completions, sync and async. Text and image input, function tools, structured output and streaming events work; server-side state, stored objects and hosted tools raise `UnsupportedCapabilityError`. A native OpenAI or Azure OpenAI primary can fail over to an emulated backup and the reverse.
- Async chat for Vertex AI. `AsyncLLMSwitch` now supports `vertexai` through the google-genai async client, including streaming, tool calls and structured output. It shares its request and response mapping with the sync provider.
- Native async embeddings for Azure AI Foundry. `AsyncLLMSwitch.embeddings` awaits the SDK's async client for `azure_foundry` and keeps the same rate limiting, batching, retries and failover. The other providers still run in a worker thread.
- `share_clients=True` to reuse one SDK client across clients that have the same configuration.
- `token_counter` to count tokens with your own function.
- `on_event` callback that receives `request`, `response`, `error`, `retry` and `failover` events with timing and token usage.
- `live_logs=True` on a call logs the request at INFO with secrets redacted. It is off by default and works the same for sync and async clients.
- `max_tokens`, `max_completion_tokens` and `max_output_tokens` are interchangeable on every provider. The library sends the name each provider expects.
- Every error has `provider`, `user_id` and `retry_after` attributes next to `status_code`.
- Exceptions `ContextLengthExceededError` and `RateLimitBackendUnavailableError`.
- Python 3.13 support.

### Changed

- Provider SDKs are loaded the first time a provider is used, so `import llmswitch` takes about 0.2 s instead of about 1.7 s.
- Unknown constructor options are rejected with a suggestion for the closest valid name, so a typo such as `max_retires` is no longer ignored. Options that belong to another provider are still accepted.
- Token counting uses `tiktoken`, which is now a core dependency. The encoding is chosen from the model name: OpenAI model names use their own encoding, and any other name uses the newest tiktoken encoding (`o200k_harmony`). Counts for ordinary-sized texts are cached.

### Fixed

- Vertex AI responses without an id, model version or creation time no longer fail. The creation time is now read as UTC.
- Azure OpenAI deployments with custom names no longer fail when a model needs `max_completion_tokens`.
- Unrecognized Vertex AI errors keep their `status_code`.
- Async chat and Responses calls now honor `live_logs`.

## [0.1.1] - 2026-10-22

### Added

- Responses API through `client.responses` on `LLMSwitch` and `AsyncLLMSwitch`, for `openai` and `azure_openai`:
  - `create`, `stream` and `parse` (structured output with a Pydantic model class).
  - `retrieve`, `delete`, `cancel`, `compact`.
  - `input_items.list` and `input_tokens.count`.
  - `conversations.create`, `retrieve`, `update`, `delete`, `items.list`, `items.retrieve` and `items.delete`.
- Responses request validation: unknown request fields are rejected before dispatch.
- `responses_capabilities` to turn individual Responses features on or off per client.
- `mcp_policy` that must explicitly allow remote MCP servers and connectors before they can be sent.
- `responses_api_version` for Azure OpenAI, which overrides the API version used for Responses calls.
- Responses tool helpers: `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`, `ComputerTool`, `McpTool`, `ConnectorTool` and `McpApprovalDecision`.
- Response models and error types: `Response`, `ResponseStreamEvent`, `ResponsesError`, `InvalidResponseRequestError`, `UnsupportedCapabilityError` and `ResponseNotFoundError`.
- A provider whose SDK fails to import now reports the import error and the upgrade command, instead of "Unsupported provider".

### Changed

- All provider SDKs are installed by default. `pip install llmswitch-sdk` now installs the Anthropic, Azure AI Inference, aiohttp and Google GenAI SDKs along with OpenAI.
- The `anthropic`, `azure-foundry`, `vertexai` and `all` extras are kept as empty groups so existing install commands keep working.

## [0.1.0] - 2026-10-22

### Added

- `LLMSwitch` (synchronous) and `AsyncLLMSwitch` (asynchronous) clients with an OpenAI-style
  `chat.completions.create`, `chat.completions.parse` and `embeddings.create` surface.
- Providers: `openai`, `azure_openai`, `azure_foundry`, `anthropic`, `vertexai`.
  - Chat: all five providers synchronously; native async chat for all except `vertexai`.
  - Embeddings: `openai`, `azure_openai`, `azure_foundry`. Async embeddings run the sync provider in a worker thread.
- Streaming chat completions (sync iterator, async iterator).
- Structured output through `parse()` with a Pydantic model class.
- Typed configuration per provider, validated at construction; unknown constructor fields are ignored.
- `client_options` to pass extra keyword arguments to the native SDK client constructor; reserved unified fields are rejected.
- Error hierarchy rooted at `LLMSwitchError`; provider SDK errors are translated into it.
- Optional extras: `anthropic`, `azure-foundry`, `vertexai`, `all`.
- MIT license.
