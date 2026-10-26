# llmswitch

One Python client for chat completions, embeddings and the Responses API across OpenAI,
Azure OpenAI, Azure AI Foundry, Anthropic and Vertex AI, with rate limits, retries, a circuit
breaker and backup providers. Switch providers by changing the constructor
arguments; the call sites stay the same.

Supported in 0.1.3: synchronous and asynchronous chat completions (including streaming and
structured output), embeddings, the Responses API for OpenAI, Azure OpenAI, Anthropic and Vertex AI, rate limiting,
retries, a circuit breaker, backup failover, context truncation, and an agentic chat model with a
bounded tool loop.

## Installation

```bash
pip install llmswitch-sdk
```

This installs every supported provider SDK: OpenAI, Azure OpenAI, Azure AI Foundry, Anthropic and
Vertex AI. Supports Python 3.11, 3.12 and 3.13.

A provider's SDK is loaded the first time you use that provider, so `import llmswitch` is fast and
only the providers you use are imported.

Shared rate limits across processes need Redis: `pip install "llmswitch-sdk[redis]"`.

Agentic features need the `agentic` extra: `pip install "llmswitch-sdk[agentic]"`. See
[Agentic features](#agentic-features).

The optional extras `anthropic`, `azure-foundry` and `vertexai` are still accepted, so existing
install commands keep working, but they add nothing new. `all` installs Redis and the agentic extra.

## Quick start

```python
from llmswitch import LLMSwitch

client = LLMSwitch(provider="openai", api_key="sk-...", model="gpt-4o")

reply = client.chat.completions.create(
    messages=[{"role": "user", "content": "Say hello in French."}],
)
print(reply.choices[0].message.content)

vectors = client.embeddings.create(["first text", "second text"], model="text-embedding-3-small")
print(len(vectors.data), len(vectors.data[0].embedding))
```

Async:

```python
import asyncio
from llmswitch import AsyncLLMSwitch

async def main():
    async with AsyncLLMSwitch(provider="anthropic", api_key="sk-ant-...", model="claude-opus-4-5") as client:
        reply = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(reply.choices[0].message.content)

asyncio.run(main())
```

Streaming returns an iterator of chunks:

```python
for chunk in client.chat.completions.create(messages=messages, stream=True):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

With `AsyncLLMSwitch`, await the call first, then iterate with `async for`:

```python
stream = await client.chat.completions.create(messages=messages, stream=True)
async for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Structured output takes a Pydantic model class:

```python
from pydantic import BaseModel

class City(BaseModel):
    name: str
    country: str

parsed = client.chat.completions.parse(
    messages=[{"role": "user", "content": "Largest city in Japan?"}],
    response_format=City,
)
print(parsed.choices[0].message.parsed)
```

## Providers

| `provider` | Required settings | Chat (sync / async) | Embeddings |
|---|---|---|---|
| `openai` | `api_key` | yes / yes | yes |
| `azure_openai` | `api_key`, `endpoint`, `api_version`, `deployment_name` | yes / yes | yes |
| `azure_foundry` | `api_key`, `endpoint`, `api_version`, `deployment_name` | yes / yes | yes |
| `anthropic` | `api_key` | yes / yes | no |
| `vertexai` | `location`, `credentials_path` | yes / yes | no |

Async embeddings use the provider's native async client for `azure_foundry`; `openai` and `azure_openai` run in a worker thread.

Optional settings: `timeout` (seconds) for every provider; `endpoint`, `organization` and
`project` for `openai`; `endpoint` and `enable_prompt_caching` (default `True`) for `anthropic`;
`scopes` for `vertexai`.

Azure OpenAI example:

```python
client = LLMSwitch(
    provider="azure_openai",
    api_key="...",
    endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-10-21",
    deployment_name="your-gpt4o-deployment",
)
```

When `deployment_name` is set it is always used as the model. A per-call `model` argument
only overrides the configured model when no deployment is configured.

Token limits can be given as `max_tokens`, `max_completion_tokens` or `max_output_tokens` on every
provider. The library sends the name that provider expects. This matters for newer OpenAI models,
which reject `max_tokens`, and for Anthropic, which requires it.

Vertex AI example:

```python
client = LLMSwitch(
    provider="vertexai",
    location="us-central1",
    credentials_path="/path/to/service_account.json",
    model="gemini-2.5-pro",
)
```

Gemini 2.5 models use part of the output budget for internal reasoning before they write the answer.
A small limit can therefore end the reply early: the response has `finish_reason="length"` in chat,
`status="incomplete"` in Responses, and `parse()` may return no parsed value. In testing, a
`max_output_tokens` of 300 was not enough for a short structured answer, while 2000 was. Set a
generous limit for Gemini calls, and check `finish_reason` or `status` before trusting the output.

Credentials are never read from the environment or from `.env` files by the library. Load
them in your application and pass them to the constructor.

## Native SDK options

Pass extra arguments for the provider's native client with `client_options`. They are
given to the SDK client constructor as keyword arguments:

```python
client = LLMSwitch(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
    client_options={"max_retries": 3, "default_headers": {"X-Team": "search"}},
)
```

Options that the unified constructor sets itself are rejected with a `ConfigurationError`:
`api_key`, `api_version`, `azure_endpoint`, `azure_deployment`, `base_url`, `endpoint`,
`organization`, `project`, `timeout`, `credential`, `credentials`, `location`, `vertexai`
and `http_options`. Use the unified argument for those instead.

Constructor options are checked. A name that no provider defines, such as the typo `max_retires`,
raises a `ConfigurationError` that suggests the closest valid name. Options that belong to another
provider are accepted, so one set of settings works with any provider.

For chat completions and embeddings, per-call keyword arguments are passed to the provider request
unchanged, so provider-specific parameters such as `seed`, `top_k` or `extra_body` can be given on
`create()`. Responses requests are validated first, as described below.

## Responses API

The Responses API is available through `client.responses` for `openai`, `azure_openai`, `anthropic` and `vertexai`. OpenAI and Azure OpenAI use the service's own Responses API; Anthropic and Vertex AI [are emulated](#responses-on-anthropic-and-vertex-ai) on top of chat completions.

```python
from pydantic import BaseModel

client = LLMSwitch(provider="openai", api_key="sk-...", model="gpt-4o")

reply = client.responses.create(input="Summarize the llmswitch README in one line.")
print(reply.output_text)

for event in client.responses.stream(input="Write a haiku about Python."):
    if event.type == "response.output_text.delta":
        print(event.delta, end="")

class City(BaseModel):
    name: str
    country: str

parsed = client.responses.parse(input="Largest city in Japan?", text_format=City)
print(parsed.output_parsed)
```

Other calls: `retrieve(response_id)`, `delete(response_id)`, `cancel(response_id)`,
`compact()`, `input_items.list(response_id)`, `input_tokens.count()`, and `conversations`.

Which of these calls work depends on the model and endpoint. Token counting and the conversations
API are provided by the service and may return an error for a given deployment. `compact()` also
depends on the service. Errors from the service are raised as `ResponsesError` with the service's
message.

Responses requests are checked before they are sent. Unknown request fields raise
`InvalidResponseRequestError`. Remote MCP servers and connectors are rejected unless the client
allows them, for example:

```python
client = LLMSwitch(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
    mcp_policy={"enabled": True, "allowed_server_labels": ["docs"]},
)
```

Azure OpenAI calls to `/responses` need API version `2025-03-01-preview` or later. Older
`api_version` values are raised to the minimum automatically. Set `responses_api_version` to choose
a different version for Responses calls only.

`azure_foundry` has no Responses API and raises `ResponsesError` with status 501.

### Responses on Anthropic and Vertex AI

For `anthropic` and `vertexai`, `client.responses.create()` (and `stream()`, `parse()`, and the async client) translate the request into a chat completion and the result back into a `Response`, so the same code runs on every Responses-capable provider:

```python
client = LLMSwitch(provider="anthropic", api_key="sk-ant-...", model="claude-opus-4-5")
reply = client.responses.create(input="Hello", instructions="Be brief.")
print(reply.output_text, reply.usage.total_tokens)
```

What works: text and image input, `instructions`, function tools and `tool_choice`, function-call round trips (`function_call` and `function_call_output` items), `text` formats and `text_format` for structured output, `temperature`, `top_p`, `max_output_tokens`, streaming, and the usual rate limits, retries and failover. Streams produce the Responses events (`response.created`, `response.output_text.delta`, `response.function_call_arguments.delta`, the `.done` events, and `response.completed` or `response.incomplete`). A response cut off by the token limit has `status="incomplete"`.

What does not work, because chat completions cannot provide it: server-side state (`previous_response_id`, `conversation`, `store=True`, `background`), stored objects (`retrieve`, `delete`, `cancel`, `compact`, `input_items`, `input_tokens.count`, `conversations`) and hosted tools (web search, file search, code interpreter, MCP and the rest). These raise `UnsupportedCapabilityError` before any request is sent, so keep the conversation history in your application and pass it as `input` items. `client.responses.capabilities().emulated` is `True` for these providers. Fields that have no chat equivalent (`metadata`, `user`, `truncation`, `include`, `reasoning` and similar) are accepted and ignored. Response ids are generated locally and cannot be used to retrieve anything.

## Agentic features

Install the extra, then use `ChatLLMSwitch` as a chat model:

```python
from llmswitch.agentic import ChatLLMSwitch

llm = ChatLLMSwitch(provider="anthropic", api_key="sk-ant-...", model="claude-opus-4-5", temperature=0.2)
print(llm.invoke("Say hello in French.").content)

chain = prompt | llm   # prompts, parsers and other pipeline steps compose with it
```

Options the model understands are fields: `temperature`, `max_tokens`, `stop` and `call_options`
(extra keyword arguments sent with every request). Every other keyword argument is a switch option,
passed to `LLMSwitch`, so `provider`, `api_key`, `model`, `backup`, `on_event` and the rest work the
same way as they do for `LLMSwitch`.

`bind_tools` sends tool objects, Pydantic classes, plain functions or OpenAI-style dicts:

```python
llm_with_tools = llm.bind_tools([get_weather], tool_choice="auto")
reply = llm_with_tools.invoke("Weather in Paris?")
reply.tool_calls   # [{"name": "get_weather", "args": {...}, "id": "...", "type": "tool_call"}]
```

`tool_choice` accepts `"auto"`, `"none"`, `"required"` (or `"any"`), or a tool name to force that tool.
Arguments that are not valid JSON appear in `reply.invalid_tool_calls` instead of raising.

### Tool loop

`run_tool_loop` runs the tools the model asks for and returns the whole conversation:

```python
from llmswitch.agentic import run_tool_loop

history = run_tool_loop(llm, "Weather in Paris, and 17 plus 25?", [get_weather, add], max_steps=4)
print(history[-1].content)
```

- Each turn that requests tools runs them in order, and the results go back to the model as tool messages.
- A tool that raises, an unknown tool name, and arguments that fail the tool's schema are sent back as
  error tool messages, so the model can recover.
- `max_steps` is the most model calls allowed. If the model still asks for tools on the last one,
  `ToolLoopLimitError` is raised.
- `arun_tool_loop` is the async version. Tools run one after another in both versions.

The loop is deliberately basic. It has no routing between agents, no specialists, no checkpoints and
no budgets. Those are planned for a later release.

### Message conversion limits

- Images are supported in user messages only. An image in a system, assistant or tool message raises
  `ValueError` with a clear message.
- A provider's own reasoning blocks (for example Anthropic `thinking` and `redacted_thinking`) and its
  native tool-use blocks are not sent back. Tool calls are sent through `tool_calls`. If you turn on
  extended thinking with tools, a provider can reject a continued tool loop, because the signed thinking
  block is missing from the history.
- `tool_choice=False` is sent as `"none"`, which tells the model not to call tools.

### Closing clients

Call `llm.close()`, or `await llm.aclose()` for async use, to close the provider connections. Put the call
in a `finally` block, as the examples do.

### Responses with the agentic model

`ChatLLMSwitch` uses chat completions only. For the Responses API, use `LLMSwitch` directly.

## Rate limits, retries and failover

Limits are tracked per `user_id` (default `"global"`):

```python
client = LLMSwitch(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
    user_id="team-search",
    rpm=60,                 # requests per minute
    tpm=200_000,            # tokens per minute
    embedding_rpm=300,
    embedding_tpm=1_000_000,
)
```

A request over the limit waits for the window to reopen. If the budget is still exhausted, it fails
with `RateLimitExceededError`. By default the limits live in the process. To share them across
processes, set `redis_url`. If Redis is unavailable, `rate_limit_failure_policy` decides what happens:
`"memory_fallback"` (default) uses in-process limits, `"fail_open"` allows the request, and
`"fail_closed"` raises `RateLimitBackendUnavailableError`.

A request that is denied does not use up budget: its reservation is given back, so a later request
that fits in the window is still allowed. In-process limits keep at most 10,000 user windows. When that
table is full, the oldest window is dropped and a warning is logged once. That user's window then
restarts, so the limit is weaker for that window, but no user is refused for lack of room. Use Redis
for exact limits with many users.

Streams that do not report token usage count zero completion tokens against `tpm`.

To fail instead of waiting, set `on_rate_limit="raise"`. To wait only for short delays, set
`rate_limit_max_wait` (seconds): a longer wait raises `RateLimitExceededError` immediately.

Transient errors are retried: rate limits, connection errors and 5xx responses. Timeouts are not
retried, because the provider may already have processed the request and a retry could bill it twice.
A timeout does count as a failure for the circuit breaker, and it does trigger backup failover.
`max_retries` (default 3) and `retry_base_delay` control the exponential backoff. `retry_deadline` (seconds) caps how
long one provider call keeps retrying: when the next wait would pass it, the last error is raised.

```python
client = LLMSwitch(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
    max_retries=2,
    circuit_breaker={"failure_threshold": 5, "reset_timeout": 30},
    backup={"provider": "azure_openai", "api_key": "...", "endpoint": "...",
            "api_version": "2024-10-21", "deployment_name": "gpt-4o"},
)
```

The circuit breaker stops calls to a deployment after repeated failures (timeouts, connection errors and
5xx responses) and retries it after `reset_timeout` seconds. If you stop reading a stream before it ends, call `close()` on it (or let it go out of scope), so the
breaker's trial slot is released. `backup` is used when the primary provider
fails with a retryable error or times out.
The backup provider must support the same operation.

Embedding failover is all or nothing. If the primary fails part-way through a batched request, the
backup embeds the whole input again, so every vector comes from one model. Mixing vectors from two
embedding models would make them incomparable.

`client_options` retries are separate: `max_retries` inside `client_options` is passed to the SDK,
while `max_retries` on the client is the library's retry loop.

Chat requests can be trimmed to fit the context window with `truncate=True` (set
`completion_context_window`). Without it, a request that is too long raises
`ContextLengthExceededError`. Embeddings can be sent in batches:

```python
vectors = client.embeddings.create(texts, batching=True, batchsize=32)
```

Pass `share_clients=True` to reuse one SDK client across clients with the same configuration.

Token counts use `tiktoken`. OpenAI model names use their own encoding, and any other name uses the
newest tiktoken encoding, so counts for other providers are estimates. To count with your own
function, pass `token_counter=fn`, where `fn(text, model_name)` returns an integer. It is used for
limits and context checks; truncation still cuts text with tiktoken.

## Observability

Pass `on_event` to receive a dict for each step of a call:

```python
def log_event(event):
    print(event["event"], event["operation"], event["provider"], event.get("latency"))

client = LLMSwitch(provider="openai", api_key="sk-...", model="gpt-4o", on_event=log_event)
```

| `event` | When | Extra fields |
|---|---|---|
| `request` | a call is about to be sent | `stream` |
| `response` | the call returned (for a stream, once it is established) | `stream`, `latency`, `usage` |
| `error` | the call failed | `error`, `status_code`, `latency` |
| `retry` | a transient error is about to be retried | `attempt`, `max_retries`, `delay`, `error`, `status_code` |
| `failover` | the primary provider failed and the backup is used | `error`, `status_code`, and `from_provider` and `to_provider` for calls that are not streams |

Every event also has `operation` (`chat.completions`, `embeddings`, `responses`, or
`responses.management` with a `verb` such as `retrieve`), `provider` and `model`. The callback runs on the calling thread. If it raises, the error is logged and the call
continues.

Request details are not logged by default. Pass `live_logs=True` on a call to log it at INFO. The
messages or input are replaced with a placeholder, and the other parameters are logged with secrets
redacted, so keep sensitive text out of parameters such as `instructions` when you turn it on.

## Errors

All errors derive from `LLMSwitchError`:

- `ConfigurationError`: invalid or missing constructor settings.
- `RateLimitBackendUnavailableError`: Redis is unavailable and `rate_limit_failure_policy` is `"fail_closed"`.
- `ContextLengthExceededError`: the request is too long for the context window and `truncate` is off.
- `ChatCompletionError`, `EmbeddingGenerationError`: provider failures. `status_code` carries the HTTP status when known.
- `AuthenticationError`, `APIConnectionError`, `RateLimitExceededError`, `TimeOutError`, `ContentFilterError`, `ModelNotFoundError`, `InvalidResponseError`.
- `ToolLoopLimitError`: `run_tool_loop` reached `max_steps` while the model still asked for tools.
- `ResponsesError`: Responses API failures and unsupported providers. `InvalidResponseRequestError` for invalid Responses requests, `UnsupportedCapabilityError` for features turned off for the target.

Every error has `status_code`, `provider`, `user_id` and, for rate limits, `retry_after`. These are `None`
when the value is not known.

Calling an operation a provider does not support (for example embeddings on `anthropic`)
raises `LLMSwitchError` before any request is sent.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest
ruff check src tests
```

The tests use mocks and do not contact any provider. The agentic tests skip when the agentic extra
is not installed, so `pip install -e ".[redis,dev]"` runs the core suite on its own.

## License

MIT. See [LICENSE](LICENSE).
