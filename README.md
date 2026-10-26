# llmswitch

One Python client for chat completions, embeddings and the Responses API across OpenAI, Azure OpenAI,
Azure AI Foundry, Anthropic and Vertex AI. Switch providers by changing the constructor arguments; the
call sites stay the same. Rate limits, retries, a circuit breaker, backup failover and context handling
are built in, and they behave the same for every provider.

Supported in 0.1.3: sync and async chat completions (streaming and structured output), embeddings, the
Responses API, rate limiting, retries, a circuit breaker, backup failover, context truncation, batching,
observability hooks, and an agentic chat model with a bounded tool loop.

Supports Python 3.11, 3.12 and 3.13.

## Contents

- [Which feature to use](#which-feature-to-use)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Providers and their settings](#providers-and-their-settings)
- [Chat completions](#chat-completions)
- [Embeddings](#embeddings)
- [Responses API](#responses-api)
- [Agentic features](#agentic-features)
- [Rate limits](#rate-limits)
- [Retries, timeouts and the circuit breaker](#retries-timeouts-and-the-circuit-breaker)
- [Backup failover](#backup-failover)
- [Context, truncation and token counting](#context-truncation-and-token-counting)
- [Performance: clients, sharing and lazy loading](#performance-clients-sharing-and-lazy-loading)
- [Observability](#observability)
- [Configuration reference](#configuration-reference)
- [Errors](#errors)
- [Development](#development)
- [License](#license)

## Which feature to use

| If you need to... | Use | Why |
|---|---|---|
| Get a reply from a model | `chat.completions.create` | The same call works for every provider. |
| Show the reply as it is generated | `create(..., stream=True)` | The first words arrive sooner, and long answers feel faster. |
| Get a typed object back | `chat.completions.parse` with a Pydantic class | You get validated data instead of parsing text yourself. |
| Turn text into vectors | `embeddings.create` | For search, clustering and similarity. Use `batching=True` for large lists. |
| Use OpenAI's newer Responses API | `responses.create` | Stateful and tool-rich features. Native for OpenAI and Azure OpenAI; emulated for Anthropic and Vertex AI. |
| Run tools in a loop with a model | `ChatLLMSwitch` and `run_tool_loop` | A bounded loop that runs the tools the model asks for, and feeds the results back. |
| Limit cost and traffic per user | `rpm`, `tpm`, `user_id` | Stops one user or job from using the whole quota. |
| Share limits across many processes | `redis_url` | In-process limits cannot see other processes. |
| Survive short provider outages | `max_retries`, `retry_base_delay` | Retries transient errors with exponential backoff. |
| Stop hammering a broken provider | `circuit_breaker=True` | Fails fast after repeated failures, then probes again later. |
| Keep serving when a provider is down | `backup={...}` | Moves the request to a second provider. |
| Fit long conversations into a context window | `truncate=True` | Trims the oldest content instead of failing. |
| Measure what happens on each call | `on_event` | Gives you retries, failovers, latency and token usage. |

## Installation

```bash
pip install llmswitch-sdk
```

This installs the SDKs for OpenAI, Azure OpenAI and Foundry, Anthropic and Vertex AI. An SDK is loaded
the first time you use its provider, so `import llmswitch` stays fast.

Optional extras:

| Extra | Install | Use it when |
|---|---|---|
| `redis` | `pip install "llmswitch-sdk[redis]"` | You run several processes that must share rate limits. |
| `agentic` | `pip install "llmswitch-sdk[agentic]"` | You want the agentic chat model and tool loop. |
| `all` | `pip install "llmswitch-sdk[all]"` | You want Redis and the agentic extra together. |

The extras `anthropic`, `azure-foundry` and `vertexai` are still accepted so existing install commands
keep working. They add nothing new.

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

client.close()
```

`LLMSwitch` can also be used as a context manager, which closes it on exit:

```python
with LLMSwitch(provider="openai", api_key="sk-...", model="gpt-4o") as client:
    print(client.chat.completions.create(messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```

Async works the same way with `AsyncLLMSwitch`:

```python
import asyncio
from llmswitch import AsyncLLMSwitch

async def main():
    async with AsyncLLMSwitch(provider="anthropic", api_key="sk-ant-...", model="claude-opus-4-5") as client:
        reply = await client.chat.completions.create(messages=[{"role": "user", "content": "Hello"}])
        print(reply.choices[0].message.content)

asyncio.run(main())
```

## Providers and their settings

| `provider` | Required settings | Chat (sync / async) | Embeddings | Responses |
|---|---|---|---|---|
| `openai` | `api_key`, `model` | yes / yes | yes | native |
| `azure_openai` | `api_key`, `endpoint`, `api_version`, `deployment_name` | yes / yes | yes | native |
| `azure_foundry` | `api_key`, `endpoint`, `api_version`, `deployment_name` | yes / yes | yes | no |
| `anthropic` | `api_key`, `model` | yes / yes | no | emulated |
| `vertexai` | `location`, `credentials_path`, `model` | yes / yes | no | emulated |

Asynchronous embeddings use the native async client for `azure_foundry`. The other providers run their
synchronous embedding code in a worker thread, so the event loop stays free.

**Credentials.** The library never reads environment variables or `.env` files. Load your keys in your
application and pass them to the constructor. `credentials_path` for Vertex AI points to a service account
JSON file.

**Azure OpenAI example:**

```python
client = LLMSwitch(
    provider="azure_openai",
    api_key="...",
    endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-10-21",
    deployment_name="your-gpt4o-deployment",
)
```

**Vertex AI example:**

```python
client = LLMSwitch(
    provider="vertexai",
    location="us-central1",
    credentials_path="/path/to/service_account.json",
    model="gemini-2.5-pro",
)
```

**Model selection.** A configured `deployment_name` is always the model used, and a per-call `model` cannot
replace it. Without a deployment name, a per-call `model` overrides the configured `model`. Use this to
send one client's requests to several models.

**Token limits.** `max_tokens`, `max_completion_tokens` and `max_output_tokens` are interchangeable on every
provider. The library sends the name each provider expects. This matters for newer OpenAI models, which
reject `max_tokens`, and for Anthropic, which requires it.

**Gemini and the output budget.** Gemini 2.5 models use part of the output budget for internal reasoning.
A small limit can end the reply early: the response has `finish_reason="length"` in chat,
`status="incomplete"` in Responses, and `parse()` may return no parsed value. In testing, a
`max_output_tokens` of 300 was not enough for a short structured answer, while 2000 was. Set a generous
limit for Gemini calls, and check `finish_reason` or `status` before trusting the output.

**Anthropic prompt caching.** `enable_prompt_caching` (default `True`) adds cache markers to system messages
and tool definitions, so repeated prompts cost less. Set it to `False` if you need the raw request.

## Chat completions

**When to use it:** any time you need a model reply. Use `create` for a complete reply, `create(stream=True)`
for progressive output, and `parse` when you need a typed object.

```python
reply = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "Answer in one sentence."},
        {"role": "user", "content": "Why is the sky blue?"},
    ],
    temperature=0.2,
    max_tokens=200,
)
print(reply.choices[0].message.content, reply.usage.total_tokens)
```

**Streaming** returns an iterator of chunks. Stop early by breaking out of the loop, or call `close()` on
the stream:

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

**Structured output** takes a Pydantic model class. The parsed object is on `choice.message.parsed`:

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

**Provider-specific arguments.** Per-call keyword arguments such as `seed`, `top_k` or `extra_body` are
passed to the provider request unchanged. Constructor options that no provider defines are rejected with a
suggestion, so a typo such as `max_retires` fails at construction time.

**Per-call overrides.** `max_retries`, `truncate` and `live_logs` can be passed to any single call.

## Embeddings

**When to use it:** to turn text into vectors for search, clustering or comparison. Pass a string or a list.

```python
vectors = client.embeddings.create(
    ["first text", "second text"],
    model="text-embedding-3-small",
)
print(vectors.data[0].embedding[:5], vectors.usage.total_tokens)
```

**Batching** splits a large list by token budget and by `batchsize`, so each request stays within the
provider's limits. Results come back in input order unless you pass `restore_order=False`:

```python
vectors = client.embeddings.create(texts, batching=True, batchsize=32)
```

**Failover for embeddings is all or nothing.** If the primary fails part-way through a batched request, the
backup embeds the whole input again. Every vector then comes from one model, because mixing vectors from two
embedding models would make them incomparable.

Embeddings are available for `openai`, `azure_openai` and `azure_foundry`.

## Responses API

**When to use it:** when you need the Responses API's features: server-side conversation state, background
responses, hosted tools, or its structured output and streaming events. For `openai` and `azure_openai` these
are the service's own features. For `anthropic` and `vertexai`, Responses is emulated on top of chat
completions, so the same code runs on every provider that supports it.

```python
client = LLMSwitch(provider="openai", api_key="sk-...", model="gpt-4o")

reply = client.responses.create(input="Summarize the llmswitch README in one line.")
print(reply.output_text)

for event in client.responses.stream(input="Write a haiku about Python."):
    if event.type == "response.output_text.delta":
        print(event.delta, end="")

parsed = client.responses.parse(input="Largest city in Japan?", text_format=City)
print(parsed.output_parsed)
```

Other calls:

- `retrieve(response_id)`, `delete(response_id)` and `cancel(response_id)` manage stored responses.
- `compact()` shortens a conversation.
- `input_items.list(response_id)` lists the input items of a response.
- `input_tokens.count()` counts the tokens a request would use.
- `conversations` creates, reads, updates and deletes conversations and their items.
- `create_background(...)` starts a background response. `wait(response_id)` polls it until it finishes.
- `submit_mcp_approval(previous_response_id, approval)` answers an MCP approval request.

Which of these work depends on the model and the endpoint. Token counting and the conversations API are
provided by the service and may return an error for a given deployment. Service errors are raised as
`ResponsesError` with the service's message.

**Checks before sending.** Responses requests are validated first. Unknown request fields raise
`InvalidResponseRequestError`. Remote MCP servers and connectors are rejected unless you allow them:

```python
client = LLMSwitch(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
    mcp_policy={
        "enabled": True,
        "allowed_server_labels": ["docs"],        # or allowed_connector_ids / allowed_origins
    },
)
```

**Capabilities.** `client.responses.capabilities()` reports what the configured target supports. A feature
can be turned on or off with `responses_capabilities`, for example `{"background": True}`.

**API version.** Azure OpenAI calls to `/responses` need API version `2025-03-01-preview` or later. Older
`api_version` values are raised to that minimum automatically. Set `responses_api_version` to choose a
different version for Responses calls only.

`azure_foundry` has no Responses API. Calls to it raise `ResponsesError` with status 501.

### Responses on Anthropic and Vertex AI

For `anthropic` and `vertexai`, `responses.create()` (and `stream()`, `parse()` and the async client)
translate the request into a chat completion and the result back into a `Response`:

```python
client = LLMSwitch(provider="anthropic", api_key="sk-ant-...", model="claude-opus-4-5")
reply = client.responses.create(input="Hello", instructions="Be brief.")
print(reply.output_text, reply.usage.total_tokens)
```

**What works:** text and image input, `instructions`, function tools and `tool_choice`, function-call round
trips (`function_call` and `function_call_output` items), `text` formats and `text_format` for structured
output, `temperature`, `top_p`, `max_output_tokens`, streaming, and the usual rate limits, retries and
failover. Streams produce the Responses events (`response.created`, `response.output_text.delta`,
`response.function_call_arguments.delta`, the `.done` events, and `response.completed` or
`response.incomplete`). A response cut off by the token limit has `status="incomplete"`.

**What does not work**, because chat completions cannot provide it: server-side state
(`previous_response_id`, `conversation`, `store=True`, `background`), stored objects (`retrieve`, `delete`,
`cancel`, `compact`, `input_items`, `input_tokens.count`, `conversations`) and hosted tools (web search, file
search, code interpreter, MCP and the rest). These raise `UnsupportedCapabilityError` before any request is
sent. Keep the conversation history in your application and pass it as `input` items.

Fields with no chat equivalent (`metadata`, `user`, `truncation`, `include`, `reasoning` and similar) are
accepted and ignored. Response ids are generated locally and cannot be used to retrieve anything.

`client.responses.capabilities().emulated` is `True` for these providers.

## Agentic features

**When to use it:** when a model should call your functions in a loop, for example to look things up and
then answer. Use `ChatLLMSwitch` when you want a chat model object; use `run_tool_loop` when you want the
whole loop run for you. Install the `agentic` extra first.

```python
from llmswitch.agentic import ChatLLMSwitch

llm = ChatLLMSwitch(provider="anthropic", api_key="sk-ant-...", model="claude-opus-4-5", temperature=0.2)
print(llm.invoke("Say hello in French.").content)
```

`ChatLLMSwitch` accepts the same provider options as `LLMSwitch`. The model's own options are `temperature`,
`max_tokens`, `stop` and `call_options` (extra keyword arguments sent with every request). Every other
keyword argument is a switch option, so `backup`, `on_event`, the rate-limit settings and the rest work the
same way as they do for `LLMSwitch`. It supports `invoke`, `ainvoke`, `stream`, `astream`, `bind_tools`
and `close`/`aclose`. It is also a standard runnable, so it composes with prompts and parsers using `|`.

**Binding tools.** `bind_tools` sends tool objects, Pydantic classes, plain functions or OpenAI-style dicts
with every request:

```python
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"

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

- Each turn that requests tools runs them in order. The results go back to the model as tool messages.
- A tool that raises, an unknown tool name, and arguments that fail the tool's schema are sent back as
  error tool messages, so the model can recover.
- `max_steps` is the most model calls allowed. If the model still asks for tools on the last call,
  `ToolLoopLimitError` is raised and no tools run for that last turn.
- `arun_tool_loop` is the async version. Tools run one after another in both versions, so their order and
  side effects are predictable.
- Plain functions need a name. Functions created with `functools.partial` have none, so wrap them in a
  `def` or use a tool object.

The loop is deliberately basic. It has no routing between agents, no specialists, no checkpoints and no
budgets. Those are planned for a later release.

### Message conversion limits

- Images are supported in user messages only. An image in a system, assistant or tool message raises
  `ValueError` with a clear message.
- A provider's own reasoning blocks (for example Anthropic `thinking` and `redacted_thinking`) and its
  native tool-use blocks are not sent back. Tool calls are sent through `tool_calls`. If you turn on
  extended thinking with tools, a provider can reject a continued tool loop, because the signed thinking
  block is missing from the history.
- `tool_choice=False` is sent as `"none"`, which tells the model not to call tools.

### Closing the agentic model

Call `llm.close()`, or `await llm.aclose()` for async use, to close the provider connections. Put the call in
a `finally` block, as the examples do.

## Rate limits

**When to use it:** always, when more than one user, job or thread shares a provider account. Limits stop one
caller from exhausting the quota, and keep requests inside the provider's own limits.

Limits are counted per `user_id` (default `"global"`) in a fixed one-minute window:

```python
client = LLMSwitch(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
    user_id="team-search",
    rpm=60,                  # requests per minute
    tpm=200_000,             # tokens per minute
    embedding_rpm=300,
    embedding_tpm=1_000_000,
)
```

The defaults are `rpm=10000` and `tpm=1000000`, for chat and for embeddings. Lower them for shared accounts.

**What happens when a limit is reached.** By default the request waits for the window to reopen. If the budget
is still exhausted, it fails with `RateLimitExceededError`:

- `on_rate_limit="raise"` fails immediately instead of waiting.
- `rate_limit_max_wait` (seconds) waits only for short delays. A longer wait raises `RateLimitExceededError`.

**Denied requests do not use budget.** A denied request gives its reservation back, so a later request that
fits in the window is still allowed.

**Shared limits across processes.** In-process limits are per process. To share them, set `redis_url` and
install the `redis` extra. `redis_connect_timeout` (default 1.0 second) bounds how long each check waits for
Redis. If Redis is unavailable, `rate_limit_failure_policy` decides what happens:

- `"memory_fallback"` (default): use in-process limits, so you keep protection without Redis.
- `"fail_open"`: allow the request.
- `"fail_closed"`: raise `RateLimitBackendUnavailableError`.

```python
client = LLMSwitch(
    provider="openai", api_key="sk-...", model="gpt-4o",
    redis_url="redis://cache.internal:6379/0",
    rate_limit_failure_policy="memory_fallback",
)
```

**In-process table size.** In-process limits keep at most 10,000 user windows. When that table is full, the
oldest window is dropped and a warning is logged once. That user's window restarts, so the limit is weaker for
that window, but no user is refused for lack of room. Use Redis for exact limits with many users.

**Streams without usage.** A stream that does not report token usage counts zero completion tokens against
`tpm`.

## Retries, timeouts and the circuit breaker

**Retries** repeat a call after a transient error. This absorbs short provider hiccups without bothering the
caller.

- Retried: rate limits, connection errors and 5xx responses.
- Not retried: timeouts. The provider may already have processed the request, so a retry could bill it twice.
- `max_retries` (default 3) is the number of retries, not attempts. `retry_base_delay` (default 1.0 seconds)
  doubles each time, with jitter.
- `retry_deadline` (seconds) caps the total time one provider call may keep retrying. When the next wait would
  pass the deadline, the last error is raised.

```python
client = LLMSwitch(provider="openai", api_key="sk-...", model="gpt-4o",
                   max_retries=2, retry_base_delay=0.5, retry_deadline=30)
```

**Timeouts** still count as failures for the circuit breaker, and they trigger backup failover, so a
timed-out provider is treated as unavailable.

**The circuit breaker** stops calls to a provider that keeps failing. Use it when you would rather fail fast
than wait through retries against a provider that is down.

```python
client = LLMSwitch(provider="openai", api_key="sk-...", model="gpt-4o",
                   circuit_breaker={"failure_threshold": 5, "reset_timeout": 30})
```

- It counts timeouts, connection errors and 5xx responses. Client errors (4xx) and rate limits do not count.
- After `failure_threshold` failures it opens and refuses calls. After `reset_timeout` seconds it lets one trial
  call through. If the trial succeeds, the breaker closes; if it fails, it opens again.
- A stream you stop reading early still holds the trial slot until you call `close()` on it or let it go out of
  scope. Close streams you abandon.

`circuit_breaker=True` uses the default thresholds.

## Backup failover

**When to use it:** when a second provider can serve the same operation and you want requests to keep working
through a primary outage.

```python
client = LLMSwitch(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
    backup={
        "provider": "azure_openai",
        "api_key": "...",
        "endpoint": "...",
        "api_version": "2024-10-21",
        "deployment_name": "gpt-4o",
    },
)
```

- The backup is used when the primary fails with a retryable error or times out.
- The backup must support the same operation. This is checked before the primary is called, so a backup that cannot serve the operation fails clearly.
- For streams, the backup is used only if the primary fails before the first output reaches the caller. After
  that, switching providers would repeat text, so the stream fails instead.
- Embedding failover is all or nothing, as described in [Embeddings](#embeddings).
- Responses requests that create side effects (background requests, MCP calls, computer use) are not retried or
  failed over, so the same action is not performed twice.

`client_options` retries are separate. A `max_retries` inside `client_options` goes to the SDK, while
`max_retries` on the client is the library's own retry loop.

## Context, truncation and token counting

**When to use it:** long conversations or documents that might exceed the model's context window.

By default a request that is too long raises `ContextLengthExceededError`. With `truncate=True`, the oldest
content is trimmed so the request fits:

```python
reply = client.chat.completions.create(messages=long_history, truncate=True)
```

The window is controlled by:

- `completion_context_window` (default 123000 tokens): the budget for input and output together.
- `context_output_reserve_tokens` (default 0): output room kept free for every request, so the answer is not cut.
- `context_overhead_tokens` (default 0): extra protocol overhead you know about, counted against the budget.

**Token counting** uses `tiktoken`. OpenAI model names use their own encoding. Any other name uses the newest
tiktoken encoding, so counts for other providers are estimates. For an exact or provider-specific count, pass
your own function with `token_counter=fn`, where `fn(text, model_name)` returns an integer. It is used for
rate limits and context checks. Truncation still cuts text with tiktoken.

## Performance: clients, sharing and lazy loading

- **Client reuse.** Create one client per configuration and reuse it. Each client keeps its SDK connection pool.
- **`share_clients=True`** reuses one SDK client across clients that have the same configuration. Use it when
  you create many clients, for example one per request.
- **Lazy loading.** A provider's SDK is loaded the first time that provider is used, so unused providers cost
  nothing at import time.
- **Batching** for embeddings (see [Embeddings](#embeddings)) keeps large jobs inside provider limits.
- **Async** clients let one process serve many concurrent requests. Embeddings for providers without native
  async run in worker threads, so they do not block the event loop.

## Observability

**When to use it:** to log, measure or alert on what the library does: retries, failovers, latency and tokens.

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

Every event also has `operation` (`chat.completions`, `embeddings`, `responses`, or `responses.management`
with a `verb` such as `retrieve`), `provider` and `model`. The callback runs on the calling thread. If it
raises, the error is logged and the call continues.

**Request logging.** Request details are not logged by default. Pass `live_logs=True` on a call to log it at
INFO. The messages or input are replaced with a placeholder, and the other parameters are logged with secrets
redacted. Keep sensitive text out of parameters such as `instructions` when you turn it on.

## Configuration reference

Constructor options for `LLMSwitch` and `AsyncLLMSwitch`. Options that a provider does not use are accepted, so
the same settings work with any provider.

**Identity and provider**

| Option | Default | Meaning |
|---|---|---|
| `provider` | required | One of `openai`, `azure_openai`, `azure_foundry`, `anthropic`, `vertexai`. |
| `model` | provider default | Model name. A `deployment_name` takes precedence. |
| `deployment_name` | none | Azure deployment name. Always used as the model when set. |
| `api_key` | none | Provider API key. |
| `endpoint` | provider default | Custom endpoint or gateway URL. |
| `api_version` | provider default | API version (Azure). |
| `responses_api_version` | derived | API version for Responses calls only. |
| `location`, `credentials_path`, `scopes` | none | Vertex AI settings. |
| `organization`, `project` | none | OpenAI organization and project. |
| `timeout` | SDK default | Request timeout in seconds. |
| `enable_prompt_caching` | `True` | Anthropic cache markers on system messages and tools. |
| `client_options` | none | Extra keyword arguments for the native SDK client, for example `{"max_retries": 3}`. These cannot set `api_key`, `api_version`, `azure_endpoint`, `azure_deployment`, `base_url`, `endpoint`, `organization`, `project`, `timeout`, `credential`, `credentials`, `location`, `vertexai`, `http_options` or `responses_api_version`; use the unified options for those. |
| `user_id` | `"global"` | The key that rate limits are counted against. |

**Rate limits**

| Option | Default | Meaning |
|---|---|---|
| `rpm`, `tpm` | 10000, 1000000 | Requests and tokens per minute for chat. |
| `embedding_rpm`, `embedding_tpm` | 10000, 1000000 | Requests and tokens per minute for embeddings. |
| `on_rate_limit` | `"wait"` | `"wait"` or `"raise"`. |
| `rate_limit_max_wait` | none | Longest wait, in seconds, before raising. |
| `redis_url` | none | Shared limits across processes. |
| `redis_connect_timeout` | 1.0 | Seconds to wait for Redis. |
| `rate_limit_failure_policy` | `"memory_fallback"` | `"memory_fallback"`, `"fail_open"` or `"fail_closed"`. |

**Reliability**

| Option | Default | Meaning |
|---|---|---|
| `max_retries` | 3 | Retries for transient errors. |
| `retry_base_delay` | 1.0 | Seconds before the first retry; doubles each time. |
| `retry_deadline` | none | Maximum total seconds one call may keep retrying. |
| `circuit_breaker` | `False` | `True`, or a dict with `failure_threshold` (5) and `reset_timeout` (30). |
| `backup` | none | Backup provider settings, used on failover. |

**Context and tokens**

| Option | Default | Meaning |
|---|---|---|
| `completion_context_window` | 123000 | Token budget for input and output together. |
| `context_output_reserve_tokens` | 0 | Output room kept free for every request. |
| `context_overhead_tokens` | 0 | Extra protocol overhead to count. |
| `token_counter` | none | Your own function `fn(text, model_name)` for counting. |

**Responses**

| Option | Default | Meaning |
|---|---|---|
| `mcp_policy` | none | Allow-list for remote MCP servers and connectors. |
| `responses_capabilities` | none | Turn Responses features on or off, for example `{"background": True}`. |

**Other**

| Option | Default | Meaning |
|---|---|---|
| `on_event` | none | Callback that receives one dict per event. |
| `share_clients` | `False` | Reuse SDK clients across clients with the same configuration. |

Per-call arguments on `create()` include `stream`, `truncate`, `live_logs`, `max_retries` and provider-specific
keyword arguments. Embedding calls also accept `batching`, `batchsize` and `restore_order`.

## Errors

All errors derive from `LLMSwitchError`. Every error has `status_code`, `provider`, `user_id` and, for rate
limits, `retry_after`. These are `None` when the value is not known.

| Error | Raised when |
|---|---|
| `ConfigurationError` | Constructor settings are invalid, missing, or name an option no provider defines. |
| `RateLimitExceededError` | The budget is exhausted and waiting is not allowed, or the wait is too long. |
| `RateLimitBackendUnavailableError` | Redis is unavailable and the policy is `fail_closed`, or the rate-limit backend is disabled. |
| `ContextLengthExceededError` | The request is too long and `truncate` is off. |
| `ChatCompletionError`, `EmbeddingGenerationError` | A provider failed. `status_code` carries the HTTP status when known. |
| `AuthenticationError` | The provider rejected the credentials. |
| `APIConnectionError` | The provider could not be reached. Retried, and counted by the breaker. |
| `TimeOutError` | The provider did not answer in time. Not retried; fails over to the backup. |
| `ContentFilterError` | The provider blocked the content, including a prompt Vertex AI blocked. |
| `InvalidResponseError` | The provider returned something the library cannot read. |
| `ResponsesError` | A Responses call failed, or the provider has no Responses API (status 501). |
| `InvalidResponseRequestError` | A Responses request is invalid, or uses an MCP server that the policy does not allow. |
| `UnsupportedCapabilityError` | A Responses feature is not available on the configured provider. |
| `ToolLoopLimitError` | `run_tool_loop` reached `max_steps` while the model still asked for tools. |

Calling an operation a provider does not support (for example embeddings on `anthropic`) raises
`LLMSwitchError` before any request is sent.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest
ruff check src tests examples
mypy
```

The tests use mocks and do not contact any provider. The agentic tests skip when the `agentic` extra is not
installed, so `pip install -e ".[redis,dev]"` runs the core suite on its own. CI runs the full suite, the core
suite without the extra, types and lint. The release workflow also tests the built wheel in a clean environment
before anything is published.

## License

MIT. See [LICENSE](LICENSE).
