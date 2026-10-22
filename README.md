# llmswitch

One Python client for chat completions, embeddings and the Responses API across OpenAI,
Azure OpenAI, Azure AI Foundry, Anthropic and Vertex AI. Switch providers by changing the constructor
arguments; the call sites stay the same.

Supported in 0.1.1: synchronous and asynchronous chat completions (including streaming and
structured output), embeddings, and the Responses API for OpenAI and Azure OpenAI.

## Installation

```bash
pip install llmswitch-sdk
```

This installs every supported provider SDK: OpenAI, Azure OpenAI, Azure AI Foundry, Anthropic and
Vertex AI. Requires Python 3.11 or later.

The optional extras `anthropic`, `azure-foundry`, `vertexai` and `all` are still accepted, so existing
install commands keep working, but they add nothing new.

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
| `vertexai` | `location`, `credentials_path` | yes / no | no |

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

Vertex AI example:

```python
client = LLMSwitch(
    provider="vertexai",
    location="us-central1",
    credentials_path="/path/to/service_account.json",
    model="gemini-2.5-pro",
)
```

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

For chat completions and embeddings, per-call keyword arguments are passed to the provider request
unchanged, so provider-specific parameters such as `seed`, `top_k` or `extra_body` can be given on
`create()`. Responses requests are validated first, as described below.

## Responses API

The Responses API is available through `client.responses` for `openai` and `azure_openai`:

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

Other providers raise `ResponsesError` with status 501 for Responses calls.

## Errors

All errors derive from `LLMSwitchError`:

- `ConfigurationError`: invalid or missing constructor settings.
- `ChatCompletionError`, `EmbeddingGenerationError`: provider failures. `status_code` carries the HTTP status when known.
- `AuthenticationError`, `APIConnectionError`, `RateLimitExceededError`, `TimeOutError`, `ContentFilterError`, `ModelNotFoundError`, `InvalidResponseError`.
- `ResponsesError`: Responses API failures and unsupported providers. `InvalidResponseRequestError` for invalid Responses requests, `UnsupportedCapabilityError` for features turned off for the target.

Calling an operation a provider does not support (for example embeddings on `anthropic`)
raises `LLMSwitchError` before any request is sent.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest
ruff check src tests
```

The tests use mocks and do not contact any provider.

## License

MIT. See [LICENSE](LICENSE).
