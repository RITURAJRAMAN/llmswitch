# llmswitch

One Python client for chat completions and embeddings across OpenAI, Azure OpenAI,
Azure AI Foundry, Anthropic and Vertex AI. Switch providers by changing the constructor
arguments; the call sites stay the same.

Supported in 0.1.0: synchronous and asynchronous chat completions (including streaming and
structured output) and embeddings.

## Installation

```bash
pip install llmswitch-sdk                  # core: OpenAI and Azure OpenAI
pip install "llmswitch-sdk[anthropic]"       # adds Anthropic
pip install "llmswitch-sdk[azure-foundry]"   # adds Azure AI Foundry
pip install "llmswitch-sdk[vertexai]"        # adds Vertex AI
pip install "llmswitch-sdk[all]"             # every provider
```

Requires Python 3.11 or later.

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

Per-call keyword arguments are passed to the provider request unchanged, so provider-specific
parameters such as `seed`, `top_k` or `extra_body` can be given on `create()`.

## Errors

All errors derive from `LLMSwitchError`:

- `ConfigurationError`: invalid or missing constructor settings.
- `ChatCompletionError`, `EmbeddingGenerationError`: provider failures. `status_code` carries the HTTP status when known.
- `AuthenticationError`, `APIConnectionError`, `RateLimitExceededError`, `TimeOutError`, `ContentFilterError`, `ModelNotFoundError`, `InvalidResponseError`.

Calling an operation a provider does not support (for example embeddings on `anthropic`)
raises `LLMSwitchError` before any request is sent.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest
ruff check src tests
```

The tests use mocks and do not contact any provider. Live tests are not part of 0.1.0.

## License

MIT. See [LICENSE](LICENSE).
