# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
