"""
core.embedding_flow

Pure helpers shared by the sync and async embeddings pipelines: absorbing one
batch's result and assembling the aggregated response.
"""
from __future__ import annotations

import time
from typing import Any

from ..errors import EmbeddingGenerationError
from ..models import CreateEmbeddingResponse, ProviderExtensions, Usage


class EmbeddingAccumulator:
    """Collects batch results for one target and builds the final response."""

    def __init__(self, provider: str, model: str | None, restore_order: bool) -> None:
        self.provider = provider
        self.model = model
        self.restore_order = restore_order
        self.items: list[Any] = []
        self.usages: list[Any] = []
        self.extensions: list[tuple[list[int], ProviderExtensions]] = []

    def add(self, result: Any, indices: list[int]) -> None:
        data = getattr(result, "data", result)
        if len(data) != len(indices):
            raise EmbeddingGenerationError(f"Provider returned {len(data)} embeddings for {len(indices)} inputs")
        for index, embedding in zip(indices, data, strict=True):
            embedding.index = index
            self.items.append(embedding)
        usage = getattr(result, "usage", None)
        if usage is not None:
            self.usages.append(usage)
        extension = getattr(result, "extensions", None)
        if extension is not None:
            self.extensions.append((indices, extension))
        self.model = getattr(result, "model", None) or self.model

    @property
    def total_tokens(self) -> int:
        return sum(getattr(usage, "total_tokens", 0) for usage in self.usages)

    def build(self) -> CreateEmbeddingResponse:
        if self.restore_order:
            self.items.sort(key=lambda item: item.index)
        extensions = None
        if len(self.extensions) == 1:
            extensions = self.extensions[0][1]
        elif self.extensions:
            extensions = ProviderExtensions(
                provider=self.provider,
                fields={"batches": [
                    {"input_indices": indices, "fields": extension.fields}
                    for indices, extension in self.extensions
                ]},
            )
        return CreateEmbeddingResponse(
            data=self.items,
            model=self.model,
            object="list",
            usage=Usage(prompt_tokens=sum(getattr(u, "prompt_tokens", 0) for u in self.usages),
                        total_tokens=self.total_tokens),
            extensions=extensions,
            created=int(time.time()),
        )
