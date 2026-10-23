"""
ratelimit.batching

Batching helpers for embedding generation (token-aware First-Fit Decreasing)
and a small progress-bar logger.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class BatchUtility:
    """Helper functions for token-aware embedding batching."""

    def plan_batches(self, token_counts: list[int], batch_size: int, tpm: int,
                     batching: bool, restore_order: bool = True) -> list[list[int]]:
        """
        Plan embedding batches as lists of ORIGINAL input indices using token-aware
        First-Fit Decreasing. Working in index space lets callers re-map provider
        results back to the input order with correct, non-colliding indices and
        handles duplicate texts correctly.

        Args:
            token_counts: Per-input token counts (aligned to the input list).
            batch_size: Max number of items per batch.
            tpm: Max tokens per batch (token-based batching).
            batching: If False, returns a single batch of all indices in order.
            restore_order: If True, sorts indices ascending within each batch.

        Returns:
            List of batches, each a list of original input indices.
        """
        n = len(token_counts)
        if not batching:
            return [list(range(n))]
        # FFD: place the largest items first.
        order = sorted(range(n), key=lambda i: -token_counts[i])
        batches: list[list[int]] = []
        totals: list[int] = []
        for i in order:
            tokens = token_counts[i]
            placed = False
            for b, members in enumerate(batches):
                if len(members) < batch_size and totals[b] + tokens <= tpm:
                    members.append(i)
                    totals[b] += tokens
                    placed = True
                    break
            if not placed:
                batches.append([i])
                totals.append(tokens)
        if restore_order:
            return [sorted(members) for members in batches]
        return batches

    def batch_texts_for_embedding(self, texts: list[str], batch_size: int, tpm: int,
                                  tokenizer: Callable[[str], int], batching: bool,
                                  restore_order: bool = True) -> list[list[str]]:
        """
        Token-aware FFD batching returning batches of texts. Thin wrapper over
        :meth:`plan_batches` (which works in index space, so duplicate texts are
        handled correctly). Retained for backward compatibility.
        """
        token_counts = [tokenizer(text) for text in texts]
        return [[texts[i] for i in idxs]
                for idxs in self.plan_batches(token_counts, batch_size, tpm, batching, restore_order)]

    def print_progress_bar(self, current: int, total: int, bar_length: int = 60) -> None:
        """Log a simple progress bar for batching operations."""
        percent = float(current) / total
        completed = int(percent * bar_length)
        bar = "█" * completed + " " * (bar_length - completed)
        logger.info("Batching Progress: |%s| [%d/%d] %d%%", bar, current, total, int(percent * 100))
