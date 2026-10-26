"""
llmswitch.agentic.aggregation

Combining the rows from several specialists into one answer.

Each aggregator takes the rows that :func:`invoke_specialists` returns and gives an
:class:`AggregatedResult`:

- :class:`MajorityVoteAggregator` picks the output that the most specialists gave.
- :class:`WeightedScoreAggregator` picks the output with the largest total weight. A row's ``weight``
  comes from its specialist's contract (see :func:`attach_contract_weights`); a row without one counts as 1.
- :class:`JsonMergeAggregator` merges outputs that are JSON objects, key by key, and falls back to a
  majority vote when no output is an object.

Rows that did not finish (``error``, ``timeout``, ``interrupted``, ``stopped``) are left out unless
``include_failed=True``. An output that is a JSON string is parsed first, so ``'{"a": 1}'`` and
``{"a": 1}`` count as the same answer, and so does a JSON object wrapped in a Markdown code fence. An output
of ``None`` is not a vote, but it still counts in the
denominator of ``confidence``.

Ties go to the answer that appeared first in the rows.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .specialists import ainvoke_specialists, invoke_specialists

if TYPE_CHECKING:
    from .agent import Agent
    from .contracts import SharedState, SpecialistContract

_FAILED_STATUSES = frozenset({"error", "timeout", "interrupted", "stopped"})
_CONFLICT_POLICIES = frozenset({"last_writer_wins", "first_writer_wins", "error"})
_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class AggregatedResult:
    """The consensus value, the rows it was built from, the strategy used, and how sure it is."""

    consensus: Any
    inputs: list[dict[str, Any]]
    strategy: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAggregator:
    """Base class for aggregators. Subclasses implement :meth:`aggregate`."""

    def aggregate(self, results: list[dict[str, Any]]) -> AggregatedResult:
        raise NotImplementedError


def _eligible(results: list[dict[str, Any]], include_failed: bool) -> list[dict[str, Any]]:
    if include_failed:
        return list(results)
    return [row for row in results if row.get("status") not in _FAILED_STATUSES]


def _parsed(value: Any) -> Any:
    if isinstance(value, str):
        fenced = _FENCED_JSON.fullmatch(value.strip())
        try:
            return json.loads(fenced.group(1) if fenced else value)
        except json.JSONDecodeError:
            return value
    return value


def _bucket_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _answers(rows: list[dict[str, Any]], value_key: str) -> list[tuple[str, Any]]:
    """The bucket key and parsed value of each row that has an answer."""
    answers = []
    for row in rows:
        value = _parsed(row.get(value_key))
        if value is not None:
            answers.append((_bucket_key(value), value))
    return answers


def _first_max(totals: Mapping[str, float]) -> str:
    """The key with the largest total; on a tie, the one inserted first. ``totals`` must not be empty."""
    winner = next(iter(totals))
    for key, total in totals.items():
        if total > totals[winner]:
            winner = key
    return winner


def _weight(row: dict[str, Any], weight_key: str) -> float:
    raw = row.get(weight_key)
    if raw is None:
        return 1.0
    try:
        weight = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Aggregation weights must be finite non-negative numbers") from exc
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Aggregation weights must be finite non-negative numbers")
    return weight


class MajorityVoteAggregator(BaseAggregator):
    """Pick the output that the most specialists gave."""

    def __init__(self, *, value_key: str = "output", include_failed: bool = False) -> None:
        self.value_key = value_key
        self.include_failed = include_failed

    def aggregate(self, results: list[dict[str, Any]]) -> AggregatedResult:
        rows = _eligible(results, self.include_failed)
        answers = _answers(rows, self.value_key)
        if not answers:
            return AggregatedResult(consensus=None, inputs=rows, strategy="majority_vote", confidence=0.0)
        counts: dict[str, int] = {}
        values: dict[str, Any] = {}
        for key, value in answers:
            counts[key] = counts.get(key, 0) + 1
            values.setdefault(key, value)
        winner = _first_max(counts)
        return AggregatedResult(
            consensus=values[winner],
            inputs=rows,
            strategy="majority_vote",
            confidence=counts[winner] / len(rows),
            metadata={"votes": counts},
        )


class WeightedScoreAggregator(BaseAggregator):
    """Pick the output with the largest total weight."""

    def __init__(self, *, value_key: str = "output", weight_key: str = "weight", include_failed: bool = False) -> None:
        self.value_key = value_key
        self.weight_key = weight_key
        self.include_failed = include_failed

    def aggregate(self, results: list[dict[str, Any]]) -> AggregatedResult:
        rows = _eligible(results, self.include_failed)
        totals: dict[str, float] = {}
        values: dict[str, Any] = {}
        total_weight = 0.0
        for row in rows:
            weight = _weight(row, self.weight_key)
            total_weight += weight
            value = _parsed(row.get(self.value_key))
            if value is None:
                continue
            key = _bucket_key(value)
            totals[key] = totals.get(key, 0.0) + weight
            values.setdefault(key, value)
        if not totals:
            return AggregatedResult(consensus=None, inputs=rows, strategy="weighted_score", confidence=0.0)
        winner = _first_max(totals)
        confidence = totals[winner] / total_weight if total_weight > 0 else 0.0
        return AggregatedResult(
            consensus=values[winner],
            inputs=rows,
            strategy="weighted_score",
            confidence=confidence,
            metadata={"weights": totals},
        )


class JsonMergeAggregator(BaseAggregator):
    """Merge the JSON-object outputs key by key. With no object outputs, fall back to a majority vote.

    ``conflict_policy`` decides a key that two specialists set differently: ``last_writer_wins`` keeps the
    later value, ``first_writer_wins`` keeps the earlier one, and ``error`` raises ``ValueError``. Every
    conflict is listed in the result's ``metadata["conflicts"]``.
    """

    def __init__(self, *, conflict_policy: str = "last_writer_wins", include_failed: bool = False) -> None:
        if conflict_policy not in _CONFLICT_POLICIES:
            raise ValueError("conflict_policy must be one of 'last_writer_wins', 'first_writer_wins' or 'error'")
        self.conflict_policy = conflict_policy
        self.include_failed = include_failed

    def aggregate(self, results: list[dict[str, Any]]) -> AggregatedResult:
        rows = _eligible(results, self.include_failed)
        if not rows:
            return AggregatedResult(consensus=None, inputs=[], strategy="json_merge", confidence=0.0)
        merged: dict[str, Any] = {}
        conflicts: dict[str, list[Any]] = {}
        merged_count = 0
        for row in rows:
            payload = _parsed(row.get("output"))
            if not isinstance(payload, dict):
                continue
            merged_count += 1
            for key, value in payload.items():
                if key in merged and merged[key] != value:
                    conflicts.setdefault(key, [merged[key]]).append(value)
                    if self.conflict_policy == "error":
                        raise ValueError(f"JSON merge conflict for key '{key}'")
                    if self.conflict_policy == "first_writer_wins":
                        continue
                merged[key] = value
        if merged_count == 0:
            majority = MajorityVoteAggregator().aggregate(rows)
            return AggregatedResult(
                consensus=majority.consensus,
                inputs=rows,
                strategy="json_merge_fallback_majority",
                confidence=majority.confidence,
                metadata={"fallback": True, "votes": majority.metadata.get("votes", {})},
            )
        return AggregatedResult(
            consensus=merged,
            inputs=rows,
            strategy="json_merge",
            confidence=merged_count / len(rows),
            metadata={
                "merged_items": merged_count,
                "conflicts": conflicts,
                "conflict_policy": self.conflict_policy,
            },
        )


def aggregate_specialists(
    *,
    task: str,
    specialists: Mapping[str, Agent],
    aggregator: BaseAggregator | None = None,
    parallel: bool = True,
    max_concurrency: int = 4,
    max_runtime_seconds: float | None = None,
    shared_state: SharedState | None = None,
    shared_state_keys: set[str] | None = None,
    shared_state_max_bytes: int = 16_000,
    contracts: Mapping[str, SpecialistContract] | None = None,
    required_capabilities: set[str] | None = None,
) -> AggregatedResult:
    """Run the specialists on ``task`` and aggregate their rows. The default aggregator is :class:`JsonMergeAggregator`.

    To weigh the specialists by their contracts, pass ``contracts`` and a :class:`WeightedScoreAggregator`.
    """
    rows = invoke_specialists(
        task=task,
        specialists=specialists,
        parallel=parallel,
        max_concurrency=max_concurrency,
        max_runtime_seconds=max_runtime_seconds,
        shared_state=shared_state,
        shared_state_keys=shared_state_keys,
        shared_state_max_bytes=shared_state_max_bytes,
        contracts=contracts,
        required_capabilities=required_capabilities,
    )
    return (aggregator or JsonMergeAggregator()).aggregate(rows)


async def aaggregate_specialists(
    *,
    task: str,
    specialists: Mapping[str, Agent],
    aggregator: BaseAggregator | None = None,
    parallel: bool = True,
    max_concurrency: int = 4,
    max_runtime_seconds: float | None = None,
    shared_state: SharedState | None = None,
    shared_state_keys: set[str] | None = None,
    shared_state_max_bytes: int = 16_000,
    contracts: Mapping[str, SpecialistContract] | None = None,
    required_capabilities: set[str] | None = None,
) -> AggregatedResult:
    """Async :func:`aggregate_specialists`. Specialists that run out of time are cancelled."""
    rows = await ainvoke_specialists(
        task=task,
        specialists=specialists,
        parallel=parallel,
        max_concurrency=max_concurrency,
        max_runtime_seconds=max_runtime_seconds,
        shared_state=shared_state,
        shared_state_keys=shared_state_keys,
        shared_state_max_bytes=shared_state_max_bytes,
        contracts=contracts,
        required_capabilities=required_capabilities,
    )
    return (aggregator or JsonMergeAggregator()).aggregate(rows)
