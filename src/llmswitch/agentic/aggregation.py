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
    """The result of combining specialist answers.

    Attributes:
        consensus (object): The combined answer. None when no specialist gave an answer.
        inputs (list[dict]): The rows the answer was built from.
        strategy (str): The aggregator that combined them, such as ``majority_vote``.
        confidence (float, optional): How strongly the answer is supported, from 0 to 1.
        metadata (dict): Extra details, such as the vote counts or the merge conflicts.
    """

    consensus: Any
    inputs: list[dict[str, Any]]
    strategy: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAggregator:
    """Base class for aggregators. Subclasses implement :meth:`aggregate`."""

    def aggregate(self, results: list[dict[str, Any]]) -> AggregatedResult:
        """Combine specialist rows into one result. Subclasses implement this.

        Args:
            results (list[dict]): The rows from ``invoke_specialists``, each with ``specialist``, ``status`` and
                ``output``.

        Returns:
            AggregatedResult: The combined answer.
        """
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
    """Picks the answer that the most specialists gave.

    Answers are compared as JSON, so ``{"a": 1}`` and ``'{"a": 1}'`` count as the same answer. A tie goes to the answer
    that appeared first.
    """

    def __init__(self, *, value_key: str = "output", include_failed: bool = False) -> None:
        """Create the aggregator.

        Args:
            value_key (str): The row key that holds the answer. Defaults to ``output``.
            include_failed (bool): Count rows whose status is ``error``, ``timeout``, ``interrupted`` or ``stopped``.
                Defaults to False.
        """
        self.value_key = value_key
        self.include_failed = include_failed

    def aggregate(self, results: list[dict[str, Any]]) -> AggregatedResult:
        """Pick the answer that the most rows gave.

        Args:
            results (list[dict]): The specialist rows.

        Returns:
            AggregatedResult: The most common answer. ``confidence`` is its share of the rows, and ``metadata["votes"]``
            counts each answer.
        """
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
    """Picks the answer with the largest total weight.

    Each row's ``weight`` comes from its specialist's contract. A row without a weight counts as 1.
    """

    def __init__(self, *, value_key: str = "output", weight_key: str = "weight", include_failed: bool = False) -> None:
        """Create the aggregator.

        Args:
            value_key (str): The row key that holds the answer. Defaults to ``output``.
            weight_key (str): The row key that holds the weight. Defaults to ``weight``.
            include_failed (bool): Count rows whose status is ``error``, ``timeout``, ``interrupted`` or ``stopped``.
                Defaults to False.
        """
        self.value_key = value_key
        self.weight_key = weight_key
        self.include_failed = include_failed

    def aggregate(self, results: list[dict[str, Any]]) -> AggregatedResult:
        """Pick the answer with the largest total weight.

        Args:
            results (list[dict]): The specialist rows.

        Returns:
            AggregatedResult: The heaviest answer. ``confidence`` is its share of the total weight.

        Raises:
            ValueError: A weight is negative, infinite or not a number.
        """
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
    """Merges specialists' JSON-object answers into one object, key by key.

    When no answer is an object, it falls back to a majority vote.
    """

    def __init__(self, *, conflict_policy: str = "last_writer_wins", include_failed: bool = False) -> None:
        """Create the aggregator.

        Args:
            conflict_policy (str): What to do when two specialists give the same key different values.
                ``last_writer_wins``
                (default) keeps the later value, ``first_writer_wins`` keeps the earlier one, and ``error`` raises.
            include_failed (bool): Count rows whose status is ``error``, ``timeout``, ``interrupted`` or ``stopped``.
                Defaults to False.

        Raises:
            ValueError: ``conflict_policy`` is not one of the three names above.
        """
        if conflict_policy not in _CONFLICT_POLICIES:
            raise ValueError("conflict_policy must be one of 'last_writer_wins', 'first_writer_wins' or 'error'")
        self.conflict_policy = conflict_policy
        self.include_failed = include_failed

    def aggregate(self, results: list[dict[str, Any]]) -> AggregatedResult:
        """Merge the object answers into one object.

        Args:
            results (list[dict]): The specialist rows.

        Returns:
            AggregatedResult: The merged object, with ``metadata["conflicts"]`` listing every key that was set
                differently.
            When no answer is an object, the majority vote instead.

        Raises:
            ValueError: A key conflicts and the policy is ``error``.
        """
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
    """Run the specialists on a task, then combine their answers.

    Args:
        task (str): The task that every specialist receives.
        specialists (dict[str, Agent]): The specialists, by name.
        aggregator (BaseAggregator, optional): How to combine the answers. Defaults to ``JsonMergeAggregator``.
        parallel (bool): Run the specialists at the same time. Defaults to True.
        max_concurrency (int): The most specialists running at once. Defaults to 4.
        max_runtime_seconds (float, optional): The shared time limit for all the specialists.
        shared_state (SharedState, optional): Shared state that each specialist can read and write.
        shared_state_keys (set, optional): The keys of the shared state to include in each prompt.
        shared_state_max_bytes (int): The most bytes of shared state put in a prompt. Defaults to 16000.
        contracts (dict, optional): Each specialist's :class:`SpecialistContract`, by name.
        required_capabilities (set, optional): Only specialists whose contracts cover these capabilities are asked.

    Returns:
        AggregatedResult: The combined answer.
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
    """Async :func:`aggregate_specialists`. Specialists that run out of time are cancelled.

    Args:
        task (str): The task that every specialist receives.
        specialists (dict[str, Agent]): The specialists, by name.
        aggregator (BaseAggregator, optional): How to combine the answers. Defaults to ``JsonMergeAggregator``.
        parallel (bool): Run the specialists at the same time. Defaults to True.
        max_concurrency (int): The most specialists running at once. Defaults to 4.
        max_runtime_seconds (float, optional): The shared time limit for all the specialists.
        shared_state (SharedState, optional): Shared state that each specialist can read and write.
        shared_state_keys (set, optional): The keys of the shared state to include in each prompt.
        shared_state_max_bytes (int): The most bytes of shared state put in a prompt. Defaults to 16000.
        contracts (dict, optional): Each specialist's :class:`SpecialistContract`, by name.
        required_capabilities (set, optional): Only specialists whose contracts cover these capabilities are asked.

    Returns:
        An awaitable that gives the combined :class:`AggregatedResult`.
    """
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
