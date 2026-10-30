"""
llmswitch.agentic.contracts

Capability contracts and shared state for coordinating specialist agents.

- :class:`SpecialistContract` declares what a specialist can do (its capabilities) and how much its
  answer should count when results are aggregated (its weight).
- :class:`SharedState` is a small thread-safe store that specialists read and write. Its
  :meth:`~SharedState.project_for_prompt` returns a bounded, JSON-safe copy for a prompt, so a large
  state cannot overflow the model's context.
"""
from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


class SharedState:
    """A thread-safe key and value store that specialists read and write.

    Use it to pass results from one specialist to the others. Its prompt projection keeps a large state from overflowing
    a model's context.
    """

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        """Create the store.

        Args:
            initial (dict, optional): The values to start with.
        """
        self._lock = threading.RLock()
        self._data: dict[str, Any] = dict(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        """Read a value.

        Args:
            key (str): The key to read.
            default (object): What to return when the key is missing. Defaults to None.

        Returns:
            object: The stored value, or ``default``.
        """
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a value under a key, replacing any earlier value.

        Args:
            key (str): The key.
            value (object): The value.

        Returns:
            None
        """
        with self._lock:
            self._data[key] = value

    def update(self, values: Mapping[str, Any]) -> None:
        """Store several values at once.

        Args:
            values (dict): The keys and values to store.

        Returns:
            None
        """
        with self._lock:
            self._data.update(values)

    def snapshot(self) -> dict[str, Any]:
        """Copy everything in the store.

        Returns:
            dict: A shallow copy of all the keys and values.
        """
        with self._lock:
            return dict(self._data)

    def project_for_prompt(self, *, allowed_keys: Iterable[str] | None = None,
                           max_bytes: int = 16_000) -> dict[str, Any]:
        """Return a JSON-safe copy of the state, small enough for a prompt.

        Entries are taken in the order they were added. An entry that cannot be encoded as JSON, or that would push the
            copy
        over ``max_bytes``, is left out, so the result is always valid JSON.

        Args:
            allowed_keys (iterable, optional): Only these keys are considered. None means all keys.
            max_bytes (int): The most bytes the encoded copy may use. Defaults to 16000.

        Returns:
            dict: The keys and values that fit.

        Raises:
            ValueError: ``max_bytes`` is not positive.
        """
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        allowed = set(allowed_keys) if allowed_keys is not None else None
        with self._lock:
            source = {k: v for k, v in self._data.items() if allowed is None or k in allowed}
        projected: dict[str, Any] = {}
        for key, value in source.items():
            candidate = {**projected, key: value}
            try:
                size = len(json.dumps(candidate, default=str, sort_keys=True).encode("utf-8"))
            except (TypeError, ValueError):
                continue
            if size <= max_bytes:
                projected[key] = value
        return projected


@dataclass
class SpecialistContract:
    """What a specialist can do, and how much its answer counts when answers are combined.

    Attributes:
        capabilities (set[str]): The capabilities of the specialist, such as ``review`` or ``legal``.
        weight (float): How much its answer counts in a weighted aggregation. A finite, non-negative number. Defaults
            to 1.0.
        metadata (dict): Any extra information about the specialist.

    Raises:
        ValueError: ``weight`` is negative, infinite or not a number.
    """

    capabilities: set[str] = field(default_factory=set)
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("SpecialistContract.weight must be a finite non-negative number")


def filter_specialists_by_capabilities(
    specialists: Mapping[str, Any],
    contracts: Mapping[str, SpecialistContract] | None,
    required_capabilities: set[str] | None,
) -> dict[str, Any]:
    """Select the specialists whose contracts cover every required capability.

    Args:
        specialists (dict): The specialists, by name.
        contracts (dict, optional): Each specialist's :class:`SpecialistContract`, by name.
        required_capabilities (set, optional): The capabilities that every selected specialist must have.

    Returns:
        dict: The selected specialists. With no required capabilities, all of them. With requirements, a specialist
        without a contract is never selected.
    """
    if not required_capabilities:
        return dict(specialists)
    if not contracts:
        return {}
    return {
        name: specialist for name, specialist in specialists.items()
        if (contract := contracts.get(name)) is not None and required_capabilities.issubset(contract.capabilities)
    }


def attach_contract_weights(rows: list[dict[str, Any]],
                            contracts: Mapping[str, SpecialistContract] | None) -> list[dict[str, Any]]:
    """Add each row's ``weight`` and ``capabilities`` from its specialist's contract.

    Args:
        rows (list[dict]): The rows from ``invoke_specialists``.
        contracts (dict, optional): Each specialist's :class:`SpecialistContract`, by name.

    Returns:
        list[dict]: Copies of the rows, with ``weight`` and ``capabilities`` added where a contract exists.
    """
    if not contracts:
        return rows
    enriched: list[dict[str, Any]] = []
    for row in rows:
        contract = contracts.get(row.get("specialist") or "")
        item = dict(row)
        if contract is not None:
            item["weight"] = contract.weight
            item["capabilities"] = sorted(contract.capabilities)
        enriched.append(item)
    return enriched
