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
    """Thread-safe key/value store shared by specialists."""

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, Any] = dict(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def update(self, values: Mapping[str, Any]) -> None:
        with self._lock:
            self._data.update(values)

    def snapshot(self) -> dict[str, Any]:
        """A shallow copy of everything in the state."""
        with self._lock:
            return dict(self._data)

    def project_for_prompt(self, *, allowed_keys: Iterable[str] | None = None,
                           max_bytes: int = 16_000) -> dict[str, Any]:
        """A JSON-safe copy holding as many entries as fit in ``max_bytes`` once encoded.

        Entries are taken in insertion order. An entry that cannot be encoded, or that would push the
        projection over the limit, is left out, so the projection is always valid JSON.
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
    """A specialist's capabilities and its weight in aggregation (a finite, non-negative number)."""

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
    """The specialists whose contracts cover every required capability.

    With no requirements, every specialist is returned. With requirements, a specialist without a
    contract is never selected.
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
    """Add each row's ``weight`` and ``capabilities`` from its specialist's contract, when it has one."""
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
