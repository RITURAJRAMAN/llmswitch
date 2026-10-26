"""
llmswitch.agentic.specialists

Running several specialist agents on one task, and exposing an agent as a tool for another agent.

:func:`invoke_specialists` (and :func:`ainvoke_specialists`) run each selected specialist and return one
row per specialist, in the order the specialists were given:

- ``status``: the specialist's status (``completed``, ``stopped``, ``interrupted``...), or ``error`` when
  it raised, or ``timeout`` when the shared time limit ran out before it finished.
- ``output``, ``reason``, ``steps``, ``checkpoint_id``, ``raw``: from the specialist's result.
- ``weight`` and ``capabilities``: added from the specialist's contract, when it has one.

The time limit is shared: every specialist gets whatever is left of it. A specialist that is still running
when the limit ends is reported as ``timeout``. In the async runner it is cancelled. In the sync runner its
thread keeps running in the background, because Python cannot stop a running thread.

:func:`make_subagent_tool` wraps an agent as a plain function, so another agent can call it as a tool.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from typing import TYPE_CHECKING, Any, Callable

from .contracts import SharedState, SpecialistContract, attach_contract_weights, filter_specialists_by_capabilities

if TYPE_CHECKING:
    from .agent import Agent


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _task_payload(task: str, shared_state: SharedState | None, keys: set[str] | None, max_bytes: int) -> str:
    if shared_state is None:
        return task
    context = shared_state.project_for_prompt(allowed_keys=keys, max_bytes=max_bytes)
    return f"{task}\n\nSharedStateContext(JSON): {json.dumps(context, sort_keys=True, default=str)}"


def _publish(shared_state: SharedState | None, rows: list[dict[str, Any]]) -> None:
    """Write the outputs of the specialists that finished to the shared state.

    This runs in the caller, after the rows are collected. A specialist that timed out may still be running in
    its thread, and it must not write to the state after the caller has reported it as a timeout.
    """
    if shared_state is None:
        return
    for row in rows:
        if row["raw"] is not None:
            shared_state.set(f"specialist_outputs.{row['specialist']}", row["output"])


def _row(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "specialist": name,
        "status": result.get("status"),
        "output": result.get("output"),
        "reason": result.get("reason"),
        "steps": len(result.get("steps", [])),
        "checkpoint_id": result.get("checkpoint_id"),
        "raw": result,
    }


def _error_row(name: str, exc: Exception) -> dict[str, Any]:
    return {"specialist": name, "status": "error", "output": None, "reason": str(exc), "steps": 0,
            "checkpoint_id": None, "raw": None}


def _timeout_row(name: str) -> dict[str, Any]:
    return {"specialist": name, "status": "timeout", "output": None, "reason": "max_runtime_seconds_exceeded",
            "steps": 0, "checkpoint_id": None, "raw": None}


def _check_runtime(max_runtime_seconds: float | None) -> float | None:
    if max_runtime_seconds is not None and max_runtime_seconds < 0:
        raise ValueError("max_runtime_seconds must not be negative")
    return None if max_runtime_seconds is None else time.monotonic() + max_runtime_seconds


def invoke_specialists(
    *,
    task: str,
    specialists: Mapping[str, Agent],
    parallel: bool = True,
    max_concurrency: int = 4,
    max_runtime_seconds: float | None = None,
    shared_state: SharedState | None = None,
    shared_state_keys: set[str] | None = None,
    shared_state_max_bytes: int = 16_000,
    contracts: Mapping[str, SpecialistContract] | None = None,
    required_capabilities: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Run the selected specialists on ``task`` and return one row per specialist.

    ``shared_state`` gets each specialist's output under ``specialist_outputs.<name>``, and the state (or
    the keys in ``shared_state_keys``, bounded by ``shared_state_max_bytes``) is added to the task prompt.
    """
    deadline = _check_runtime(max_runtime_seconds)
    items = list(filter_specialists_by_capabilities(specialists, contracts, required_capabilities).items())
    if not items:
        return []

    def run(name: str, agent: Agent) -> dict[str, Any]:
        try:
            payload = _task_payload(task, shared_state, shared_state_keys, shared_state_max_bytes)
            result = agent.invoke(payload, max_runtime_seconds=_remaining(deadline))
            return _row(name, result)
        except Exception as exc:  # noqa: BLE001 - one specialist failing must not stop the others
            return _error_row(name, exc)

    if not parallel or len(items) == 1:
        rows = []
        for name, agent in items:
            if deadline is not None and _remaining(deadline) == 0:
                rows.append(_timeout_row(name))
            else:
                rows.append(run(name, agent))
        _publish(shared_state, rows)
        return attach_contract_weights(rows, contracts)

    pool = ThreadPoolExecutor(max_workers=max(1, min(max_concurrency, len(items))),
                              thread_name_prefix="llmswitch-specialist")
    try:
        futures = {
            pool.submit(copy_context().run, run, name, agent): index
            for index, (name, agent) in enumerate(items)
        }
        ordered: dict[int, dict[str, Any]] = {}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=_remaining(deadline), return_when=FIRST_COMPLETED)
            for future in done:
                ordered[futures[future]] = future.result()
            if not done:  # the shared time limit ended
                break
        for future in pending:
            future.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    rows = [ordered.get(index) or _timeout_row(name) for index, (name, _) in enumerate(items)]
    _publish(shared_state, rows)
    return attach_contract_weights(rows, contracts)


async def ainvoke_specialists(
    *,
    task: str,
    specialists: Mapping[str, Agent],
    parallel: bool = True,
    max_concurrency: int = 4,
    max_runtime_seconds: float | None = None,
    shared_state: SharedState | None = None,
    shared_state_keys: set[str] | None = None,
    shared_state_max_bytes: int = 16_000,
    contracts: Mapping[str, SpecialistContract] | None = None,
    required_capabilities: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Async :func:`invoke_specialists`. A specialist that runs out of time is cancelled."""
    deadline = _check_runtime(max_runtime_seconds)
    items = list(filter_specialists_by_capabilities(specialists, contracts, required_capabilities).items())
    if not items:
        return []

    async def run(name: str, agent: Agent) -> dict[str, Any]:
        remaining = _remaining(deadline)
        if remaining == 0:
            return _timeout_row(name)
        try:
            payload = _task_payload(task, shared_state, shared_state_keys, shared_state_max_bytes)
            call = agent.ainvoke(payload, max_runtime_seconds=remaining)
            result = await (asyncio.wait_for(call, remaining) if remaining is not None else call)
            return _row(name, result)
        except TimeoutError:
            return _timeout_row(name)
        except Exception as exc:  # noqa: BLE001 - one specialist failing must not stop the others
            return _error_row(name, exc)

    if not parallel:
        rows = [await run(name, agent) for name, agent in items]
    else:
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def gated(name: str, agent: Agent) -> dict[str, Any]:
            async with semaphore:
                return await run(name, agent)

        rows = list(await asyncio.gather(*(gated(name, agent) for name, agent in items)))
    _publish(shared_state, rows)
    return attach_contract_weights(rows, contracts)


def make_subagent_tool(*, name: str, description: str, agent: Agent) -> Callable[[str], str]:
    """Wrap an agent as a function that takes a task and returns a JSON summary of the run.

    The summary has ``status``, ``output``, ``reason`` and ``steps``. Give the function to another
    agent's ``tools`` list, and the calling model decides when to delegate.
    """

    def delegate(task: str) -> str:
        result = agent.invoke(task)
        payload = {"status": result.get("status"), "output": result.get("output"),
                   "reason": result.get("reason"), "steps": len(result.get("steps", []))}
        return json.dumps(payload, default=str)

    delegate.__name__ = name
    delegate.__doc__ = description
    return delegate


def make_specialist_tools(specs: Mapping[str, tuple[str, Agent]]) -> list[Callable[[str], str]]:
    """Wrap several agents as tools. ``specs`` maps each tool name to ``(description, agent)``."""
    return [make_subagent_tool(name=name, description=description, agent=agent)
            for name, (description, agent) in specs.items()]
