"""
llmswitch.agentic.checkpoints

Checkpoint stores for pausing and resuming an agent run.

A store keeps JSON-compatible state under an id. The agent writes a checkpoint at each phase of a
run, so an interrupted run can be resumed from the last phase it completed. Two stores are included:

- :class:`InMemoryCheckpointStore` keeps checkpoints in the process. It is lost when the process ends.
- :class:`FileCheckpointStore` keeps one JSON file per checkpoint. Each write is atomic and carries a
  checksum, so a damaged or half-written file is detected rather than resumed.

Any object with ``save``, ``load`` and ``delete`` methods works as a store.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = 1
_CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def validate_checkpoint_id(checkpoint_id: str) -> str:
    """Return the id if it is a safe name (letters, digits, ``-`` and ``_``), otherwise raise ``ValueError``."""
    if not isinstance(checkpoint_id, str) or not _CHECKPOINT_ID.fullmatch(checkpoint_id):
        raise ValueError("checkpoint_id must be 1 to 128 letters, digits, '-' or '_', starting with a letter or digit")
    return checkpoint_id


@runtime_checkable
class CheckpointStore(Protocol):
    """Where checkpoints are kept."""

    def save(self, checkpoint_id: str, state: dict[str, Any]) -> None: ...

    def load(self, checkpoint_id: str) -> dict[str, Any] | None: ...

    def delete(self, checkpoint_id: str) -> None: ...


class InMemoryCheckpointStore:
    """Checkpoints kept in memory. Each save and load works on a copy, so later changes do not leak in."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def save(self, checkpoint_id: str, state: dict[str, Any]) -> None:
        validate_checkpoint_id(checkpoint_id)
        with self._lock:
            self._data[checkpoint_id] = copy.deepcopy(state)

    def load(self, checkpoint_id: str) -> dict[str, Any] | None:
        validate_checkpoint_id(checkpoint_id)
        with self._lock:
            state = self._data.get(checkpoint_id)
            return copy.deepcopy(state) if state is not None else None

    def delete(self, checkpoint_id: str) -> None:
        validate_checkpoint_id(checkpoint_id)
        with self._lock:
            self._data.pop(checkpoint_id, None)


def _digest(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class FileCheckpointStore:
    """Checkpoints as JSON files in one directory.

    A write goes to a temporary file in the same directory, is flushed to disk, and then replaces the
    checkpoint in one step. A load checks the schema version and the checksum, and raises ``ValueError``
    if either does not match.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, checkpoint_id: str) -> Path:
        return self.directory / f"{validate_checkpoint_id(checkpoint_id)}.json"

    def save(self, checkpoint_id: str, state: dict[str, Any]) -> None:
        path = self._path(checkpoint_id)
        payload = {"schema_version": SCHEMA_VERSION, "state": state}
        envelope = {"checksum": _digest(payload), "payload": payload}
        text = json.dumps(envelope, sort_keys=True, ensure_ascii=False)
        with self._lock:
            descriptor, temporary = tempfile.mkstemp(dir=self.directory, prefix=".checkpoint-", suffix=".tmp")
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise

    def load(self, checkpoint_id: str) -> dict[str, Any] | None:
        path = self._path(checkpoint_id)
        with self._lock:
            if not path.exists():
                return None
            envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope.get("payload")
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported or invalid checkpoint schema.")
        if envelope.get("checksum") != _digest(payload):
            raise ValueError("Checkpoint checksum validation failed.")
        state = payload.get("state")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be an object.")
        return state

    def delete(self, checkpoint_id: str) -> None:
        path = self._path(checkpoint_id)
        with self._lock:
            path.unlink(missing_ok=True)
