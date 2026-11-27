from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from .executions import (
    CellObservation,
    ExecutionState,
    ExecutionStore,
    mark_receipt_unknown,
    parse_cell_id,
    project_cell,
    project_receipt,
    project_session_snapshot,
)
from .interpreter_runner import AsyncInterpreterRunner, BusyError
from .utils import precheck_syntax

logger = logging.getLogger(__name__)

# Teardown is bounded: a wedged interpreter must never take the event loop with it.
# Keep this well above the runner's own worst case (one reader poll + the terminate
# wait, ~1.2s) so it only ever fires for a genuinely wedged teardown.
SESSION_TEARDOWN_TIMEOUT_SECONDS = 10.0


class PythonSessionError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class _CellBinding:
    execution_id: str
    source: str = ""


@dataclass
class _Session:
    session_id: str
    runner: AsyncInterpreterRunner
    owner: str | None = None
    label: str | None = None
    description: str | None = None
    ephemeral: bool = False
    created_at: float = field(default_factory=time.time)
    cell_bindings: dict[int, _CellBinding] = field(default_factory=dict)

    def bind_cell(self, cell_id: int, execution_id: str, *, source: str = "") -> None:
        """Bind one cell to its immutable execution_id and remember its source."""
        existing = self.cell_bindings.get(cell_id)
        if existing is not None:
            if existing.execution_id != execution_id:
                raise ValueError(
                    f"Cell {cell_id} in session {self.session_id} "
                    f"is already bound to {existing.execution_id}"
                )
            if existing.source or not source:
                return
        self.cell_bindings[cell_id] = _CellBinding(execution_id=execution_id, source=source)

    def cell_source(self, cell_id: int) -> str:
        runner_source = self.runner.get_cell_source(cell_id)
        if runner_source is not None:
            return runner_source
        binding = self.cell_bindings.get(cell_id)
        return binding.source if binding is not None else ""


class NativePythonService:
    """Own Python interpreter sessions for one workspace process."""

    def __init__(self, *, receipt_path: Path | None = None) -> None:
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()
        self._execution_lock = asyncio.Lock()
        self._executions = ExecutionStore(receipt_path)
        self._submitting: set[str] = set()
        self._running: dict[str, tuple[_Session, int]] = {}
        self._completion_tasks: dict[str, asyncio.Task[None]] = {}

    async def close(self) -> None:
        tasks = list(self._completion_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await asyncio.gather(
            *(self._teardown_session(session) for session in sessions),
            return_exceptions=True,
        )
        self._executions.close()

    async def create_session(
        self,
        *,
        owner: str | None = None,
        label: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        return _session_payload(await self._create_session(owner=owner, label=label, description=description))

    async def _create_session(
        self, *, owner: str | None, label: str | None = None,
        description: str | None = None, ephemeral: bool = False,
    ) -> _Session:
        _validate_source_key(owner)
        async with self._lock:
            session_id = uuid4().hex
            runner = AsyncInterpreterRunner(
                name=f"session-{session_id}",
                loop=asyncio.get_running_loop(),
            )
            session = _Session(
                session_id=session_id,
                runner=runner,
                owner=owner,
                label=_optional_text(label),
                description=_optional_text(description),
                ephemeral=ephemeral,
            )
            self._sessions[session_id] = session
        return session

    async def list_sessions(self, *, owner: str | None = None) -> list[dict[str, Any]]:
        async with self._lock:
            sessions = sorted(self._sessions.values(), key=lambda item: item.created_at)
            _validate_source_key(owner)
            return [
                _session_payload(session) for session in sessions
                if not session.ephemeral and session.owner == owner
            ]

    async def update_session(
        self,
        session_id: str,
        *,
        owner: str | None = None,
        label: str | None,
        description: str | None,
        update_label: bool,
        update_description: bool,
    ) -> dict[str, Any]:
        session = await self._get(session_id, owner=owner)
        if update_label:
            session.label = _optional_text(label)
        if update_description:
            session.description = _optional_text(description)
        return _session_payload(session)

    async def delete_session(self, session_id: str, *, owner: str | None = None) -> dict[str, str]:
        normalized = _session_id(session_id)
        async with self._lock:
            session = self._sessions.get(normalized)
            if session is not None:
                _check_owner(session.owner, owner)
                del self._sessions[normalized]
        if session is None:
            raise PythonSessionError("SESSION_NOT_FOUND", "Python session was not found.")
        execution_ids = [
            execution_id for execution_id, _ in self._session_running_executions(session)
        ]
        tasks = [
            self._completion_tasks[execution_id]
            for execution_id in execution_ids
            if execution_id in self._completion_tasks
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for execution_id in execution_ids:
            self._completion_tasks.pop(execution_id, None)
            self._running.pop(execution_id, None)
            previous = self._executions.get(execution_id)
            if previous is not None and mark_receipt_unknown(previous[1]):
                # Cancellation may happen before the watcher coroutine starts at all.
                self._executions.update(execution_id, previous[1])
        await self._teardown_session(session)
        return {"session_id": normalized}

    async def submit_cell(
        self,
        session_id: str,
        source: str,
        *,
        wait_seconds: float,
        owner: str | None = None,
        execution_context: str | None = None,
    ) -> dict[str, Any]:
        source = str(source or "")
        if not source.strip():
            raise PythonSessionError("INVALID_SOURCE", "Python source is required.")
        _validate_source(source)
        session = await self._get(session_id, owner=owner)
        try:
            cell_id = session.runner.start_cell(source, execution_context=execution_context)
        except BusyError as exc:
            raise PythonSessionError(
                "SESSION_BUSY",
                "The Python session is already running a cell.",
                retryable=True,
            ) from exc
        if wait_seconds <= 0:
            return _cell_snapshot(session, cell_id)
        try:
            await session.runner.wait_cell(cell_id, timeout=wait_seconds)
        except (TimeoutError, ValueError):
            pass
        return _cell_snapshot(session, cell_id)

    async def execute(
        self,
        execution_id: str,
        source: str,
        *,
        session_id: str | None = None,
        wait_seconds: float = 330,
        meta: dict[str, Any] | None = None,
        owner: str | None = None,
        origin: str | None = None,
        execution_context: str | None = None,
    ) -> dict[str, Any]:
        """Submit once, then observe that submission even if the caller disconnects."""
        _validate_wait(wait_seconds)
        started = time.monotonic()
        execution_id = str(UUID(execution_id))
        if not source.strip():
            raise PythonSessionError("INVALID_SOURCE", "Python source is required.")
        _validate_source(source)
        _validate_source_key(owner)
        _validate_source_key(origin)
        if (owner is None) != (origin is None):
            raise PythonSessionError(
                "EXECUTION_SOURCE_REQUIRED", "Execution owner and origin must be supplied together."
            )
        fingerprint = _execution_fingerprint(session_id, source, owner, origin)
        async with self._execution_lock:
            previous = self._executions.get(execution_id)
            if previous is not None:
                _check_execution_owner(previous[1], owner, origin)
                if previous[0] != fingerprint:
                    raise PythonSessionError(
                        "EXECUTION_ID_CONFLICT", "Execution ID already identifies different code."
                    )
            else:
                # Validate explicit sessions before recording acceptance. A durable reservation
                # precedes every worker dispatch; a crash in between becomes unknown, not replay.
                session = await self._get(session_id, owner=owner) if session_id else None
                if session is not None and session.runner.is_running:
                    raise PythonSessionError("SESSION_BUSY", "Python session is running a cell.")
                receipt: dict[str, Any] = {
                    "execution_id": execution_id,
                    "state": ExecutionState.ACCEPTED.value,
                    "cell": None,
                    "meta": dict(meta) if meta is not None else {},
                }
                if owner is not None:
                    receipt.update(_execution_owner=owner, _runtime_origin=origin)
                if self._executions.reserve(execution_id, fingerprint, receipt):
                    self._submitting.add(execution_id)
                    owned_session = None
                    watcher_started = False
                    try:
                        if session is None:
                            owned_session = await self._create_session(owner=owner, ephemeral=True)
                            session = owned_session
                        cell = await self.submit_cell(
                            session.session_id, source, wait_seconds=0, owner=owner,
                            execution_context=execution_context,
                        )
                        cell_id = _require_cell_id(cell["cell_id"])
                        session.bind_cell(cell_id, execution_id, source=source)
                        self._running[execution_id] = (session, cell_id)
                        receipt = project_receipt(
                            {**receipt, "state": ExecutionState.RUNNING.value, "cell": cell},
                            execution_id=execution_id,
                            ephemeral=session.ephemeral,
                        )
                        self._executions.update(execution_id, receipt)
                        self._completion_tasks[execution_id] = asyncio.create_task(
                            self._complete_execution(
                                execution_id, session, cell_id, receipt, started
                            )
                        )
                        watcher_started = True
                    except BaseException:
                        receipt.update(state=ExecutionState.UNKNOWN.value, cell=None)
                        self._executions.update(execution_id, receipt)
                        raise
                    finally:
                        try:
                            if not watcher_started:
                                self._running.pop(execution_id, None)
                                if owned_session is not None:
                                    await self._release_session(owned_session)
                        finally:
                            self._submitting.discard(execution_id)
        return await self.execution(execution_id, wait_seconds=wait_seconds, owner=owner, origin=origin)

    async def execution(
        self, execution_id: str, *, wait_seconds: float = 0,
        owner: str | None = None, origin: str | None = None,
    ) -> dict[str, Any]:
        _validate_wait(wait_seconds)
        previous = self._executions.get(execution_id)
        if previous is None:
            raise PythonSessionError("EXECUTION_NOT_FOUND", "Python execution receipt not found.")
        _check_execution_owner(previous[1], owner, origin)
        task = self._completion_tasks.get(execution_id)
        if task is not None and wait_seconds > 0:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=wait_seconds)
            except TimeoutError:
                pass
        previous = self._executions.get(execution_id)
        if previous is None:
            raise PythonSessionError("EXECUTION_NOT_FOUND", "Python execution receipt not found.")
        result = self._project_stored_receipt(execution_id, previous[0], previous[1])
        result.pop("_execution_owner", None)
        result.pop("_runtime_origin", None)
        return result

    async def _complete_execution(
        self, execution_id: str, session: _Session, cell_id: int,
        receipt: dict[str, Any], started: float
    ) -> None:
        try:
            await session.runner.wait_cell(cell_id)
            observation = _read_runner_observation(session, cell_id)
            terminal_state = observation.state
            if terminal_state not in {ExecutionState.SUCCEEDED, ExecutionState.FAILED}:
                receipt.update(state=ExecutionState.UNKNOWN.value, cell=None)
            else:
                receipt.update(
                    state=terminal_state.value,
                    cell=project_cell(
                        observation,
                        execution_id=execution_id,
                        ephemeral=session.ephemeral,
                    ),
                )
        except (ValueError, OSError, asyncio.CancelledError):
            receipt.update(state=ExecutionState.UNKNOWN.value, cell=None)
        finally:
            receipt["meta"]["duration_ms"] = max(0, int((time.monotonic() - started) * 1000))
            try:
                self._executions.update(execution_id, receipt)
            finally:
                self._running.pop(execution_id, None)
                try:
                    if session.ephemeral:
                        await self._release_session(session)
                finally:
                    self._completion_tasks.pop(execution_id, None)

    async def _release_session(self, session: _Session) -> None:
        async with self._lock:
            self._sessions.pop(session.session_id, None)
        await self._teardown_session(session)

    async def _teardown_session(self, session: _Session) -> None:
        """Close one session's interpreter with a hard upper bound.

        The runner already bounds its own cleanup; this is the backstop that keeps a
        future regression from turning a single wedged teardown into a dead workspace.
        """
        try:
            await asyncio.wait_for(
                session.runner.aclose(), timeout=SESSION_TEARDOWN_TIMEOUT_SECONDS
            )
        except TimeoutError:
            logger.error(
                "session=%s event=interpreter_teardown_timeout", session.session_id
            )

    async def snapshot(
        self,
        session_id: str,
        *,
        include_all: bool,
        owner: str | None = None,
        wait_seconds: float = 330,
    ) -> dict[str, Any]:
        _validate_wait(wait_seconds)
        session = await self._get(session_id, owner=owner)
        if wait_seconds > 0:
            tasks = self._snapshot_wait_tasks(session, include_all=include_all)
            if tasks:
                done, _ = await asyncio.wait(tasks, timeout=wait_seconds)
                for task in done:
                    if not task.cancelled() and (exc := task.exception()) is not None:
                        logger.warning(
                            "Execution completion task failed before snapshot",
                            exc_info=exc,
                        )
                session = await self._get(session_id, owner=owner)

        persisted_by_cell = {
            cell_id: (execution_id, fingerprint, receipt)
            for cell_id, (execution_id, fingerprint, receipt) in self._executions.find_by_session(
                session.session_id
            ).items()
            if not _is_legacy_one_shot_receipt(fingerprint, receipt)
        }
        cell_ids = sorted(self._session_cell_ids(session, persisted_by_cell))
        if not cell_ids:
            return project_session_snapshot(_session_payload(session), [])

        selected = cell_ids if include_all else [cell_ids[-1]]
        projected_cells = [
            self._project_session_cell(session, cell_id, persisted_by_cell)
            for cell_id in selected
        ]
        return project_session_snapshot(_session_payload(session), projected_cells)

    async def send_input(self, session_id: str, data: str, *, owner: str | None = None) -> dict[str, Any]:
        session = await self._get(session_id, owner=owner)
        if not session.runner.is_running:
            raise PythonSessionError("NO_RUNNING_CELL", "No Python cell is awaiting input.")
        if not data:
            raise PythonSessionError("INVALID_INPUT", "Input data is required.")
        session.runner.send_stdin(data)
        return {"session_id": session.session_id, "bytes_sent": len(data.encode("utf-8"))}

    async def interrupt(self, session_id: str, *, owner: str | None = None) -> dict[str, Any]:
        session = await self._get(session_id, owner=owner)
        if not session.runner.cancel_current_cell():
            raise PythonSessionError("NO_RUNNING_CELL", "No Python cell is running.")
        return {"session_id": session.session_id, "interrupt_sent": True}

    async def interrupt_execution(self, execution_id: str, *, owner: str | None = None) -> dict[str, Any]:
        receipt = await self.execution(execution_id, owner=owner)
        active = self._active_execution(execution_id)
        cell = receipt["cell"]
        # A completed cell's watcher may still be pending while this explicit session
        # starts another cell. Only interrupt the observed target, with no intervening await.
        sent = (
            active is not None and cell is not None and cell["running"] and not cell["done"]
            and active[0].runner.cancel_current_cell()
        )
        return {
            "execution_id": execution_id,
            "session_id": None,
            "state": receipt["state"],
            "interrupt_sent": sent,
        }

    def _session_running_executions(self, session: _Session) -> list[tuple[str, int]]:
        return [
            (execution_id, cell_id)
            for execution_id, (owner, cell_id) in self._running.items()
            if owner is session
        ]

    def _session_cell_ids(
        self,
        session: _Session,
        persisted_by_cell: Mapping[int, Any] | None = None,
    ) -> set[int]:
        cell_ids = {_require_cell_id(cell["cell_id"]) for cell in session.runner.list_cells()}
        cell_ids.update(cell_id for _, cell_id in self._session_running_executions(session))
        cell_ids.update(session.cell_bindings)
        if persisted_by_cell:
            cell_ids.update(persisted_by_cell)
        return cell_ids

    def _snapshot_wait_tasks(
        self, session: _Session, *, include_all: bool
    ) -> list[asyncio.Task[None]]:
        running = self._session_running_executions(session)
        if not running:
            return []
        if include_all:
            target_ids = [execution_id for execution_id, _ in running]
        else:
            latest_execution_id, latest_running_cell_id = max(running, key=lambda item: item[1])
            known_cell_ids = self._session_cell_ids(session)
            if known_cell_ids and latest_running_cell_id < max(known_cell_ids):
                return []
            target_ids = [latest_execution_id]
        return [
            self._completion_tasks[execution_id]
            for execution_id in target_ids
            if execution_id in self._completion_tasks
        ]

    def _active_execution(self, execution_id: str) -> tuple[_Session, int] | None:
        active = self._running.get(execution_id)
        if active is None or active[0].session_id not in self._sessions:
            return None
        return active

    def _project_stored_receipt(
        self, execution_id: str, fingerprint: str, receipt: dict[str, Any]
    ) -> dict[str, Any]:
        active = self._active_execution(execution_id)
        live_observation: CellObservation | None = None
        ephemeral = False
        if active is not None:
            session, cell_id = active
            ephemeral = session.ephemeral
            try:
                live_observation = _read_runner_observation(session, cell_id)
            except ValueError:
                # The worker exited without a final cell result, before its watcher resumed.
                receipt.update(state=ExecutionState.UNKNOWN.value, cell=None)
        else:
            # Submission or execution ended without persisting a final receipt.
            # No live owner remains to wait for; the durable outcome is unknown.
            self._reconcile_orphaned_receipt(execution_id, receipt)
            ephemeral = _is_legacy_one_shot_receipt(fingerprint, receipt)
        return project_receipt(
            receipt,
            execution_id=execution_id,
            ephemeral=ephemeral,
            live_observation=live_observation,
        )

    def _resolve_cell_receipt(
        self,
        session: _Session,
        cell_id: int,
        persisted_by_cell: Mapping[int, tuple[str, str, dict[str, Any]]],
    ) -> tuple[str, str, dict[str, Any]] | None:
        persisted = persisted_by_cell.get(cell_id)
        binding = session.cell_bindings.get(cell_id)
        if binding is None or (persisted is not None and persisted[0] == binding.execution_id):
            entry = persisted
        else:
            stored = self._executions.get(binding.execution_id)
            entry = (binding.execution_id, stored[0], stored[1]) if stored is not None else None
        if entry is None:
            return None

        execution_id, _, receipt = entry
        raw_cell = receipt.get("cell")
        source = str(raw_cell.get("source") or "") if isinstance(raw_cell, dict) else ""
        session.bind_cell(cell_id, execution_id, source=source)
        return entry

    def _project_session_cell(
        self,
        session: _Session,
        cell_id: int,
        persisted_by_cell: Mapping[int, tuple[str, str, dict[str, Any]]],
    ) -> dict[str, Any]:
        entry = self._resolve_cell_receipt(session, cell_id, persisted_by_cell)
        if entry is None:
            return _cell_snapshot(session, cell_id)

        execution_id, fingerprint, raw_receipt = entry
        projected_receipt = self._project_stored_receipt(execution_id, fingerprint, raw_receipt)
        if projected_receipt["cell"] is not None:
            return projected_receipt["cell"]

        return project_cell(
            CellObservation(
                session_id=session.session_id,
                cell_id=cell_id,
                source=session.cell_source(cell_id),
            ),
            execution_id=execution_id,
            state=projected_receipt["state"],
        )

    def _reconcile_orphaned_receipt(self, execution_id: str, receipt: dict[str, Any]) -> None:
        if (
            execution_id in self._submitting
            or execution_id in self._completion_tasks
            or not mark_receipt_unknown(receipt)
        ):
            return
        try:
            self._executions.update(execution_id, receipt)
        except (OSError, sqlite3.Error):
            logger.warning("Failed to persist reconciled unknown receipt %s", execution_id, exc_info=True)

    async def _get(self, session_id: str, *, owner: str | None) -> _Session:
        normalized = _session_id(session_id)
        async with self._lock:
            session = self._sessions.get(normalized)
        if session is None:
            raise PythonSessionError("SESSION_NOT_FOUND", "Python session was not found.")
        _check_owner(session.owner, owner)
        return session


def _session_id(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise PythonSessionError("INVALID_SESSION_ID", "session_id is required.")
    return normalized


def _validate_source_key(key: str | None) -> None:
    if key is not None and (not isinstance(key, str) or not key.strip() or len(key) > 8192):
        raise PythonSessionError("EXECUTION_SOURCE_REQUIRED", "A bounded execution source is required.")


def _check_owner(actual: str | None, expected: str | None) -> None:
    _validate_source_key(expected)
    if actual != expected:
        raise PythonSessionError("SESSION_SOURCE_MISMATCH", "Python session belongs to another execution source.")


def _check_execution_owner(receipt: Mapping[str, Any], owner: str | None, origin: str | None) -> None:
    _check_owner(receipt.get("_execution_owner"), owner)
    _validate_source_key(origin)
    if origin is not None and receipt.get("_runtime_origin") != origin:
        raise PythonSessionError("EXECUTION_SOURCE_MISMATCH", "Execution source differs from its original submission.")


def _execution_fingerprint(
    session_id: str | None, source: str, owner: str | None = None, origin: str | None = None,
) -> str:
    fields = [session_id, source] if owner is None else [session_id, source, owner, origin]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def _is_legacy_one_shot_receipt(fingerprint: str, receipt: dict[str, Any]) -> bool:
    """Older receipts persist the submitted owner choice in their fingerprint.

    Normalize only a proven one-shot submission, never guess from cell IDs.
    """
    cell = receipt.get("cell")
    return (
        isinstance(cell, dict)
        and cell.get("session_id") is not None
        and isinstance(cell.get("source"), str)
        and fingerprint == _execution_fingerprint(
            None, cell["source"], receipt.get("_execution_owner"), receipt.get("_runtime_origin"),
        )
    )


def _require_cell_id(value: object) -> int:
    parsed = parse_cell_id(value)
    if parsed is None:
        raise TypeError(f"Expected integer cell_id, got {value!r}")
    return parsed


def _validate_wait(wait_seconds: float) -> None:
    if not 0 <= wait_seconds <= 330:
        raise PythonSessionError("INVALID_WAIT", "wait_seconds must be between 0 and 330.")


def _validate_source(source: str) -> None:
    try:
        precheck_syntax(source)
    except (SyntaxError, ValueError) as exc:
        raise PythonSessionError("INVALID_SOURCE", str(exc)) from exc


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _session_payload(session: _Session) -> dict[str, Any]:
    running_source = session.runner.get_active_source() if session.runner.is_running else None
    return {
        "session_id": session.session_id,
        "label": session.label,
        "description": session.description,
        "running": session.runner.is_running,
        "running_source": running_source,
    }


def _read_runner_observation(session: _Session, cell_id: int) -> CellObservation:
    try:
        snapshot = session.runner.get_cell_snapshot(cell_id)
    except ValueError:
        if not session.runner.is_running:
            raise
        output = session.runner.get_current_output()
        return CellObservation(
            session_id=session.session_id,
            cell_id=cell_id,
            source=session.cell_source(cell_id),
            stdout=output.stdout,
            stderr=output.stderr,
            result=output.result,
            running=True,
            done=False,
        )
    observation = CellObservation.from_mapping(
        snapshot,
        session_id=session.session_id,
        cell_id=_require_cell_id(snapshot["cell_id"]),
    )
    # Guard against stale `_active_cid` left behind if the worker exited without `cell_end`.
    if not observation.done and observation.running and not session.runner.is_running:
        raise ValueError(f"Cell {cell_id} exited without completion")
    return observation


def _cell_snapshot(session: _Session, cell_id: int) -> dict[str, Any]:
    try:
        observation = _read_runner_observation(session, cell_id)
    except ValueError:
        if session.runner.get_cell_source(cell_id) is None and cell_id not in session.cell_bindings:
            raise
        observation = CellObservation(
            session_id=session.session_id,
            cell_id=cell_id,
            source=session.cell_source(cell_id),
        )
    binding = session.cell_bindings.get(cell_id)
    return project_cell(
        observation,
        execution_id=binding.execution_id if binding is not None else None,
    )
