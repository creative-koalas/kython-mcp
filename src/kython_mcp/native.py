from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from .executions import ExecutionStore
from .interpreter_runner import AsyncInterpreterRunner, BusyError
from .utils import precheck_syntax


class PythonSessionError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass
class _Session:
    session_id: str
    runner: AsyncInterpreterRunner
    label: str | None = None
    description: str | None = None
    created_at: float = field(default_factory=time.time)


class NativePythonService:
    """Own Python interpreter sessions for one workspace process."""

    def __init__(self, *, receipt_path: Path | None = None) -> None:
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()
        self._execution_lock = asyncio.Lock()
        self._executions = ExecutionStore(receipt_path)
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
            *(session.runner.aclose() for session in sessions),
            return_exceptions=True,
        )
        self._executions.close()

    async def create_session(
        self,
        *,
        label: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            session_id = uuid4().hex
            runner = AsyncInterpreterRunner(
                name=f"session-{session_id}",
                loop=asyncio.get_running_loop(),
            )
            session = _Session(
                session_id=session_id,
                runner=runner,
                label=_optional_text(label),
                description=_optional_text(description),
            )
            self._sessions[session_id] = session
        return _session_payload(session)

    async def list_sessions(self) -> list[dict[str, Any]]:
        async with self._lock:
            sessions = sorted(self._sessions.values(), key=lambda item: item.created_at)
            return [_session_payload(session) for session in sessions]

    async def update_session(
        self,
        session_id: str,
        *,
        label: str | None,
        description: str | None,
        update_label: bool,
        update_description: bool,
    ) -> dict[str, Any]:
        session = await self._get(session_id)
        if update_label:
            session.label = _optional_text(label)
        if update_description:
            session.description = _optional_text(description)
        return _session_payload(session)

    async def delete_session(self, session_id: str) -> dict[str, str]:
        normalized = _session_id(session_id)
        async with self._lock:
            session = self._sessions.pop(normalized, None)
        if session is None:
            raise PythonSessionError("SESSION_NOT_FOUND", "Python session was not found.")
        execution_ids = [
            execution_id
            for execution_id, (owner, _) in self._running.items()
            if owner is session
        ]
        tasks = [self._completion_tasks[execution_id] for execution_id in execution_ids]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for execution_id in execution_ids:
            self._completion_tasks.pop(execution_id, None)
            self._running.pop(execution_id, None)
            previous = self._executions.get(execution_id)
            if previous is not None and previous[1]["state"] in {"accepted", "running"}:
                # Cancellation may happen before the watcher coroutine starts at all.
                receipt = previous[1]
                receipt.update(state="unknown", cell=None)
                self._executions.update(execution_id, receipt)
        await session.runner.aclose()
        return {"session_id": normalized}

    async def submit_cell(
        self,
        session_id: str,
        source: str,
        *,
        wait_seconds: float,
    ) -> dict[str, Any]:
        source = str(source or "")
        if not source.strip():
            raise PythonSessionError("INVALID_SOURCE", "Python source is required.")
        _validate_source(source)
        session = await self._get(session_id)
        try:
            cell_id = session.runner.start_cell(source)
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
        except TimeoutError:
            pass
        return _cell_snapshot(session, cell_id)

    async def execute(
        self,
        execution_id: str,
        source: str,
        *,
        session_id: str | None = None,
        wait_seconds: float = 5,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit once, then observe that submission even if the caller disconnects."""
        started = time.monotonic()
        execution_id = str(UUID(execution_id))
        if not source.strip():
            raise PythonSessionError("INVALID_SOURCE", "Python source is required.")
        _validate_source(source)
        fingerprint = hashlib.sha256(
            json.dumps([session_id, source], ensure_ascii=False).encode()
        ).hexdigest()
        async with self._execution_lock:
            previous = self._executions.get(execution_id)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise PythonSessionError(
                        "EXECUTION_ID_CONFLICT", "Execution ID already identifies different code."
                    )
            else:
                # Validate explicit sessions before recording acceptance. A durable reservation
                # precedes every worker dispatch; a crash in between becomes unknown, not replay.
                session = await self._get(session_id) if session_id else None
                if session is not None and session.runner.is_running:
                    raise PythonSessionError("SESSION_BUSY", "Python session is running a cell.")
                receipt = {
                    "execution_id": execution_id,
                    "state": "accepted",
                    "cell": None,
                    "meta": meta or {},
                }
                if self._executions.reserve(execution_id, fingerprint, receipt):
                    try:
                        if session is None:
                            created = await self.create_session()
                            session = await self._get(created["session_id"])
                        cell = await self.submit_cell(session.session_id, source, wait_seconds=0)
                        cell_id = int(cell["cell_id"])
                        self._running[execution_id] = (session, cell_id)
                        receipt.update(state="running", cell=cell)
                        self._executions.update(execution_id, receipt)
                        self._completion_tasks[execution_id] = asyncio.create_task(
                            self._complete_execution(
                                execution_id, session, cell_id, not session_id, started
                            )
                        )
                    except BaseException:
                        receipt["state"] = "unknown"
                        receipt["cell"] = None
                        self._executions.update(execution_id, receipt)
                        raise
        task = self._completion_tasks.get(execution_id)
        if task is not None and wait_seconds > 0:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=wait_seconds)
            except TimeoutError:
                pass
        return await self.execution(execution_id)

    async def execution(self, execution_id: str) -> dict[str, Any]:
        previous = self._executions.get(execution_id)
        if previous is None:
            raise PythonSessionError("EXECUTION_NOT_FOUND", "Python execution receipt not found.")
        receipt = previous[1]
        active = self._running.get(execution_id)
        if active is not None:
            try:
                receipt["cell"] = _cell_snapshot(*active)
            except ValueError:
                # The worker exited without a final cell result, before its watcher resumed.
                receipt.update(state="unknown", cell=None)
        return receipt

    async def _complete_execution(
        self, execution_id: str, session: _Session, cell_id: int, ephemeral: bool, started: float
    ) -> None:
        previous = self._executions.get(execution_id)
        assert previous is not None
        receipt = previous[1]
        try:
            await session.runner.wait_cell(cell_id)
            cell = _cell_snapshot(session, cell_id)
            receipt.update(state="failed" if cell["exception"] else "succeeded", cell=cell)
        except (ValueError, OSError, asyncio.CancelledError):
            receipt["state"] = "unknown"
            receipt["cell"] = None
        finally:
            receipt["meta"]["duration_ms"] = max(0, int((time.monotonic() - started) * 1000))
            self._executions.update(execution_id, receipt)
            self._running.pop(execution_id, None)
            if ephemeral:
                async with self._lock:
                    self._sessions.pop(session.session_id, None)
                await session.runner.aclose()
            self._completion_tasks.pop(execution_id, None)

    async def snapshot(
        self,
        session_id: str,
        *,
        include_all: bool,
    ) -> dict[str, Any]:
        session = await self._get(session_id)
        cells = session.runner.list_cells()
        if not cells:
            return {"session": _session_payload(session), "cells": []}
        cell_ids = sorted({int(cell["cell_id"]) for cell in cells})
        selected = cell_ids if include_all else [cell_ids[-1]]
        return {
            "session": _session_payload(session),
            "cells": [_cell_snapshot(session, cell_id) for cell_id in selected],
        }

    async def send_input(self, session_id: str, data: str) -> dict[str, Any]:
        session = await self._get(session_id)
        if not session.runner.is_running:
            raise PythonSessionError("NO_RUNNING_CELL", "No Python cell is awaiting input.")
        if not data:
            raise PythonSessionError("INVALID_INPUT", "Input data is required.")
        session.runner.send_stdin(data)
        return {"session_id": session.session_id, "bytes_sent": len(data.encode("utf-8"))}

    async def interrupt(self, session_id: str) -> dict[str, Any]:
        session = await self._get(session_id)
        if not session.runner.cancel_current_cell():
            raise PythonSessionError("NO_RUNNING_CELL", "No Python cell is running.")
        return {"session_id": session.session_id, "interrupt_sent": True}

    async def _get(self, session_id: str) -> _Session:
        normalized = _session_id(session_id)
        async with self._lock:
            session = self._sessions.get(normalized)
        if session is None:
            raise PythonSessionError("SESSION_NOT_FOUND", "Python session was not found.")
        return session


def _session_id(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise PythonSessionError("INVALID_SESSION_ID", "session_id is required.")
    return normalized


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


def _cell_snapshot(session: _Session, cell_id: int) -> dict[str, Any]:
    try:
        snapshot = session.runner.get_cell_snapshot(cell_id)
    except ValueError:
        if not session.runner.is_running:
            raise
        output = session.runner.get_current_output()
        snapshot = {
            "cell_id": cell_id,
            "source": session.runner.get_cell_source(cell_id),
            "stdout": output.stdout,
            "stderr": output.stderr,
            "result": output.result,
            "exception": None,
            "running": True,
            "done": False,
        }
    return {
        "session_id": session.session_id,
        "cell_id": int(snapshot["cell_id"]),
        "source": str(snapshot.get("source") or ""),
        "stdout": str(snapshot.get("stdout") or ""),
        "stderr": str(snapshot.get("stderr") or ""),
        "result": str(snapshot.get("result") or ""),
        "exception": snapshot.get("exception"),
        "running": bool(snapshot.get("running")),
        "done": bool(snapshot.get("done")),
    }
