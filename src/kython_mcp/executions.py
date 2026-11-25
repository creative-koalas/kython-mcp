"""Durable submission identity and canonical execution projection."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class ExecutionState(StrEnum):
    """Canonical execution state vocabulary shared by receipts and cell snapshots."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


INFLIGHT_RECEIPT_STATES = frozenset(
    {ExecutionState.ACCEPTED.value, ExecutionState.RUNNING.value, "starting"}
)


@dataclass(frozen=True, slots=True)
class CellObservation:
    """Raw cell state observed from a runner or restored from a persisted receipt."""

    session_id: str | None = None
    cell_id: int | None = None
    source: str = ""
    stdout: str = ""
    stderr: str = ""
    result: str = ""
    exception: str | None = None
    running: bool = False
    done: bool = False

    @property
    def state(self) -> ExecutionState:
        """Derive canonical execution state from observed cell execution flags."""
        if self.done:
            return ExecutionState.FAILED if self.exception else ExecutionState.SUCCEEDED
        if self.running:
            return ExecutionState.RUNNING
        return ExecutionState.UNKNOWN

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        session_id: str | None = None,
        cell_id: int | None = None,
        fallback_state: ExecutionState | None = None,
    ) -> CellObservation:
        """Build a CellObservation from a runner snapshot or persisted cell mapping."""
        raw_session_id = session_id if session_id is not None else raw.get("session_id")
        raw_exception = raw.get("exception")
        raw_running = raw.get("running")
        raw_done = raw.get("done")
        default_running = fallback_state == ExecutionState.RUNNING
        default_done = fallback_state in {ExecutionState.SUCCEEDED, ExecutionState.FAILED}
        return cls(
            session_id=str(raw_session_id) if raw_session_id is not None else None,
            cell_id=cell_id if cell_id is not None else parse_cell_id(raw.get("cell_id")),
            source=str(raw.get("source") or ""),
            stdout=str(raw.get("stdout") or ""),
            stderr=str(raw.get("stderr") or ""),
            result=str(raw.get("result") or ""),
            exception=str(raw_exception) if raw_exception is not None else None,
            running=default_running if raw_running is None else bool(raw_running),
            done=default_done if raw_done is None else bool(raw_done),
        )


def normalize_execution_state(state: object) -> ExecutionState:
    """Normalize a stored or legacy receipt state into the canonical vocabulary."""
    if isinstance(state, ExecutionState):
        return state
    if state == "starting":
        return ExecutionState.RUNNING
    if isinstance(state, str):
        try:
            return ExecutionState(state)
        except ValueError:
            return ExecutionState.UNKNOWN
    return ExecutionState.UNKNOWN


def parse_cell_id(value: object) -> int | None:
    """Parse a persisted or runner cell_id into an integer, returning None if invalid."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def mark_receipt_unknown(receipt: dict[str, Any]) -> bool:
    """Transition an orphaned in-flight receipt to unknown; return whether it changed."""
    if receipt.get("state") not in INFLIGHT_RECEIPT_STATES:
        return False
    receipt["state"] = ExecutionState.UNKNOWN.value
    receipt["cell"] = None
    return True


def project_cell(
    observation: CellObservation,
    *,
    execution_id: str | None = None,
    state: ExecutionState | str | None = None,
    ephemeral: bool = False,
) -> dict[str, Any]:
    """Build the single canonical cell projection shared by snapshot() and execution()."""
    resolved_state = normalize_execution_state(state) if state is not None else observation.state
    payload: dict[str, Any] = {
        "session_id": None if ephemeral else observation.session_id,
        "cell_id": None if ephemeral else observation.cell_id,
        "source": observation.source,
        "stdout": observation.stdout,
        "stderr": observation.stderr,
        "result": observation.result,
        "exception": observation.exception,
        "running": observation.running,
        "done": observation.done,
        "state": resolved_state.value,
    }
    if execution_id:
        payload["execution_id"] = execution_id
    return payload


def project_receipt(
    receipt: Mapping[str, Any],
    *,
    execution_id: str,
    ephemeral: bool = False,
    live_observation: CellObservation | None = None,
) -> dict[str, Any]:
    """Project a stored receipt and its embedded cell through the canonical model."""
    if not execution_id:
        raise ValueError("execution_id is required to project a receipt")
    state = normalize_execution_state(receipt.get("state"))
    projected: dict[str, Any] = {**receipt, "execution_id": execution_id, "state": state.value}
    if state in {ExecutionState.ACCEPTED, ExecutionState.UNKNOWN}:
        projected["cell"] = None
        return projected
    if live_observation is not None:
        projected["cell"] = project_cell(
            live_observation,
            execution_id=execution_id,
            ephemeral=ephemeral,
        )
    elif isinstance(raw_cell := receipt.get("cell"), Mapping):
        projected["cell"] = project_cell(
            CellObservation.from_mapping(raw_cell, fallback_state=state),
            execution_id=execution_id,
            state=state,
            ephemeral=ephemeral,
        )
    else:
        projected["cell"] = None
    return projected


def project_session_snapshot(
    session_payload: dict[str, Any],
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the canonical session snapshot payload from session metadata and projected cells."""
    payload: dict[str, Any] = {
        "session": session_payload,
        "cells": cells,
    }
    if cells:
        latest = cells[-1]
        latest_execution_id = latest.get("execution_id")
        if isinstance(latest_execution_id, str) and latest_execution_id:
            payload["execution_id"] = latest_execution_id
            payload["state"] = latest["state"]
    return payload


class ExecutionStore:
    def __init__(self, path: Path | None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._db = sqlite3.connect(str(path) if path else ":memory:")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS executions "
            "(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, receipt TEXT NOT NULL)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_executions_session_id "
            "ON executions (json_extract(receipt, '$.cell.session_id')) "
            "WHERE json_extract(receipt, '$.cell.session_id') IS NOT NULL"
        )
        # A new owner cannot infer whether the previous worker produced effects.
        self.reconcile_inflight_receipts()

    def reconcile_inflight_receipts(self) -> int:
        """Mark durable in-flight receipts unknown when no live worker owns them."""
        inflight = tuple(sorted(INFLIGHT_RECEIPT_STATES))
        placeholders = ", ".join("?" for _ in inflight)
        rows = self._db.execute(
            f"SELECT id, receipt FROM executions "
            f"WHERE json_extract(receipt, '$.state') IN ({placeholders})",
            inflight,
        ).fetchall()
        reconciled = 0
        with self._db:
            for execution_id, raw in rows:
                receipt = json.loads(raw)
                if mark_receipt_unknown(receipt):
                    self._db.execute(
                        "UPDATE executions SET receipt = ? WHERE id = ?",
                        (json.dumps(receipt), execution_id),
                    )
                    reconciled += 1
        return reconciled

    def reserve(self, execution_id: str, fingerprint: str, receipt: dict[str, Any]) -> bool:
        with self._db:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO executions VALUES (?, ?, ?)",
                (execution_id, fingerprint, json.dumps(receipt)),
            )
        return cursor.rowcount == 1

    def get(self, execution_id: str) -> tuple[str, dict[str, Any]] | None:
        row = self._db.execute(
            "SELECT fingerprint, receipt FROM executions WHERE id = ?", (execution_id,)
        ).fetchone()
        return (str(row[0]), json.loads(row[1])) if row else None

    def find_by_session(self, session_id: str) -> dict[int, tuple[str, str, dict[str, Any]]]:
        """Return persisted receipts indexed by cell_id for one explicit session."""
        rows = self._db.execute(
            "SELECT id, fingerprint, receipt FROM executions "
            "WHERE json_extract(receipt, '$.cell.session_id') = ? "
            "ORDER BY rowid",
            (session_id,),
        ).fetchall()
        by_cell: dict[int, tuple[str, str, dict[str, Any]]] = {}
        for execution_id, fingerprint, raw in rows:
            receipt = json.loads(raw)
            cell = receipt.get("cell")
            if not isinstance(cell, dict) or cell.get("session_id") != session_id:
                continue
            cell_id = parse_cell_id(cell.get("cell_id"))
            if cell_id is not None:
                by_cell[cell_id] = (str(execution_id), str(fingerprint), receipt)
        return by_cell

    def update(self, execution_id: str, receipt: dict[str, Any]) -> None:
        with self._db:
            self._db.execute(
                "UPDATE executions SET receipt = ? WHERE id = ?",
                (json.dumps(receipt), execution_id),
            )

    def close(self) -> None:
        self._db.close()
