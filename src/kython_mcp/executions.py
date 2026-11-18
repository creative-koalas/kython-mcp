"""Durable submission identity; interrupted submissions are never replayed."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

INFLIGHT_RECEIPT_STATES = frozenset({"starting", "accepted", "running"})


def mark_receipt_unknown(receipt: dict[str, Any]) -> bool:
    """Transition an orphaned in-flight receipt to unknown; return whether it changed."""
    if receipt.get("state") not in INFLIGHT_RECEIPT_STATES:
        return False
    receipt["state"] = "unknown"
    receipt["cell"] = None
    return True


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
        # A new owner cannot infer whether the previous worker produced effects.
        self.reconcile_inflight_receipts()

    def reconcile_inflight_receipts(self) -> int:
        """Mark durable in-flight receipts unknown when no live worker owns them."""
        rows = self._db.execute("SELECT id, receipt FROM executions").fetchall()
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

    def update(self, execution_id: str, receipt: dict[str, Any]) -> None:
        with self._db:
            self._db.execute(
                "UPDATE executions SET receipt = ? WHERE id = ?",
                (json.dumps(receipt), execution_id),
            )

    def close(self) -> None:
        self._db.close()
