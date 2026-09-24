"""Durable submission identity; interrupted submissions are never replayed."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


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
        for execution_id, raw in self._db.execute("SELECT id, receipt FROM executions"):
            receipt = json.loads(raw)
            if receipt["state"] in {"accepted", "running"}:
                receipt["state"] = "unknown"
                receipt["cell"] = None
                self.update(execution_id, receipt)
        self._db.commit()

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
