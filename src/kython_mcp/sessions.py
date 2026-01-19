from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from mcp.server.fastmcp import Context

from .interpreter_runner import AsyncInterpreterRunner
from .local_log import get_logger, get_session_logger

logger = get_logger("kython_mcp.sessions")


@dataclass
class SessionRecord:
    """Snapshot of a Python session."""

    runner: AsyncInterpreterRunner
    logger: object
    ctx_key: int
    public_id: str
    label: str | None = None
    description: str | None = None
    python_executable: str | None = None
    created_at: float = field(default_factory=time.time)


class InterpreterSessionStore:
    """Manage Python sessions under a single MCP client context."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._ctx_index: dict[int, set[str]] = {}
        self._ctx_public_map: dict[int, dict[str, str]] = {}
        self._ctx_public_counter: dict[int, int] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        ctx: Context,
        label: str | None = None,
        description: str | None = None,
        python_executable: str | None = None,
    ) -> tuple[str, SessionRecord]:
        key = self._session_key(ctx)
        async with self._lock:
            session_id = uuid.uuid4().hex
            counter = self._ctx_public_counter.get(key, 0) + 1
            self._ctx_public_counter[key] = counter
            public_id = str(counter)
            session_name = f"session-{public_id}"
            loop = asyncio.get_running_loop()
            runner = AsyncInterpreterRunner(
                name=session_name,
                loop=loop,
                python_executable=python_executable,
            )
            record = SessionRecord(
                runner=runner,
                logger=get_session_logger(session_name),
                ctx_key=key,
                public_id=public_id,
                label=label,
                description=description,
                python_executable=python_executable,
                created_at=time.time(),
            )
            self._sessions[session_id] = record
            self._ctx_index.setdefault(key, set()).add(session_id)
            self._ctx_public_map.setdefault(key, {})[public_id] = session_id
            logger.info(
                "create_session session_id=%s ctx_key=%s label=%s description=%s",
                session_id,
                key,
                label,
                description,
            )
            return session_id, record

    async def list_sessions(self, ctx: Context) -> list[tuple[str, SessionRecord]]:
        key = self._session_key(ctx)
        async with self._lock:
            session_ids = list(self._ctx_index.get(key, set()))
            records = [
                (self._sessions[sid].public_id, self._sessions[sid])
                for sid in session_ids
                if sid in self._sessions
            ]
        return sorted(records, key=lambda item: item[1].created_at)

    async def get_session(
        self, ctx: Context, session_id: str
    ) -> tuple[str, SessionRecord]:
        key = self._session_key(ctx)
        async with self._lock:
            internal_id = self._resolve_internal_id_locked(key, session_id)
            record = self._sessions.get(internal_id)
        if record is None:
            raise ValueError("Session not found or already closed")
        if record.ctx_key != key:
            raise ValueError("Session does not belong to current MCP context")
        return internal_id, record

    async def reset_session(
        self, ctx: Context, session_id: str
    ) -> tuple[str, SessionRecord]:
        internal_id, record = await self.get_session(ctx, session_id)
        loop = asyncio.get_running_loop()
        runner_label = f"session-{record.public_id}"
        new_runner = AsyncInterpreterRunner(
            name=runner_label,
            loop=loop,
            python_executable=record.python_executable,
        )
        await record.runner.aclose()
        new_record = SessionRecord(
            runner=new_runner,
            logger=record.logger,
            ctx_key=record.ctx_key,
            public_id=record.public_id,
            label=record.label,
            description=record.description,
            python_executable=record.python_executable,
            created_at=record.created_at,
        )
        async with self._lock:
            self._sessions[internal_id] = new_record
        logger.info("reset_session session_id=%s ctx_key=%s", internal_id, record.ctx_key)
        return internal_id, new_record

    async def close_session(self, ctx: Context, session_id: str) -> tuple[str, str]:
        key = self._session_key(ctx)
        async with self._lock:
            internal_id = self._resolve_internal_id_locked(key, session_id)
            record = self._sessions.pop(internal_id, None)
            if record is None:
                raise ValueError("Session not found or already closed")
            if record.ctx_key != key:
                raise ValueError("Session does not belong to current MCP context")
            if key in self._ctx_index:
                self._ctx_index[key].discard(internal_id)
            if key in self._ctx_public_map:
                self._ctx_public_map[key].pop(record.public_id, None)
        await record.runner.aclose()
        logger.info("close_session session_id=%s ctx_key=%s", internal_id, key)
        return internal_id, record.public_id

    async def close_all(self) -> None:
        async with self._lock:
            records = list(self._sessions.values())
            self._sessions.clear()
            self._ctx_index.clear()
            self._ctx_public_map.clear()
            self._ctx_public_counter.clear()
        for record in records:
            await record.runner.aclose()
        logger.info("close_all_sessions count=%d", len(records))

    async def update_description(
        self, ctx: Context, session_id: str, description: str | None
    ) -> tuple[str, SessionRecord]:
        internal_id, record = await self.get_session(ctx, session_id)
        record.description = description
        logger.info(
            "update_session_description session_id=%s ctx_key=%s",
            internal_id,
            record.ctx_key,
        )
        return internal_id, record

    async def update_label(
        self, ctx: Context, session_id: str, label: str | None
    ) -> tuple[str, SessionRecord]:
        internal_id, record = await self.get_session(ctx, session_id)
        record.label = label
        logger.info(
            "update_session_label session_id=%s ctx_key=%s",
            internal_id,
            record.ctx_key,
        )
        return internal_id, record

    @staticmethod
    def _session_key(ctx: Context) -> int:
        return id(ctx.request_context.session)

    def _resolve_internal_id_locked(self, key: int, public_id: str) -> str:
        public_map = self._ctx_public_map.get(key, {})
        internal_id = public_map.get(public_id)
        if internal_id is None:
            raise ValueError("Session not found or already closed")
        return internal_id


def _running_cell_snippet(source: str | None, max_len: int = 80) -> str | None:
    if not source:
        return None
    lines = source.splitlines()
    first = lines[0] if lines else ""
    truncated = first
    if len(first) > max_len:
        truncated = first[:max_len] + "..."
    elif len(lines) > 1:
        truncated = first + "..."
    return truncated
    

def session_payload(record: SessionRecord) -> dict[str, Any]:
    running_snippet = None
    if record.runner.is_running:
        running_snippet = _running_cell_snippet(record.runner.get_active_source())

    return {
        "id": record.public_id,
        "metadata": {
            "label": record.label,
            "description": record.description,
            "running_cell": running_snippet,
            "python_executable": record.python_executable,
        },
    }
