from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Set, Tuple

import yaml
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from .interpreter_runner import AsyncInterpreterRunner, BusyError
from .local_log import get_logger, get_session_logger

logger = get_logger("kython_mcp.server")


def _precheck_syntax(code: str) -> None:
    """在主进程侧进行语法预检，确保尽早失败。

    策略与子解释器保持一致：优先尝试 "single"，失败再尝试 "exec"。
    若两者均失败，抛出包含行列与上下文的 ValueError。
    """
    if not isinstance(code, str):
        raise ValueError("code 必须为字符串")
    # 允许空字符串/空白：不视为语法错误
    try:
        compile(code, "<mcp-client-code>", "single")
        return
    except SyntaxError:
        pass
    try:
        compile(code, "<mcp-client-code>", "exec")
    except SyntaxError as e:
        text = e.text or ""
        where = f"第 {e.lineno} 行, 第 {e.offset} 列" if e.lineno else "未知位置"
        msg = f"语法错误: {e.msg} ({where})\n{text}"
        raise ValueError(msg) from e


def _dump_yaml(data: object) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, indent=2)


class StartCellResult(BaseModel):
    session_id: str = Field(description="所属 session ID")
    cell_id: int = Field(description="启动的 cell 序号")
    status: str = Field(description="状态说明，例如 started")


class CreateSessionResult(BaseModel):
    session_id: str = Field(description="新建 session 的唯一 ID")
    name: str = Field(description="session 的显示名称")


class PythonSessionInfo(BaseModel):
    session_id: str = Field(description="session ID")
    name: str = Field(description="session 的显示名称")
    running: bool = Field(description="是否有代码正在运行")


class CellSnapshot(BaseModel):
    cell_id: int = Field(description="cell 序号")
    stdout: str = Field(description="当前/最终标准输出")
    stderr: str = Field(description="当前/最终标准错误")
    result: str = Field(description="当前/最终的 displayhook repr")
    running: bool = Field(description="是否仍在运行")
    done: bool = Field(description="是否已结束")
    exception: str | None = Field(description="若已结束则可能包含异常堆栈")


class CellInfo(BaseModel):
    cell_id: int = Field(description="cell 序号")
    status: str = Field(description="cell 状态: running 或 completed")
    has_exception: bool = Field(description="是否有异常")


@dataclass
class _SessionRecord:
    runner: AsyncInterpreterRunner
    logger: object
    ctx_key: int
    public_id: str


def _session_payload(record: _SessionRecord) -> Dict[str, Any]:
    return {"sid": record.public_id, "run": record.runner.is_running}


def _tool_response(action: str, **payload: Any) -> str:
    data: Dict[str, Any] = {"action": action}
    data.update(payload)
    return _dump_yaml(data)


class InterpreterSessionStore:
    """管理单个 MCP 会话下的多个 Python session。"""

    def __init__(self):
        self._sessions: Dict[str, _SessionRecord] = {}
        self._ctx_index: Dict[int, Set[str]] = {}
        self._ctx_public_map: Dict[int, Dict[str, str]] = {}
        self._ctx_public_counter: Dict[int, int] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, ctx: Context) -> Tuple[str, _SessionRecord]:
        key = self._session_key(ctx)
        async with self._lock:
            session_id = uuid.uuid4().hex
            counter = self._ctx_public_counter.get(key, 0) + 1
            self._ctx_public_counter[key] = counter
            public_id = str(counter)
            session_label = f"session-{public_id}"
            loop = asyncio.get_running_loop()
            runner = AsyncInterpreterRunner(name=session_label, loop=loop)
            record = _SessionRecord(
                runner=runner,
                logger=get_session_logger(session_label),
                ctx_key=key,
                public_id=public_id,
            )
            self._sessions[session_id] = record
            self._ctx_index.setdefault(key, set()).add(session_id)
            self._ctx_public_map.setdefault(key, {})[public_id] = session_id
            logger.info(
                "create_session session_id=%s session_label=%s ctx_key=%s",
                session_id,
                session_label,
                key,
            )
            return session_id, record

    async def list_sessions(self, ctx: Context) -> List[Tuple[str, _SessionRecord]]:
        key = self._session_key(ctx)
        async with self._lock:
            session_ids = list(self._ctx_index.get(key, set()))
            records = [
                (self._sessions[sid].public_id, self._sessions[sid])
                for sid in session_ids
                if sid in self._sessions
            ]
        return records

    async def get_session(
        self, ctx: Context, session_id: str
    ) -> Tuple[str, _SessionRecord]:
        key = self._session_key(ctx)
        async with self._lock:
            internal_id = self._resolve_internal_id_locked(key, session_id)
            record = self._sessions.get(internal_id)
        if record is None:
            raise ValueError("指定的 session 不存在或已关闭")
        if record.ctx_key != key:
            raise ValueError("session 不属于当前 MCP 会话")
        return internal_id, record

    async def reset_session(
        self, ctx: Context, session_id: str
    ) -> Tuple[str, _SessionRecord]:
        internal_id, record = await self.get_session(ctx, session_id)
        loop = asyncio.get_running_loop()
        session_label = f"session-{record.public_id}"
        new_runner = AsyncInterpreterRunner(name=session_label, loop=loop)
        await record.runner.aclose()
        new_record = _SessionRecord(
            runner=new_runner,
            logger=record.logger,
            ctx_key=record.ctx_key,
            public_id=record.public_id,
        )
        async with self._lock:
            self._sessions[internal_id] = new_record
        logger.info(
            "reset_session session_id=%s ctx_key=%s", internal_id, record.ctx_key
        )
        return internal_id, new_record

    async def close_session(self, ctx: Context, session_id: str) -> Tuple[str, str]:
        key = self._session_key(ctx)
        async with self._lock:
            internal_id = self._resolve_internal_id_locked(key, session_id)
            record = self._sessions.pop(internal_id, None)
            if record is None:
                raise ValueError("指定的 session 不存在或已关闭")
            if record.ctx_key != key:
                raise ValueError("session 不属于当前 MCP 会话")
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

    @staticmethod
    def _session_key(ctx: Context) -> int:
        return id(ctx.request_context.session)

    def _resolve_internal_id_locked(self, key: int, public_id: str) -> str:
        public_map = self._ctx_public_map.get(key, {})
        internal_id = public_map.get(public_id)
        if internal_id is None:
            raise ValueError("指定的 session 不存在或已关闭")
        return internal_id


session_store = InterpreterSessionStore()

server = FastMCP(
    name="Kython Subinterpreter",
    instructions="使用 create_python_session 创建多个独立 Python session，start_python_cell 异步执行代码，并通过快照/事件轮询结果。",
)


def main() -> None:
    """以 stdio 传输启动 MCP Server，并在退出时关闭所有会话。"""
    try:
        server.run(transport="stdio")
    finally:
        asyncio.run(session_store.close_all())


@server.tool(
    name="create_python_session",
    description="创建一个新的 Python 子解释器 session，并返回 session_id。",
)
async def create_python_session(ctx: Context | None = None) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    internal_id, record = await session_store.create_session(ctx)
    record.logger.info("create_python_session session_id=%s", internal_id)
    logger.info(
        "create_python_session client=%s session_id=%s", ctx.client_id, internal_id
    )
    return _tool_response(
        "create_python_session", session=_session_payload(record), stat="new"
    )


@server.tool(
    name="list_python_sessions",
    description="列出当前 MCP 会话下已创建的所有 Python session。",
)
async def list_python_sessions(ctx: Context | None = None) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    sessions = await session_store.list_sessions(ctx)
    logger.info("list_python_sessions client=%s count=%d", ctx.client_id, len(sessions))
    session_entries = [_session_payload(record) for _, record in sessions]
    return _tool_response(
        "list_python_sessions", count=len(session_entries), session=session_entries
    )


@server.tool(
    name="close_python_session",
    description="关闭并移除指定的 Python session。",
)
async def close_python_session(
    session_id: Annotated[str, Field(description="要关闭的 session ID")],
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    internal_id, public_id = await session_store.close_session(ctx, session_id)
    logger.info("close_python_session client=%s session=%s", ctx.client_id, internal_id)
    return _tool_response("close_python_session", sid=public_id, stat="closed")


@server.tool(
    name="start_python_cell",
    description="非阻塞启动执行 Python 代码，立即返回 cell_id。",
)
async def start_python_cell(
    session_id: Annotated[str, Field(description="要运行的 session ID")],
    code: Annotated[str, Field(description="要执行的 Python 代码")],
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    # 语法预检：发现语法问题直接返回错误
    _precheck_syntax(code)

    internal_id, record = await session_store.get_session(ctx, session_id)
    runner = record.runner
    slog = record.logger
    try:
        logger.info(
            "start_python_cell start client=%s session=%s", ctx.client_id, internal_id
        )
        slog.info(
            "调用 start_python_cell, 代码长度=%d\n代码:\n%s", len(code or ""), code
        )
        cid = runner.start_cell(code)
    except BusyError as exc:
        logger.warning(
            "start_python_cell busy client=%s session=%s", ctx.client_id, internal_id
        )
        slog.warning("start_python_cell 忙，拒绝执行")
        raise RuntimeError(f"当前会话正在执行其他代码: {exc}") from exc
    except Exception:
        logger.exception(
            "start_python_cell error client=%s session=%s", ctx.client_id, internal_id
        )
        slog.exception("start_python_cell 异常")
        raise
    logger.info(
        "start_python_cell done client=%s session=%s cid=%s",
        ctx.client_id,
        internal_id,
        cid,
    )
    slog.info("start_python_cell 已启动 cid=%s", cid)
    return _tool_response(
        "start_python_cell",
        session=_session_payload(record),
        cell={"cid": cid, "stat": "start"},
    )


@server.tool(
    name="get_python_cell_snapshot",
    description="获取指定或当前活动 cell 的输出快照（运行中或已完成）。",
)
async def get_python_cell_snapshot(
    session_id: Annotated[str, Field(description="要查询的 session ID")],
    cell_id: Annotated[
        int | None,
        Field(
            description="可选指定 cell_id；为空则优先返回当前活动 cell，否则返回最近完成者"
        ),
    ] = None,
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")
    internal_id, record = await session_store.get_session(ctx, session_id)
    runner = record.runner
    slog = record.logger
    snap = runner.get_cell_snapshot(cell_id)
    logger.info(
        "get_python_cell_snapshot client=%s session=%s cid=%s running=%s done=%s",
        ctx.client_id,
        internal_id,
        snap["cell_id"],
        snap["running"],
        snap["done"],
    )
    slog.info(
        "调用 get_python_cell_snapshot cid=%s, running=%s, done=%s",
        snap["cell_id"],
        snap["running"],
        snap["done"],
    )
    stdout_text = snap.get("stdout") or ""
    stderr_text = snap.get("stderr") or ""
    result_text = snap.get("result") or ""
    exception_text = snap.get("exception") or ""
    cell_info = {
        "cid": snap["cell_id"],
        "run": bool(snap["running"]),
        "dn": bool(snap["done"]),
        "out": stdout_text,
        "err": stderr_text,
        "val": result_text,
    }
    if exception_text:
        cell_info["exc"] = exception_text
    return _tool_response(
        "get_python_cell_snapshot", session=_session_payload(record), cell=cell_info
    )


@server.tool(
    name="reset_python_session",
    description="为指定 session 重建子解释器，以获得全新的命名空间。",
)
async def reset_python_session(
    session_id: Annotated[str, Field(description="要重置的 session ID")],
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    internal_id, record = await session_store.reset_session(ctx, session_id)
    record.logger.info("reset_python_session 重置 session=%s", internal_id)
    logger.info("reset_python_session client=%s session=%s", ctx.client_id, internal_id)
    return _tool_response(
        "reset_python_session", session=_session_payload(record), stat="reset"
    )


@server.tool(
    name="send_python_stdin",
    description="向当前会话的子解释器写入 stdin。通常在代码使用 input() 时调用。",
)
async def send_python_stdin(
    session_id: Annotated[str, Field(description="目标 session ID")],
    chunk: Annotated[
        str | None, Field(description="要写入的内容，可为空字符串")
    ] = None,
    append_newline: Annotated[bool, Field(description="写入后自动追加换行符")] = True,
    send_eof: Annotated[bool, Field(description="是否在写入后发送 EOF")] = False,
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    internal_id, record = await session_store.get_session(ctx, session_id)
    runner = record.runner
    slog = record.logger

    data = chunk or ""
    if data or append_newline:
        payload = data + ("\n" if append_newline else "")
        if payload:
            runner.send_stdin(payload)

    if send_eof:
        runner.send_stdin_eof()
    logger.info(
        "send_python_stdin client=%s session=%s len=%d newline=%s eof=%s",
        ctx.client_id,
        internal_id,
        len(chunk or ""),
        append_newline,
        send_eof,
    )
    slog.info(
        "调用 send_python_stdin, 数据长度=%d, 追加换行=%s, EOF=%s\n数据:\n%s",
        len(chunk or ""),
        append_newline,
        send_eof,
        chunk or "",
    )
    return _tool_response(
        "send_python_stdin",
        session=_session_payload(record),
        inp={
            "len": len(chunk or ""),
            "txt": chunk or "",
            "nl": append_newline,
            "eof": send_eof,
        },
    )


@server.tool(
    name="cancel_python_cell",
    description="尝试中断当前执行中的代码，触发 KeyboardInterrupt。",
)
async def cancel_python_cell(
    session_id: Annotated[str, Field(description="目标 session ID")],
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    internal_id, record = await session_store.get_session(ctx, session_id)
    runner = record.runner
    slog = record.logger
    if not runner.is_running:
        logger.info(
            "cancel_python_cell no_running client=%s session=%s",
            ctx.client_id,
            internal_id,
        )
        slog.info("cancel_python_cell 当前无运行任务")
        return _dump_yaml("当前无运行任务")

    success = runner.cancel_current_cell()
    logger.info(
        "cancel_python_cell client=%s session=%s success=%s",
        ctx.client_id,
        internal_id,
        success,
    )
    slog.info("cancel_python_cell 已发送中断信号, 成功=%s", success)
    return _tool_response(
        "cancel_python_cell",
        session=_session_payload(record),
        success=success,
    )


@server.tool(
    name="list_python_cells",
    description="列出当前会话中所有可用的 Python cell，包括已完成和正在运行的。",
)
async def list_python_cells(
    session_id: Annotated[str, Field(description="要查询的 session ID")],
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    internal_id, record = await session_store.get_session(ctx, session_id)
    runner = record.runner
    slog = record.logger
    cells = runner.list_cells()
    logger.info(
        "list_python_cells client=%s session=%s count=%d",
        ctx.client_id,
        internal_id,
        len(cells),
    )
    slog.info("调用 list_python_cells, 总数=%d", len(cells))

    status_map = {"running": "run", "completed": "done"}
    cells_data = [
        {
            "cid": cell["cell_id"],
            "stat": status_map.get(cell["status"], cell["status"]),
            "exc": bool(cell["has_exception"]),
        }
        for cell in cells
    ]
    return _tool_response(
        "list_python_cells",
        session=_session_payload(record),
        tot=len(cells_data),
        cell=cells_data,
    )


__all__ = ["server", "main"]


if __name__ == "__main__":
    main()
