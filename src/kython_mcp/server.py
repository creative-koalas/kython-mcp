from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any

import yaml
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from .interpreter_runner import AsyncInterpreterRunner, BusyError
from .local_log import get_logger, get_session_logger

logger = get_logger("kython_mcp.server")

_DEFAULT_SETTINGS = {
    "FASTMCP_DEBUG": "false",
    "FASTMCP_LOG_LEVEL": "INFO",
    "FASTMCP_HOST": "127.0.0.1",
    "FASTMCP_PORT": "8000",
    "FASTMCP_MOUNT_PATH": "/",
    "FASTMCP_SSE_PATH": "/sse",
    "FASTMCP_MESSAGE_PATH": "/messages/",
    "FASTMCP_STREAMABLE_HTTP_PATH": "/mcp",
    "FASTMCP_JSON_RESPONSE": "true",
    "FASTMCP_STATELESS_HTTP": "false",
    "FASTMCP_WARN_ON_DUPLICATE_RESOURCES": "false",
    "FASTMCP_WARN_ON_DUPLICATE_TOOLS": "false",
    "FASTMCP_WARN_ON_DUPLICATE_PROMPTS": "false",
    "FASTMCP_DEPENDENCIES": "[]",
    "FASTMCP_LIFESPAN": "null",
    "FASTMCP_AUTH": "null",
    "FASTMCP_TRANSPORT_SECURITY": "null",
}


def _ensure_fastmcp_env() -> None:
    for key, value in _DEFAULT_SETTINGS.items():
        os.environ.setdefault(key, value)


def str_presenter(dumper, data):
    """强制多行字符串使用 YAML 的 | 样式，更易读"""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)


def _precheck_syntax(code: str) -> None:
    """在主进程侧进行语法预检，确保尽早失败。"""
    if not isinstance(code, str):
        raise ValueError("code 必须为字符串")
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
    return yaml.dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        indent=2,
    )


class StartCellResult(BaseModel):
    session_id: str = Field(description="所属 session ID")
    cell_id: int = Field(description="启动的 cell 序号")
    status: str = Field(description="状态说明，例如 started")


class CreateSessionResult(BaseModel):
    session_id: str = Field(description="新建 session 的唯一 ID")
    description: str | None = Field(description="session 的描述信息")


class PythonSessionInfo(BaseModel):
    session_id: str = Field(description="session ID")
    description: str | None = Field(description="session 的描述信息")
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
    description: str | None = None
    python_executable: str | None = None
    created_at: float = field(default_factory=time.time)


def _session_payload(record: _SessionRecord) -> dict[str, Any]:
    return {
        "id": record.public_id,
        "metadata": {
            "description": record.description,
            "running": record.runner.is_running,
            "python_executable": record.python_executable,
        },
    }


def _format_blocks(blocks: list[tuple[str, str, object]]) -> str:
    sections = []
    for title, tag, payload in blocks:
        body = _dump_yaml(payload).rstrip()
        sections.append(f"{title}\n<{tag}>\n{body}\n</{tag}>")
    return "\n\n".join(sections)


class InterpreterSessionStore:
    """管理单个 MCP 会话下的多个 Python session。"""

    def __init__(self):
        self._sessions: dict[str, _SessionRecord] = {}
        self._ctx_index: dict[int, set[str]] = {}
        self._ctx_public_map: dict[int, dict[str, str]] = {}
        self._ctx_public_counter: dict[int, int] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        ctx: Context,
        description: str | None = None,
        python_executable: str | None = None,
    ) -> tuple[str, _SessionRecord]:
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
            record = _SessionRecord(
                runner=runner,
                logger=get_session_logger(session_name),
                ctx_key=key,
                public_id=public_id,
                description=description,
                python_executable=python_executable,
                created_at=time.time(),
            )
            self._sessions[session_id] = record
            self._ctx_index.setdefault(key, set()).add(session_id)
            self._ctx_public_map.setdefault(key, {})[public_id] = session_id
            logger.info(
                "create_session session_id=%s ctx_key=%s description=%s",
                session_id,
                key,
                description,
            )
            return session_id, record

    async def list_sessions(self, ctx: Context) -> list[tuple[str, _SessionRecord]]:
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
    ) -> tuple[str, _SessionRecord]:
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
    ) -> tuple[str, _SessionRecord]:
        internal_id, record = await self.get_session(ctx, session_id)
        loop = asyncio.get_running_loop()
        runner_label = f"session-{record.public_id}"
        new_runner = AsyncInterpreterRunner(
            name=runner_label,
            loop=loop,
            python_executable=record.python_executable,
        )
        await record.runner.aclose()
        new_record = _SessionRecord(
            runner=new_runner,
            logger=record.logger,
            ctx_key=record.ctx_key,
            public_id=record.public_id,
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

    async def update_description(
        self, ctx: Context, session_id: str, description: str | None
    ) -> tuple[str, _SessionRecord]:
        internal_id, record = await self.get_session(ctx, session_id)
        record.description = description
        logger.info(
            "update_session_description session_id=%s ctx_key=%s",
            internal_id,
            record.ctx_key,
        )
        return internal_id, record


session_store = InterpreterSessionStore()

_ensure_fastmcp_env()

server = FastMCP(
    name="Kython Process Interpreter",
    instructions="使用 create_python_session 创建多个独立 Python session，start_python_cell 异步执行代码，并通过快照/事件轮询结果。",
    debug=os.environ.get("FASTMCP_DEBUG", "false").lower() == "true",
    log_level=os.environ.get("FASTMCP_LOG_LEVEL", "INFO"),
    host=os.environ.get("FASTMCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("FASTMCP_PORT", "8000")),
    mount_path=os.environ.get("FASTMCP_MOUNT_PATH", "/"),
    sse_path=os.environ.get("FASTMCP_SSE_PATH", "/sse"),
    message_path=os.environ.get("FASTMCP_MESSAGE_PATH", "/messages/"),
    streamable_http_path=os.environ.get("FASTMCP_STREAMABLE_HTTP_PATH", "/mcp"),
    json_response=os.environ.get("FASTMCP_JSON_RESPONSE", "false").lower() == "true",
    stateless_http=os.environ.get("FASTMCP_STATELESS_HTTP", "false").lower() == "true",
    warn_on_duplicate_resources=os.environ.get("FASTMCP_WARN_ON_DUPLICATE_RESOURCES", "true").lower() == "true",
    warn_on_duplicate_tools=os.environ.get("FASTMCP_WARN_ON_DUPLICATE_TOOLS", "true").lower() == "true",
    warn_on_duplicate_prompts=os.environ.get("FASTMCP_WARN_ON_DUPLICATE_PROMPTS", "true").lower() == "true",
)


def main() -> None:
    """以 stdio 传输启动 MCP Server，并在退出时关闭所有会话。"""
    _ensure_fastmcp_env()
    try:
        server.run(transport="stdio")
    finally:
        asyncio.run(session_store.close_all())


def main_http() -> None:
    """以 streamable-http 传输启动 MCP Server，并在退出时关闭所有会话。"""
    _ensure_fastmcp_env()
    try:
        server.run(transport="streamable-http")
    finally:
        asyncio.run(session_store.close_all())


@server.tool(
    name="create_python_session",
    description="创建一个新的 Python 解释器 session，并返回 session_id。支持指定 python_executable。",
)
async def create_python_session(
    description: Annotated[
        str | None, Field(description="可选的 session 描述信息，方便区分用途")
    ] = None,
    python_executable: Annotated[
        str | None,
        Field(description="可选 Python 可执行路径，用于指定不同环境"),
    ] = None,
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    internal_id, record = await session_store.create_session(
        ctx, description=description, python_executable=python_executable
    )
    record.logger.info(
        "create_python_session description=%s session_id=%s",
        record.description,
        internal_id,
    )
    logger.info(
        "create_python_session client=%s session_id=%s", ctx.client_id, internal_id
    )
    return f"New python session created, ID:{record.public_id}"


@server.tool(
    name="update_python_session_description",
    description="更新指定 session 的描述信息，方便区分不同会话。",
)
async def update_python_session_description(
    session_id: Annotated[str, Field(description="需要更新的 session ID")],
    description: Annotated[
        str | None, Field(description="新的描述信息，传入 None 表示清除")
    ] = None,
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    internal_id, record = await session_store.update_description(
        ctx, session_id, description
    )
    record.logger.info("update_description description=%s", description)
    logger.info(
        "update_python_session_description client=%s session=%s",
        ctx.client_id,
        internal_id,
    )
    return f"Session ID:{record.public_id} description updated to:{description}"


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
    return _format_blocks([
        ("Current active python sessions:", "sessions", session_entries)
    ])


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
    return f"Session ID:{public_id} closed"


@server.tool(
    name="start_python_cell",
    description="""在指定 session 中启动执行 Python 代码.配置timeout<=0的时候,程序将在后台执行并立即返回ID.配置timeout>=0的时候,程序运行完毕后会直接返回结果;如果超时,程序不会立即停止,而是在后台继续执行.""",
)
async def start_python_cell(
    session_id: Annotated[str, Field(description="要运行的 session ID")],
    code: Annotated[str, Field(description="要执行的 Python 代码")],
    timeout: Annotated[
        float | None,
        Field(description="阻塞等待的秒数，<=0 或 None 表示立即返回并在后台继续运行"),
    ] = None,
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    _precheck_syntax(code)

    internal_id, record = await session_store.get_session(ctx, session_id)
    runner = record.runner
    slog = record.logger
    try:
        logger.info(
            "start_python_cell start client=%s session=%s", ctx.client_id, internal_id
        )
        slog.info("调用 start_python_cell, 代码长度=%d\n代码:\n%s", len(code or ""), code)
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

    wait_timeout = None
    if timeout is not None and timeout > 0:
        wait_timeout = timeout

    if wait_timeout:
        try:
            result = await runner.wait_cell(cid, timeout=wait_timeout)
        except asyncio.TimeoutError:
            logger.info(
                "start_python_cell wait timeout client=%s session=%s cid=%s timeout=%s",
                ctx.client_id,
                internal_id,
                cid,
                wait_timeout,
            )
            slog.info("start_python_cell 超时，cid=%s timeout=%s，后台继续运行", cid, wait_timeout)
            return f"Code in Session ID:{record.public_id} Cell ID:{cid} timeout, but still running in background"
        else:
            stdout_text = result.get("stdout") or ""
            stderr_text = result.get("stderr") or ""
            exception_text = result.get("exception")
            logger.info(
                "start_python_cell completed client=%s session=%s cid=%s",
                ctx.client_id,
                internal_id,
                cid,
            )
            slog.info("start_python_cell 在超时内完成 cid=%s", cid)
            cell_info = {
                "cell_id": cid,
                "status": "completed",
                "stdout": stdout_text,
                "stderr": stderr_text,
                "exception": exception_text,
            }

            return (
                f"Code in Session ID:{record.public_id}Cell ID:{cid} complete:\n"
                + _format_blocks([("Cell Result", "result", cell_info)])
            )

    logger.info("start_python_cell started client=%s session=%s cid=%s", ctx.client_id, internal_id, cid)
    slog.info("start_python_cell 已启动 cid=%s", cid)
    return f"Code in Session ID:{record.public_id}Cell ID:{cid} started"


@server.tool(
    name="get_python_cell_snapshot",
    description="获取指定 cell 或最近/全部已执行 cell 的输出快照（运行中或已完成）。",
)
async def get_python_cell_snapshot(
    session_id: Annotated[str, Field(description="要查询的 session ID")],
    cell_id: Annotated[
        int | None,
        Field(description="可选指定 cell_id；为空则优先返回当前活动 cell，否则返回最近完成者"),
    ] = None,
    n_cells: Annotated[
        int | None,
        Field(
            description=(
                "查询最近的 n 个 cell 快照，仅在未指定 cell_id 时生效；为空则返回全部 cell"
            )
        ),
    ] = None,
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")
    if n_cells is not None and n_cells <= 0:
        raise ValueError("n_cells 需要为正整数")
    internal_id, record = await session_store.get_session(ctx, session_id)
    runner = record.runner
    slog = record.logger
    snapshots: list[dict[str, Any]] = []
    if cell_id is not None:
        snapshots = [runner.get_cell_snapshot(cell_id)]
    else:
        cells = runner.list_cells()
        if not cells:
            raise ValueError("没有可用的 cell")
        sorted_ids = sorted({cell["cell_id"] for cell in cells}, reverse=True)
        if n_cells is None:
            target_ids = sorted_ids
        else:
            target_ids = sorted_ids[:n_cells]
        snapshots = [runner.get_cell_snapshot(cid) for cid in target_ids]
    cell_infos = []
    for snap in reversed(snapshots):
        stdout_text = snap.get("stdout") or ""
        stderr_text = snap.get("stderr") or ""
        result_text = snap.get("result") or ""
        exception_text = snap.get("exception") or None
        cell_infos.append(
            {
                "cell_id": snap["cell_id"],
                "running": bool(snap["running"]),
                "done": bool(snap["done"]),
                "stdout": stdout_text,
                "stderr": stderr_text,
                "result": result_text,
                "exception": exception_text,
            }
        )
    logger.info(
        "get_python_cell_snapshot client=%s session=%s cells=%s",
        ctx.client_id,
        internal_id,
        [info["cell_id"] for info in cell_infos],
    )
    slog.info(
        "调用 get_python_cell_snapshot 汇总 %s 个 cell: %s",
        len(cell_infos),
        [info["cell_id"] for info in cell_infos],
    )
    blocks = [
        (f"Cell {info['cell_id']} snapshot", "result", info) for info in cell_infos
    ]
    return (
        f"Session ID:{record.public_id} 最新 {len(cell_infos)} 个 cell 快照\n"
        + _format_blocks(blocks)
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
    stdin_info = {
        "length": len(chunk or ""),
        "content": chunk or "",
        "append_newline": append_newline,
        "eof_sent": send_eof,
    }

    return f"Send stdin to Session ID:{record.public_id}\n" + _format_blocks(
        [("stdin payload:", "stdin", stdin_info)]
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
        return f"No cells running in Session ID:{record.public_id}"

    success = runner.cancel_current_cell()
    logger.info(
        "cancel_python_cell client=%s session=%s success=%s",
        ctx.client_id,
        internal_id,
        success,
    )
    slog.info("cancel_python_cell 已发送中断信号, 成功=%s", success)
    return f"Try to cancel Session ID:{record.public_id}\n" + _format_blocks(
        [
            (
                "Cancel result:",
                "state",
                {"state": "running", "interrupt_acknowledged": success},
            )
        ]
    )


__all__ = ["server", "main", "main_http"]


if __name__ == "__main__":
    main()
