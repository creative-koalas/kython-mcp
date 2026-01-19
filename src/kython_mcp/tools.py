from __future__ import annotations

import asyncio
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from .interpreter_runner import BusyError
from .sessions import InterpreterSessionStore, SessionRecord, session_payload
from .utils import format_blocks, precheck_syntax
from .local_log import get_logger

logger = get_logger("kython_mcp.tools")


def register_tools(server: FastMCP, session_store: InterpreterSessionStore) -> None:
    @server.tool(
        name="create_session",
        description="创建一个新的 Python 会话（对齐 kmux create_session）。",
    )
    async def create_session(
        label: Annotated[
            str | None, Field(description="可选会话名称，类似 kmux 的 label")
        ] = None,
        description: Annotated[
            str | None, Field(description="可选会话描述")
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
            ctx, label=label, description=description, python_executable=python_executable
        )
        record.logger.info(
            "create_session label=%s description=%s session_id=%s",
            record.label,
            record.description,
            internal_id,
        )
        logger.info("create_session client=%s session_id=%s", ctx.client_id, internal_id)
        return f"New session created, ID:{record.public_id}"

    @server.tool(
        name="list_sessions",
        description="列出当前 MCP 会话下已创建的所有 session（对齐 kmux list_sessions）。",
    )
    async def list_sessions(ctx: Context | None = None) -> str:
        if ctx is None:
            raise ValueError("Context 注入失败")

        sessions = await session_store.list_sessions(ctx)
        logger.info("list_sessions client=%s count=%d", ctx.client_id, len(sessions))
        session_entries = [session_payload(record) for _, record in sessions]
        return format_blocks([("Current sessions:", "sessions", session_entries)])

    @server.tool(
        name="update_session_label",
        description="更新指定 session 的 label（对齐 kmux update_session_label）。",
    )
    async def update_session_label(
        session_id: Annotated[str, Field(description="需要更新的 session ID")],
        label: Annotated[
            str | None, Field(description="新的 label，传入 None 表示清除")
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context 注入失败")

        internal_id, record = await session_store.update_label(ctx, session_id, label)
        record.logger.info("update_label label=%s", label)
        logger.info("update_session_label client=%s session=%s", ctx.client_id, internal_id)
        return f"Session ID:{record.public_id} label updated to:{label}"

    @server.tool(
        name="update_session_description",
        description="更新指定 session 的描述信息（对齐 kmux update_session_description）。",
    )
    async def update_session_description(
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
            "update_session_description client=%s session=%s",
            ctx.client_id,
            internal_id,
        )
        return f"Session ID:{record.public_id} description updated to:{description}"

    @server.tool(
        name="delete_session",
        description="关闭并移除指定 session（对齐 kmux delete_session）。",
    )
    async def delete_session(
        session_id: Annotated[str, Field(description="要关闭的 session ID")],
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context 注入失败")

        internal_id, public_id = await session_store.close_session(ctx, session_id)
        logger.info("delete_session client=%s session=%s", ctx.client_id, internal_id)
        return f"Session ID:{public_id} closed"

    @server.tool(
        name="submit_command",
        description="""在指定 session 中执行 Python 代码（对齐 kmux submit_command）。
        timeout_seconds <=0 或 None：立即返回 cell_id 并在后台执行。
        timeout_seconds >0：等待完成并返回结果；超时则后台继续运行。""",
    )
    async def submit_command(
        session_id: Annotated[str, Field(description="要运行的 session ID")],
        command: Annotated[str, Field(description="要执行的 Python 代码")],
        timeout_seconds: Annotated[
            float | None,
            Field(description="阻塞等待的秒数，<=0 或 None 表示立即返回并在后台继续运行"),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context 注入失败")

        precheck_syntax(command)

        internal_id, record = await session_store.get_session(ctx, session_id)
        runner = record.runner
        slog = record.logger
        try:
            logger.info(
                "submit_command start client=%s session=%s",
                ctx.client_id,
                internal_id,
            )
            slog.info(
                "调用 submit_command, 代码长度=%d\n代码:\n%s",
                len(command or ""),
                command,
            )
            cid = runner.start_cell(command)
        except BusyError as exc:
            logger.warning(
                "submit_command busy client=%s session=%s",
                ctx.client_id,
                internal_id,
            )
            slog.warning("submit_command 忙，拒绝执行")
            raise RuntimeError(f"当前会话正在执行其他代码: {exc}") from exc
        except Exception:
            logger.exception(
                "submit_command error client=%s session=%s",
                ctx.client_id,
                internal_id,
            )
            slog.exception("submit_command 异常")
            raise

        wait_timeout = None
        if timeout_seconds is not None and timeout_seconds > 0:
            wait_timeout = timeout_seconds

        if wait_timeout:
            try:
                result = await runner.wait_cell(cid, timeout=wait_timeout)
            except asyncio.TimeoutError:
                logger.info(
                    "submit_command wait timeout client=%s session=%s cid=%s timeout=%s",
                    ctx.client_id,
                    internal_id,
                    cid,
                    wait_timeout,
                )
                slog.info(
                    "submit_command 超时，cid=%s timeout=%s，后台继续运行",
                    cid,
                    wait_timeout,
                )
                return (
                    f"Command in Session ID:{record.public_id} Cell ID:{cid} timeout, "
                    "but still running in background"
                )
            else:
                stdout_text = result.get("stdout") or ""
                stderr_text = result.get("stderr") or ""
                exception_text = result.get("exception")
                logger.info(
                    "submit_command completed client=%s session=%s cid=%s",
                    ctx.client_id,
                    internal_id,
                    cid,
                )
                slog.info("submit_command 在超时内完成 cid=%s", cid)
                cell_info = {
                    "cell_id": cid,
                    "status": "completed",
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "exception": exception_text,
                }

                return (
                    f"Command in Session ID:{record.public_id} Cell ID:{cid} complete:\n"
                    + format_blocks([("Cell Result", "result", cell_info)])
                )

        logger.info(
            "submit_command started client=%s session=%s cid=%s",
            ctx.client_id,
            internal_id,
            cid,
        )
        slog.info("submit_command 已启动 cid=%s", cid)
        return f"Command in Session ID:{record.public_id} Cell ID:{cid} started"

    @server.tool(
        name="snapshot",
        description="获取指定 session 的输出快照（对齐 kmux snapshot）。",
    )
    async def snapshot(
        session_id: Annotated[str, Field(description="要查询的 session ID")],
        include_all: Annotated[
            bool, Field(description="是否返回所有 cell；false 则返回最新一个")
        ] = False,
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context 注入失败")

        internal_id, record = await session_store.get_session(ctx, session_id)
        runner = record.runner
        slog = record.logger

        cells = runner.list_cells()
        if not cells:
            raise ValueError("没有可用的 cell")

        sorted_ids = sorted({cell["cell_id"] for cell in cells})
        if include_all:
            target_ids = sorted_ids
        else:
            target_ids = [sorted_ids[-1]]

        snapshots = [runner.get_cell_snapshot(cid) for cid in target_ids]
        cell_infos = []
        for snap in snapshots:
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
            "snapshot client=%s session=%s cells=%s",
            ctx.client_id,
            internal_id,
            [info["cell_id"] for info in cell_infos],
        )
        slog.info("调用 snapshot 汇总 %s 个 cell", len(cell_infos))
        blocks = [
            (f"Cell {info['cell_id']} snapshot", "result", info) for info in cell_infos
        ]
        return f"Session ID:{record.public_id} snapshot\n" + format_blocks(blocks)

    @server.tool(
        name="send_keys",
        description="向 Python stdin 写入内容（对齐 kmux send_keys）。",
    )
    async def send_keys(
        session_id: Annotated[str, Field(description="目标 session ID")],
        keys: Annotated[str, Field(description="要写入的内容，支持转义字符")],
        append_newline: Annotated[
            bool, Field(description="写入后自动追加换行符")
        ] = False,
        send_eof: Annotated[bool, Field(description="是否在写入后发送 EOF")] = False,
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context 注入失败")

        internal_id, record = await session_store.get_session(ctx, session_id)
        runner = record.runner
        slog = record.logger

        payload = keys + ("\n" if append_newline else "")
        if payload:
            runner.send_stdin(payload)
        if send_eof:
            runner.send_stdin_eof()

        logger.info(
            "send_keys client=%s session=%s len=%d newline=%s eof=%s",
            ctx.client_id,
            internal_id,
            len(keys or ""),
            append_newline,
            send_eof,
        )
        slog.info(
            "调用 send_keys, 数据长度=%d, 追加换行=%s, EOF=%s\n数据:\n%s",
            len(keys or ""),
            append_newline,
            send_eof,
            keys,
        )
        stdin_info = {
            "length": len(keys or ""),
            "content": keys,
            "append_newline": append_newline,
            "eof_sent": send_eof,
        }
        return f"Send keys to Session ID:{record.public_id}\n" + format_blocks(
            [("stdin payload:", "stdin", stdin_info)]
        )
