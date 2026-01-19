from __future__ import annotations

import asyncio
import re
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from .interpreter_runner import BusyError
from .sessions import InterpreterSessionStore, session_payload
from .utils import format_blocks, precheck_syntax
from .local_log import get_logger

logger = get_logger("kython_mcp.tools")


def register_tools(server: FastMCP, session_store: InterpreterSessionStore) -> None:
    """Register all MCP tools on the server."""

    @server.tool(
        name="create_session",
        description="Create a new Python session (kmux-style).",
    )
    async def create_session(
        label: Annotated[
            str | None, Field(description="Optional session label (kmux style).")
        ] = None,
        description: Annotated[
            str | None, Field(description="Optional session description.")
        ] = None,
        python_executable: Annotated[
            str | None,
            Field(description="Optional Python executable path for custom envs."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context injection failed")

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
        description="List active sessions for current MCP connection.",
    )
    async def list_sessions(ctx: Context | None = None) -> str:
        if ctx is None:
            raise ValueError("Context injection failed")

        sessions = await session_store.list_sessions(ctx)
        logger.info("list_sessions client=%s count=%d", ctx.client_id, len(sessions))
        session_entries = [session_payload(record) for _, record in sessions]
        return format_blocks([("Current sessions:", "sessions", session_entries)])

    @server.tool(
        name="update_session_label",
        description="Update session label (kmux-style).",
    )
    async def update_session_label(
        session_id: Annotated[str, Field(description="Target session ID.")],
        label: Annotated[
            str | None, Field(description="New label. Use null to clear.")
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context injection failed")

        internal_id, record = await session_store.update_label(ctx, session_id, label)
        record.logger.info("update_label label=%s", label)
        logger.info("update_session_label client=%s session=%s", ctx.client_id, internal_id)
        return f"Session ID:{record.public_id} label updated to:{label}"

    @server.tool(
        name="update_session_description",
        description="Update session description (kmux-style).",
    )
    async def update_session_description(
        session_id: Annotated[str, Field(description="Target session ID.")],
        description: Annotated[
            str | None, Field(description="New description. Use null to clear.")
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context injection failed")

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
        description="Close and remove a session (kmux-style).",
    )
    async def delete_session(
        session_id: Annotated[str, Field(description="Target session ID.")],
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context injection failed")

        internal_id, public_id = await session_store.close_session(ctx, session_id)
        logger.info("delete_session client=%s session=%s", ctx.client_id, internal_id)
        return f"Session ID:{public_id} closed"

    @server.tool(
        name="submit_cell",
        description=(
            "Execute Python code in a session (kmux-style). "
            "timeout_seconds<=0 returns immediately; >0 waits until completion."
        ),
    )
    async def submit_cell(
        session_id: Annotated[str, Field(description="Target session ID.")],
        command: Annotated[str, Field(description="Python code to execute.")],
        timeout_seconds: Annotated[
            float | None,
            Field(
                description=(
                    "Seconds to wait. <=0 or null returns immediately while the cell keeps running."
                )
            ),
        ] = 5.0,
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context injection failed")

        precheck_syntax(command)

        internal_id, record = await session_store.get_session(ctx, session_id)
        runner = record.runner
        slog = record.logger
        try:
            logger.info(
                "submit_cell start client=%s session=%s",
                ctx.client_id,
                internal_id,
            )
            slog.info(
                "submit_cell code_len=%d\ncode:\n%s",
                len(command or ""),
                command,
            )
            cid = runner.start_cell(command)
        except BusyError as exc:
            logger.warning(
                "submit_cell busy client=%s session=%s",
                ctx.client_id,
                internal_id,
            )
            slog.warning("submit_cell busy, rejected")
            raise RuntimeError(f"Session is busy: {exc}") from exc
        except Exception:
            logger.exception(
                "submit_cell error client=%s session=%s",
                ctx.client_id,
                internal_id,
            )
            slog.exception("submit_cell exception")
            raise

        if timeout_seconds is None:
            timeout_seconds = 5.0

        wait_timeout = None
        if timeout_seconds is not None and timeout_seconds > 0:
            wait_timeout = timeout_seconds

        if wait_timeout:
            try:
                result = await runner.wait_cell(cid, timeout=wait_timeout)
            except asyncio.TimeoutError:
                logger.info(
                    "submit_cell wait timeout client=%s session=%s cid=%s timeout=%s",
                    ctx.client_id,
                    internal_id,
                    cid,
                    wait_timeout,
                )
                slog.info(
                    "submit_cell timeout cid=%s timeout=%s, continue in background",
                    cid,
                    wait_timeout,
                )
                current = runner.get_current_output()
                output = "".join(
                    [
                        current.stdout or "",
                        current.result or "",
                        current.stderr or "",
                    ]
                )
                return (
                    "Command is still running after "
                    f"{wait_timeout:.2f} seconds;\n"
                    "this could mean the command is doing blocking operations "
                    "(e.g., disk reading, downloading)\n"
                    "or is awaiting input (e.g., password, confirmation).\n\n"
                    "Currently executing command buffer:\n"
                    f"<command>\n{command}\n</command>\n\n"
                    "Current command output:\n\n"
                    f"<command-output>\n{output}\n</command-output>\n\n"
                    "It is recommended to use `snapshot` on this session later to see command status,\n"
                    "and use `send_keys` to interact with the session if necessary.\n"
                    "You cannot execute another command on this session until the current command finishes or get terminated."
                )
            else:
                stdout_text = result.get("stdout") or ""
                stderr_text = result.get("stderr") or ""
                exception_text = result.get("exception") or ""
                result_text = result.get("result") or ""
                duration_seconds = result.get("duration_seconds")
                logger.info(
                    "submit_cell completed client=%s session=%s cid=%s",
                    ctx.client_id,
                    internal_id,
                    cid,
                )
                slog.info("submit_cell completed cid=%s", cid)
                output_parts = []
                if stdout_text:
                    output_parts.append(stdout_text)
                if result_text:
                    output_parts.append(result_text)
                if stderr_text:
                    output_parts.append(stderr_text)
                if exception_text:
                    output_parts.append(exception_text)
                output = "".join(output_parts)
                duration_display = "unknown"
                if isinstance(duration_seconds, (float, int)):
                    duration_display = f"{duration_seconds:.2f}"
                return (
                    f"Command finished in {duration_display} seconds.\n\n"
                    "Executed command buffer:\n"
                    f"<command>\n{command}\n</command>\n\n"
                    "Command output:\n\n"
                    f"<command-output>\n{output}\n</command-output>"
                )

        logger.info(
            "submit_cell started client=%s session=%s cid=%s",
            ctx.client_id,
            internal_id,
            cid,
        )
        slog.info("submit_cell started cid=%s", cid)
        return f"Command in Session ID:{record.public_id} Cell ID:{cid} started"

    @server.tool(
        name="snapshot",
        description="Fetch output snapshots for a session (kmux-style).",
    )
    async def snapshot(
        session_id: Annotated[str, Field(description="Target session ID.")],
        include_all: Annotated[
            bool, Field(description="Return all cells instead of latest.")
        ] = False,
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context injection failed")

        internal_id, record = await session_store.get_session(ctx, session_id)
        runner = record.runner
        slog = record.logger

        cells = runner.list_cells()
        if not cells:
            raise ValueError("No cells available")

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
        slog.info("snapshot collected %s cells", len(cell_infos))
        blocks = [
            (f"Cell {info['cell_id']} snapshot", "result", info) for info in cell_infos
        ]
        snapshot_text = format_blocks(blocks)
        header = (
            "Terminal snapshot (including all outputs):"
            if include_all
            else "Terminal snapshot (starting from last command input):"
        )
        return f"{header}\n<snapshot>\n{snapshot_text}\n</snapshot>"

    @server.tool(
        name="send_keys",
        description="Send data to Python stdin (kmux-style).",
    )
    async def send_keys(
        session_id: Annotated[str, Field(description="Target session ID.")],
        keys: Annotated[str, Field(description="Payload to send (supports escapes).")],
        append_newline: Annotated[
            bool, Field(description="Append newline after payload.")
        ] = False,
        send_eof: Annotated[bool, Field(description="Send EOF after payload.")] = False,
        ctx: Context | None = None,
    ) -> str:
        if ctx is None:
            raise ValueError("Context injection failed")

        internal_id, record = await session_store.get_session(ctx, session_id)
        runner = record.runner
        slog = record.logger

        if not runner.is_running:
            raise RuntimeError("No running cell in this session")

        def _unescape(s: str) -> str:
            def repl(match: re.Match[str]) -> str:
                token = match.group(1)
                if token == "n":
                    return "\n"
                if token == "r":
                    return "\r"
                if token == "t":
                    return "\t"
                if token == "\\":
                    return "\\"
                if token.startswith("x"):
                    return bytes.fromhex(token[1:]).decode("latin-1")
                if token.startswith("u"):
                    return chr(int(token[1:], 16))
                if token.startswith("U"):
                    return chr(int(token[1:], 16))
                return match.group(0)

            return re.sub(r"\\(\\|n|r|t|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8})", repl, s)

        decoded_keys = _unescape(keys or "")

        payload = decoded_keys + ("\n" if append_newline else "")
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
            "send_keys len=%d newline=%s eof=%s\npayload:\n%s",
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
