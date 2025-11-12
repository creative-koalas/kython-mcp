from __future__ import annotations

import asyncio
from typing import Annotated, Dict

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kython_mcp.interpreter_runner import AsyncInterpreterRunner, BusyError


class RunCellResult(BaseModel):
    cell_id: int = Field(description="执行的 cell 序号")
    stdout: str = Field(description="标准输出内容")
    stderr: str = Field(description="标准错误内容")
    result: str = Field(description="displayhook 捕获的 repr")
    exception: str | None = Field(description="异常堆栈，如果成功则为 None")


class StartCellResult(BaseModel):
    cell_id: int = Field(description="启动的 cell 序号")
    status: str = Field(description="状态说明，例如 started")


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


class InterpreterSessionStore:
    """为每个 MCP 会话维持一个子解释器实例。"""

    def __init__(self):
        self._runners: Dict[int, AsyncInterpreterRunner] = {}
        self._lock = asyncio.Lock()

    async def get_runner(self, ctx: Context) -> AsyncInterpreterRunner:
        key = self._session_key(ctx)
        runner = self._runners.get(key)
        if runner is not None:
            return runner

        async with self._lock:
            runner = self._runners.get(key)
            if runner is None:
                loop = asyncio.get_running_loop()
                session_name = ctx.client_id or f"session-{key}"
                runner = AsyncInterpreterRunner(name=session_name, loop=loop)
                self._runners[key] = runner
            return runner

    async def reset_runner(self, ctx: Context) -> None:
        key = self._session_key(ctx)
        runner = self._runners.pop(key, None)
        if runner is not None:
            await runner.aclose()

    async def close_all(self) -> None:
        runners = list(self._runners.values())
        self._runners.clear()
        for runner in runners:
            await runner.aclose()

    @staticmethod
    def _session_key(ctx: Context) -> int:
        return id(ctx.request_context.session)


session_store = InterpreterSessionStore()

server = FastMCP(
    name="Kython Subinterpreter",
    instructions="使用 run_python_cell 在隔离子解释器中执行 Python 代码；每个 MCP 会话共享同一命名空间。",
)


@server.tool(
    name="run_python_cell",
    description="执行任意 Python 代码，并返回 stdout/stderr/displayhook/异常信息。会话内变量会被复用。",
)
async def run_python_cell(
    code: Annotated[str, Field(description="要执行的 Python 代码")],
    timeout: Annotated[float | None, Field(description="可选超时时间，单位秒", ge=0)] = None,
    ctx: Context | None = None,
) -> RunCellResult:
    if ctx is None:
        raise ValueError("Context 注入失败")

    runner = await session_store.get_runner(ctx)

    try:
        result = await runner.run_cell(code, timeout=timeout)
    except BusyError as exc:
        raise RuntimeError(f"当前会话正在执行其他代码: {exc}") from exc

    return RunCellResult(
        cell_id=result["cell_id"],
        stdout=result["stdout"],
        stderr=result["stderr"],
        result=result["result"],
        exception=result["exception"],
    )


@server.tool(
    name="start_python_cell",
    description="非阻塞启动执行 Python 代码，立即返回 cell_id。",
)
async def start_python_cell(
    code: Annotated[str, Field(description="要执行的 Python 代码")],
    ctx: Context | None = None,
) -> StartCellResult:
    if ctx is None:
        raise ValueError("Context 注入失败")
    runner = await session_store.get_runner(ctx)
    try:
        cid = runner.start_cell(code)
    except BusyError as exc:
        raise RuntimeError(f"当前会话正在执行其他代码: {exc}") from exc
    return StartCellResult(cell_id=cid, status="started")


@server.tool(
    name="get_python_cell_snapshot",
    description="获取指定或当前活动 cell 的输出快照（运行中或已完成）。",
)
async def get_python_cell_snapshot(
    cell_id: Annotated[int | None, Field(description="可选指定 cell_id；为空则优先返回当前活动 cell，否则返回最近完成者")]=None,
    ctx: Context | None = None,
) -> CellSnapshot:
    if ctx is None:
        raise ValueError("Context 注入失败")
    runner = await session_store.get_runner(ctx)
    snap = runner.get_cell_snapshot(cell_id)
    return CellSnapshot(
        cell_id=snap["cell_id"],
        stdout=snap["stdout"],
        stderr=snap["stderr"],
        result=snap["result"],
        running=snap["running"],
        done=snap["done"],
        exception=snap.get("exception"),
    )


@server.tool(
    name="wait_python_cell",
    description="等待指定或当前活动 cell 完成；可设置超时，超时则返回当前快照并标注未完成。",
)
async def wait_python_cell(
    cell_id: Annotated[int | None, Field(description="可选指定 cell_id；为空则等待当前活动 cell 或直接返回最近完成者")]=None,
    timeout: Annotated[float | None, Field(description="可选超时时间（秒）", ge=0)] = None,
    ctx: Context | None = None,
) -> CellSnapshot:
    if ctx is None:
        raise ValueError("Context 注入失败")
    runner = await session_store.get_runner(ctx)

    # 若未指定，则选用当前活动或最近完成者
    if cell_id is None:
        snap0 = runner.get_cell_snapshot(None)
        if snap0.get("done"):
            return CellSnapshot(
                cell_id=snap0["cell_id"],
                stdout=snap0["stdout"],
                stderr=snap0["stderr"],
                result=snap0["result"],
                running=False,
                done=True,
                exception=snap0.get("exception"),
            )
        cell_id = snap0["cell_id"]

    try:
        res = await runner.wait_cell(cell_id, timeout=timeout)
        return CellSnapshot(
            cell_id=res["cell_id"],
            stdout=res["stdout"],
            stderr=res["stderr"],
            result=res["result"],
            running=False,
            done=True,
            exception=res.get("exception"),
        )
    except asyncio.TimeoutError:
        snap = runner.get_cell_snapshot(cell_id)
        return CellSnapshot(
            cell_id=snap["cell_id"],
            stdout=snap["stdout"],
            stderr=snap["stderr"],
            result=snap["result"],
            running=True,
            done=False,
            exception=None,
        )


@server.tool(
    name="reset_python_session",
    description="销毁当前会话绑定的子解释器，以便获得全新的命名空间。",
)
async def reset_python_session(ctx: Context | None = None) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    await session_store.reset_runner(ctx)
    return "已重置当前会话的 Python 子解释器。"


@server.tool(
    name="send_python_stdin",
    description="向当前会话的子解释器写入 stdin。通常在代码使用 input() 时调用。",
)
async def send_python_stdin(
    chunk: Annotated[str | None, Field(description="要写入的内容，可为空字符串")] = None,
    append_newline: Annotated[bool, Field(description="写入后自动追加换行符")] = True,
    send_eof: Annotated[bool, Field(description="是否在写入后发送 EOF")] = False,
    ctx: Context | None = None,
) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    runner = await session_store.get_runner(ctx)

    data = chunk or ""
    if data or append_newline:
        payload = data + ("\n" if append_newline else "")
        if payload:
            runner.send_stdin(payload)

    if send_eof:
        runner.send_stdin_eof()

    return "stdin 写入完成。"


@server.tool(
    name="cancel_python_cell",
    description="尝试中断当前执行中的代码，触发 KeyboardInterrupt。",
)
async def cancel_python_cell(ctx: Context | None = None) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    runner = await session_store.get_runner(ctx)
    if not runner.is_running:
        return "当前没有正在执行的代码。"

    success = runner.cancel_current_cell()
    return "已发送中断信号。" if success else "中断失败，请稍后重试。"


@server.tool(
    name="list_python_cells",
    description="列出当前会话中所有可用的 Python cell，包括已完成和正在运行的。",
)
async def list_python_cells(ctx: Context | None = None) -> list[CellInfo]:
    if ctx is None:
        raise ValueError("Context 注入失败")

    runner = await session_store.get_runner(ctx)
    cells = runner.list_cells()

    return [
        CellInfo(
            cell_id=cell["cell_id"],
            status=cell["status"],
            has_exception=cell["has_exception"],
        )
        for cell in cells
    ]


async def _shutdown():
    await session_store.close_all()


def main() -> None:
    try:
        server.run(transport="stdio")
    finally:
        asyncio.run(_shutdown())


__all__ = ["main", "server"]
