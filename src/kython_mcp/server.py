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
    name="reset_python_session",
    description="销毁当前会话绑定的子解释器，以便获得全新的命名空间。",
)
async def reset_python_session(ctx: Context | None = None) -> str:
    if ctx is None:
        raise ValueError("Context 注入失败")

    await session_store.reset_runner(ctx)
    return "已重置当前会话的 Python 子解释器。"


async def _shutdown():
    await session_store.close_all()


def main() -> None:
    try:
        server.run(transport="stdio")
    finally:
        asyncio.run(_shutdown())


__all__ = ["main", "server"]
