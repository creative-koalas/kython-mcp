"""
LLM-REPL Agent as an MCP server

- Transport: stdio (default) or SSE
- Tools exposed:
    * solve_task(user_request: str, session_id: str = "llm_agent") -> dict
    * reset_session(session_id: str = "llm_agent") -> str
    * token_stats(session_id: str = "llm_agent") -> dict
    * close_session(session_id: str = "llm_agent") -> str

Run examples:
    python -m src.kython_mcp.llm_repl_mcp_server              # stdio
    python -m src.kython_mcp.llm_repl_mcp_server --transport sse --host 127.0.0.1 --port 8001

Claude Desktop example (mcp.json):
{
  "mcpServers": {
    "llm-repl": {
      "command": "python",
      "args": ["/abs/path/to/llm_repl_mcp_server.py"]
    }
  }
}
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    # Prefer the dedicated fastmcp package if available
    from fastmcp import FastMCP
except Exception:  # pragma: no cover - fallback to official SDK shim if present
    from mcp.server.fastmcp import FastMCP  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kython_mcp.llm_repl_agent import LLMREPLAgent


# ------------------------------
# MCP server wrapper
# ------------------------------


mcp = FastMCP(name="LLMREPLAgentServer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# session -> agent 映射（含最近一次访问时间）


@dataclass
class _SessionEntry:
    agent: LLMREPLAgent
    last_used: float


_AGENTS: Dict[str, _SessionEntry] = {}
_AGENT_LOCK = asyncio.Lock()
_SESSION_TTL_SECONDS = int(os.getenv("LLM_MCP_SESSION_TTL", "1800"))


def _log_tool_event(tool: str, session_id: str | None, status: str, **extra: object) -> None:
    extras = " ".join(f"{k}={v}" for k, v in extra.items())
    logging.info("tool=%s session=%s status=%s %s", tool, session_id or "-", status, extras.strip())


def _pop_expired_sessions_locked(now: float) -> List[Tuple[str, _SessionEntry]]:
    if _SESSION_TTL_SECONDS <= 0:
        return []
    expired: List[Tuple[str, _SessionEntry]] = []
    for session_id, entry in list(_AGENTS.items()):
        if now - entry.last_used >= _SESSION_TTL_SECONDS:
            expired.append((session_id, _AGENTS.pop(session_id)))
    return expired


async def _close_session_entries(entries: List[Tuple[str, _SessionEntry]]) -> None:
    for session_id, entry in entries:
        try:
            await entry.agent.aclose()
            logging.info("session=%s status=EXPIRED", session_id)
        except Exception:  # pragma: no cover - defensive logging only
            logging.exception("session=%s close_failed", session_id)


async def _expire_idle_sessions() -> None:
    entries: List[Tuple[str, _SessionEntry]] = []
    async with _AGENT_LOCK:
        entries = _pop_expired_sessions_locked(time.monotonic())
    await _close_session_entries(entries)


async def _get_or_create_agent(session_id: str) -> Tuple[LLMREPLAgent, bool]:
    now = time.monotonic()
    entries_to_close: List[Tuple[str, _SessionEntry]] = []
    async with _AGENT_LOCK:
        entries_to_close = _pop_expired_sessions_locked(now)
        entry = _AGENTS.get(session_id)
        created = False
        if entry is None:
            entry = _SessionEntry(agent=LLMREPLAgent(session_id=session_id), last_used=now)
            _AGENTS[session_id] = entry
            created = True
            logging.info("session=%s status=CREATED", session_id)
        else:
            entry.last_used = now
    await _close_session_entries(entries_to_close)
    return entry.agent, created


@mcp.tool()
async def solve_task(user_request: str, session_id: str = "llm_agent") -> dict:
    """执行一次“需求→生成代码→执行→反馈→修复”的闭环。

    Args:
        user_request: 任务描述（自然语言）
        session_id: 会话ID（同ID共享同一REPL命名空间）
    Returns:
        执行结果字典，包含 success/result/code/attempts/error/execution_history
    """
    _log_tool_event("solve_task", session_id, "START", snippet=user_request[:80])
    agent, _ = await _get_or_create_agent(session_id)
    try:
        result = await agent.solve_task(user_request)
    except Exception as exc:
        logging.exception("solve_task_failed", extra={"session_id": session_id})
        return {
            "success": False,
            "error": str(exc),
            "tool": "solve_task",
            "session_id": session_id,
        }
    _log_tool_event("solve_task", session_id, "SUCCESS", attempts=result.get("attempts"))
    # 确保可序列化
    return json.loads(json.dumps(result, ensure_ascii=False))


@mcp.tool()
async def reset_session(session_id: str = "llm_agent") -> str:
    """重置指定会话（清空对话历史与REPL命名空间）。"""
    _log_tool_event("reset_session", session_id, "START")
    agent, created = await _get_or_create_agent(session_id)
    if created:
        _log_tool_event("reset_session", session_id, "CREATED")
        return "CREATED"
    await agent.reset()
    _log_tool_event("reset_session", session_id, "OK")
    return "OK"


@mcp.tool()
async def token_stats(session_id: str = "llm_agent") -> dict:
    """查看指定会话的token统计。"""
    _log_tool_event("token_stats", session_id, "START")
    agent, _ = await _get_or_create_agent(session_id)
    stats = agent.get_token_stats()
    _log_tool_event("token_stats", session_id, "SUCCESS", total_tokens=stats.get("total"))
    return stats


@mcp.tool()
async def close_session(session_id: str = "llm_agent") -> str:
    """关闭并移除指定会话。"""
    _log_tool_event("close_session", session_id, "START")
    async with _AGENT_LOCK:
        entry = _AGENTS.pop(session_id, None)
    if entry is None:
        _log_tool_event("close_session", session_id, "NOT_FOUND")
        return "NOT_FOUND"
    await entry.agent.aclose()
    _log_tool_event("close_session", session_id, "CLOSED")
    return "CLOSED"


@mcp.tool()
async def list_sessions() -> list:
    """列出当前已创建的会话ID。"""
    await _expire_idle_sessions()
    async with _AGENT_LOCK:
        sessions = list(_AGENTS.keys())
    _log_tool_event("list_sessions", None, "SUCCESS", count=len(sessions))
    return sessions


# ------------------------------
# Entrypoint
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-REPL MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.transport == "stdio":
        # stdio 适配编辑器/桌面客户端
        mcp.run(transport="stdio")
    else:
        # SSE 适合远程多客户端
        mcp.run(transport="sse", host=args.host, port=args.port)
