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
import json
import logging
from pathlib import Path
from typing import Dict
import sys

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

# session -> agent 映射
_AGENTS: Dict[str, LLMREPLAgent] = {}


def _get_or_create_agent(session_id: str) -> LLMREPLAgent:
    agent = _AGENTS.get(session_id)
    if agent is None:
        agent = LLMREPLAgent(session_id=session_id)
        _AGENTS[session_id] = agent
    return agent


@mcp.tool()
async def solve_task(user_request: str, session_id: str = "llm_agent") -> dict:
    """执行一次“需求→生成代码→执行→反馈→修复”的闭环。

    Args:
        user_request: 任务描述（自然语言）
        session_id: 会话ID（同ID共享同一REPL命名空间）
    Returns:
        执行结果字典，包含 success/result/code/attempts/error/execution_history
    """
    agent = _get_or_create_agent(session_id)
    result = await agent.solve_task(user_request)
    # 确保可序列化
    return json.loads(json.dumps(result, ensure_ascii=False))


@mcp.tool()
async def reset_session(session_id: str = "llm_agent") -> str:
    """重置指定会话（清空对话历史与REPL命名空间）。"""
    agent = _AGENTS.get(session_id)
    if agent is None:
        _AGENTS[session_id] = LLMREPLAgent(session_id=session_id)
        return "CREATED"
    await agent.reset()
    return "OK"


@mcp.tool()
async def token_stats(session_id: str = "llm_agent") -> dict:
    """查看指定会话的token统计。"""
    agent = _get_or_create_agent(session_id)
    return agent.get_token_stats()


@mcp.tool()
async def close_session(session_id: str = "llm_agent") -> str:
    """关闭并移除指定会话。"""
    agent = _AGENTS.pop(session_id, None)
    if agent is None:
        return "NOT_FOUND"
    await agent.aclose()
    return "CLOSED"


@mcp.tool()
def list_sessions() -> list:
    """列出当前已创建的会话ID。"""
    return list(_AGENTS.keys())


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
