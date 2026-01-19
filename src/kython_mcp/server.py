from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

from .local_log import get_logger
from .sessions import InterpreterSessionStore
from .tools import register_tools
from .utils import bool_env, ensure_fastmcp_env, int_env, str_env

logger = get_logger("kython_mcp.server")

session_store = InterpreterSessionStore()

server = FastMCP(
    name="Kython Process Interpreter",
    instructions="使用 create_session 创建独立 Python session，submit_command 执行代码，snapshot 获取输出快照。",
    debug=bool_env("FASTMCP_DEBUG", "false"),
    log_level=str_env("FASTMCP_LOG_LEVEL", "INFO"),
    host=str_env("FASTMCP_HOST", "127.0.0.1"),
    port=int_env("FASTMCP_PORT", "8000"),
    mount_path=str_env("FASTMCP_MOUNT_PATH", "/"),
    sse_path=str_env("FASTMCP_SSE_PATH", "/sse"),
    message_path=str_env("FASTMCP_MESSAGE_PATH", "/messages/"),
    streamable_http_path=str_env("FASTMCP_STREAMABLE_HTTP_PATH", "/mcp"),
    json_response=bool_env("FASTMCP_JSON_RESPONSE", "true"),
    stateless_http=bool_env("FASTMCP_STATELESS_HTTP", "false"),
    warn_on_duplicate_resources=bool_env("FASTMCP_WARN_ON_DUPLICATE_RESOURCES", "false"),
    warn_on_duplicate_tools=bool_env("FASTMCP_WARN_ON_DUPLICATE_TOOLS", "false"),
    warn_on_duplicate_prompts=bool_env("FASTMCP_WARN_ON_DUPLICATE_PROMPTS", "false"),
)

register_tools(server, session_store)


def main() -> None:
    """以 stdio 传输启动 MCP Server，并在退出时关闭所有会话。"""
    ensure_fastmcp_env()
    try:
        server.run(transport="stdio")
    finally:
        asyncio.run(session_store.close_all())


def main_http() -> None:
    """以 streamable-http 传输启动 MCP Server，并在退出时关闭所有会话。"""
    ensure_fastmcp_env()
    try:
        server.run(transport="streamable-http")
    finally:
        asyncio.run(session_store.close_all())


__all__ = ["server", "main", "main_http"]


if __name__ == "__main__":
    main()
