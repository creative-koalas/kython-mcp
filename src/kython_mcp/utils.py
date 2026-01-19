from __future__ import annotations

import os
from typing import Any

import yaml

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


def ensure_fastmcp_env() -> None:
    for key, value in _DEFAULT_SETTINGS.items():
        os.environ.setdefault(key, value)


def format_blocks(blocks: list[tuple[str, str, object]]) -> str:
    sections = []
    for title, tag, payload in blocks:
        body = dump_yaml(payload).rstrip()
        sections.append(f"{title}\n<{tag}>\n{body}\n</{tag}>")
    return "\n\n".join(sections)


def dump_yaml(data: object) -> str:
    return yaml.dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        indent=2,
    )


def str_presenter(dumper, data):
    """强制多行字符串使用 YAML 的 | 样式，更易读"""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)


def precheck_syntax(code: str) -> None:
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


def bool_env(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() == "true"


def int_env(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


def str_env(name: str, default: str) -> str:
    return os.environ.get(name, default)
