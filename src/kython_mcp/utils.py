from __future__ import annotations


def precheck_syntax(code: str) -> None:
    """Validate code syntax early to fail fast."""
    if not isinstance(code, str):
        raise TypeError("Code must be a string")
    try:
        compile(code, "<workspace-python>", "single")
        return
    except SyntaxError:
        pass
    try:
        compile(code, "<workspace-python>", "exec")
    except SyntaxError as e:
        text = e.text or ""
        where = f"line {e.lineno}, column {e.offset}" if e.lineno else "unknown location"
        msg = f"Syntax error: {e.msg} ({where})\n{text}"
        raise ValueError(msg) from e
