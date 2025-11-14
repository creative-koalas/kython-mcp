import logging
import os
import re
from logging.handlers import RotatingFileHandler


def get_logger(name: str = "kython_mcp", log_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_dir = (
        log_dir
        or os.environ.get("KYTHON_MCP_LOG_DIR")
        or os.path.join(os.getcwd(), "logs")
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "kython_mcp.log")

    handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _sanitize(name: str) -> str:
    name = name.strip() or "unknown"
    # 仅保留字母数字、连字符和下划线
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:128]


def get_session_logger(session_name: str, log_dir: str | None = None) -> logging.Logger:
    safe = _sanitize(session_name)
    logger = logging.getLogger(f"kython_mcp.session.{safe}")
    if logger.handlers:
        return logger

    log_dir = (
        log_dir
        or os.environ.get("KYTHON_MCP_LOG_DIR")
        or os.path.join(os.getcwd(), "logs")
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"session-{safe}.log")

    handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
