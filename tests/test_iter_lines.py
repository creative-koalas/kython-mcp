"""Framing tests for _iter_lines.

A worker message is one JSON line, and nothing caps how long that line is, so the reader
has to behave exactly like readline() across chunk boundaries, EOF, and foreign bytes.
"""

from __future__ import annotations

import os
import threading
from typing import IO

from kython_mcp.interpreter_runner import AsyncInterpreterRunner


def _write_all(fd: int, data: bytes) -> None:
    """os.write may short-write on a pipe; keep going until everything is out."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]
    os.close(fd)


def _reader_runner() -> AsyncInterpreterRunner:
    """A runner without a worker process: _iter_lines only needs `_stop`."""
    runner = AsyncInterpreterRunner.__new__(AsyncInterpreterRunner)
    runner.name = "framing"
    runner._stop = False
    return runner


def _lines_for(parts: bytes, *, stop_after: int | None = None) -> list[str]:
    read_fd, write_fd = os.pipe()
    threading.Thread(target=_write_all, args=(write_fd, parts), daemon=True).start()
    stream: IO[str] = os.fdopen(read_fd, "r", encoding="utf-8", closefd=True)
    runner = _reader_runner()
    try:
        lines: list[str] = []
        for line in runner._iter_lines(stream):
            lines.append(line)
            if stop_after is not None and len(lines) >= stop_after:
                runner._stop = True
        return lines
    finally:
        stream.close()


def test_framing_large_line_spanning_many_reads() -> None:
    """A line bigger than one os.read window still arrives whole."""
    payload = "x" * 200_000
    lines = _lines_for(f'{{"type":"stdout","chunk":"{payload}"}}\n'.encode())
    assert len(lines) == 1
    assert payload in lines[0]
    assert lines[0].endswith("\n")


def test_framing_partial_line_at_eof_is_yielded() -> None:
    """A worker killed mid-write still shows what it managed to write."""
    assert _lines_for(b'{"type":"stdout","chunk":"half') == ['{"type":"stdout","chunk":"half']


def test_framing_foreign_bytes_are_yielded_verbatim() -> None:
    """Cell code can write to fd 1 directly; that is output, not an error."""
    assert _lines_for(b"raw output\nsecond\n") == ["raw output\n", "second\n"]


def test_framing_multibyte_utf8_survives_the_read_window() -> None:
    """Multi-byte characters split across reads must not become replacement chars."""
    text = "中文🙂" * 40_000 + "\n"
    lines = _lines_for(text.encode())
    assert len(lines) == 1
    assert lines[0] == text


def test_framing_stop_mid_stream_keeps_what_was_written() -> None:
    """Teardown stops the reader; bytes already handed over must not vanish.

    Lines already buffered are still delivered, and the trailing partial line is flushed,
    so nothing the worker wrote before the stop is dropped.
    """
    assert _lines_for(b"first\nsecond\nthi", stop_after=1) == ["first\n", "second\n", "thi"]
