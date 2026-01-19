import json
import queue
import sys
import threading
import time
import traceback


def _send(msg: dict) -> None:
    sys.__stdout__.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


class _QueueWriter:
    def __init__(self, kind: str):
        self.kind = kind

    def write(self, s: str) -> int:
        if not s:
            return 0
        _send({"type": self.kind, "chunk": s})
        return len(s)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


class _QueueReader:
    def __init__(self, in_q: queue.Queue):
        self._queue = in_q
        self._buffer = ""
        self._eof = False

    def _fill_buffer(self) -> None:
        while not self._buffer and not self._eof:
            msg = self._queue.get()
            kind = msg.get("type")
            if kind == "stdin":
                self._buffer += msg.get("chunk", "")
            elif kind == "stdin_eof":
                self._eof = True
                break

    def read(self, size: int = -1) -> str:
        if size == 0:
            return ""
        if size < 0:
            chunks = []
            while True:
                self._fill_buffer()
                if not self._buffer:
                    break
                chunks.append(self._buffer)
                self._buffer = ""
            return "".join(chunks)

        self._fill_buffer()
        if not self._buffer:
            return ""
        chunk = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return chunk

    def readline(self, size: int = -1) -> str:
        if size == 0:
            return ""
        line_parts = []
        remaining = size
        while True:
            self._fill_buffer()
            if not self._buffer:
                break
            chunk = self._buffer
            newline_pos = chunk.find("\n")
            take = len(chunk) if newline_pos == -1 else newline_pos + 1
            if remaining >= 0:
                take = min(take, remaining)
            line_parts.append(chunk[:take])
            self._buffer = chunk[take:]
            if (newline_pos != -1 and take == newline_pos + 1) or (
                remaining >= 0 and take == remaining
            ):
                if remaining >= 0:
                    remaining -= take
                break
            if remaining >= 0:
                remaining -= take
                if remaining <= 0:
                    break
        return "".join(line_parts)

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self._eof = True


class _CancelFlag:
    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    def check(self) -> None:
        if self._event.is_set():
            raise KeyboardInterrupt("执行已被取消")


class _ExecutionState:
    def __init__(self) -> None:
        self.running = False
        self.cell_id = 0
        self.thread: threading.Thread | None = None


cancel_flag = _CancelFlag()
stdin_queue: queue.Queue = queue.Queue()
state = _ExecutionState()
_globals = {"__name__": "__main__", "__package__": None}

sys.stdout = _QueueWriter("stdout")
sys.stderr = _QueueWriter("stderr")

def _displayhook(value):
    if value is None:
        return
    _send({"type": "result", "repr": repr(value)})


sys.displayhook = _displayhook
sys.stdin = _QueueReader(stdin_queue)


def _run_cell(cell_id: int, source: str) -> None:
    cancel_flag.clear()
    _send({"type": "cell_start", "cell_id": cell_id})
    t0 = time.perf_counter()
    exc_text = None

    try:
        cancel_flag.check()
        try:
            code = compile(source, "<cell>", "single")
        except SyntaxError:
            code = compile(source, "<cell>", "exec")

        def _trace(frame, event, arg):
            cancel_flag.check()
            return _trace

        sys.settrace(_trace)
        try:
            exec(code, _globals, _globals)
        finally:
            sys.settrace(None)
    except BaseException as exc:
        exc_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    dt_ms = (time.perf_counter() - t0) * 1000.0
    _send(
        {
            "type": "cell_end",
            "cell_id": cell_id,
            "exception": exc_text,
            "timing_ms": round(dt_ms, 3),
        }
    )
    state.running = False
    state.thread = None
    cancel_flag.clear()


def _start_cell(cell_id: int, source: str) -> None:
    if state.running:
        _send({"type": "cell_rejected", "cell_id": cell_id, "reason": "busy"})
        return
    state.running = True
    state.thread = threading.Thread(target=_run_cell, args=(cell_id, source), daemon=True)
    state.thread.start()


def main() -> None:
    control_in = sys.__stdin__
    for raw in control_in:
        if not raw.strip():
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        kind = msg.get("type")
        if kind == "run_cell":
            _start_cell(int(msg.get("cell_id")), msg.get("source", ""))
        elif kind == "stdin":
            stdin_queue.put({"type": "stdin", "chunk": msg.get("chunk", "")})
        elif kind == "stdin_eof":
            stdin_queue.put({"type": "stdin_eof"})
        elif kind == "cancel":
            cancel_flag.request()
        elif kind == "close":
            break


if __name__ == "__main__":
    main()
