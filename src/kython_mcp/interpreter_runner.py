"""
Process-based Python interpreter runner.
Each session spawns its own Python process, enabling custom Python executables.
"""

import asyncio
import json
import os
import select
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Dict, List, Optional

# A reader checks `_stop` after every poll, so this is also the bound on how long
# stop-then-join can take.
_READER_POLL_SECONDS = 0.2


class BusyError(RuntimeError):
    """Raised when the interpreter is busy executing code."""


class AsyncInterpreterRunner:
    def __init__(
        self,
        name: str = "default",
        loop: Optional[asyncio.AbstractEventLoop] = None,
        python_executable: Optional[str] = None,
        worker_path: Optional[str] = None,
    ):
        self.name = name
        self._loop = loop or asyncio.get_event_loop()
        self.python_executable = python_executable
        self.worker_path = worker_path

        self._stdin_lock = threading.Lock()
        self._stop = False
        self._stderr_lines: List[str] = []

        self._running = False
        self._cell_id = 0
        self._cell_done = asyncio.Event()
        self._last_exc: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._current_buf: Dict[str, List[str]] = {
            "stdout": [],
            "stderr": [],
            "result": [],
        }
        self._results: Dict[int, Dict[str, object]] = {}
        self._active_cid: Optional[int] = None
        self._cell_sources: Dict[int, str] = {}
        self._active_source: Optional[str] = None

        # 事件流 (支持多订阅者的广播)
        self.events: asyncio.Queue = asyncio.Queue()
        self._event_subscribers: List[asyncio.Queue] = []

        self._start_process()

    @dataclass(frozen=True)
    class OutputSnapshot:
        stdout: str
        stderr: str
        result: str

    def _start_process(self) -> None:
        python_executable = self.python_executable or "python"
        worker_path = (
            Path(self.worker_path)
            if self.worker_path
            else Path(__file__).with_name("worker.py")
        )

        self._proc = subprocess.Popen(
            [python_executable, "-u", str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._stdout_thread = threading.Thread(
            target=self._stdout_reader_loop, daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_reader_loop, daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    # ---------- Reader loops ----------
    def _iter_lines(self, stream: IO[str] | None) -> Iterator[str]:
        """Yield lines from a reader pipe without ever waiting on an EOF that may not come.

        Two rules this exists for:

        * A cell can leave a detached process holding the pipe's write end, so the pipe
          may never reach EOF. ``select`` bounds every wait, which is what makes
          ``_request_reader_stop()`` effective instead of a flag nobody observes.
        * A concurrent ``close()`` on a buffered wrapper waits for the reader sitting in
          it, so readers must not park inside ``readline()``. Raw ``os.read`` keeps the
          buffered wrapper out of the read path entirely.

        Line framing matches ``readline()``: complete lines keep their terminator, and a
        trailing partial line is yielded as-is — both at EOF (the worker died mid-line)
        and when ``_stop`` arrives, so nothing the worker already wrote is dropped.

        Requires a POSIX ``select`` on the pipe fds: workspace agents run in Linux
        containers, and macOS is fine for development; Windows pipes are not supported.
        """
        if stream is None:
            return
        try:
            fd = stream.fileno()
        except (OSError, ValueError):
            return
        buffer = b""
        while not self._stop:
            try:
                ready, _, _ = select.select([fd], [], [], _READER_POLL_SECONDS)
            except (OSError, ValueError):
                break
            if not ready:
                continue
            try:
                chunk = os.read(fd, 65536)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                yield raw.decode("utf-8", errors="replace") + "\n"
        if buffer:
            yield buffer.decode("utf-8", errors="replace")

    def _publish(self, msg: dict) -> None:
        """Hand one worker message to the loop, tagged with its session."""
        msg["session"] = self.name
        self._loop.call_soon_threadsafe(asyncio.create_task, self._handle_msg(msg))

    def _stdout_reader_loop(self) -> None:
        for line in self._iter_lines(self._proc.stdout):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                msg = {"type": "process_stdout", "chunk": line}
            self._publish(msg)
        if not self._stop:
            self._loop.call_soon_threadsafe(self._worker_exited)

    def _worker_exited(self) -> None:
        # EOF without cell_end supplies no evidence of the execution outcome.
        # Wake waiters so the receipt owner can report unknown instead of waiting forever.
        self._running = False
        self._cell_done.set()

    def _stderr_reader_loop(self) -> None:
        for line in self._iter_lines(self._proc.stderr):
            self._stderr_lines.append(line)
            self._publish({"type": "process_stderr", "chunk": line})

    def _send_control(self, payload: dict) -> None:
        if not self._proc or not self._proc.stdin:
            return
        line = json.dumps(payload, ensure_ascii=False)
        try:
            with self._stdin_lock:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
        except ValueError as exc:
            # A closed stream raises ValueError, not OSError; callers only know BrokenPipe.
            raise BrokenPipeError(str(exc)) from exc

    def _request_reader_stop(self) -> None:
        self._stop = True
        if self._proc.poll() is not None:
            return
        try:
            self._send_control({"type": "close"})
        except Exception:
            pass

    def get_active_source(self) -> Optional[str]:
        return self._active_source

    def get_cell_source(self, cid: int) -> Optional[str]:
        return self._cell_sources.get(cid)


    async def _handle_msg(self, msg: dict):
        """Apply one worker message to the cell state, then broadcast it.

        Messages arrive either as parsed worker JSON or as a raw ``process_stdout``
        fallback for a line that was not JSON (teardown cut it in half, or something
        outside the worker's stdout wrapper wrote to fd 1); neither is an error.
        """
        msg_type = msg.get("type")

        if msg_type == "stdout":
            self._current_buf["stdout"].append(msg.get("chunk", ""))
        elif msg_type == "stderr":
            self._current_buf["stderr"].append(msg.get("chunk", ""))
        elif msg_type == "result":
            self._current_buf["result"].append(msg.get("repr", ""))
        elif msg_type == "cell_start":
            self._current_buf = {"stdout": [], "stderr": [], "result": []}
            self._last_exc = None
            self._active_cid = msg.get("cell_id")
        elif msg_type == "cell_end":
            self._last_exc = msg.get("exception")
            cid = msg.get("cell_id")
            out = self.get_current_output()
            duration_seconds = None
            if msg.get("timing_ms") is not None:
                duration_seconds = float(msg.get("timing_ms")) / 1000.0
            self._results[cid] = {
                "cell_id": cid,
                "stdout": out.stdout,
                "stderr": out.stderr,
                "result": out.result,
                "exception": self._last_exc,
                "duration_seconds": duration_seconds,
            }
            self._cell_done.set()
            self._running = False
            self._worker_thread = None
            self._active_cid = None
            self._active_source = None
        elif msg_type == "cell_rejected":
            pass

        await self.events.put(msg)
        for subscriber in self._event_subscribers:
            try:
                subscriber.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    # ---------- Public API ----------
    @property
    def is_running(self) -> bool:
        return self._running

    def get_current_output(self) -> "AsyncInterpreterRunner.OutputSnapshot":
        return AsyncInterpreterRunner.OutputSnapshot(
            stdout="".join(self._current_buf["stdout"]),
            stderr="".join(self._current_buf["stderr"]),
            result="".join(self._current_buf["result"]),
        )

    def start_cell(self, source: str) -> int:
        if self._running:
            raise BusyError(f"Session {self.name} is already running cell {self._cell_id}")

        self._running = True
        self._cell_done = asyncio.Event()
        self._cell_id += 1
        cid = self._cell_id

        self._cell_sources[cid] = source
        self._active_source = source

        self._send_control({"type": "run_cell", "cell_id": cid, "source": source})
        return cid

    async def wait_cell(self, cid: int, timeout: Optional[float] = None) -> Dict[str, object]:
        if cid in self._results:
            return self._results[cid]
        if cid != self._active_cid:
            await asyncio.wait_for(self._cell_done.wait(), timeout=timeout)
            if cid in self._results:
                return self._results[cid]
            raise ValueError("Cell ID not found or not the active cell")
        await asyncio.wait_for(self._cell_done.wait(), timeout=timeout)
        return self._results.get(
            cid,
            {
                "cell_id": cid,
                "stdout": "",
                "stderr": "",
                "result": "",
                "exception": "Result unavailable",
            },
        )

    def get_cell_snapshot(self, cid: Optional[int] = None) -> Dict[str, object]:
        if cid is not None and cid in self._results:
            r = self._results[cid]
            return {
                "cell_id": r["cell_id"],
                "stdout": r["stdout"],
                "stderr": r["stderr"],
                "result": r["result"],
                "exception": r.get("exception"),
                "running": False,
                "done": True,
                "source": self._cell_sources.get(cid),
            }

        target_cid = cid
        if target_cid is None:
            if self._active_cid is not None:
                target_cid = self._active_cid
            elif self._results:
                target_cid = max(self._results)
            else:
                raise ValueError("No cells available")

        if target_cid == self._active_cid:
            out = self.get_current_output()
            return {
                "cell_id": target_cid,
                "stdout": out.stdout,
                "stderr": out.stderr,
                "result": out.result,
                "exception": None,
                "running": True,
                "done": False,
                "source": self._active_source,
            }

        if target_cid in self._results:
            r = self._results[target_cid]
            return {
                "cell_id": r["cell_id"],
                "stdout": r["stdout"],
                "stderr": r["stderr"],
                "result": r["result"],
                "exception": r.get("exception"),
                "running": False,
                "done": True,
                "source": self._cell_sources.get(target_cid),
            }

        raise ValueError("Cell ID not found")

    def list_cells(self) -> List[Dict[str, object]]:
        cells = []
        for cid in sorted(self._results.keys()):
            r = self._results[cid]
            cells.append(
                {
                    "cell_id": cid,
                    "status": "completed",
                    "has_exception": r.get("exception") is not None,
                }
            )

        if self._active_cid is not None and self._active_cid not in self._results:
            cells.append(
                {
                    "cell_id": self._active_cid,
                    "status": "running",
                    "has_exception": False,
                }
            )

        return cells

    async def run_cell(self, source: str, timeout: Optional[float] = None) -> Dict[str, object]:
        if self._running:
            raise BusyError(f"Session {self.name} is already running cell {self._cell_id}")

        self._running = True
        self._cell_done = asyncio.Event()
        self._cell_id += 1
        cid = self._cell_id

        self._cell_sources[cid] = source
        self._active_source = source

        self._send_control({"type": "run_cell", "cell_id": cid, "source": source})

        try:
            await asyncio.wait_for(self._cell_done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise

        out = self.get_current_output()
        return {
            "cell_id": cid,
            "stdout": out.stdout,
            "stderr": out.stderr,
            "result": out.result,
            "exception": self._last_exc,
        }

    def send_stdin(self, chunk: str):
        if not isinstance(chunk, str):
            raise TypeError("stdin payload must be a string")
        self._send_control({"type": "stdin", "chunk": chunk})

    def send_stdin_eof(self):
        self._send_control({"type": "stdin_eof"})

    def cancel_current_cell(self) -> bool:
        if not self._running:
            return False
        self._send_control({"type": "cancel"})
        return True

    def subscribe_events(self) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=100)
        self._event_subscribers.append(queue)
        return queue

    def unsubscribe_events(self, queue: asyncio.Queue):
        if queue in self._event_subscribers:
            self._event_subscribers.remove(queue)

    async def aclose(self):
        """Teardown blocks by nature, so it runs off the loop.

        The caller bounds the *wait*, not the work: ``asyncio.wait_for`` can stop waiting,
        but the teardown thread finishes on its own — a pipe close or a process wait has
        nothing cancellable in it.
        """
        await asyncio.to_thread(self.close)

    def close(self):
        """Stop the readers, the worker, then its pipes — in that order.

        The joins are deliberately unbounded: a reader only ever waits
        ``_READER_POLL_SECONDS`` before re-checking ``_stop``, and closing a pipe while a
        reader could still be inside it is how the workspace agent used to wedge.
        """
        self._request_reader_stop()
        self._stdout_thread.join()
        self._stderr_thread.join()
        self._terminate_process()
        if self._proc is None:
            return
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def _terminate_process(self) -> None:
        """Stop the worker process; ``close()`` closes the pipes once this returns."""
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=1)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
