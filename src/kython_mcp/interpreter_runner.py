"""
Process-based Python interpreter runner.
Each session spawns its own Python process, enabling custom Python executables.
"""

import asyncio
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


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
    def _stdout_reader_loop(self) -> None:
        while True:
            if self._stop:
                break
            line = self._proc.stdout.readline() if self._proc.stdout else ""
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                msg = {"type": "process_stdout", "chunk": line}
            msg["session"] = self.name
            self._loop.call_soon_threadsafe(
                asyncio.create_task, self._handle_msg(msg)
            )

    def _stderr_reader_loop(self) -> None:
        while True:
            if self._stop:
                break
            line = self._proc.stderr.readline() if self._proc.stderr else ""
            if not line:
                break
            self._stderr_lines.append(line)
            msg = {"type": "process_stderr", "chunk": line, "session": self.name}
            self._loop.call_soon_threadsafe(
                asyncio.create_task, self._handle_msg(msg)
            )

    def _send_control(self, payload: dict) -> None:
        if not self._proc or not self._proc.stdin:
            return
        line = json.dumps(payload, ensure_ascii=False)
        with self._stdin_lock:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()

    def _request_reader_stop(self) -> None:
        self._stop = True
        try:
            self._send_control({"type": "close"})
        except Exception:
            pass

    def get_active_source(self) -> Optional[str]:
        return self._active_source

    def get_cell_source(self, cid: int) -> Optional[str]:
        return self._cell_sources.get(cid)


    async def _handle_msg(self, msg: dict):
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
        self._request_reader_stop()
        if self._stdout_thread.is_alive():
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._stdout_thread.join), timeout=1.0
                )
            except (asyncio.TimeoutError, RuntimeError):
                pass
        if self._stderr_thread.is_alive():
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._stderr_thread.join), timeout=1.0
                )
            except (asyncio.TimeoutError, RuntimeError):
                pass
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass

    def close(self):
        self._request_reader_stop()
        if self._stdout_thread.is_alive():
            self._stdout_thread.join(timeout=1.0)
        if self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)
        if self._proc:
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
