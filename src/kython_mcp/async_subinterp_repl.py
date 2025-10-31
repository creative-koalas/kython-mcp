import asyncio
import json
import threading
import time
from typing import Dict, List, Optional

try:
    import concurrent.interpreters as ip  # Python 3.12+ (provisional)
except Exception as e:
    raise SystemExit("This requires the provisional `interpreters` module (PEP 554).") from e


# ---------- Code bootstrapped inside each subinterpreter ----------
_BOOTSTRAP = r"""
import sys, builtins, time, traceback
import interpreters as ip

cur = ip.get_current()
out = cur.get_channel(OUT_ID)   # subinterp -> host (events)
inn = cur.get_channel(IN_ID)    # host -> subinterp (commands, input replies)

# Streaming writers
class _ChanWriter:
    def __init__(self, kind): self.kind = kind
    def write(self, s):
        if not s: return 0
        out.send({"type": self.kind, "chunk": s})
        return len(s)
    def flush(self): pass
    def isatty(self): return False

sys.stdout = _ChanWriter("stdout")
sys.stderr = _ChanWriter("stderr")

# input() handshake
def _input(prompt=""):
    out.send({"type":"input_request", "prompt": prompt})
    v = inn.recv()  # host must send a string
    return v
builtins.input = _input

# last-expression hook: stream repr of final expr
def _displayhook(value):
    if value is None: return
    out.send({"type":"result", "repr": repr(value)})
sys.displayhook = _displayhook

# Persistent namespace
_G = {"__name__":"__main__", "__package__": None}

# Single-flight guard inside the subinterp (belt & suspenders)
_RUNNING = False

def run_cell(cell_id, source):
    global _RUNNING
    if _RUNNING:
        out.send({"type":"cell_rejected", "cell_id": cell_id, "reason": "busy"})
        return
    _RUNNING = True
    out.send({"type":"cell_start", "cell_id": cell_id})
    t0 = time.perf_counter()
    exc = None
    try:
        code = compile(source, "<cell>", "single")
        exec(code, _G, _G)
    except BaseException as e:
        exc = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    dt = (time.perf_counter() - t0) * 1000.0
    out.send({"type":"cell_end", "cell_id": cell_id, "exception": exc, "timing_ms": round(dt, 3)})
    _RUNNING = False
"""


class BusyError(RuntimeError):
    pass


class AsyncSubinterpSession:
    """
    Async facade over a single subinterpreter.
    Features:
      - run_cell(source, timeout): returns when finished or raises asyncio.TimeoutError
      - live event stream via self.events (asyncio.Queue of dicts)
      - single-flight execution enforced (BusyError)
      - get_current_output(): snapshot of collected stdout/stderr/result so far for active cell
      - reply_input(value): satisfy a pending input() in the subinterp
    """
    def __init__(self, name: str, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.name = name
        self._loop = loop or asyncio.get_event_loop()

        # Subinterpreter + channels
        self._sid = ip.create()
        self._out_send, self._out_recv = ip.create_channel()  # sub -> host
        self._in_send,  self._in_recv  = ip.create_channel()  # host -> sub

        # Bootstrap
        code = _BOOTSTRAP.replace("OUT_ID", str(self._out_send.id)).replace("IN_ID", str(self._in_recv.id))
        ip.run_string(self._sid, code)

        # Reader thread -> puts messages into asyncio.Queue
        self.events: asyncio.Queue = asyncio.Queue()
        self._stop = False
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        # Execution state
        self._running = False
        self._cell_id = 0
        self._cell_done = asyncio.Event()
        self._last_exc: Optional[str] = None
        self._current_buf: Dict[str, List[str]] = {"stdout": [], "stderr": [], "result": []}

    # ---------- Private plumbing ----------

    def _reader_loop(self):
        """Blocking recv() in a thread; forward to asyncio loop."""
        while not self._stop:
            msg = self._out_recv.recv()  # dict from subinterp
            msg["session"] = self.name
            # schedule handling in the event loop
            self._loop.call_soon_threadsafe(asyncio.create_task, self._handle_msg(msg))

    async def _handle_msg(self, msg: dict):
        # Maintain current buffers for "peek while running"
        t = msg.get("type")
        if t == "stdout":
            self._current_buf["stdout"].append(msg.get("chunk", ""))
        elif t == "stderr":
            self._current_buf["stderr"].append(msg.get("chunk", ""))
        elif t == "result":
            self._current_buf["result"].append(msg.get("repr", ""))
        elif t == "cell_start":
            # Reset buffers at start of a cell
            self._current_buf = {"stdout": [], "stderr": [], "result": []}
            self._last_exc = None
        elif t == "cell_end":
            self._last_exc = msg.get("exception")
            # signal completion
            self._cell_done.set()
            self._running = False
        elif t == "cell_rejected":
            # If subinterp rejected due to busy (shouldn’t happen if host enforces)
            pass

        # Always publish to the public stream queue (for UIs / logs)
        await self.events.put(msg)

    # ---------- Public API ----------

    @property
    def is_running(self) -> bool:
        return self._running

    def get_current_output(self) -> Dict[str, str]:
        """Return concatenated stdout/stderr/results collected so far for the active cell."""
        return {
            "stdout": "".join(self._current_buf["stdout"]),
            "stderr": "".join(self._current_buf["stderr"]),
            "result": "".join(self._current_buf["result"]),
        }

    async def run_cell(self, source: str, timeout: Optional[float] = None) -> Dict[str, object]:
        """
        Execute a cell. If finishes within `timeout`, return a summary dict:
           {"cell_id", "stdout", "stderr", "result", "exception", "timing_ms"}
        Else raise asyncio.TimeoutError. Single-flight enforced.
        """
        if self._running:
            raise BusyError(f"Session {self.name} is already running cell {self._cell_id}")
        self._running = True
        self._cell_done = asyncio.Event()
        self._cell_id += 1
        cid = self._cell_id

        # Kick off execution (fire-and-forget)
        ip.run_string(self._sid, f"run_cell({cid}, {source!r})")

        # Wait for done or timeout
        try:
            await asyncio.wait_for(self._cell_done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Still running; keep it running, caller can cancel/ignore or later await again
            # (Hard cancellation inside a subinterp is non-trivial; you can add a cooperative check.)
            raise

        # Snapshot and return
        out = self.get_current_output()
        return {
            "cell_id": cid,
            "stdout": out["stdout"],
            "stderr": out["stderr"],
            "result": out["result"],
            "exception": self._last_exc,
            # timing_ms is carried on the final event; optionally parse it back:
            # for simplicity we don't parse—caller can read the final cell_end event from self.events
        }

    async def reply_input(self, text: str):
        """
        Respond to the latest input_request.
        Note: This is immediate; pair it with reading events to know *when* to reply.
        """
        self._in_send.send(text)

    async def aclose(self):
        self._stop = True
        # No official destroy API in provisional module; you can add cleanup if available.


# ---------- Demo / example usage ----------

async def demo():
    session = AsyncSubinterpSession("S")
    # Start a task to print events as they stream
    async def printer():
        while True:
            ev = await session.events.get()
            print("\x1e" + json.dumps(ev, ensure_ascii=False))  # framed JSON
    asyncio.create_task(printer())

    # 1) Quick cell (finish within timeout)
    res = await session.run_cell("x = 1\nx", timeout=2.0)
    print("RES1:", json.dumps(res, ensure_ascii=False))

    # 2) A cell that asks for input, while we "peek" mid-flight
    exec_task = asyncio.create_task(session.run_cell("name = input('Name? ')\nprint('Hi', name)\n'OK'", timeout=5.0))
    # Wait until we see the input_request
    while True:
        snap = session.get_current_output()
        # (Optional) inspect snap["stdout"] while pending input
        await asyncio.sleep(0.05)
        # Feed input and break (real code would wait for an input_request event)
        await session.reply_input("Trent")
        break
    res2 = await exec_task
    print("RES2:", json.dumps(res2, ensure_ascii=False))

    # 3) Enforce single-flight
    t1 = asyncio.create_task(session.run_cell("import time; time.sleep(0.6); 41+1", timeout=2))
    await asyncio.sleep(0.05)
    try:
        await session.run_cell("print('should fail')", timeout=1)
    except BusyError as e:
        print("BusyError caught:", e)
    print("RES3:", json.dumps(await t1, ensure_ascii=False))

    await session.aclose()


if __name__ == "__main__":
    asyncio.run(demo())
