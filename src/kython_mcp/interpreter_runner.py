"""
Python 3.14+ Interpreters 异步执行器
移植自 async_subinterp_repl.py 的特性

展示功能:
1. 异步代码执行 (asyncio 集成)
2. 事件流系统 (stdout/stderr/result/cell_start/cell_end)
3. displayhook 捕获最后表达式结果
4. 单次执行保护 (BusyError)
5. 实时输出快照 (get_current_output)
6. 持久化命名空间 (跨 cell 保持变量)
7. 超时控制和异常处理
"""

import asyncio
import ctypes
import json
import threading
import time
import traceback
from typing import Dict, List, Optional

from concurrent import interpreters


# ---------- 子解释器引导代码 ----------
# 注意: out_queue 会通过 prepare_main() 注入
_BOOTSTRAP = r"""
import sys, time, traceback

# 流式输出写入器
class _QueueWriter:
    def __init__(self, kind, out_q):
        self.kind = kind
        self.out_q = out_q

    def write(self, s):
        if not s:
            return 0
        self.out_q.put({"type": self.kind, "chunk": s})
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False

sys.stdout = _QueueWriter("stdout", out_queue)
sys.stderr = _QueueWriter("stderr", out_queue)

class _QueueReader:
    def __init__(self, in_q):
        self.in_q = in_q
        self._buffer = ""
        self._eof = False

    def _fill_buffer(self):
        while not self._buffer and not self._eof:
            msg = self.in_q.get()
            kind = msg.get("type")
            if kind == "stdin":
                self._buffer += msg.get("chunk", "")
            elif kind == "stdin_eof":
                self._eof = True
                break

    def read(self, size=-1):
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

    def readline(self, size=-1):
        if size == 0:
            return ""
        line = []
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
            line.append(chunk[:take])
            self._buffer = chunk[take:]
            if (newline_pos != -1 and take == newline_pos + 1) or (remaining >= 0 and take == remaining):
                if remaining >= 0:
                    remaining -= take
                break
            if remaining >= 0:
                remaining -= take
                if remaining <= 0:
                    break
        return "".join(line)

    def readable(self):
        return True

    def close(self):
        self._eof = True

sys.stdin = _QueueReader(in_queue)

# displayhook: 捕获最后表达式的值
def _displayhook(value):
    if value is None:
        return
    out_queue.put({"type": "result", "repr": repr(value)})

sys.displayhook = _displayhook

# 持久化命名空间
_GLOBALS = {"__name__": "__main__", "__package__": None}

# 单次执行保护
_RUNNING = False

# 取消标志
_CANCEL_REQUESTED = False

def _check_cancel():
    global _CANCEL_REQUESTED
    # 非阻塞地检查 in_queue 中的取消消息
    try:
        while True:
            msg = in_queue.get(block=False, timeout=0)
            if msg.get("type") == "cancel":
                _CANCEL_REQUESTED = True
                break
    except:
        pass  # 队列为空或超时

    if _CANCEL_REQUESTED:
        raise KeyboardInterrupt("执行已被取消")

def run_cell(cell_id, source):
    global _RUNNING, _CANCEL_REQUESTED
    if _RUNNING:
        out_queue.put({"type": "cell_rejected", "cell_id": cell_id, "reason": "busy"})
        return

    _RUNNING = True
    _CANCEL_REQUESTED = False  # 重置取消标志
    out_queue.put({"type": "cell_start", "cell_id": cell_id})
    t0 = time.perf_counter()
    exc = None

    try:
        # 执行前检查取消
        _check_cancel()

        # 尝试用 "single" 模式编译(单语句/表达式)
        try:
            code = compile(source, "<cell>", "single")
        except SyntaxError:
            # 多语句情况:使用 "exec" 模式
            code = compile(source, "<cell>", "exec")

        # 设置 trace 函数以便在执行过程中检查取消
        def _trace_cancel(frame, event, arg):
            _check_cancel()
            return _trace_cancel

        sys.settrace(_trace_cancel)
        try:
            exec(code, _GLOBALS, _GLOBALS)
        finally:
            sys.settrace(None)

    except BaseException as e:
        exc = "".join(traceback.format_exception(type(e), e, e.__traceback__))

    dt = (time.perf_counter() - t0) * 1000.0
    out_queue.put({
        "type": "cell_end",
        "cell_id": cell_id,
        "exception": exc,
        "timing_ms": round(dt, 3)
    })
    _RUNNING = False
    _CANCEL_REQUESTED = False
"""


class BusyError(RuntimeError):
    """当解释器正在执行时尝试运行新代码"""
    pass


class AsyncInterpreterRunner:
    def __init__(self, name: str = "default", loop: Optional[asyncio.AbstractEventLoop] = None):
        self.name = name
        self._loop = loop or asyncio.get_event_loop()

        # 创建子解释器和通信队列
        self.interp = interpreters.create()
        self.out_queue = interpreters.create_queue()  # 子解释器 -> 主解释器
        self.in_queue = interpreters.create_queue()   # 主解释器 -> 子解释器

        # 引导子解释器
        self._bootstrap()

        # 事件流 (支持多订阅者的广播)
        self.events: asyncio.Queue = asyncio.Queue()
        self._event_subscribers: List[asyncio.Queue] = []

        # 读取线程
        self._stop = False
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        # 执行状态
        self._running = False
        self._cell_id = 0
        self._cell_done = asyncio.Event()
        self._last_exc: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._current_buf: Dict[str, List[str]] = {
            "stdout": [],
            "stderr": [],
            "result": []
        }
        # 结果存档与当前活动 cell
        self._results: Dict[int, Dict[str, object]] = {}
        self._active_cid: Optional[int] = None

    def _bootstrap(self):
        """注入引导代码到子解释器"""
        self.interp.prepare_main(out_queue=self.out_queue, in_queue=self.in_queue)
        self.interp.exec(_BOOTSTRAP)

    # ---------- 私有方法 ----------

    def _reader_loop(self):
        """阻塞读取线程，将队列消息转发到 asyncio"""
        while not self._stop:
            try:
                msg = self.out_queue.get(timeout=0.1)  # 字典消息
                msg["session"] = self.name
                # 调度到事件循环
                self._loop.call_soon_threadsafe(
                    asyncio.create_task,
                    self._handle_msg(msg)
                )
            except:
                continue

    async def _handle_msg(self, msg: dict):
        """处理来自子解释器的消息"""
        msg_type = msg.get("type")

        # 维护当前缓冲区
        if msg_type == "stdout":
            self._current_buf["stdout"].append(msg.get("chunk", ""))
        elif msg_type == "stderr":
            self._current_buf["stderr"].append(msg.get("chunk", ""))
        elif msg_type == "result":
            self._current_buf["result"].append(msg.get("repr", ""))
        elif msg_type == "cell_start":
            # 重置缓冲区
            self._current_buf = {"stdout": [], "stderr": [], "result": []}
            self._last_exc = None
            self._active_cid = msg.get("cell_id")
        elif msg_type == "cell_end":
            self._last_exc = msg.get("exception")
            # 存档本次执行的最终输出
            cid = msg.get("cell_id")
            out = self.get_current_output()
            self._results[cid] = {
                "cell_id": cid,
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "result": out.get("result", ""),
                "exception": self._last_exc,
            }
            self._cell_done.set()
            self._running = False
            self._worker_thread = None
            self._active_cid = None
        elif msg_type == "cell_rejected":
            pass  # 子解释器拒绝执行

        # 广播事件到所有订阅者
        await self.events.put(msg)
        for subscriber in self._event_subscribers:
            try:
                subscriber.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # 订阅者队列满,丢弃事件

    # ---------- 公共 API ----------

    @property
    def is_running(self) -> bool:
        """检查是否正在执行"""
        return self._running

    def get_current_output(self) -> Dict[str, str]:
        """获取当前 cell 的输出快照"""
        return {
            "stdout": "".join(self._current_buf["stdout"]),
            "stderr": "".join(self._current_buf["stderr"]),
            "result": "".join(self._current_buf["result"]),
        }

    # ---------- 非阻塞执行与结果查询 ----------

    def start_cell(self, source: str) -> int:
        """
        非阻塞地启动代码单元执行，立即返回 cell_id。

        异常:
            BusyError: 正在执行其他代码
        """
        if self._running:
            raise BusyError(f"会话 {self.name} 正在执行 cell {self._cell_id}")

        self._running = True
        self._cell_done = asyncio.Event()
        self._cell_id += 1
        cid = self._cell_id

        # 启动执行线程
        self._worker_thread = threading.Thread(
            target=self._exec_cell,
            args=(cid, source),
            daemon=True,
            name=f"{self.name}-cell-{cid}"
        )
        self._worker_thread.start()
        return cid

    async def wait_cell(self, cid: int, timeout: Optional[float] = None) -> Dict[str, object]:
        """
        等待指定 cell 完成，返回最终结果。

        - 若已完成，立即返回存档结果。
        - 若 cid 不是当前活动且不存在存档，则报错。
        - 若等待超时，抛出 asyncio.TimeoutError。
        """
        if cid in self._results:
            return self._results[cid]
        if cid != self._active_cid:
            raise ValueError("指定的 cell_id 不存在或不是当前活动 cell")
        await asyncio.wait_for(self._cell_done.wait(), timeout=timeout)
        return self._results.get(cid, {
            "cell_id": cid,
            "stdout": "",
            "stderr": "",
            "result": "",
            "exception": "执行结果不可用",
        })

    def get_cell_snapshot(self, cid: Optional[int] = None) -> Dict[str, object]:
        """
        获取指定 cell 的输出快照。

        - 若未传 cid，优先返回当前活动 cell；若无活动 cell 则返回最新完成的一个。
        - 返回结构包含: cell_id, stdout, stderr, result, exception(若已结束), running(bool), done(bool)
        """
        # 已完成直接返回并标注 done
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
            }

        # 选择目标 cell
        target_cid = cid
        if target_cid is None:
            if self._active_cid is not None:
                target_cid = self._active_cid
            elif self._results:
                # 返回最近一个完成的（最大 cid）
                target_cid = max(self._results)
            else:
                raise ValueError("没有可用的 cell")

        # 若是当前活动 cell，返回实时快照
        if target_cid == self._active_cid:
            out = self.get_current_output()
            return {
                "cell_id": target_cid,
                "stdout": out["stdout"],
                "stderr": out["stderr"],
                "result": out["result"],
                "exception": None,
                "running": True,
                "done": False,
            }

        # 否则尝试返回已完成存档
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
            }

        raise ValueError("指定的 cell_id 不存在")

    def list_cells(self) -> List[Dict[str, object]]:
        """
        列出当前会话中所有可用的 cell。

        返回:
            List[Dict]: 包含所有cell信息的列表，每个元素包含:
                - cell_id: int
                - status: str ("running" 或 "completed")
                - has_exception: bool (是否有异常)
        """
        cells = []

        # 添加所有已完成的cell（按cell_id排序）
        for cid in sorted(self._results.keys()):
            r = self._results[cid]
            cells.append({
                "cell_id": cid,
                "status": "completed",
                "has_exception": r.get("exception") is not None,
            })

        # 添加当前正在运行的cell（如果存在且不在已完成列表中）
        if self._active_cid is not None and self._active_cid not in self._results:
            cells.append({
                "cell_id": self._active_cid,
                "status": "running",
                "has_exception": False,
            })

        return cells

    async def run_cell(
        self,
        source: str,
        timeout: Optional[float] = None
    ) -> Dict[str, object]:
        """
        异步执行代码单元

        返回:
            {"cell_id", "stdout", "stderr", "result", "exception", "timing_ms"}

        异常:
            BusyError: 正在执行其他代码
            asyncio.TimeoutError: 超时
        """
        if self._running:
            raise BusyError(f"会话 {self.name} 正在执行 cell {self._cell_id}")

        self._running = True
        self._cell_done = asyncio.Event()
        self._cell_id += 1
        cid = self._cell_id

        # 触发执行
        self._worker_thread = threading.Thread(
            target=self._exec_cell,
            args=(cid, source),
            daemon=True,
            name=f"{self.name}-cell-{cid}"
        )
        self._worker_thread.start()

        # 等待完成或超时
        try:
            await asyncio.wait_for(self._cell_done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise

        # 返回结果快照
        out = self.get_current_output()
        return {
            "cell_id": cid,
            "stdout": out["stdout"],
            "stderr": out["stderr"],
            "result": out["result"],
            "exception": self._last_exc,
        }

    def _exec_cell(self, cid: int, source: str):
        try:
            self.interp.exec(f"run_cell({cid}, {source!r})")
        except Exception as exc:
            err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            msg = {
                "type": "cell_end",
                "cell_id": cid,
                "exception": err,
                "timing_ms": 0.0,
                "session": self.name,
                "synthetic": True,
            }
            self._loop.call_soon_threadsafe(
                asyncio.create_task,
                self._handle_msg(msg)
            )

    def send_stdin(self, chunk: str):
        """写入 stdin 队列"""
        if not isinstance(chunk, str):
            raise TypeError("stdin 数据必须为字符串")
        self.in_queue.put({"type": "stdin", "chunk": chunk})

    def send_stdin_eof(self):
        """发送 EOF 标记"""
        self.in_queue.put({"type": "stdin_eof"})

    def cancel_current_cell(self) -> bool:
        """
        通过消息通道向子解释器发送取消信号。

        首先尝试通过 in_queue 发送取消消息，让子解释器主动响应。
        如果子解释器支持，这是最可靠的方式。

        作为后备方案，同时向执行线程注入 KeyboardInterrupt。
        """
        if not self._running or self._worker_thread is None:
            return False

        # 方法1: 通过消息队列发送取消信号（推荐方式）
        try:
            self.in_queue.put({"type": "cancel"})
        except Exception:
            pass  # 队列可能已满或不可用

        # 方法2: 向执行线程注入异常（后备方案）
        ident = self._worker_thread.ident
        if ident is None:
            return False

        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(ident),
            ctypes.py_object(KeyboardInterrupt)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), None)
            raise RuntimeError("取消执行失败")

        return True  # 已发送取消信号（至少一种方式）

    def subscribe_events(self) -> asyncio.Queue:
        """
        创建新的事件订阅者队列

        返回:
            asyncio.Queue: 接收所有事件副本的队列
        """
        queue = asyncio.Queue(maxsize=100)
        self._event_subscribers.append(queue)
        return queue

    def unsubscribe_events(self, queue: asyncio.Queue):
        """取消事件订阅"""
        if queue in self._event_subscribers:
            self._event_subscribers.remove(queue)

    async def aclose(self):
        """关闭会话"""
        self._stop = True
        try:
            self.interp.close()
        except:
            pass

    def close(self):
        """同步关闭"""
        self._stop = True
        try:
            self.interp.close()
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
