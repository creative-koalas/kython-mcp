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

def run_cell(cell_id, source):
    global _RUNNING
    if _RUNNING:
        out_queue.put({"type": "cell_rejected", "cell_id": cell_id, "reason": "busy"})
        return

    _RUNNING = True
    out_queue.put({"type": "cell_start", "cell_id": cell_id})
    t0 = time.perf_counter()
    exc = None

    try:
        # 尝试用 "single" 模式编译(单语句/表达式)
        try:
            code = compile(source, "<cell>", "single")
        except SyntaxError:
            # 多语句情况:使用 "exec" 模式
            code = compile(source, "<cell>", "exec")

        exec(code, _GLOBALS, _GLOBALS)
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
        self._current_buf: Dict[str, List[str]] = {
            "stdout": [],
            "stderr": [],
            "result": []
        }

    def _bootstrap(self):
        """注入引导代码到子解释器"""
        self.interp.prepare_main(out_queue=self.out_queue)
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
        elif msg_type == "cell_end":
            self._last_exc = msg.get("exception")
            self._cell_done.set()
            self._running = False
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
        self.interp.exec(f"run_cell({cid}, {source!r})")

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

if __name__ == "__main__":
    # ---------- 异步版本演示 ----------
    async def demo_async():
        print("=== AsyncInterpreterRunner 演示 ===\n")

        session = AsyncInterpreterRunner("demo-session")

        # 启动事件监听任务 (使用独立订阅队列)
        event_queue = session.subscribe_events()

        async def event_printer():
            while True:
                try:
                    ev = await event_queue.get()
                    print(f"[事件] {json.dumps(ev, ensure_ascii=False)}")
                except:
                    break

        asyncio.create_task(event_printer())

        # 1) 简单执行
        print("1. 执行简单代码:")
        res = await session.run_cell("x = 42\nprint(f'x = {x}')\nx * 2", timeout=2.0)
        print(f"结果: {json.dumps(res, ensure_ascii=False)}\n")

        # 2) 持久化命名空间
        print("2. 验证持久化命名空间:")
        res = await session.run_cell("print(f'x 仍然是: {x}')", timeout=2.0)
        print(f"结果: {json.dumps(res, ensure_ascii=False)}\n")

        # 3) 异常处理
        print("3. 异常捕获:")
        res = await session.run_cell("1 / 0", timeout=2.0)
        print(f"结果: {json.dumps(res, ensure_ascii=False, indent=2)}\n")

        # 4) 单次执行保护
        print("4. 单次执行保护 (BusyError):")
        task = asyncio.create_task(
            session.run_cell("import time; time.sleep(0.5); print('完成')", timeout=2.0)
        )
        await asyncio.sleep(0.1)

        try:
            await session.run_cell("print('这应该失败')", timeout=1.0)
        except BusyError as e:
            print(f"捕获到 BusyError: {e}")

        await task
        print()

        # 5) 实时输出快照
        print("5. 实时输出快照:")
        task = asyncio.create_task(
            session.run_cell(
                "import time\nfor i in range(3):\n    print(f'Step {i}')\n    time.sleep(0.1)",
                timeout=5.0
            )
        )

        for _ in range(5):
            await asyncio.sleep(0.15)
            snapshot = session.get_current_output()
            print(f"快照: stdout={repr(snapshot['stdout'])}")

        await task
        print()

        await session.aclose()
        print("=== 演示完成 ===")

    asyncio.run(demo_async())
