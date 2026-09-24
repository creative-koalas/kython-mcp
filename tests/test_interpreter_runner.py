from __future__ import annotations

import asyncio
import os
import signal
import sys
import time

import pytest

from kython_mcp.interpreter_runner import AsyncInterpreterRunner


def _runner() -> AsyncInterpreterRunner:
    return AsyncInterpreterRunner(
        name="teardown-test",
        loop=asyncio.get_running_loop(),
        python_executable=sys.executable,
    )


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


@pytest.mark.asyncio
async def test_teardown_returns_when_a_cell_leaves_a_detached_process() -> None:
    """A cell can leave a process holding the worker's stdout pipe open.

    Teardown used to wait on that pipe forever inside aclose() — in production the
    workspace stayed Running with zero restarts while /health stopped answering — so the
    assertions here are about the event loop surviving, not about the pipe reaching EOF.
    """
    runner = _runner()
    child_pid = 0
    try:
        cell_id = runner.start_cell(
            "import subprocess\n"
            "child = subprocess.Popen(['sleep', '30'])\n"
            "print(child.pid)"
        )
        cell = await asyncio.wait_for(runner.wait_cell(cell_id, timeout=20), timeout=30)
        assert cell["exception"] is None
        child_pid = int((cell["stdout"] or "").strip())

        started = time.monotonic()
        await asyncio.wait_for(runner.aclose(), timeout=10)
        elapsed = time.monotonic() - started
        # Unfixed code only returned once the child exited and closed the pipe (30s here);
        # a detached process must never be able to hold teardown hostage.
        assert elapsed < 5, f"teardown waited on a detached process for {elapsed:.1f}s"

        # The loop still schedules tasks afterwards: a deadlock would have taken it with it.
        assert await asyncio.wait_for(asyncio.sleep(0), timeout=1) is None
        # The invariant the whole fix rests on: both readers are really gone.
        assert not runner._stdout_thread.is_alive()
        assert not runner._stderr_thread.is_alive()
        # And teardown owns the pipes it created.
        assert runner._proc is not None and runner._proc.stdout is not None
        assert runner._proc.stdout.closed is True
        # Deliberate: teardown closes pipes, it does not police processes a cell started.
        assert os.kill(child_pid, 0) is None
    finally:
        _kill(child_pid)
        runner.close()


@pytest.mark.asyncio
async def test_aclose_does_not_block_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contract is "teardown runs off the loop", not "teardown happens to be fast".

    A synchronous aclose() is just as quick in the happy path, so only a deliberately
    slow teardown can tell the two apart — and a blocked loop is the production outage.
    """
    runner = _runner()
    real_terminate = runner._terminate_process

    def slow_terminate() -> None:
        time.sleep(0.5)
        real_terminate()

    monkeypatch.setattr(runner, "_terminate_process", slow_terminate)
    try:
        teardown = asyncio.create_task(runner.aclose())
        await asyncio.sleep(0)
        assert not teardown.done()
        # Teardown is still running, and the loop is still scheduling work.
        await asyncio.wait_for(asyncio.sleep(0.05), timeout=0.3)
        assert not teardown.done()
        await asyncio.wait_for(teardown, timeout=5)
    finally:
        runner.close()


@pytest.mark.asyncio
async def test_runner_rejects_a_submission_after_teardown() -> None:
    """A closed worker must fail loudly; silently dropping the cell would hang the caller."""
    runner = _runner()
    try:
        runner.close()
        with pytest.raises(OSError):
            runner.start_cell("print('never runs')")
        runner.close()  # idempotent: a second teardown is a no-op, not an error
    finally:
        runner.close()


@pytest.mark.asyncio
async def test_teardown_closes_a_plain_session() -> None:
    runner = _runner()
    try:
        cell_id = runner.start_cell("value = 21 * 2\nprint(value)")
        cell = await asyncio.wait_for(runner.wait_cell(cell_id, timeout=20), timeout=30)
        assert cell["stdout"] == "42\n"

        await asyncio.wait_for(runner.aclose(), timeout=10)
        assert runner._proc is not None and runner._proc.poll() is not None
    finally:
        runner.close()
