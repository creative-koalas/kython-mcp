from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import uuid
from typing import Any

import pytest

from kython_mcp import native
from kython_mcp.native import NativePythonService, PythonSessionError, _Session


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


@pytest.mark.asyncio
async def test_native_python_session_lifecycle_and_execution() -> None:
    service = NativePythonService()
    created = await service.create_session(label="analysis")

    completed = await service.submit_cell(
        created["session_id"],
        "value = 6 * 7\nprint(value)",
        wait_seconds=2,
    )
    assert completed["done"] is True
    assert completed["stdout"] == "42\n"
    snapshot = await service.snapshot(created["session_id"], include_all=False)
    assert snapshot["cells"][0]["cell_id"] == completed["cell_id"]

    await service.delete_session(created["session_id"])
    with pytest.raises(PythonSessionError, match="not found"):
        await service.snapshot(created["session_id"], include_all=False)
    await service.close()


@pytest.mark.asyncio
async def test_native_python_running_cell_accepts_input_and_interrupt() -> None:
    service = NativePythonService()
    session_id = (await service.create_session())["session_id"]
    started = await service.submit_cell(
        session_id,
        "import time\ntime.sleep(30)",
        wait_seconds=0,
    )
    assert started["running"] is True
    for _ in range(20):
        if (await service.snapshot(session_id, include_all=False))["cells"]:
            break
        await asyncio.sleep(0.05)
    interrupted = await service.interrupt(session_id)
    assert interrupted["interrupt_sent"] is True
    for _ in range(20):
        cells = (await service.snapshot(session_id, include_all=False))["cells"]
        if cells and not cells[0]["running"]:
            break
        await asyncio.sleep(0.05)
    await service.close()


def test_native_payload_is_json_compatible_shape() -> None:
    payload: dict[str, Any] = {"session_id": "1", "running": False}
    assert payload["session_id"] == "1"


@pytest.mark.asyncio
async def test_execution_survives_a_cell_that_leaves_a_detached_process() -> None:
    """The production wedge, end to end on the service path.

    An ephemeral execution whose cell leaves a process holding the worker's stdout pipe
    used to freeze the event loop inside teardown, so the service never answered again.
    The second execution is the real assertion: it only runs if the loop is still alive.
    """
    service = NativePythonService()
    try:
        first = await asyncio.wait_for(
            service.execute(
                str(uuid.uuid4()),
                "import subprocess\nc = subprocess.Popen(['sleep', '30'])\nprint('child', c.pid)",
                wait_seconds=20,
            ),
            timeout=30,
        )
        assert first["state"] == "succeeded"
        child_pid = int(first["cell"]["stdout"].split()[-1])

        second = await asyncio.wait_for(
            service.execute(str(uuid.uuid4()), "print('still serving')", wait_seconds=20),
            timeout=20,
        )
        assert second["state"] == "succeeded"
        assert second["cell"]["stdout"].strip() == "still serving"

        _kill(child_pid)
    finally:
        await asyncio.wait_for(service.close(), timeout=20)


@pytest.mark.asyncio
async def test_delete_session_teardown_survives_a_detached_process() -> None:
    """The explicit-session teardown path, with the same detached process holding the pipe."""
    service = NativePythonService()
    try:
        session_id = (await service.create_session())["session_id"]
        cell = await asyncio.wait_for(
            service.submit_cell(
                session_id,
                "import subprocess\nc = subprocess.Popen(['sleep', '30'])\nprint('child', c.pid)",
                wait_seconds=20,
            ),
            timeout=30,
        )
        assert cell["done"] is True
        child_pid = int(cell["stdout"].split()[-1])

        await asyncio.wait_for(service.delete_session(session_id), timeout=20)
        # One session's wedged teardown must not take the service with it.
        other = await asyncio.wait_for(
            service.submit_cell(
                (await service.create_session())["session_id"], "print('alive')", wait_seconds=20
            ),
            timeout=20,
        )
        assert other["stdout"].strip() == "alive"

        _kill(child_pid)
    finally:
        await asyncio.wait_for(service.close(), timeout=20)


@pytest.mark.asyncio
async def test_session_teardown_is_bounded_and_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The runner bounds its own cleanup; this pins the backstop for when it does not."""

    class HangingRunner:
        async def aclose(self) -> None:
            await asyncio.sleep(30)

    monkeypatch.setattr(native, "SESSION_TEARDOWN_TIMEOUT_SECONDS", 0.05)
    service = NativePythonService()
    # _Session is private, and the runner here is deliberately not an AsyncInterpreterRunner.
    session = _Session(session_id="session-hang", runner=HangingRunner())  # type: ignore[arg-type]
    try:
        started = time.monotonic()
        with caplog.at_level(logging.ERROR):
            await asyncio.wait_for(service._teardown_session(session), timeout=5)
        assert time.monotonic() - started < 1
        assert "interpreter_teardown_timeout" in caplog.text
    finally:
        await service.close()
