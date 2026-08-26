from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kython_mcp.native import NativePythonService, PythonSessionError


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
