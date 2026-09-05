from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from kython_mcp.native import NativePythonService, PythonSessionError


@pytest.mark.asyncio
async def test_stateless_calls_discard_interpreter_but_keep_receipt(tmp_path: Path) -> None:
    service = NativePythonService(receipt_path=tmp_path / "receipts.db")
    first_id, second_id = str(uuid4()), str(uuid4())
    first = await service.execute(first_id, "value = 42\nprint(value)", wait_seconds=2)
    assert first["state"] == "succeeded"
    assert first["cell"]["stdout"] == "42\n"
    assert await service.list_sessions() == []
    second = await service.execute(second_id, "print('value' in globals())", wait_seconds=2)
    assert second["cell"]["stdout"] == "False\n"
    assert (await service.execution(first_id))["cell"]["stdout"] == "42\n"
    await service.close()


@pytest.mark.asyncio
async def test_cancelled_wait_and_concurrent_retries_dispatch_once(tmp_path: Path) -> None:
    service = NativePythonService(receipt_path=tmp_path / "receipts.db")
    execution_id = str(uuid4())
    effect = tmp_path / "effect.txt"
    source = f"import time\ntime.sleep(.15)\nopen({str(effect)!r}, 'a').write('once\\n')"
    first = asyncio.create_task(service.execute(execution_id, source, wait_seconds=2))
    await asyncio.sleep(0)
    assert (await service.execution(execution_id))["state"] == "running"
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    results = await asyncio.gather(*(
        service.execute(execution_id, source, wait_seconds=2) for _ in range(5)
    ))
    assert all(result["state"] == "succeeded" for result in results)
    assert len({result["cell"]["cell_id"] for result in results}) == 1
    assert effect.read_text() == "once\n"
    with pytest.raises(PythonSessionError, match="different code"):
        await service.execute(execution_id, "print('another effect')")
    await service.close()


@pytest.mark.asyncio
async def test_restart_keeps_completed_receipt_and_never_replays_uncertain_work(tmp_path: Path) -> None:
    db = tmp_path / "receipts.db"
    service = NativePythonService(receipt_path=db)
    completed_id, running_id = str(uuid4()), str(uuid4())
    await service.execute(completed_id, "print('done')", wait_seconds=2)
    source = "import time\ntime.sleep(30)"
    await service.execute(running_id, source, wait_seconds=0)
    await service.close()
    restarted = NativePythonService(receipt_path=db)
    assert (await restarted.execution(completed_id))["state"] == "succeeded"
    assert (await restarted.execute(running_id, source))["state"] == "unknown"
    assert await restarted.list_sessions() == []
    await restarted.close()


@pytest.mark.asyncio
async def test_explicit_sessions_preserve_variables_without_reusing_ids_after_restart() -> None:
    service = NativePythonService()
    session_id = (await service.create_session())["session_id"]
    await service.execute(str(uuid4()), "value = 41", session_id=session_id, wait_seconds=2)
    second = await service.execute(
        str(uuid4()), "print(value + 1)", session_id=session_id, wait_seconds=2
    )
    assert second["cell"]["stdout"] == "42\n"
    assert len(await service.list_sessions()) == 1
    await service.close()
    restarted = NativePythonService()
    assert (await restarted.create_session())["session_id"] != session_id
    await restarted.close()


@pytest.mark.asyncio
async def test_worker_exit_marks_unknown_instead_of_infinite_running() -> None:
    service = NativePythonService()
    result = await service.execute(str(uuid4()), "import os\nos._exit(0)", wait_seconds=2)
    assert result["state"] == "unknown"
    assert await service.list_sessions() == []
    await service.close()


@pytest.mark.asyncio
async def test_invalid_source_never_reserves_execution() -> None:
    service = NativePythonService()
    execution_id = str(uuid4())
    with pytest.raises(PythonSessionError):
        await service.execute(execution_id, "this is invalid python syntax")
    with pytest.raises(PythonSessionError, match="not found"):
        await service.execution(execution_id)
    await service.close()


@pytest.mark.asyncio
async def test_deleting_running_explicit_session_finishes_receipt_as_unknown() -> None:
    service = NativePythonService()
    session_id = (await service.create_session())["session_id"]
    execution_id = str(uuid4())
    source = "import time\ntime.sleep(30)"
    await service.execute(execution_id, source, session_id=session_id, wait_seconds=0)
    await service.delete_session(session_id)
    receipt = await service.execute(execution_id, source, session_id=session_id)
    assert receipt["state"] == "unknown"
    assert receipt["cell"] is None
    await service.close()
