from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from kython_mcp.native import NativePythonService, PythonSessionError


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
    assert all(result["cell"]["session_id"] is None and result["cell"]["cell_id"] is None for result in results)
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
    unknown_interrupt = await restarted.interrupt_execution(running_id)
    assert unknown_interrupt["state"] == "unknown" and unknown_interrupt["interrupt_sent"] is False
    assert await restarted.list_sessions() == []
    await restarted.close()


@pytest.mark.asyncio
async def test_continuous_receipt_update_failure_ends_submission_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = NativePythonService(receipt_path=tmp_path / "receipts.db")
    execution_id = str(uuid4())
    source = "print('one shot')"
    creating = asyncio.Event()
    release_create = asyncio.Event()
    runners = []
    create = service._create_session

    async def gated_create(**kwargs):
        creating.set()
        await release_create.wait()
        session = await create(**kwargs)
        runners.append(session.runner)
        return session

    def fail_update(*args):
        raise OSError("receipt storage remains unavailable")

    monkeypatch.setattr(service, "_create_session", gated_create)
    monkeypatch.setattr(service._executions, "update", fail_update)
    submission = asyncio.create_task(service.execute(execution_id, source, wait_seconds=0))
    try:
        await asyncio.wait_for(creating.wait(), timeout=2)
        assert (await service.execution(execution_id))["state"] == "accepted"
        release_create.set()
        with pytest.raises(OSError, match="storage remains unavailable"):
            await submission
        assert not service._submitting and not service._running and not service._sessions
        assert not service._completion_tasks
        assert len(runners) == 1 and runners[0]._proc.poll() is not None
        assert service._executions.get(execution_id)[1]["state"] == "accepted"
        assert (await service.execution(execution_id))["state"] == "unknown"
        assert (await service.execute(execution_id, source))["state"] == "unknown"
        assert len(runners) == 1
    finally:
        release_create.set()
        await asyncio.gather(submission, return_exceptions=True)
        await service.close()


@pytest.mark.asyncio
async def test_interrupt_completed_execution_does_not_cancel_new_cell_in_same_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = NativePythonService()
    old_id, new_id = str(uuid4()), str(uuid4())
    release_old_watcher = asyncio.Event()
    complete = service._complete_execution

    async def delayed_complete(execution_id, *args):
        if execution_id == old_id:
            await release_old_watcher.wait()
        await complete(execution_id, *args)

    monkeypatch.setattr(service, "_complete_execution", delayed_complete)
    try:
        session_id = (await service.create_session())["session_id"]
        await service.execute(old_id, "print('old')", session_id=session_id, wait_seconds=0)
        owner, cell_id = service._running[old_id]
        await owner.runner.wait_cell(cell_id, timeout=2)
        await service.execute(
            new_id, "import time\nprint('new', flush=True)\ntime.sleep(.2)",
            session_id=session_id, wait_seconds=0,
        )
        for _ in range(100):
            if (await service.execution(new_id))["cell"]["stdout"] == "new\n":
                break
            await asyncio.sleep(.01)
        else:
            pytest.fail("new cell did not start")
        assert old_id in service._running
        assert (await service.interrupt_execution(old_id))["interrupt_sent"] is False
        assert (await service.execution(new_id, wait_seconds=2))["state"] == "succeeded"
        release_old_watcher.set()
        assert (await service.execution(old_id, wait_seconds=2))["state"] == "succeeded"
    finally:
        release_old_watcher.set()
        await service.close()
