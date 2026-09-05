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
async def test_explicit_sessions_preserve_variables_without_reusing_ids_after_restart() -> None:
    service = NativePythonService()
    session_id = (await service.create_session())["session_id"]
    await service.execute(str(uuid4()), "value = 41", session_id=session_id, wait_seconds=2)
    second = await service.execute(
        str(uuid4()), "print(value + 1)", session_id=session_id, wait_seconds=2
    )
    assert second["cell"]["stdout"] == "42\n"
    assert second["cell"]["session_id"] == session_id
    assert second["cell"]["cell_id"] == 2
    assert len(await service.list_sessions()) == 1
    await service.close()
    restarted = NativePythonService()
    assert (await restarted.create_session())["session_id"] != session_id
    await restarted.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["submit", "running_receipt", "terminal_receipt"])
async def test_one_shot_owner_is_released_when_submission_or_persistence_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    service = NativePythonService(receipt_path=tmp_path / "receipts.db")
    runners = []
    create = service._create_session

    async def capture_owner(**kwargs):
        session = await create(**kwargs)
        runners.append(session.runner)
        return session

    monkeypatch.setattr(service, "_create_session", capture_owner)
    if failure == "submit":
        async def fail_submit(*args, **kwargs):
            raise OSError("submission unavailable")
        monkeypatch.setattr(service, "submit_cell", fail_submit)
    else:
        update = service._executions.update
        failed = False

        def fail_update(execution_id, receipt):
            nonlocal failed
            target = "running" if failure == "running_receipt" else "succeeded"
            if not failed and receipt["state"] == target:
                failed = True
                raise OSError("receipt persistence unavailable")
            update(execution_id, receipt)

        monkeypatch.setattr(service._executions, "update", fail_update)
    execution_id = str(uuid4())
    source = "print('one shot')"
    with pytest.raises(OSError):
        await service.execute(execution_id, source, wait_seconds=2)
    assert not service._sessions and not service._running and not service._completion_tasks
    assert len(runners) == 1 and runners[0]._proc.poll() is not None
    assert (await service.execution(execution_id))["state"] == "unknown"
    assert (await service.execute(execution_id, source))["state"] == "unknown"
    assert len(runners) == 1
    await service.close()


@pytest.mark.asyncio
async def test_bounded_receipt_wait_and_cancel_do_not_replay_or_interrupt(tmp_path: Path) -> None:
    service = NativePythonService(receipt_path=tmp_path / "receipts.db")
    execution_id = str(uuid4())
    effect = tmp_path / "once.txt"
    source = f"import time\ntime.sleep(.3)\nopen({str(effect)!r}, 'a').write('once\\n')"
    await service.execute(execution_id, source, wait_seconds=0)
    assert await service.list_sessions() == []
    waiting = asyncio.create_task(service.execution(execution_id, wait_seconds=2))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    pending = await service.execution(execution_id, wait_seconds=.01)
    assert pending["state"] == "running"
    assert pending["cell"]["session_id"] is None and pending["cell"]["cell_id"] is None
    complete = await service.execution(execution_id, wait_seconds=2)
    assert complete["state"] == "succeeded" and effect.read_text() == "once\n"
    assert not service._sessions
    await service.close()


@pytest.mark.asyncio
async def test_interrupt_by_execution_id_reports_request_then_observed_result() -> None:
    service = NativePythonService()
    execution_id = str(uuid4())
    await service.execute(execution_id, "import time\nprint('started', flush=True)\ntime.sleep(30)", wait_seconds=0)
    for _ in range(100):
        if (await service.execution(execution_id))["cell"]["stdout"] == "started\n":
            break
        await asyncio.sleep(.02)
    else:
        pytest.fail("worker did not start")
    requested = await service.interrupt_execution(execution_id)
    assert requested == {"execution_id": execution_id, "session_id": None, "state": "running", "interrupt_sent": True}
    finished = await service.execution(execution_id, wait_seconds=2)
    assert finished["state"] == "failed"
    assert "KeyboardInterrupt" in finished["cell"]["exception"]
    again = await service.interrupt_execution(execution_id)
    assert again["interrupt_sent"] is False and again["state"] == "failed"
    assert not service._sessions
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


@pytest.mark.asyncio
async def test_older_receipt_owner_is_normalized_from_stored_fingerprint(tmp_path: Path) -> None:
    import hashlib
    import json

    service = NativePythonService(receipt_path=tmp_path / "receipts.db")
    for explicit_session in (None, "explicit-session"):
        execution_id = str(uuid4())
        source = "print('completed')"
        fingerprint = hashlib.sha256(json.dumps([explicit_session, source], ensure_ascii=False).encode()).hexdigest()
        service._executions.reserve(execution_id, fingerprint, {
            "execution_id": execution_id, "state": "succeeded", "meta": {},
            "cell": {"source": source, "session_id": explicit_session or "old-internal-owner", "cell_id": 1},
        })
        receipt = await service.execution(execution_id)
        assert receipt["cell"]["session_id"] == explicit_session
        assert receipt["cell"]["cell_id"] == (1 if explicit_session else None)
    await service.close()


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
