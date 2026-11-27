from __future__ import annotations

import asyncio
import json
import sqlite3
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


@pytest.mark.asyncio
async def test_snapshot_contract_projects_execution_id_and_terminal_states(tmp_path: Path) -> None:
    service = NativePythonService(receipt_path=tmp_path / "receipts.db")
    try:
        session_id = (await service.create_session(label="analysis"))["session_id"]
        empty = await service.snapshot(session_id, include_all=False, wait_seconds=0)
        assert empty["cells"] == []
        assert "execution_id" not in empty and "state" not in empty

        exec_ok = str(uuid4())
        await service.execute(
            exec_ok,
            "import time\ntime.sleep(.15)\nprint('ok')",
            session_id=session_id,
            wait_seconds=0,
        )
        running_snap = await service.snapshot(session_id, include_all=False, wait_seconds=0)
        assert running_snap["execution_id"] == exec_ok
        assert running_snap["state"] == "running"
        assert running_snap["cells"][0]["execution_id"] == exec_ok
        assert running_snap["cells"][0]["state"] == "running"

        done_snap = await asyncio.wait_for(
            service.snapshot(session_id, include_all=False, wait_seconds=5),
            timeout=5,
        )
        ok_receipt = await service.execution(exec_ok)
        assert done_snap["execution_id"] == exec_ok
        assert done_snap["state"] == "succeeded"
        assert done_snap["cells"] == [ok_receipt["cell"]]

        exec_fail = str(uuid4())
        fail_receipt = await service.execute(
            exec_fail, "raise ValueError('boom')", session_id=session_id, wait_seconds=5
        )
        all_snap = await service.snapshot(session_id, include_all=True, wait_seconds=0)
        assert all_snap["execution_id"] == exec_fail
        assert all_snap["state"] == "failed"
        assert all_snap["cells"] == [ok_receipt["cell"], fail_receipt["cell"]]
        assert [cell["execution_id"] for cell in all_snap["cells"]] == [exec_ok, exec_fail]
        assert [cell["state"] for cell in all_snap["cells"]] == ["succeeded", "failed"]

        direct = await service.submit_cell(session_id, "print('direct')", wait_seconds=5)
        assert "execution_id" not in direct and direct["state"] == "succeeded"
        direct_snap = await service.snapshot(session_id, include_all=False, wait_seconds=0)
        assert "execution_id" not in direct_snap and "state" not in direct_snap
        assert "execution_id" not in direct_snap["cells"][0]
        assert direct_snap["cells"][0]["state"] == "succeeded"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_snapshot_after_worker_exit_is_idempotent_and_preserves_execution_identity(
    tmp_path: Path,
) -> None:
    db = tmp_path / "receipts.db"
    service = NativePythonService(receipt_path=db)
    exec_ok, exec_crash = str(uuid4()), str(uuid4())
    crash_source = "import os, time\nprint('before', flush=True)\ntime.sleep(.05)\nos._exit(0)"
    try:
        session_id = (await service.create_session())["session_id"]
        ok_receipt = await service.execute(
            exec_ok, "print('ok')", session_id=session_id, wait_seconds=5
        )
        crash_receipt = await service.execute(
            exec_crash, crash_source, session_id=session_id, wait_seconds=5
        )
        assert ok_receipt["state"] == "succeeded"
        assert crash_receipt["state"] == "unknown" and crash_receipt["cell"] is None

        first_snap = await service.snapshot(session_id, include_all=True, wait_seconds=0)
        second_snap = await service.snapshot(session_id, include_all=True, wait_seconds=0)
        assert second_snap == first_snap
        assert first_snap["execution_id"] == exec_crash
        assert first_snap["state"] == "unknown"
        assert [cell["execution_id"] for cell in first_snap["cells"]] == [exec_ok, exec_crash]
        assert [cell["state"] for cell in first_snap["cells"]] == ["succeeded", "unknown"]
        crashed_cell = first_snap["cells"][1]
        assert crashed_cell["source"] == crash_source
        assert crashed_cell["running"] is False and crashed_cell["done"] is False
    finally:
        await service.close()

    restarted = NativePythonService(receipt_path=db)
    try:
        reloaded_ok = await restarted.execution(exec_ok)
        assert reloaded_ok["state"] == "succeeded"
        assert reloaded_ok["cell"]["execution_id"] == exec_ok
        assert reloaded_ok["cell"]["state"] == "succeeded"
        reloaded_crash = await restarted.execution(exec_crash)
        assert reloaded_crash["state"] == "unknown" and reloaded_crash["cell"] is None
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_snapshot_reconciles_orphaned_sqlite_receipt_idempotently(tmp_path: Path) -> None:
    db = tmp_path / "receipts.db"
    service = NativePythonService(receipt_path=db)
    try:
        session_id = (await service.create_session())["session_id"]
        exec_live = str(uuid4())
        await service.execute(exec_live, "print('live')", session_id=session_id, wait_seconds=5)

        legacy_id, orphan_id = str(uuid4()), str(uuid4())
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO executions VALUES (?, ?, ?)",
                (
                    legacy_id,
                    "fp-" + legacy_id,
                    json.dumps(
                        {
                            "execution_id": legacy_id,
                            "state": "succeeded",
                            "cell": {
                                "session_id": session_id,
                                "cell_id": 2,
                                "source": "print('legacy')",
                                "stdout": "legacy\n",
                            },
                            "meta": {},
                        }
                    ),
                ),
            )
            conn.execute(
                "INSERT INTO executions VALUES (?, ?, ?)",
                (
                    orphan_id,
                    "fp-" + orphan_id,
                    json.dumps(
                        {
                            "execution_id": orphan_id,
                            "state": "starting",
                            "cell": {
                                "session_id": session_id,
                                "cell_id": 3,
                                "source": "print('orphaned')",
                                "running": True,
                                "done": False,
                            },
                            "meta": {},
                        }
                    ),
                ),
            )

        first_snap = await service.snapshot(session_id, include_all=True, wait_seconds=0)
        assert [cell["execution_id"] for cell in first_snap["cells"]] == [
            exec_live,
            legacy_id,
            orphan_id,
        ]
        assert [cell["state"] for cell in first_snap["cells"]] == [
            "succeeded",
            "succeeded",
            "unknown",
        ]
        assert first_snap["cells"][1]["done"] is True and first_snap["cells"][1]["running"] is False
        assert first_snap["cells"][2]["source"] == "print('orphaned')"
        assert first_snap["cells"][2]["running"] is False and first_snap["cells"][2]["done"] is False
        assert first_snap["execution_id"] == orphan_id
        assert first_snap["state"] == "unknown"

        orphan_receipt = await service.execution(orphan_id)
        assert orphan_receipt["state"] == "unknown" and orphan_receipt["cell"] is None
        with sqlite3.connect(db) as conn:
            stored_orphan = json.loads(
                conn.execute("SELECT receipt FROM executions WHERE id = ?", (orphan_id,)).fetchone()[0]
            )
        assert stored_orphan["state"] == "unknown" and stored_orphan["cell"] is None

        interrupt = await service.interrupt_execution(orphan_id)
        assert interrupt["state"] == "unknown" and interrupt["interrupt_sent"] is False

        # Consecutive snapshots remain identical even after SQLite reconciliation clears receipt.cell.
        assert await service.snapshot(session_id, include_all=True, wait_seconds=0) == first_snap
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_owned_receipt_and_transient_proof_survive_upgrade(tmp_path: Path) -> None:
    db = tmp_path / "receipts.db"
    service = NativePythonService(receipt_path=db)
    execution_id = str(uuid4())
    owner, origin = "session-principal", "event-origin"
    proof = "temporary-proof-for-one-cell"
    source = "import os; print(len(os.environ.get('WORKSPACE_EXECUTION_CONTEXT', '')))"
    try:
        session = (await service.create_session(owner=owner))["session_id"]
        original = await service.execute(
            execution_id, source, session_id=session, owner=owner, origin=origin,
            execution_context=proof,
        )
        assert original["cell"]["stdout"].strip() == str(len(proof))
        next_cell = await service.submit_cell(session, source, owner=owner, wait_seconds=3)
        assert next_cell["stdout"].strip() == "0"
        assert await service.list_sessions() == []
        assert (await service.snapshot(session, owner=owner, include_all=True))["cells"][0]["execution_id"] == execution_id
        assert not {"_execution_owner", "_runtime_origin"} & original.keys()
    finally:
        await service.close()
    assert proof.encode() not in db.read_bytes()

    restarted = NativePythonService(receipt_path=db)
    try:
        for request in ({}, {"owner": "other"}, {"owner": owner, "origin": "another-event"}):
            with pytest.raises(PythonSessionError, match="source"):
                await restarted.execution(execution_id, **request)
        assert await restarted.execute(
            execution_id, source, session_id=session, owner=owner, origin=origin,
            execution_context="rotated-proof",
        ) == original
    finally:
        await restarted.close()
