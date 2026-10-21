from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from kython_mcp.native import NativePythonService, PythonSessionError
from kython_mcp.redaction import SecretRedactor


def test_secret_redactor_retains_split_prefixes_and_masks_repr() -> None:
    redactor = SecretRedactor(("credential-example-123", "line\nsecret"))
    assert redactor.feed("safe: credent") == "safe: "
    assert redactor.feed("ial-example-") == ""
    assert redactor.feed("123 done\n") == "[REDACTED] done\n"
    assert redactor.feed("line\\nsecret") == "[REDACTED]"
    assert redactor.feed("credent") == ""
    assert redactor.feed("", final=True) == "[REDACTED]"
    overlapping = SecretRedactor(("abc", "abcdef"))
    assert overlapping.feed("abc") == ""
    assert overlapping.feed("def", final=True) == "[REDACTED]"


@pytest.mark.asyncio
async def test_execute_binds_three_material_types_without_global_environment(
    tmp_path: Path,
) -> None:
    service = NativePythonService(receipt_path=tmp_path / "receipts.db")
    material = tmp_path / "material"
    material.write_text("synthetic-file-secret")
    cleanup = []
    environment = {
        "BOUND_TEXT": "synthetic-text-secret",
        "BOUND_USER": "synthetic-user",
        "BOUND_PASSWORD": "synthetic-password-secret",
        "BOUND_FILE": str(material),
    }
    try:
        result = await service.execute(
            str(uuid4()),
            "import os\nfrom pathlib import Path\n"
            "assert all(os.environ.get(key) for key in "
            "('BOUND_TEXT', 'BOUND_USER', 'BOUND_PASSWORD', 'BOUND_FILE'))\n"
            "print(Path(os.environ['BOUND_FILE']).read_text())\n"
            "print(os.environ['BOUND_TEXT'], os.environ['BOUND_PASSWORD'])",
            execution_env=environment,
            output_secrets=tuple(environment.values()) + (material.read_text(),),
            binding_fingerprint="metadata-only-one",
            on_execution_finished=lambda: cleanup.append("finished"),
        )
        assert result["state"] == "succeeded"
        assert "synthetic" not in result["cell"]["stdout"]
        assert cleanup == ["finished"]
        assert all(key not in os.environ for key in environment)
        next_result = await service.execute(
            str(uuid4()), "import os\nprint('BOUND_TEXT' in os.environ)"
        )
        assert next_result["cell"]["stdout"] == "False\n"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_execute_masks_running_chunks_exceptions_events_and_durable_receipts(
    tmp_path: Path,
) -> None:
    database = tmp_path / "receipts.db"
    service = NativePythonService(receipt_path=database)
    execution_id = str(uuid4())
    secret = "split-output-credential-example"
    try:
        result = await service.execute(
            execution_id,
            "import os, time, sys\nvalue = os.environ['BOUND_TEXT']\n"
            "sys.stdout.write(value[:12]); sys.stdout.flush()\n"
            "time.sleep(.15)\n"
            "sys.stdout.write(value[12:] + '\\n'); sys.stdout.flush()\n"
            "os.write(2, (value + '\\n').encode())\n"
            "raise ValueError(value)",
            execution_env={"BOUND_TEXT": secret}, output_secrets=(secret,),
            binding_fingerprint="metadata-only-two", wait_seconds=0,
        )
        assert result["state"] == "running"
        runner = service._running[execution_id][0].runner
        await asyncio.sleep(.08)
        running = await service.execution(execution_id)
        assert secret[:12] not in running["cell"]["stdout"]
        completed = await service.execution(execution_id, wait_seconds=3)
        assert completed["state"] == "failed"
        assert secret not in str(completed)
        assert "[REDACTED]" in completed["cell"]["exception"]
        events = []
        while not runner.events.empty():
            events.append(runner.events.get_nowait())
        assert secret not in str(events)
        assert secret not in str(runner._stderr_lines)
    finally:
        await service.close()
    assert secret.encode() not in database.read_bytes()
    restarted = NativePythonService(receipt_path=database)
    try:
        assert secret not in str(await restarted.execution(execution_id))
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_execute_retains_binding_until_completion_and_cleans_duplicate_inputs() -> None:
    service = NativePythonService()
    execution_id = str(uuid4())
    cleaned = []
    source = "import time\ntime.sleep(.15)\nprint('done')"
    try:
        await service.execute(
            execution_id, source, wait_seconds=0, execution_env={},
            binding_fingerprint="binding-a",
            on_execution_finished=lambda: cleaned.append("original"),
        )
        assert cleaned == []
        await service.execute(
            execution_id, source, wait_seconds=0, execution_env={},
            binding_fingerprint="binding-a",
            on_execution_finished=lambda: cleaned.append("duplicate"),
        )
        assert cleaned == ["duplicate"]
        with pytest.raises(PythonSessionError, match="different code"):
            await service.execute(
                execution_id, source, execution_env={}, binding_fingerprint="binding-b",
                on_execution_finished=lambda: cleaned.append("conflict"),
            )
        assert cleaned == ["duplicate", "conflict"]
        await service.execution(execution_id, wait_seconds=3)
        assert cleaned == ["duplicate", "conflict", "original"]
    finally:
        await service.close()
    assert cleaned.count("original") == 1


@pytest.mark.asyncio
async def test_execute_binding_rejects_persistent_session_and_cleans_validation_failure() -> None:
    service = NativePythonService()
    cleaned = []
    try:
        session_id = (await service.create_session())["session_id"]
        with pytest.raises(PythonSessionError, match="one-shot"):
            await service.execute(
                str(uuid4()), "pass", session_id=session_id, execution_env={},
                on_execution_finished=lambda: cleaned.append("session"),
            )
        with pytest.raises(PythonSessionError):
            await service.execute(
                str(uuid4()), "", execution_env={},
                on_execution_finished=lambda: cleaned.append("source"),
            )
        assert cleaned == ["session", "source"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_close_cleans_binding_even_before_completion_watcher_starts() -> None:
    service = NativePythonService()
    cleaned = []
    await service.execute(
        str(uuid4()), "import time\ntime.sleep(30)", wait_seconds=0,
        execution_env={}, on_execution_finished=lambda: cleaned.append("closed"),
    )
    assert cleaned == []
    await service.close()
    assert cleaned == ["closed"]


@pytest.mark.asyncio
async def test_execute_cleans_material_when_interpreter_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kython_mcp import native

    def fail_start(*args, **kwargs):
        raise OSError("synthetic worker startup failure")

    service = NativePythonService()
    cleaned = []
    monkeypatch.setattr(native, "AsyncInterpreterRunner", fail_start)
    try:
        with pytest.raises(OSError, match="startup failure"):
            await service.execute(
                str(uuid4()), "pass", execution_env={},
                on_execution_finished=lambda: cleaned.append("failed"),
            )
        assert cleaned == ["failed"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_cleanup_callback_failure_never_logs_material(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_cleanup() -> None:
        raise RuntimeError("synthetic-private-cleanup-value")

    service = NativePythonService()
    try:
        completed = await service.execute(
            str(uuid4()), "pass", execution_env={}, on_execution_finished=fail_cleanup,
        )
        assert completed["state"] == "succeeded"
        assert "execution_binding_cleanup_failed" in caplog.text
        assert "synthetic-private-cleanup-value" not in caplog.text
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_bound_execution_reaps_inherited_environment_process_group() -> None:
    service = NativePythonService()
    child_pid = 0
    try:
        completed = await service.execute(
            str(uuid4()),
            "import subprocess\nchild = subprocess.Popen(['sleep', '30'])\nprint(child.pid)",
            execution_env={"BOUND_TEXT": "process-group-secret"},
            output_secrets=("process-group-secret",),
        )
        child_pid = int(completed["cell"]["stdout"].strip())
        process_state = await asyncio.to_thread(
            subprocess.run, ["ps", "-o", "stat=", "-p", str(child_pid)],
            text=True, capture_output=True, check=False,
        )
        state = process_state.stdout.strip()
        assert not state or state.startswith("Z")
        assert (await service.execute(str(uuid4()), "print('alive')"))["state"] == "succeeded"
    finally:
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await service.close()
