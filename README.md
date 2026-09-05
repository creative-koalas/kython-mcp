# kython-mcp

Native, process-isolated Python execution for PsyGo workspaces. The historical
package name remains; workspace capabilities call `NativePythonService` directly.

`execute(invocation_id, source)` runs self-contained code in a fresh interpreter
and releases that interpreter when execution ends. Pass an explicitly created
`session_id` only when multiple cells need shared variables. Process IDs are not
part of this API. Session IDs are unique across service restarts.

The trusted invocation ID identifies one execution. With a persistent
`receipt_path`, SQLite commits its reservation before dispatching code. Repeated
submissions with the same source and session return the original receipt; a
different source or session under that ID is rejected. Cancelling an HTTP wait
does not cancel the execution. `execution(invocation_id)` reads its receipt.
Pass `wait_seconds` (0 through 30) to wait for the existing execution; a cancelled
wait never interrupts the worker. `interrupt_execution(invocation_id)` requests
an interrupt and returns `interrupt_sent` plus the observed receipt state. A sent
interrupt is not evidence of termination; read the receipt for the final result.

One-shot cell receipts have null `session_id` and `cell_id`: their only public
handle is `execution_id`. Explicit sessions retain their session and cell IDs.
Session listings contain only explicitly created sessions. Older receipts are
normalized only when their stored source/session fingerprint proves one-shot
ownership. A temporary interpreter is closed even if submission or receipt
persistence fails before its completion watcher takes ownership.

Receipts report `accepted`, `running`, `succeeded`, `failed`, or `unknown`.
Completed receipts survive service restarts. Incomplete records become `unknown`
after restart and are never replayed: neither a lost worker nor a lost response
proves that code had no effects. The workspace must persist the receipt file with
its task data; deleting that state removes the execution history.

The lower-level `submit_cell` and session APIs manage explicitly owned interpreter
sessions; network providers should use `execute` to bind trusted invocation IDs.

```bash
uv run pytest -q
```
