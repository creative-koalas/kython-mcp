# Kython MCP Design

## What is kython-mcp?

kython-mcp (koala + python = kython) is a Python interpreter MCP server designed for LLM and agentic systems.

It offers:

- Multiple Python interpreter sessions with isolated stdin/stdout/stderr
- Block-delimited execution (cell-based)
- Optional custom Python executable per session

## Architecture

Each session runs in its own Python process. The parent process orchestrates sessions and routes stdin/stdout/stderr through structured messages, providing cell-level snapshots and cooperative interruption.

The module layout mirrors kmux (thin server module + focused submodules):

- `server.py`: MCP server wiring
- `tools.py`: tool definitions
- `sessions.py`: session lifecycle + registry
- `utils.py`: env + formatting helpers

## API Design (kmux-style)

The API mirrors kmux semantics so agents can reuse the same mental model.

### Tools

- **create_session**
  - **Purpose**: Create a new Python session (like kmux create_session)
  - **Input**: `label?: string`, `description?: string`, `python_executable?: string`
  - **Output**: `"New session created, ID:{id}"`

- **list_sessions**
  - **Purpose**: List active sessions (like kmux list_sessions)
  - **Input**: none
  - **Output**: YAML block with `id`, `metadata.label`, `metadata.description`, `metadata.running`, `metadata.python_executable`

- **update_session_label**
  - **Purpose**: Update session label (like kmux update_session_label)
  - **Input**: `session_id: string`, `label?: string`
  - **Output**: `"Session ID:{id} label updated to:{label}"`

- **update_session_description**
  - **Purpose**: Update session description (like kmux update_session_description)
  - **Input**: `session_id: string`, `description?: string`
  - **Output**: `"Session ID:{id} description updated to:{description}"`

- **submit_command**
  - **Purpose**: Execute Python code (like kmux submit_command)
  - **Input**: `session_id: string`, `command: string`, `timeout_seconds?: number`
  - **Behavior**:
    - `timeout_seconds <= 0 or None`: return immediately with cell_id
    - `timeout_seconds > 0`: wait for completion, return result; if timeout, continues in background
  - **Output**:
    - Started: `"Command in Session ID:{id} Cell ID:{cid} started"`
    - Completed: `"Command finished in {seconds} seconds"` + command buffer + output block
    - Timeout: running status + command buffer + partial output

- **snapshot**
  - **Purpose**: Get cell output snapshot (like kmux snapshot)
  - **Input**: `session_id: string`, `include_all?: boolean`
  - **Behavior**:
    - `include_all=false` returns latest cell only
    - `include_all=true` returns all cells
  - **Output**: YAML blocks with stdout/stderr/result/exception

- **send_keys**
  - **Purpose**: Send input to stdin (like kmux send_keys)
  - **Input**: `session_id: string`, `keys: string`, `append_newline?: boolean`, `send_eof?: boolean`
  - **Output**: YAML block with stdin payload

- **delete_session**
  - **Purpose**: Close and remove a session (like kmux delete_session)
  - **Input**: `session_id: string`
  - **Output**: `"Session ID:{id} closed"`

## Notes

- Cell outputs are segmented to make reasoning easier.
- Interrupts are handled by sending control sequences via `send_keys` (e.g. `\x03`).
