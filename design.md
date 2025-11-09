# Kython MCP Design

## What is kython-mcp?

kython-mcp (koala + python = kython)
is a Python interpreter MCP server designed for LLM and agentic systems.

It offers:

- Multiple Python interpreter sessions with virtualized, independent stdin/stdout/stderr
- Block-delimited execution
- Specifying custom Python interpreter executable for custom environments

## Architecture

The most important piece of the puzzle is
how to effectively handle multiple Python interpreter sessions
with independent stdin/stdout/stderr,
and to capture them in a robust way.
Specifically, it is crucial to find a way to handle stdin correctly.

We will use a single-process architecture with subinterpreters.
Concretely, we will create channels in the "master interpreter" for communication,
then override `sys.stdin`, `sys.stdout`, and `sys.stderr` in each subinterpreter.
We'll send JSON messages instead of raw strings over the channels;
this makes it easy to see the boundaries of each command execution.

## Features

`kython-mcp` offers the following features:

- Cell-based execution:
executions and outputs are automatically segmented into cells;
this makes it easy to see which code produced which output.
This is done by sending markers at specific points
(e.g., before execution, after execution, etc.)
to the message channels in the subinterpreters.
- Interruption (cooperative):
Cells that take too long to execute can be interrupted.

The design philosophy of `kython-mcp` is similar to that of [`kmux`](https://github.com/creative-koalas/kmux.git);
block-oriented execution, AI-friendly ergonomics, etc.
