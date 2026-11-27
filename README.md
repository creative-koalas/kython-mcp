# kython-mcp

PsyGo 工作区沙箱的进程隔离 Python 执行引擎库（`NativePythonService`）。仓库保留历史包名，但**不是**独立运行的 Kubernetes 服务或 MCP Server，而是由 `vm-agent` 构建沙箱镜像（`WORKSPACE_IMAGE`）时直接打包进虚拟环境并在进程内调用。

## 执行模型与回执语义

宿主验证请求后提供不透明的 `owner` 和 `origin`。会话按所有者隔离，执行回执绑定首次提交的来源。临时 `execution_context` 仅供本次工作进程使用。省略身份的本地调用只访问同样未绑定身份的旧会话和回执。

- **一次性隔离执行与显式会话**：`execute(invocation_id, source)` 默认在全新独立子进程解释器中运行自包含代码并在结束后立即释放。仅当多步计算确需共享内存变量时才传入显式创建的跨重启唯一 `session_id`。
- **派发前落盘与不确定态封顶**：配置持久 `receipt_path` 后，SQLite `ExecutionStore` 在派发代码前先提交执行预留记录。相同 `(invocation_id, source, session, owner, origin)` 重复提交幂等返回原回执，同 ID 异参直接拒绝。取消外层等待绝不终止后台子进程，`interrupt_execution` 仅发送协作中断信号而不冒充已终止证据。已完成回执跨重启保留，重启前未收敛的在途记录统一标记为 `unknown` 且永不自动重放。

```bash
uv sync --frozen
uv run pytest -q
```

- [`AGENTS.md`](AGENTS.md) — 库职责边界、宿主集成与研发验证规范。
