"""
Kython MCP Server - Python REPL in Subinterpreter

提供基于Python 3.13+子解释器的安全REPL执行环境
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from interpreter_runner import AsyncInterpreterRunner, BusyError
from mcp.server import Server
from mcp.types import Tool, TextContent
from .llm_repl_agent import LLMREPLAgent
from .plan_reflect_agent import PlanReflectAgent


class KythonMCPServer:
    """Kython MCP服务器 - 管理Python子解释器REPL会话"""

    def __init__(self):
        self.app = Server("kython-mcp")
        self.sessions: Dict[str, AsyncInterpreterRunner] = {}
        self.llm_agents: Dict[str, LLMREPLAgent] = {}  # LLM代理会话
        self._setup_handlers()

    def _setup_handlers(self):
        """设置MCP处理器"""

        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            """列出可用工具"""
            return [
                Tool(
                    name="python_repl_execute",
                    description=(
                        "在子解释器REPL会话中执行Python代码。"
                        "代码在隔离的子解释器中运行，支持持久化命名空间。"
                        "返回stdout、stderr、结果值和异常信息。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "要执行的Python代码"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "会话ID，默认为'default'",
                                "default": "default"
                            },
                            "timeout": {
                                "type": "number",
                                "description": "超时时间（秒），默认30秒",
                                "default": 30.0
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="python_repl_create_session",
                    description=(
                        "创建新的Python REPL会话。"
                        "每个会话有独立的命名空间和状态。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "新会话的ID"
                            }
                        },
                        "required": ["session_id"]
                    }
                ),
                Tool(
                    name="python_repl_list_sessions",
                    description="列出所有活动的REPL会话及其状态",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="python_repl_close_session",
                    description="关闭指定的REPL会话，释放资源",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "要关闭的会话ID"
                            }
                        },
                        "required": ["session_id"]
                    }
                ),
                Tool(
                    name="llm_solve_task",
                    description=(
                        "【核心功能】使用LLM自动编写Python代码并执行来完成任务。"
                        "只需提供自然语言需求描述，LLM会自动生成代码、执行、"
                        "并在出错时自动修复。适合快速实现算法、数据处理等任务。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "任务需求描述(自然语言)"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "LLM代理会话ID，默认为'default_llm'",
                                "default": "default_llm"
                            },
                            "max_retries": {
                                "type": "number",
                                "description": "错误修复最大重试次数，默认3次",
                                "default": 3
                            },
                            "timeout": {
                                "type": "number",
                                "description": "代码执行超时时间(秒)，默认30秒",
                                "default": 30.0
                            }
                        },
                        "required": ["task"]
                    }
                ),
                Tool(
                    name="llm_solve_complex_task",
                    description=(
                        "【高级功能】使用Plan-Reflect模式解决复杂任务。"
                        "适合需要多步骤规划和迭代的复杂问题。"
                        "工作流程: 规划→执行→反思→再规划(最多n轮)。"
                        "每轮使用独立的REPL会话,避免上下文污染。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "总体目标描述(自然语言)"
                            },
                            "max_rounds": {
                                "type": "number",
                                "description": "最大迭代轮次，默认5轮",
                                "default": 5
                            },
                            "step_timeout": {
                                "type": "number",
                                "description": "每个步骤的超时时间(秒)，默认30秒",
                                "default": 30.0
                            },
                            "step_max_retries": {
                                "type": "number",
                                "description": "每个步骤的最大重试次数，默认3次",
                                "default": 3
                            }
                        },
                        "required": ["goal"]
                    }
                )
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """处理工具调用"""
            try:
                if name == "python_repl_execute":
                    return await self._handle_execute(arguments)
                elif name == "python_repl_create_session":
                    return await self._handle_create_session(arguments)
                elif name == "python_repl_list_sessions":
                    return await self._handle_list_sessions(arguments)
                elif name == "python_repl_close_session":
                    return await self._handle_close_session(arguments)
                elif name == "llm_solve_task":
                    return await self._handle_llm_solve_task(arguments)
                elif name == "llm_solve_complex_task":
                    return await self._handle_llm_solve_complex_task(arguments)
                else:
                    return [TextContent(type="text", text=f"未知工具: {name}")]
            except Exception as e:
                return [TextContent(type="text", text=f"错误: {str(e)}")]

    async def _get_or_create_session(self, session_id: str) -> AsyncInterpreterRunner:
        """获取或创建会话"""
        if session_id not in self.sessions:
            loop = asyncio.get_event_loop()
            self.sessions[session_id] = AsyncInterpreterRunner(session_id, loop)
        return self.sessions[session_id]

    async def _handle_execute(self, arguments: dict) -> list[TextContent]:
        """处理代码执行"""
        code = arguments.get("code", "")
        session_id = arguments.get("session_id", "default")
        timeout = arguments.get("timeout", 30.0)

        if not code.strip():
            return [TextContent(type="text", text="错误: 代码不能为空")]

        try:
            session = await self._get_or_create_session(session_id)
            result = await session.run_cell(code, timeout=timeout)

            # 格式化输出
            output_parts = []

            if result.get("stdout"):
                output_parts.append(f"=== stdout ===\n{result['stdout']}")

            if result.get("stderr"):
                output_parts.append(f"=== stderr ===\n{result['stderr']}")

            if result.get("result"):
                output_parts.append(f"=== result ===\n{result['result']}")

            if result.get("exception"):
                output_parts.append(f"=== exception ===\n{result['exception']}")

            output_parts.append(
                f"\n执行时间: {result.get('timing_ms', 0):.3f}ms | "
                f"Cell ID: {result.get('cell_id')}"
            )

            response = "\n\n".join(output_parts) if output_parts else "执行完成（无输出）"
            return [TextContent(type="text", text=response)]

        except BusyError as e:
            return [TextContent(type="text", text=f"会话忙碌: {str(e)}")]
        except asyncio.TimeoutError:
            return [TextContent(
                type="text",
                text=f"执行超时（>{timeout}秒）。会话可能仍在运行。"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"执行错误: {str(e)}")]

    async def _handle_create_session(self, arguments: dict) -> list[TextContent]:
        """处理创建会话"""
        session_id = arguments.get("session_id")

        if not session_id:
            return [TextContent(type="text", text="错误: 需要提供session_id")]

        if session_id in self.sessions:
            return [TextContent(type="text", text=f"会话 '{session_id}' 已存在")]

        try:
            await self._get_or_create_session(session_id)
            return [TextContent(
                type="text",
                text=f"成功创建会话 '{session_id}'"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"创建会话失败: {str(e)}")]

    async def _handle_list_sessions(self, arguments: dict) -> list[TextContent]:
        """处理列出会话"""
        if not self.sessions:
            return [TextContent(type="text", text="当前无活动会话")]

        session_info = []
        for session_id, session in self.sessions.items():
            status = "运行中" if session.is_running else "空闲"
            output = session.get_current_output()
            has_output = any(output.values())
            output_status = "有输出" if has_output else "无输出"

            session_info.append(
                f"- {session_id}: {status} | {output_status} | "
                f"Cell #{session._cell_id}"
            )

        response = "活动会话:\n" + "\n".join(session_info)
        return [TextContent(type="text", text=response)]

    async def _handle_close_session(self, arguments: dict) -> list[TextContent]:
        """处理关闭会话"""
        session_id = arguments.get("session_id")

        if not session_id:
            return [TextContent(type="text", text="错误: 需要提供session_id")]

        if session_id not in self.sessions:
            return [TextContent(type="text", text=f"会话 '{session_id}' 不存在")]

        try:
            session = self.sessions[session_id]
            await session.aclose()
            del self.sessions[session_id]
            return [TextContent(type="text", text=f"成功关闭会话 '{session_id}'")]
        except Exception as e:
            return [TextContent(type="text", text=f"关闭会话失败: {str(e)}")]

    async def _get_or_create_llm_agent(
        self,
        session_id: str,
        max_retries: int = 3,
        timeout: float = 30.0
    ) -> LLMREPLAgent:
        """获取或创建LLM代理"""
        if session_id not in self.llm_agents:
            self.llm_agents[session_id] = LLMREPLAgent(
                session_id=session_id,
                max_retries=max_retries,
                timeout=timeout,
            )
        return self.llm_agents[session_id]

    async def _handle_llm_solve_task(self, arguments: dict) -> list[TextContent]:
        """处理LLM自动编码任务"""
        task = arguments.get("task", "").strip()
        session_id = arguments.get("session_id", "default_llm")
        max_retries = int(arguments.get("max_retries", 3))
        timeout = float(arguments.get("timeout", 30.0))

        if not task:
            return [TextContent(type="text", text="错误: 任务描述不能为空")]

        try:
            # 获取或创建LLM代理
            agent = await self._get_or_create_llm_agent(
                session_id, max_retries, timeout
            )

            # 执行任务
            result = await agent.solve_task(task)

            # 格式化输出
            output_parts = []
            output_parts.append(f"=== 任务 ===\n{task}\n")

            if result["success"]:
                output_parts.append(f"=== 状态 ===\n✓ 成功")
                output_parts.append(f"\n=== 生成的代码 ===\n{result['code']}")
                output_parts.append(f"\n=== 执行结果 ===\n{result['result']}")
                output_parts.append(f"\n尝试次数: {result['attempts']}")
            else:
                output_parts.append(f"=== 状态 ===\n✗ 失败")
                output_parts.append(f"\n=== 错误 ===\n{result['error']}")
                if result['code']:
                    output_parts.append(f"\n=== 最后尝试的代码 ===\n{result['code']}")
                output_parts.append(f"\n尝试次数: {result['attempts']}/{max_retries + 1}")

            # 如果有多次尝试，显示历史
            if result['attempts'] > 1:
                output_parts.append(f"\n=== 执行历史 ===")
                for i, hist in enumerate(result['execution_history'], 1):
                    has_error = bool(hist['result'].get('exception'))
                    status = "❌" if has_error else "✓"
                    output_parts.append(f"\n第{i}次尝试 {status}:")
                    if has_error:
                        output_parts.append(f"  错误: {hist['result']['exception'][:100]}...")

            response = "\n".join(output_parts)
            return [TextContent(type="text", text=response)]

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return [TextContent(
                type="text",
                text=f"LLM任务执行失败: {str(e)}\n\n详细错误:\n{error_detail}"
            )]

    async def _handle_llm_solve_complex_task(self, arguments: dict) -> list[TextContent]:
        """处理Plan-Reflect模式的复杂任务"""
        goal = arguments.get("goal", "").strip()
        max_rounds = int(arguments.get("max_rounds", 5))
        step_timeout = float(arguments.get("step_timeout", 30.0))
        step_max_retries = int(arguments.get("step_max_retries", 3))

        if not goal:
            return [TextContent(type="text", text="错误: 目标描述不能为空")]

        try:
            # 创建Plan-Reflect代理(每次调用都创建新实例)
            agent = PlanReflectAgent(
                max_rounds=max_rounds,
                step_timeout=step_timeout,
                step_max_retries=step_max_retries,
            )

            # 执行复杂任务
            result = await agent.solve_complex_task(goal)

            # 格式化输出
            output_parts = []
            output_parts.append(f"=== 总体目标 ===\n{goal}\n")

            if result["success"]:
                output_parts.append(f"=== 状态 ===\n✓ 任务完成")
                output_parts.append(f"\n使用轮次: {result['rounds']}/{max_rounds}")
            else:
                output_parts.append(f"=== 状态 ===\n✗ 未完全完成")
                output_parts.append(f"\n使用轮次: {result['rounds']}/{max_rounds}")

            output_parts.append(f"\n=== 最终结果 ===\n{result['final_result']}")
            output_parts.append(f"\n=== 分析 ===\n{result['final_analysis']}")

            # 显示每轮历史
            output_parts.append(f"\n=== 执行历史 ===")
            for hist in result['history']:
                round_num = hist['round']
                plan_steps = len(hist['plan']['steps'])
                all_success = hist['execution_result']['all_success']
                status = "✓" if all_success else "✗"

                output_parts.append(f"\n第{round_num}轮 {status}:")
                output_parts.append(f"  目标: {hist['goal']}")
                output_parts.append(f"  计划: {plan_steps}个步骤 - {hist['plan']['reasoning']}")
                output_parts.append(f"  执行结果:")
                # 显示每个步骤的结果
                for sr in hist['execution_result']['step_results']:
                    step_status = "✓" if sr['success'] else "✗"
                    output_parts.append(f"    {step_status} 步骤{sr['step_id']}: {sr['description']}")
                output_parts.append(f"  反思: {hist['reflection']['analysis']}")

            response = "\n".join(output_parts)
            return [TextContent(type="text", text=response)]

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return [TextContent(
                type="text",
                text=f"复杂任务执行失败: {str(e)}\n\n详细错误:\n{error_detail}"
            )]

    async def cleanup(self):
        """清理所有会话"""
        for session in self.sessions.values():
            try:
                await session.aclose()
            except:
                pass
        self.sessions.clear()

        # 清理LLM代理
        for agent in self.llm_agents.values():
            try:
                await agent.aclose()
            except:
                pass
        self.llm_agents.clear()


async def main():
    """主入口"""
    import mcp

    server = KythonMCPServer()

    try:
        # 使用stdio传输运行MCP服务器
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.app.run(
                read_stream,
                write_stream,
                server.app.create_initialization_options()
            )
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
