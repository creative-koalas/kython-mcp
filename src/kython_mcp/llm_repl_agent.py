"""
LLM-REPL Agent - 将大模型与Python REPL结合的自主编码代理

实现"需求→生成代码→执行→反馈→修复"的闭环
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kython_mcp.interpreter_runner import AsyncInterpreterRunner, BusyError  # type: ignore

from llm_backend import (  # type: ignore
    LLMResponse,
    Message,
    OpenAICompatibleBackend,
)


class LLMREPLAgent:
    """LLM驱动的Python REPL代理"""

    SYSTEM_PROMPT = """Python代码生成专家。

输出要求:
- 纯Python代码(无markdown/解释)
- REPL环境(变量持久化)
- 使用print()输出结果
- 简洁高效无bug
- 错误时根据反馈修复

示例:
用户: 计算1到100的和
输出: print(sum(range(1, 101)))
"""

    def __init__(
        self,
        llm_backend: Optional[OpenAICompatibleBackend] = None,
        session_id: str = "llm_agent",
        max_retries: int = 3,
        timeout: float = 30.0,
        temperature: float = 0.3,  # 低温度以获得更确定性的代码
        max_history_messages: int = 10,  # 最大保留历史消息数
    ):
        """
        初始化LLM-REPL代理

        Args:
            llm_backend: LLM后端,为None则自动创建
            session_id: REPL会话ID
            max_retries: 错误修复最大重试次数
            timeout: 代码执行超时时间(秒)
            temperature: LLM采样温度
            max_history_messages: 最大保留历史消息数（不含system prompt）
        """
        self.llm = llm_backend or OpenAICompatibleBackend()
        self.session_id = session_id
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.max_history_messages = max_history_messages

        # 创建REPL会话
        loop = asyncio.get_event_loop()
        self.repl = AsyncInterpreterRunner(session_id, loop)

        # 对话历史
        self.conversation: List[Message] = [
            Message(role="system", content=self.SYSTEM_PROMPT)
        ]

        # Token 使用统计
        self.token_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_requests": 0,
        }

        # 命名空间变量跟踪
        self.namespace_vars: Dict[str, str] = {}  # {变量名: 类型}

    def _trim_conversation(self):
        """修剪对话历史，保留 system prompt + 最近 N 条消息"""
        if len(self.conversation) <= self.max_history_messages + 1:
            return

        # 保留 system prompt
        system_msg = self.conversation[0]
        # 保留最近的消息
        recent_msgs = self.conversation[-(self.max_history_messages) :]
        self.conversation = [system_msg] + recent_msgs

    def _extract_variables_from_code(self, code: str) -> List[str]:
        """从代码中提取变量和函数定义

        Args:
            code: Python代码

        Returns:
            变量名列表
        """
        import re

        variables = []

        # 提取赋值语句（变量定义）
        # 匹配: var = ...
        assignments = re.findall(r"^(\w+)\s*=", code, re.MULTILINE)
        variables.extend(assignments)

        # 提取函数定义
        # 匹配: def func_name(...)
        functions = re.findall(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
        variables.extend(functions)

        # 提取类定义
        # 匹配: class ClassName(...)
        classes = re.findall(r"^class\s+(\w+)\s*[:\(]", code, re.MULTILINE)
        variables.extend(classes)

        return list(set(variables))  # 去重

    def _update_namespace(self, code: str):
        """更新命名空间变量跟踪

        Args:
            code: 成功执行的代码
        """
        variables = self._extract_variables_from_code(code)
        for var in variables:
            self.namespace_vars[var] = "user_defined"

    def _build_namespace_hint(self) -> str:
        """构建命名空间提示信息

        Returns:
            可用变量的提示文本（如果有的话）
        """
        if not self.namespace_vars:
            return ""

        var_list = ", ".join(sorted(self.namespace_vars.keys()))
        return f"\n\n可用变量: {var_list}"

    async def solve_task(self, user_request: str) -> Dict[str, any]:
        """
        解决用户任务

        Args:
            user_request: 用户需求描述

        Returns:
            {
                "success": bool,
                "result": str,  # 执行结果
                "code": str,  # 最终成功的代码
                "attempts": int,  # 尝试次数
                "error": str | None,  # 最终错误(如果失败)
                "execution_history": List[Dict]  # 执行历史
            }
        """
        # 在开始新任务前清理对话历史
        self._trim_conversation()

        # 添加用户请求到对话历史（附加命名空间提示）
        namespace_hint = self._build_namespace_hint()
        full_request = user_request + namespace_hint
        self.conversation.append(Message(role="user", content=full_request))

        execution_history = []
        last_error = None
        last_code_hash = None
        same_code_count = 0

        for attempt in range(self.max_retries + 1):
            try:
                # 生成代码
                code = await self._generate_code()

                if not code.strip():
                    last_error = "LLM生成了空代码"
                    break

                # 智能重试中断：检测重复代码
                code_hash = hash(code)
                if code_hash == last_code_hash:
                    same_code_count += 1
                    if same_code_count >= 2:
                        last_error = "LLM 持续生成相同错误代码，停止重试"
                        break
                else:
                    same_code_count = 0
                    last_code_hash = code_hash

                # 执行代码
                exec_result = await self._execute_code(code)

                # 记录执行历史
                execution_history.append(
                    {
                        "attempt": attempt + 1,
                        "code": code,
                        "result": exec_result,
                    }
                )

                # 检查是否成功
                if exec_result.get("exception"):
                    # 有异常,构造反馈让LLM修复
                    error_feedback = self._format_error_feedback(exec_result)
                    last_error = exec_result["exception"]

                    if attempt < self.max_retries:
                        # 还有重试机会,让LLM修复
                        self.conversation.append(
                            Message(role="assistant", content=code)
                        )
                        self.conversation.append(
                            Message(role="user", content=error_feedback)
                        )
                        continue
                    else:
                        # 已达最大重试次数
                        break
                else:
                    # 成功执行
                    self.conversation.append(Message(role="assistant", content=code))

                    # 更新命名空间变量跟踪
                    self._update_namespace(code)

                    return {
                        "success": True,
                        "result": self._format_success_result(exec_result),
                        "code": code,
                        "attempts": attempt + 1,
                        "error": None,
                        "execution_history": execution_history,
                    }

            except asyncio.TimeoutError:
                last_error = f"代码执行超时(>{self.timeout}秒)"
                if attempt < self.max_retries:
                    feedback = f"错误: 代码执行超时。请优化代码或减少计算量。\n\n超时的代码:\n{code}"
                    self.conversation.append(Message(role="assistant", content=code))
                    self.conversation.append(Message(role="user", content=feedback))
                else:
                    break

            except BusyError as e:
                last_error = f"REPL会话忙碌: {str(e)}"
                break

            except Exception as e:
                last_error = f"未知错误: {str(e)}"
                break

        # 所有尝试都失败了
        return {
            "success": False,
            "result": None,
            "code": execution_history[-1]["code"] if execution_history else None,
            "attempts": len(execution_history),
            "error": last_error,
            "execution_history": execution_history,
        }

    async def _generate_code(self) -> str:
        """调用LLM生成代码"""
        response = await self.llm.acomplete(
            messages=self.conversation,
            temperature=self.temperature,
            max_tokens=2000,
        )

        # 记录 token 使用
        if response.usage:
            self.token_stats["total_prompt_tokens"] += response.usage.get(
                "prompt_tokens", 0
            )
            self.token_stats["total_completion_tokens"] += response.usage.get(
                "completion_tokens", 0
            )
            self.token_stats["total_requests"] += 1

        return response.content.strip()

    async def _execute_code(self, code: str) -> Dict:
        """执行代码并返回结果"""
        return await self.repl.run_cell(code, timeout=self.timeout)

    def _format_error_feedback(self, exec_result: Dict) -> str:
        """格式化错误反馈给LLM（压缩版本，节省token）"""
        exception = exec_result.get("exception", "")

        # 提取关键信息
        lines = exception.strip().split("\n")
        error_line = lines[-1] if lines else "Unknown error"

        # 提取错误类型和消息
        error_type = error_line.split(":")[0] if ":" in error_line else error_line
        error_msg = error_line.split(":", 1)[1].strip() if ":" in error_line else ""

        # 提取关键 traceback（最多最后2行）
        traceback_lines = [l for l in lines if l.strip().startswith("File")]
        key_traceback = "\n".join(traceback_lines[-2:]) if traceback_lines else ""

        parts = [
            "执行错误:",
            f"类型: {error_type}",
        ]

        if error_msg:
            parts.append(f"消息: {error_msg}")

        if key_traceback:
            parts.append(f"位置:\n{key_traceback}")

        parts.append("\n请修复后提供完整代码(纯代码,无解释):")

        return "\n".join(parts)

    def _format_success_result(self, exec_result: Dict) -> str:
        """格式化成功结果"""
        parts = []

        if exec_result.get("stdout"):
            parts.append(f"输出:\n{exec_result['stdout']}")

        if exec_result.get("result") and exec_result["result"] != "None":
            parts.append(f"返回值: {exec_result['result']}")

        if exec_result.get("stderr"):
            parts.append(f"警告:\n{exec_result['stderr']}")

        return "\n\n".join(parts) if parts else "执行完成(无输出)"

    def get_token_stats(self) -> Dict[str, int]:
        """获取 token 使用统计

        Returns:
            {
                "total_prompt_tokens": int,
                "total_completion_tokens": int,
                "total_requests": int,
                "total_tokens": int
            }
        """
        total = (
            self.token_stats["total_prompt_tokens"]
            + self.token_stats["total_completion_tokens"]
        )
        return {
            **self.token_stats,
            "total_tokens": total,
        }

    async def reset(self):
        """重置对话历史和REPL会话"""
        self.conversation = [Message(role="system", content=self.SYSTEM_PROMPT)]
        # 重置 token 统计
        self.token_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_requests": 0,
        }
        # 重置命名空间跟踪
        self.namespace_vars = {}
        # 关闭旧会话
        await self.repl.aclose()
        # 创建新会话
        loop = asyncio.get_event_loop()
        self.repl = AsyncInterpreterRunner(self.session_id, loop)

    async def aclose(self):
        """关闭资源"""
        await self.repl.aclose()


async def demo():
    """演示用法"""
    agent = LLMREPLAgent()

    try:
        # 测试任务1: 简单计算
        print("=== 任务1: 计算1到100的和 ===")
        result = await agent.solve_task("计算1到100的和")
        print(f"成功: {result['success']}")
        print(f"结果:\n{result['result']}")
        print(f"尝试次数: {result['attempts']}")
        print()

        # 测试任务2: 稍复杂的算法
        print("=== 任务2: 生成斐波那契数列前15项 ===")
        result = await agent.solve_task("生成斐波那契数列前15项")
        print(f"成功: {result['success']}")
        print(f"结果:\n{result['result']}")
        print(f"尝试次数: {result['attempts']}")
        print()

        # 测试任务3: 故意触发错误(测试修复能力)
        print("=== 任务3: 读取并统计一段文本中的单词数 ===")
        result = await agent.solve_task(
            '统计这段文本的单词数: "Hello world, this is a test of the LLM REPL agent"'
        )
        print(f"成功: {result['success']}")
        print(f"结果:\n{result['result']}")
        print(f"尝试次数: {result['attempts']}")

    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(demo())
