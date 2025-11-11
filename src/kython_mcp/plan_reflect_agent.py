"""
Plan-Reflect Agent - 规划-执行-反思模式的多轮任务解决器

工作流程:
1. Planner: 根据总体需求生成本轮执行计划(step-by-step)
2. Executor: 使用LLMREPLAgent执行每个step
3. Reflector: 分析执行结果,决定是否需要下一轮
4. 下一轮只继承plan和result,丢弃中间对话上下文
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm_backend import OpenAICompatibleBackend, Message
from src.kython_mcp.llm_repl_agent import LLMREPLAgent


class PlanReflectAgent:
    """基于规划-执行-反思循环的复杂任务解决器"""

    PLANNER_SYSTEM_PROMPT = """你是一个任务规划专家。用户会给你一个总体需求,你需要将其分解为可执行的步骤。

输出格式(纯JSON,无markdown):
{
    "steps": [
        {"id": 1, "description": "步骤1的描述", "purpose": "为什么需要这一步"},
        {"id": 2, "description": "步骤2的描述", "purpose": "为什么需要这一步"}
    ],
    "reasoning": "整体规划思路"
}

规则:
1. 每个步骤应该是独立的、可执行的Python任务
2. 步骤间有明确的逻辑顺序
3. 避免过于细碎的步骤,每步应该有实质性产出
4. 考虑数据流:后续步骤可以访问前面步骤创建的变量
5. 每个步骤的description应该像给程序员下达的明确指令

示例:
用户需求: "分析一组销售数据,找出最畅销的产品类别"
你的输出:
{
    "steps": [
        {"id": 1, "description": "创建销售数据字典,包含产品名、类别、销量", "purpose": "准备测试数据"},
        {"id": 2, "description": "按类别分组,计算每个类别的总销量", "purpose": "聚合统计"},
        {"id": 3, "description": "找出销量最高的类别并打印结果", "purpose": "得出结论"}
    ],
    "reasoning": "先准备数据,再按类别聚合,最后找出最大值"
}
"""

    REFLECTOR_SYSTEM_PROMPT = """你是一个执行结果分析专家。你会收到一个任务的执行计划和实际结果,需要判断任务是否完成。

输出格式(纯JSON,无markdown):
{
    "task_completed": true/false,
    "analysis": "结果分析说明",
    "next_round_goal": "如果未完成,下一轮应该做什么(可为null)"
}

判断标准:
1. 所有步骤都成功执行
2. 得到了预期的最终结果
3. 没有明显的错误或缺陷

如果task_completed=false,需要在next_round_goal中说明:
- 哪里出了问题
- 下一轮需要补充什么
- 是否需要调整策略

示例:
计划: [创建数据, 分析数据, 输出结果]
结果: 步骤1成功,步骤2成功,步骤3成功,输出"最畅销类别是电子产品"

你的输出:
{
    "task_completed": true,
    "analysis": "所有步骤成功执行,得到了明确的分析结果",
    "next_round_goal": null
}

反例:
计划: [加载数据, 建模预测]
结果: 步骤1成功,步骤2失败(数据不足)

你的输出:
{
    "task_completed": false,
    "analysis": "数据准备不充分,导致建模失败",
    "next_round_goal": "生成更多样化的测试数据,并进行数据质量检查"
}
"""

    def __init__(
        self,
        max_rounds: int = 5,
        llm_backend: Optional[OpenAICompatibleBackend] = None,
        step_timeout: float = 60.0,
        step_max_retries: int = 3,
    ):
        """
        初始化Plan-Reflect代理

        Args:
            max_rounds: 最大迭代轮次
            llm_backend: LLM后端
            step_timeout: 每个step的执行超时时间
            step_max_retries: 每个step的最大重试次数
        """
        self.max_rounds = max_rounds
        self.llm = OpenAICompatibleBackend()
        self.step_timeout = step_timeout
        self.step_max_retries = step_max_retries

        # 历史记录(只保留plan和result)
        self.history: List[Dict[str, Any]] = []

    async def solve_complex_task(self, user_goal: str) -> Dict[str, Any]:
        """
        解决复杂任务

        Args:
            user_goal: 用户的总体需求

        Returns:
            {
                "success": bool,
                "final_result": str,
                "rounds": int,
                "history": List[Dict],  # 每轮的plan和result
                "final_analysis": str
            }
        """
        print(f"\n{'='*60}")
        print(f"开始解决复杂任务 (最多{self.max_rounds}轮)")
        print(f"目标: {user_goal}")
        print(f"{'='*60}\n")

        current_goal = user_goal

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- 第{round_num}轮 ---")

            # 1. 规划阶段
            plan = await self._plan(current_goal, round_num)
            if not plan:
                return self._failure_result("规划阶段失败", round_num)

            print(f"✓ 规划完成: {len(plan['steps'])}个步骤")
            print(f"  规划思路: {plan['reasoning']}")

            # 2. 执行阶段
            execution_result = await self._execute(plan, round_num)

            print(f"✓ 执行完成")

            # 3. 反思阶段
            reflection = await self._reflect(plan, execution_result, user_goal)
            if not reflection:
                return self._failure_result("反思阶段失败", round_num)

            print(f"✓ 反思完成")
            print(f"  分析: {reflection['analysis']}")

            # 记录本轮历史
            self.history.append({
                "round": round_num,
                "goal": current_goal,
                "plan": plan,
                "execution_result": execution_result,
                "reflection": reflection,
            })

            # 4. 判断是否完成
            if reflection["task_completed"]:
                print(f"\n{'='*60}")
                print(f"✓ 任务完成! (用了{round_num}轮)")
                print(f"{'='*60}")
                return {
                    "success": True,
                    "final_result": execution_result["summary"],
                    "rounds": round_num,
                    "history": self.history,
                    "final_analysis": reflection["analysis"],
                }

            # 5. 准备下一轮
            if round_num < self.max_rounds:
                current_goal = reflection["next_round_goal"]
                print(f"  → 下一轮目标: {current_goal}")
            else:
                print(f"\n{'='*60}")
                print(f"✗ 达到最大轮次({self.max_rounds}),任务未完全完成")
                print(f"{'='*60}")
                return {
                    "success": False,
                    "final_result": execution_result["summary"],
                    "rounds": round_num,
                    "history": self.history,
                    "final_analysis": f"达到最大轮次限制。最后分析: {reflection['analysis']}",
                }

        return self._failure_result("未知原因", self.max_rounds)

    async def _plan(self, goal: str, round_num: int) -> Optional[Dict]:
        """规划阶段 - 生成执行计划"""
        # 构建规划上下文(只包含历史的plan和result摘要)
        context = self._build_planning_context(goal, round_num)

        messages = [
            Message(role="system", content=self.PLANNER_SYSTEM_PROMPT),
            Message(role="user", content=context),
        ]

        try:
            response = await self.llm.acomplete(
                messages=messages,
                temperature=0.3,  # 规划需要较确定性
                max_tokens=1500,
            )

            # 提取并解析JSON
            plan = self._extract_json(response.content, "plan")
            if not plan:
                return None

            # 验证plan结构
            if "steps" not in plan or not isinstance(plan["steps"], list):
                print(f"✗ 规划失败: plan缺少steps字段")
                return None

            return plan

        except Exception as e:
            print(f"✗ 规划失败: {e}")
            return None

    async def _execute(self, plan: Dict, round_num: int) -> Dict:
        """执行阶段 - 执行计划中的每个步骤"""
        # 为每一轮创建独立的REPL会话,避免命名空间污染
        session_id = f"plan_reflect_round_{round_num}"
        agent = LLMREPLAgent(
            session_id=session_id,
            max_retries=self.step_max_retries,
            timeout=self.step_timeout,
        )

        step_results = []
        all_success = True

        try:
            for step in plan["steps"]:
                step_id = step["id"]
                step_desc = step["description"]

                print(f"  执行步骤{step_id}: {step_desc}")

                result = await agent.solve_task(step_desc)

                step_results.append({
                    "step_id": step_id,
                    "description": step_desc,
                    "success": result["success"],
                    "result": result["result"] if result["success"] else None,
                    "error": result["error"] if not result["success"] else None,
                    "code": result["code"],
                })

                if not result["success"]:
                    all_success = False
                    print(f"    ✗ 步骤{step_id}失败: {result['error']}")
                else:
                    print(f"    ✓ 步骤{step_id}成功")

        finally:
            await agent.aclose()

        # 汇总执行结果
        summary = self._summarize_execution(step_results)

        return {
            "all_success": all_success,
            "step_results": step_results,
            "summary": summary,
        }

    async def _reflect(
        self,
        plan: Dict,
        execution_result: Dict,
        original_goal: str
    ) -> Optional[Dict]:
        """反思阶段 - 分析执行结果,决定是否需要下一轮"""
        context = f"""原始目标: {original_goal}

本轮计划:
{json.dumps(plan, ensure_ascii=False, indent=2)}

执行结果:
{execution_result['summary']}

详细步骤结果:
{json.dumps(execution_result['step_results'], ensure_ascii=False, indent=2)}

请分析执行结果,判断任务是否完成。
"""

        messages = [
            Message(role="system", content=self.REFLECTOR_SYSTEM_PROMPT),
            Message(role="user", content=context),
        ]

        try:
            response = await self.llm.acomplete(
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
            )

            # 提取并解析JSON
            reflection = self._extract_json(response.content, "reflection")
            if not reflection:
                return None

            # 验证reflection结构
            if "task_completed" not in reflection:
                print(f"✗ 反思失败: 缺少task_completed字段")
                return None

            return reflection

        except Exception as e:
            print(f"✗ 反思失败: {e}")
            return None

    def _build_planning_context(self, goal: str, round_num: int) -> str:
        """构建规划上下文(只包含历史的plan和result,不含中间对话)"""
        if round_num == 1:
            return f"用户需求: {goal}\n\n请生成执行计划。"

        # 多轮情况:包含历史的plan和result摘要
        context_parts = [f"用户原始需求: {self.history[0]['goal']}\n"]

        for hist in self.history:
            context_parts.append(f"--- 第{hist['round']}轮历史 ---")
            context_parts.append(f"目标: {hist['goal']}")
            context_parts.append(f"计划: {len(hist['plan']['steps'])}个步骤")
            context_parts.append(f"结果: {hist['execution_result']['summary']}")
            context_parts.append(f"反思: {hist['reflection']['analysis']}\n")

        context_parts.append(f"--- 当前轮次(第{round_num}轮) ---")
        context_parts.append(f"新目标: {goal}")
        context_parts.append("\n请基于历史经验,生成本轮执行计划。")

        return "\n".join(context_parts)

    def _extract_json(self, content: str, source: str) -> Optional[Dict]:
        """从 LLM 响应中提取 JSON

        Args:
            content: LLM 的响应内容
            source: 来源标识（用于错误日志）

        Returns:
            提取的 JSON 对象，失败返回 None
        """
        import re

        # 策略1: 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 策略2: 尝试提取 markdown 代码块中的 JSON
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 策略3: 尝试提取第一个完整的 JSON 对象
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 策略4: 尝试查找 { 到 } 的最大匹配（处理嵌套）
        try:
            start = content.index('{')
            bracket_count = 0
            for i, char in enumerate(content[start:], start=start):
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_str = content[start:i+1]
                        return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass

        # 所有策略失败
        print(f"✗ 无法从 {source} 中提取 JSON")
        print(f"LLM 响应前 500 字符:\n{content[:500]}")
        return None

    def _summarize_execution(self, step_results: List[Dict]) -> str:
        """汇总执行结果"""
        parts = []
        for sr in step_results:
            status = "✓" if sr["success"] else "✗"
            parts.append(f"{status} 步骤{sr['step_id']}: {sr['description']}")
            if sr["success"] and sr["result"]:
                # 截取结果的前200字符
                result_preview = sr["result"][:200]
                if len(sr["result"]) > 200:
                    result_preview += "..."
                parts.append(f"  结果: {result_preview}")
            elif not sr["success"]:
                parts.append(f"  错误: {sr['error']}")

        return "\n".join(parts)

    def _failure_result(self, reason: str, rounds: int) -> Dict[str, Any]:
        """构造失败结果"""
        return {
            "success": False,
            "final_result": None,
            "rounds": rounds,
            "history": self.history,
            "final_analysis": f"任务失败: {reason}",
        }


async def demo():
    """演示Plan-Reflect模式"""
    agent = PlanReflectAgent(max_rounds=3)

    # 测试任务: 复杂的数据分析
    result = await agent.solve_complex_task(
        "创建一组销售数据(至少10条,包含产品名、类别、价格、销量),"
        "分析哪个类别最畅销,并计算该类别的平均单价"
    )

    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    print(f"成功: {result['success']}")
    print(f"轮次: {result['rounds']}")
    print(f"结果:\n{result['final_result']}")
    print(f"\n分析: {result['final_analysis']}")


if __name__ == "__main__":
    asyncio.run(demo())
