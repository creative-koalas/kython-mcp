"""
LLM Agent 使用示例 - 展示"提需求→LLM写代码→执行→反馈→修复"的完整流程
"""

import asyncio
from src.kython_mcp.llm_repl_agent import LLMREPLAgent


async def main():
    """演示LLM Agent的自主编码能力"""

    print("=" * 60)
    print("LLM-REPL Agent 演示")
    print("功能: 提需求 → LLM写代码 → REPL执行 → 自动修复错误")
    print("=" * 60)
    print()

    # 创建代理
    agent = LLMREPLAgent(
        session_id="demo_session",
        max_retries=3,  # 最多重试3次
        timeout=30.0,  # 超时30秒
        temperature=0.2,  # 低温度以获得更确定性的代码
    )

    try:
        # ========== 示例1: 简单计算任务 ==========
        res1 = await agent.solve_task("生成100个随机数(0-100)")
        res2 = await agent.solve_task("计算这些随机数的平均值")
        res3 = await agent.solve_task("找出大于平均值的随机数有多少个")
        print(res1['result'])
        print(res2['result'])
        print(res3['result'])
        # print("\n" + "=" * 60)
        # print("示例1: 简单数学计算")
        # print("=" * 60)
        # task1 = "判断2^17-1是否为素数"
        # print(f"任务: {task1}\n")

        # result1 = await agent.solve_task(task1)

        # if result1["success"]:
        #     print("✓ 任务成功完成!")
        #     print(f"\n生成的代码:\n{result1['code']}\n")
        #     print(f"执行结果:\n{result1['result']}")
        #     print(f"\n尝试次数: {result1['attempts']}")
        # else:
        #     print("✗ 任务失败")
        #     print(f"错误: {result1['error']}")

        # ========== 示例2: 数据结构算法 ==========
        # print("\n" + "=" * 60)
        # print("示例2: 算法实现")
        # print("=" * 60)
        # task2 = "生成前20个斐波那契数列,并计算它们的总和"
        # print(f"任务: {task2}\n")

        # result2 = await agent.solve_task(task2)

        # if result2["success"]:
        #     print("✓ 任务成功完成!")
        #     print(f"\n生成的代码:\n{result2['code']}\n")
        #     print(f"执行结果:\n{result2['result']}")
        #     print(f"\n尝试次数: {result2['attempts']}")
        # else:
        #     print("✗ 任务失败")
        #     print(f"错误: {result2['error']}")

        # # ========== 示例3: 字符串处理 ==========
        # print("\n" + "=" * 60)
        # print("示例3: 字符串处理")
        # print("=" * 60)
        # task3 = '分析这段文本中每个单词的出现次数: "Python is great and Python is powerful. Python is easy to learn."'
        # print(f"任务: {task3}\n")

        # result3 = await agent.solve_task(task3)

        # if result3["success"]:
        #     print("✓ 任务成功完成!")
        #     print(f"\n生成的代码:\n{result3['code']}\n")
        #     print(f"执行结果:\n{result3['result']}")
        #     print(f"\n尝试次数: {result3['attempts']}")

        #     # 显示执行历史(如果有多次尝试)
        #     if result3['attempts'] > 1:
        #         print("\n执行历史:")
        #         for i, hist in enumerate(result3['execution_history'], 1):
        #             print(f"  第{i}次尝试: {'失败 ❌' if hist['result'].get('exception') else '成功 ✓'}")
        # else:
        #     print("✗ 任务失败")
        #     print(f"错误: {result3['error']}")

        # # ========== 示例4: 数据转换 ==========
        # print("\n" + "=" * 60)
        # print("示例4: 数据转换")
        # print("=" * 60)
        # task4 = "创建一个字典,包含1-10的数字及其立方值,然后找出立方值大于500的所有数字"
        # print(f"任务: {task4}\n")

        # result4 = await agent.solve_task(task4)

        # if result4["success"]:
        #     print("✓ 任务成功完成!")
        #     print(f"\n生成的代码:\n{result4['code']}\n")
        #     print(f"执行结果:\n{result4['result']}")
        #     print(f"\n尝试次数: {result4['attempts']}")
        # else:
        #     print("✗ 任务失败")
        #     print(f"错误: {result4['error']}")

        # # ========== 示例5: 质数生成 ==========
        # print("\n" + "=" * 60)
        # print("示例5: 质数算法")
        # print("=" * 60)
        # task5 = "生成100以内的所有质数,并统计总共有多少个"
        # print(f"任务: {task5}\n")

        # result5 = await agent.solve_task(task5)

        # if result5["success"]:
        #     print("✓ 任务成功完成!")
        #     print(f"\n生成的代码:\n{result5['code']}\n")
        #     print(f"执行结果:\n{result5['result']}")
        #     print(f"\n尝试次数: {result5['attempts']}")
        # else:
        #     print("✗ 任务失败")
        #     print(f"错误: {result5['error']}")

        # # ========== 总结 ==========
        # print("\n" + "=" * 60)
        # print("演示完成!")
        # print("=" * 60)
        # print("\n核心特性:")
        # print("  ✓ 自然语言描述需求")
        # print("  ✓ LLM自动生成Python代码")
        # print("  ✓ 子解释器安全执行")
        # print("  ✓ 错误自动识别和修复")
        # print("  ✓ 会话状态保持")
        # print()

    finally:
        # 清理资源
        await agent.aclose()
        print("资源已清理")


if __name__ == "__main__":
    asyncio.run(main())
