"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-MCP

Agent 思考过程演示
------------------
工单原文明确要求："展示出 Agent 的思考过程及梳理流程，并返回结果"。

本脚本对 5 类典型场景各跑一个 query，抓取完整轨迹：
  - 用户 query
  - 每一轮 LLM 消息（含 tool_calls 决策）
  - 每次 tool 调用的参数
  - 每次 tool 的原始返回
  - LLM 综合后的自然语言回复
  - 每步耗时

结果落盘为 md + json 报告，附加到 history。

场景清单：
  1. registration —— 挂号号源查询
  2. knowledge    —— 症状咨询（需 02 起才能真答，否则演示容错）
  3. amap-hospital —— 医院搜索
  4. amap-nearby   —— 组合场景（附近餐厅）
  5. amap-route    —— 路线规划
  6. imaging       —— 未开工工单，演示 not_implemented 识别
"""
import asyncio  # 标准库：异步事件循环，用于 asyncio.wait_for（单场景超时控制）
import json  # 标准库：JSON 序列化，用于报告落盘和 tool 参数解析
import sys  # 标准库：sys.path（将 04-MCP 加入导入路径）、sys.stderr
import time  # 标准库：time.time() 计算各步耗时
from datetime import datetime  # 标准库：生成报告时间戳
from pathlib import Path  # 标准库：路径操作，用于报告文件路径

# 解决 Windows 中文终端编码问题（默认 GBK，可能无法显示中文特殊字符）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 将 04-MCP 根目录插入 Python 模块搜索路径，使 mcp_client 包可导入
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # python-dotenv：加载 .env
# openai 包：异步 OpenAI 兼容客户端，调用硅基流动 LLM
from openai import AsyncOpenAI

# 从路由 Agent 模块导入连接池、LLM 配置和系统提示词
from mcp_client.router_agent import MCPClientPool, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, SYSTEM_PROMPT

load_dotenv(_ROOT / ".env")  # 加载环境变量（确保 API Key 等已就绪）

# 报告输出目录
REPORTS_DIR = _ROOT / "tests" / "reports"


# 演示场景定义：(标题, 用户问题, 意图类别提示)
SCENARIOS = [
    ("挂号：号源查询",   "帮我查一下明天内科的号源", "registration"),
    ("健康咨询：症状",   "百日咳有什么症状", "knowledge"),
    ("地图：医院搜索",   "北京协和医院在哪里", "amap"),
    ("地图：附近餐饮",   "协和医院附近有什么好吃的餐厅", "amap"),
    ("地图：路线规划",   "从西直门到协和医院怎么走", "amap"),
    ("影像：未开工工单", "帮我看看这张 CT 片子有没有问题", "imaging"),
]

SCENARIO_TIMEOUT_S = 120  # 单场景最大等待时间（LLM 多轮推理最多 2 分钟）
CONSECUTIVE_FAIL_LIMIT = 2  # 连续异常次数达到此值时中断，疑似 API 故障


async def trace_one(client: AsyncOpenAI, pool: MCPClientPool, query: str, max_turns: int = 5) -> dict:
    """
    运行单个场景，抓取完整的 Agent 思考轨迹。

    Args:
        client: AsyncOpenAI 客户端实例
        pool: MCPClientPool 连接池
        query: 用户输入问题
        max_turns: 最大轮次

    Returns:
        {
          "query": str,          # 原始问题
          "turns": int,          # 实际轮次
          "trace": [...],        # 每轮轨迹（含 LLM 决策 + 工具调用）
          "final_reply": str,    # 最终回复
          "elapsed_s": float,    # 总耗时秒数
        }
    """
    # 初始化消息历史（系统提示 + 用户问题）
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n当前会话默认 user_id=1。"},
        {"role": "user", "content": query},
    ]
    trace: list[dict] = []  # 存储每轮的完整记录
    t_start = time.time()   # 记录开始时间
    final_reply = None

    for turn in range(max_turns):
        t_llm = time.time()  # 记录本轮 LLM 调用开始时间
        # 调用 LLM（Function Calling 模式，强制提供工具列表）
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=pool.openai_tools,   # 传入所有可用工具的 schema
            tool_choice="auto",         # auto: LLM 自行决定是否调工具
            temperature=0.3,
            max_tokens=1024,
        )
        llm_ms = int((time.time() - t_llm) * 1000)  # LLM 响应耗时（毫秒）
        msg = resp.choices[0].message

        # 构建本轮轨迹记录
        turn_record = {
            "turn": turn + 1,                     # 当前轮次编号（从 1 开始）
            "llm_ms": llm_ms,                     # LLM 耗时
            "assistant_content": msg.content,     # LLM 输出的文本（可能为 None）
            "tool_calls": [],                     # 本轮工具调用记录
        }

        # LLM 无工具调用 → 已生成最终回复，退出循环
        if not msg.tool_calls:
            final_reply = msg.content or "(空回复)"
            trace.append(turn_record)
            break

        # 将 assistant 消息（含工具决策）追加到消息历史
        # LLM 下一轮需要看到自己的决策才能正确综合工具结果
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls],
        })

        # 依次执行所有工具调用并记录轨迹
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}  # 解析失败兜底空参数
            t_tool = time.time()  # 记录工具调用开始时间
            raw = await pool.call(name, args)  # 实际调用 MCP 工具
            tool_ms = int((time.time() - t_tool) * 1000)  # 工具耗时（毫秒）
            try:
                raw_parsed = json.loads(raw)  # 尝试解析工具返回的 JSON
            except json.JSONDecodeError:
                raw_parsed = {"_raw": raw[:300]}  # 非 JSON 时截取原始文本

            # 记录本次工具调用的完整信息
            turn_record["tool_calls"].append({
                "tool_name": name,                                # 工具名
                "server": pool.tool_to_server.get(name, "?"),    # 所属 Server
                "args": args,                                     # 调用参数
                "elapsed_ms": tool_ms,                            # 工具耗时
                "result": raw_parsed,                             # 工具返回内容
            })
            # 将工具结果追加到消息历史供 LLM 综合
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": raw,
            })

        trace.append(turn_record)

    return {
        "query": query,
        "turns": len(trace),
        "trace": trace,
        "final_reply": final_reply or "(未收敛，超出 max_turns)",
        "elapsed_s": round(time.time() - t_start, 2),
    }


def _render_md(report: dict) -> str:
    """
    将演示报告 dict 渲染为 Markdown 格式字符串。

    Args:
        report: 包含 metadata 和 scenarios 列表的报告字典

    Returns:
        完整 Markdown 文本
    """
    md = report["metadata"]
    lines = [
        "# Agent 思考过程演示",
        "",
        "> 工单产出物 №2d：**展示 Agent 的思考过程及梳理流程**",
        "",
        f"- **时间**: {md['timestamp']}",
        f"- **LLM**: `{md['llm_model']}`",
        f"- **场景数**: {md['scenario_count']}",
        f"- **总耗时**: {md['total_elapsed_s']:.1f}s",
        "",
    ]
    for i, sc in enumerate(report["scenarios"], 1):
        lines += [
            f"## 场景 {i}：{sc['title']}（意图类型：`{sc['intent_hint']}`）",
            "",
            f"**用户 Query**：{sc['query']}",
            "",
            f"**总耗时**：{sc['trace_result']['elapsed_s']}s，轮次：{sc['trace_result']['turns']}",
            "",
            "### 思考轨迹",
            "",
        ]
        for tr in sc["trace_result"]["trace"]:
            lines.append(f"#### Turn {tr['turn']} · LLM 决策（{tr['llm_ms']}ms）")
            lines.append("")
            if tr["assistant_content"]:
                # 展示 LLM 的思考文本（如果有）
                lines.append("**LLM 思考文本**：")
                lines.append("")
                lines.append(f"> {tr['assistant_content']}")
                lines.append("")
            if tr["tool_calls"]:
                lines.append("**决定调用工具**：")
                lines.append("")
                for tc in tr["tool_calls"]:
                    lines.append(f"- **`{tc['tool_name']}`** (from `{tc['server']}`)，{tc['elapsed_ms']}ms")
                    lines.append(f"  - 参数：`{json.dumps(tc['args'], ensure_ascii=False)}`")
                    result_str = json.dumps(tc['result'], ensure_ascii=False)
                    # 截断过长的工具返回，避免报告过大
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "..."
                    lines.append(f"  - 返回：`{result_str}`")
                    lines.append("")
            else:
                # LLM 直接生成文本，未调用工具
                lines.append("**无工具调用（LLM 直出回复）**")
                lines.append("")
        lines += [
            "### 最终自然语言回复",
            "",
            "```text",
            sc["trace_result"]["final_reply"],
            "```",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def _append_history(report: dict):
    """
    将本次演示的摘要追加到 history.md，便于跨次运行对比。

    Args:
        report: 报告字典
    """
    hist = REPORTS_DIR / "history.md"
    md = report["metadata"]
    line = (f"\n[Agent思考演示] {md['timestamp']} · {md['scenario_count']} 场景 · "
            f"总耗时 {md['total_elapsed_s']:.1f}s\n")
    # 若 history.md 已存在则追加，否则创建新文件
    hist.write_text((hist.read_text(encoding="utf-8") if hist.exists() else "") + line, encoding="utf-8")


async def run_demo():
    """
    运行所有演示场景，生成报告。

    Returns:
        True 表示正常完成（含部分中止）；False 表示严重初始化失败
    """
    # 检查 LLM API Key 是否配置
    if not LLM_API_KEY:
        print("[ERR] SILICONFLOW_API_KEY 未配置", file=sys.stderr)
        return False

    async with MCPClientPool() as pool:
        # 检查 MCP 工具是否成功注册
        if not pool.openai_tools:
            print("[ERR] MCP Pool 未初始化", file=sys.stderr)
            return False

        client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        scenarios_out = []
        t_start = time.time()

        consecutive_errors = 0  # 连续失败计数（用于熔断）
        aborted = False
        for i, (title, query, intent_hint) in enumerate(SCENARIOS, 1):
            print(f"\n[{i}/{len(SCENARIOS)}] {title}: {query}", flush=True)
            is_error = False
            try:
                # 带超时限制地运行单个场景
                tr = await asyncio.wait_for(
                    trace_one(client, pool, query),
                    timeout=SCENARIO_TIMEOUT_S,
                )
                print(f"     → 轮次 {tr['turns']}, 耗时 {tr['elapsed_s']}s, 回复长度 {len(tr['final_reply'])}", flush=True)
                consecutive_errors = 0  # 成功则重置连续失败计数
            except asyncio.TimeoutError:
                # 场景超时，记录超时结果
                tr = {"query": query, "turns": 0, "trace": [], "final_reply": f"(超时 >{SCENARIO_TIMEOUT_S}s)", "elapsed_s": SCENARIO_TIMEOUT_S}
                print(f"     ✗ 超时 (>{SCENARIO_TIMEOUT_S}s)", flush=True)
                is_error = True
            except Exception as e:
                # 其他异常（API 错误等）
                tr = {"query": query, "turns": 0, "trace": [], "final_reply": f"(异常: {e})", "elapsed_s": 0}
                print(f"     ✗ 失败: {e}", flush=True)
                is_error = True

            if is_error:
                consecutive_errors += 1
                # 连续失败达到熔断阈值，疑似 API 或网络故障
                if consecutive_errors >= CONSECUTIVE_FAIL_LIMIT:
                    print(f"\n[ABORT] 连续 {consecutive_errors} 次异常，疑似 API 或 MCP 故障，中止演示。", flush=True)
                    print("        请检查：SILICONFLOW_API_KEY 是否有效、网络是否可达。", flush=True)
                    scenarios_out.append({
                        "title": title, "query": query, "intent_hint": intent_hint,
                        "trace_result": tr,
                    })
                    aborted = True
                    break

            scenarios_out.append({
                "title": title, "query": query, "intent_hint": intent_hint,
                "trace_result": tr,
            })

        total_elapsed = time.time() - t_start
        # 构建完整报告字典
        report = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "llm_model": LLM_MODEL,
                "scenario_count": len(scenarios_out),
                "total_elapsed_s": total_elapsed,
                "aborted": aborted,  # 标记是否因熔断中止
            },
            "scenarios": scenarios_out,
        }

        # 确保报告目录存在
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        # 时间戳作为文件名（去掉冒号/空格，避免文件名非法字符）
        ts_slug = report["metadata"]["timestamp"].replace(":", "").replace("-", "").replace(" ", "_")
        json_p = REPORTS_DIR / f"agent_thinking_{ts_slug}.json"
        md_p = REPORTS_DIR / f"agent_thinking_{ts_slug}.md"
        # 写入 JSON 格式报告（完整数据）
        json_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        # 写入 Markdown 格式报告（可读性好）
        md_p.write_text(_render_md(report), encoding="utf-8")
        # 追加到历史记录
        _append_history(report)

        # 打印最终摘要
        if aborted:
            print(f"\n{'='*60}")
            print(f"[ABORTED] 演示中止，已完成 {len(scenarios_out)}/{len(SCENARIOS)} 场景，报告已部分保存。")
        print(f"\n{'='*60}")
        print(f"Agent 思考过程演示完成：{len(scenarios_out)} 场景, 总耗时 {total_elapsed:.1f}s")
        print(f"[Report]")
        print(f"  {json_p}")
        print(f"  {md_p}")
        print(f"{'='*60}")
        return True


if __name__ == "__main__":
    # 运行演示，退出码 0=成功 1=失败
    ok = asyncio.run(run_demo())
    sys.exit(0 if ok else 1)
