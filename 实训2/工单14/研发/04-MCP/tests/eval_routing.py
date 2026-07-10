"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-MCP

路由精度评测
------------
针对 LLM function calling 架构的评测：
  1. 启动 MCPClientPool（拉起 5 个 MCP Server 收集 tool schema，不需要上游 HTTP 真的在跑）
  2. 对每个用例调 LLM 让它选工具（tool_choice=required），拿到 tool_name
  3. tool_name → 反查所属 server，与期望 server 比对
  4. 输出精度报告，命中 >= 80% 视为通过（工单验收线）

用例分布（共 20 条）：
  registration 5 / knowledge 5 / amap 5 / imaging 3 / voice 2
"""
import asyncio  # 标准库：异步事件循环
import json  # 标准库：JSON 序列化，用于报告落盘
import os  # 标准库：os.path（本文件未直接使用，保留以备扩展）
import sys  # 标准库：sys.path 路径注入
import time  # 标准库：精确计时
from datetime import datetime  # 标准库：生成报告时间戳
from pathlib import Path  # 标准库：路径操作

# Windows 中文终端默认 GBK，强制 UTF-8 输出，避免中文 query 乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 将 04-MCP/ 根目录加入 Python 模块搜索路径，使 tests/ 可导入 mcp_client 包
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # python-dotenv：加载 .env 配置

# openai 包：AsyncOpenAI 是异步 OpenAI 兼容客户端，调用硅基流动 LLM
from openai import AsyncOpenAI

# 导入路由 Agent 核心组件
from mcp_client.router_agent import MCPClientPool, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, SYSTEM_PROMPT

load_dotenv(_ROOT / ".env")  # 加载 API Key 等环境变量

REPORTS_DIR = _ROOT / "tests" / "reports"  # 报告输出目录

# ── 20 条评测用例 ──
# 格式：(query, expected_server, note)
#   expected_server: 期望由哪个 MCP Server 处理
#   note: 场景说明（用于失败明细）
TEST_CASES = [
    # ─── 挂号（registration）5 条 ───
    ("帮我查一下明天内科的号源",    "registration", "号源查询"),
    ("我想预约张医生下周三的号",    "registration", "预约"),
    ("取消我今天的挂号",            "registration", "取消预约"),
    ("查一下我的挂号记录",          "registration", "预约列表"),
    ("刘医生这周有哪些排班",        "registration", "医生排班"),

    # ─── 健康咨询（knowledge）5 条 ───
    ("百日咳有什么症状",            "knowledge", "症状咨询"),
    ("糖尿病该挂什么科",            "knowledge", "科室推荐"),
    ("高血压能吃什么药",            "knowledge", "药品用法"),
    ("肠胃炎有什么忌口",            "knowledge", "饮食建议"),
    ("肺炎会有什么并发症",          "knowledge", "疾病百科"),

    # ─── 地图/出行（amap）5 条 ───
    ("北京协和医院附近有什么酒店",  "amap", "附近酒店"),
    ("从西直门到协和医院怎么走",    "amap", "路线规划"),
    ("儿童医院附近的餐厅推荐",      "amap", "附近餐饮"),
    ("协和医院附近哪里有药店",      "amap", "附近药店"),
    ("北京大学人民医院在哪",        "amap", "医院位置"),

    # ─── 影像分析（imaging，未开工）3 条 ───
    ("帮我看看这张 CT 片子有没有问题", "imaging", "影像分析"),
    ("生成这张胸片的诊断报告",         "imaging", "报告生成"),
    ("这张 X 光片显示什么",            "imaging", "影像 VQA"),

    # ─── 语音识别（voice，未开工）2 条 ───
    ("把这段录音转成文字",             "voice", "语音转文字"),
    ("识别一下这个音频的内容",         "voice", "语音识别"),
]


async def eval_one(client: AsyncOpenAI, pool: MCPClientPool, query: str) -> str | None:
    """
    让 LLM 从 MCP tool 列表中选择一个最匹配的工具（Function Calling 模式）。

    使用 tool_choice="required" 强制 LLM 必须选一个工具，
    temperature=0.0 确保评测结果可重复。

    Args:
        client: AsyncOpenAI 客户端实例（调用硅基流动 LLM）
        pool: MCPClientPool 连接池（提供 openai_tools schema）
        query: 用户输入问题

    Returns:
        LLM 选中的 tool_name；未选任何工具（理论上不会发生）时返回 None
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},  # 路由系统提示词
        {"role": "user", "content": query},
    ]
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=pool.openai_tools,   # 传入所有 MCP 工具的 JSON Schema
        tool_choice="required",     # 强制必须选一个 tool（不允许直出文本）
        temperature=0.0,            # 温度 0 确保路由决策确定性，减少随机偏差
        max_tokens=512,             # 路由决策只需少量 token
    )
    tcs = resp.choices[0].message.tool_calls
    if not tcs:
        return None  # 理论上 tool_choice=required 不会到这里
    return tcs[0].function.name  # 返回 LLM 选中的第一个工具名


def _render_md(report: dict) -> str:
    """
    将路由精度报告 dict 渲染为 Markdown 字符串。

    Args:
        report: 包含 metadata 和 results 列表的报告字典

    Returns:
        完整 Markdown 文本
    """
    md = report["metadata"]
    rows = report["results"]
    lines = [
        "# 路由精度评测报告",
        "",
        f"- **时间**: {md['timestamp']}",
        f"- **LLM**: `{md['llm_model']}` @ `{md['llm_base_url']}`",
        f"- **MCP Tool 池**: {md['tool_count']} 个 tool，来自 {md['server_count']} 个 Server",
        f"- **总耗时**: {md['elapsed_s']:.1f}s（平均每条 {md['avg_ms_per_case']}ms）",
        "",
        "## 汇总",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 总用例 | {md['total']} |",
        f"| 命中 | {md['correct']} |",
        f"| 精度 | **{md['accuracy']:.1f}%** |",
        "| 验收线 | 80% |",
        f"| 判定 | {'✅ PASS' if md['pass'] else '❌ FAIL'} |",
        "",
        "## 逐条结果",
        "",
        "| # | 期望 | 实际 | Tool | 命中 | Query |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(rows, 1):
        mark = "✅" if r["hit"] else "❌"
        lines.append(
            f"| {i} | {r['expected']} | {r['got_server']} | "
            f"`{r['tool_name']}` | {mark} | {r['query']} |"
        )

    # 失败明细：列出未命中的用例及实际路由结果
    errs = [r for r in rows if not r["hit"]]
    if errs:
        lines += ["", f"## 失败明细（{len(errs)} 条）", ""]
        for r in errs:
            lines.append(f"- **[{r['note']}]** `{r['query']}`")
            lines.append(
                f"  - 期望 server=`{r['expected']}`，"
                f"实际 tool=`{r['tool_name']}` → server=`{r['got_server']}`"
            )
    return "\n".join(lines) + "\n"


def _append_history(report: dict):
    """
    向 history.md 追加一行路由精度汇总，方便对比多次运行趋势。
    首次调用时自动创建文件并写入 Markdown 表头。

    Args:
        report: 报告字典
    """
    hist = REPORTS_DIR / "history.md"
    md = report["metadata"]
    if not hist.exists():
        # 首次运行：创建文件并写入表格表头
        hist.write_text(
            "# 路由精度评测 · 历史记录\n\n"
            "| 时间 | LLM | 精度 | 命中/总数 | 判定 | 耗时 |\n"
            "| --- | --- | --- | --- | --- | --- |\n",
            encoding="utf-8",
        )
    # 追加模式打开，写入本次评测的汇总行
    with hist.open("a", encoding="utf-8") as f:
        f.write(
            f"| {md['timestamp']} | {md['llm_model']} | {md['accuracy']:.1f}% "
            f"| {md['correct']}/{md['total']} | {'PASS' if md['pass'] else 'FAIL'} "
            f"| {md['elapsed_s']:.1f}s |\n"
        )


def _save_reports(report: dict) -> tuple[Path, Path]:
    """
    将报告保存为 JSON 和 Markdown 文件，并追加历史记录。

    Args:
        report: 报告字典

    Returns:
        (json_path, md_path) 两个报告文件路径
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # 时间戳作为文件名（去除冒号/空格）
    ts_slug = report["metadata"]["timestamp"].replace(":", "").replace("-", "").replace(" ", "_")
    json_path = REPORTS_DIR / f"routing_eval_{ts_slug}.json"
    md_path = REPORTS_DIR / f"routing_eval_{ts_slug}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")
    _append_history(report)
    return json_path, md_path


async def run_eval():
    """
    路由精度评测主函数：
      1. 初始化 MCPClientPool（仅需 MCP Server 子进程在线，不需要上游 HTTP 服务）
      2. 逐条调用 LLM 让其从 tool 列表中挑选最匹配的工具
      3. 反查工具所属 server 并与期望 server 比对
      4. 生成报告并打印摘要

    Returns:
        True（精度 ≥ 80%，验收通过）；False（未达标或初始化失败）
    """
    if not LLM_API_KEY:
        print("[ERR] SILICONFLOW_API_KEY 未配置，无法评测", file=sys.stderr)
        return False

    t_start = time.time()
    async with MCPClientPool() as pool:
        # 检查 MCP 工具是否成功注册
        if not pool.openai_tools:
            print("[ERR] MCP Pool 初始化后无可用 tool", file=sys.stderr)
            return False
        print(
            f"\n[INIT] Pool 已加载 {len(pool.openai_tools)} 个 tool，"
            f"来自 {len(set(pool.tool_to_server.values()))} 个 Server\n"
        )

        client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        total = len(TEST_CASES)
        correct = 0
        results: list[dict] = []

        for i, (query, expected, note) in enumerate(TEST_CASES, 1):
            t_case = time.time()
            try:
                # 让 LLM 选工具（tool_choice=required，温度 0）
                tool_name = await eval_one(client, pool, query)
                # 通过 tool_to_server 反查该 tool 属于哪个 MCP Server
                got_server = pool.tool_to_server.get(tool_name, "unknown")
                err_msg = None
            except Exception as e:
                # API 异常：工具名/server 均标记为 error
                tool_name, got_server = None, "error"
                err_msg = str(e)

            hit = got_server == expected  # 实际 server == 期望 server 则命中
            if hit:
                correct += 1
            results.append({
                "query": query, "expected": expected, "note": note,
                "tool_name": tool_name, "got_server": got_server,
                "hit": hit, "error": err_msg,
                "elapsed_ms": int((time.time() - t_case) * 1000),  # 本条耗时（毫秒）
            })
            mark = "✓" if hit else "✗"
            print(f"[{i:2d}/{total}] {mark} {expected:12s} → {got_server:12s} tool={tool_name}  ｜ {query}")

        elapsed = time.time() - t_start
        acc = correct / total * 100  # 整体路由精度（百分比）
        print(f"\n{'='*60}")
        print(f"路由精度: {correct}/{total} = {acc:.1f}%")
        print(f"验收线 80%: {'PASS ✓' if acc >= 80 else 'FAIL ✗'}")
        print(f"总耗时: {elapsed:.1f}s")
        print(f"{'='*60}")

        # 打印失败用例的详细信息（便于快速定位路由偏差）
        errors = [r for r in results if not r["hit"]]
        if errors:
            print(f"\n错误明细（{len(errors)} 条）:")
            for r in errors:
                print(f"  [{r['note']}] {r['query']}")
                print(f"    期望 server={r['expected']}, 实际 tool={r['tool_name']} → server={r['got_server']}")

        # 构建报告 dict 并落盘
        report = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "llm_model": LLM_MODEL,
                "llm_base_url": LLM_BASE_URL,
                "tool_count": len(pool.openai_tools),                         # 总 tool 数量
                "server_count": len(set(pool.tool_to_server.values())),       # MCP Server 数量
                "total": total,
                "correct": correct,
                "accuracy": acc,
                "pass": acc >= 80,  # 是否通过验收线
                "elapsed_s": elapsed,
                "avg_ms_per_case": int(elapsed * 1000 / total),  # 平均每条耗时（毫秒）
            },
            "results": results,
        }
        json_p, md_p = _save_reports(report)
        print(f"\n[Report] 已保存:")
        print(f"  {json_p}")
        print(f"  {md_p}")
        print(f"  {REPORTS_DIR / 'history.md'}  (追加)")

        return acc >= 80  # 精度 >= 80% 视为通过


if __name__ == "__main__":
    ok = asyncio.run(run_eval())
    sys.exit(0 if ok else 1)
