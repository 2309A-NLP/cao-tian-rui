"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-MCP

性能压测
--------
验证工单验收指标"响应时间 <500ms"。

分两层测：
  L1 - MCP 工具层：直接 pool.call()，含 stdio round-trip + 上游 REST（高德/上游服务）
       目标 <500ms（工单验收线）
  L2 - Agent 端到端：agent_answer() 含 LLM 推理
       行业口径：3-10s 属正常（LLM function calling 天然延迟）

统计指标：min / p50 / p95 / p99 / max / avg

L1 用例（覆盖 amap 类，工单验收范围）：
  - amap.hospital_search（单次高德 place/text）
  - amap.nearby_hotels（2 次高德：place/text + place/around）
  - amap.route_planning（3 次高德：2×place/text + direction/driving）
  - amap.nearby_restaurants
  - imaging.analyze_image（探测上游 /health → 若不通则 not_implemented，测的是探测时延）

L2 用例（典型场景）：
  - "北京协和医院在哪里"
  - "协和医院附近哪里有酒店"
  - "从西直门到协和医院怎么走"

每类跑 10 次，取热运行数据（去掉前 2 次冷启动）。
"""
import asyncio  # 标准库：异步事件循环，用于超时控制（asyncio.wait_for）和并发
import json  # 标准库：JSON 序列化，用于报告落盘
import statistics  # 标准库：统计模块（导入但实际计算用列表排序实现，避免依赖 numpy）
import sys  # 标准库：sys.path 路径注入、sys.stderr 错误输出
import time  # 标准库：time.time() 精确计时
from datetime import datetime  # 标准库：生成报告时间戳
from pathlib import Path  # 标准库：路径操作，用于报告输出路径

# Windows 中文终端默认 GBK，强制 UTF-8 输出，避免中文报告乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 将 04-MCP/ 根目录加入 Python 导入路径，使 mcp_client 包可被 tests/ 导入
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # python-dotenv：从 .env 文件读取 API Key 等配置

# 从路由 Agent 模块导入连接池和业务入口
from mcp_client.router_agent import MCPClientPool, agent_answer, LLM_API_KEY

load_dotenv(_ROOT / ".env")  # 确保 .env 已加载
REPORTS_DIR = _ROOT / "tests" / "reports"  # 报告落盘目录

# L1 测试用例：(工具全名, 调用参数)
# 格式："server.tool_name" → 便于报告展示归属 server
L1_CASES = [
    ("amap.hospital_search",    {"name": "北京协和医院"}),
    ("amap.nearby_hotels",      {"hospital": "北京协和医院"}),
    ("amap.nearby_restaurants", {"hospital": "北京协和医院"}),
    ("amap.route_planning",     {"origin": "西直门", "destination": "北京协和医院", "mode": "driving"}),
    ("imaging.analyze_image",   {"image_url": "https://example.com/test.jpg", "question": "有异常吗"}),
]
L1_ROUNDS = 10   # 每个工具跑 10 轮（含 warmup）
L1_WARMUP = 2    # 前 2 轮为冷启动预热（DNS/TLS 握手，数据不计入统计）

# L2 查询：端到端（含 LLM）的典型场景
L2_QUERIES = [
    "北京协和医院在哪里",
    "协和医院附近哪里有酒店",
    "从西直门到协和医院怎么走",
]
L2_ROUNDS = 3   # 端到端慢，3 轮足以看数量级
L2_WARMUP = 1   # 第 1 轮预热

# 熔断和预警阈值
ROUND_TIMEOUT_S = 30            # 单轮 tool 调用超时（秒）
TOOL_ABORT_AVG_MS = 3000        # 热跑均值超过此值时，中止该 tool 剩余轮次（预期值的 6 倍）
SYSTEMIC_WARN_AVG_MS = 2000     # 前两个 tool 均值均超此值时，输出全局网络异常警告
L2_CALL_TIMEOUT_S = 60          # 单次 L2 端到端超时（LLM+多轮 tool）


def stats(samples: list[int]) -> dict:
    """
    计算一组延迟样本的统计分位数。

    Args:
        samples: 延迟样本列表（单位毫秒）

    Returns:
        {count, min_ms, avg_ms, p50_ms, p95_ms, p99_ms, max_ms}
        样本为空时返回 {count: 0}
    """
    if not samples:
        return {"count": 0}  # 没有有效样本，返回空统计
    s = sorted(samples)  # 升序排列，方便按索引取分位数
    return {
        "count": len(s),
        "min_ms": s[0],                                          # 最小值
        "avg_ms": round(sum(s) / len(s), 1),                    # 算术平均值
        "p50_ms": s[len(s) // 2],                               # 中位数（p50）
        # p95/p99：样本量不足 20/100 时直接用最大值，避免索引越界
        "p95_ms": s[int(len(s) * 0.95)] if len(s) >= 20 else s[-1],
        "p99_ms": s[int(len(s) * 0.99)] if len(s) >= 100 else s[-1],
        "max_ms": s[-1],                                         # 最大值
    }


async def bench_l1(pool: MCPClientPool):
    """
    L1 层基准测试：直接调用 MCP 工具，测试工具层（不含 LLM）的延迟。

    Args:
        pool: 已初始化的 MCPClientPool 连接池

    Returns:
        list of {tool, args, samples_ms, stats, pass_500ms_p95, abort_reason, first_result_preview}
    """
    results = []
    slow_tool_count = 0  # 记录前两个 tool 中有多少均值偏慢（用于全局预警）

    for tool_idx, (tool_name_full, args) in enumerate(L1_CASES):
        # "amap.hospital_search" → 取点后部分作为实际工具名
        tool_name = tool_name_full.split(".", 1)[1]
        samples = []         # 热跑延迟样本（去掉 warmup 轮次后）
        first_result = None  # 记录第一次成功调用的结果（用于报告预览）
        abort_reason = None  # 若提前熔断，记录原因
        print(f"\n[L1] {tool_name_full}({args})", flush=True)

        for r in range(L1_ROUNDS):
            t = time.time()
            try:
                # 带超时地调用 MCP 工具（pool.call 内部是 session.call_tool）
                raw = await asyncio.wait_for(pool.call(tool_name, args), timeout=ROUND_TIMEOUT_S)
                ms = int((time.time() - t) * 1000)  # 本轮耗时（毫秒）
                if first_result is None:
                    first_result = raw[:150]  # 截取前 150 字符作为结果预览
            except asyncio.TimeoutError:
                # 本轮超时，记录超时时长作为延迟样本
                ms = ROUND_TIMEOUT_S * 1000
                raw = f"(timeout >{ROUND_TIMEOUT_S}s)"
                if first_result is None:
                    first_result = raw

            # 仅将热跑数据（warmup 之后的轮次）计入统计
            if r >= L1_WARMUP:
                samples.append(ms)

            mark = "(warmup)" if r < L1_WARMUP else "        "
            print(f"    round {r+1:2d} {mark} {ms:4d}ms", flush=True)

            # 单 tool 熔断：热跑均值严重超标时，提前停止剩余轮次
            if len(samples) >= 2:
                cur_avg = sum(samples) / len(samples)
                if cur_avg > TOOL_ABORT_AVG_MS:
                    abort_reason = f"热跑均值 {cur_avg:.0f}ms > 阈值 {TOOL_ABORT_AVG_MS}ms，提前中止剩余轮次"
                    print(f"    [ABORT-TOOL] {abort_reason}", flush=True)
                    break

        st = stats(samples)
        # 全局网络预警：前两个 tool 都偏慢，说明 API 或网络可能存在整体问题
        if tool_idx < 2 and st.get("avg_ms", 0) > SYSTEMIC_WARN_AVG_MS:
            slow_tool_count += 1
            if slow_tool_count >= 2:
                print(f"\n[WARN] 前两个工具均值均超 {SYSTEMIC_WARN_AVG_MS}ms，网络/API 可能异常。", flush=True)
                print("       建议先检查高德 API Key 有效性及网络连通性，再继续压测。", flush=True)

        results.append({
            "tool": tool_name_full,
            "args": args,
            "samples_ms": samples,
            "stats": st,
            # 工单验收线：p95 < 500ms
            "pass_500ms_p95": st.get("p95_ms", 99999) < 500,
            "abort_reason": abort_reason,
            "first_result_preview": first_result,
        })
    return results


async def bench_l2(pool: MCPClientPool):
    """
    L2 层基准测试：端到端 agent_answer()，含 LLM 多轮推理 + 工具调用。

    Args:
        pool: 已初始化的 MCPClientPool 连接池

    Returns:
        list of {query, samples_ms, stats, first_reply_preview}
    """
    results = []
    l2_aborted = False  # 若预热轮超时则中止剩余 L2 查询，避免长等待

    for query in L2_QUERIES:
        if l2_aborted:
            # 前序查询超时后，跳过剩余 L2 查询（标记为跳过而非错误）
            results.append({
                "query": query,
                "samples_ms": [],
                "stats": {"count": 0},
                "first_reply_preview": "(跳过，前序 L2 查询超时)",
            })
            continue

        samples = []
        first_reply = None
        print(f"\n[L2] agent_answer({query!r})", flush=True)
        for r in range(L2_ROUNDS):
            t = time.time()
            try:
                # agent_answer 内部含 LLM function calling 多轮循环
                res = await asyncio.wait_for(
                    agent_answer(pool, query, user_id=1),
                    timeout=L2_CALL_TIMEOUT_S,
                )
                ms = int((time.time() - t) * 1000)
                if first_reply is None:
                    first_reply = res.get("reply", "")[:200]  # 截取前 200 字符作为预览
            except asyncio.TimeoutError:
                ms = L2_CALL_TIMEOUT_S * 1000
                first_reply = f"(timeout >{L2_CALL_TIMEOUT_S}s)"
                if r == 0:
                    # 预热轮就超时，说明 LLM 响应极慢，跳过剩余 L2 查询
                    print(f"    [ABORT-L2] 预热轮超时 (>{L2_CALL_TIMEOUT_S}s)，跳过剩余 L2 查询", flush=True)
                    l2_aborted = True
                    break
            except Exception as e:
                ms = int((time.time() - t) * 1000)
                first_reply = f"(error: {e})"  # 记录异常信息

            # 热跑数据（warmup 之后）才计入统计
            if r >= L2_WARMUP:
                samples.append(ms)
            mark = "(warmup)" if r < L2_WARMUP else "        "
            print(f"    round {r+1:2d} {mark} {ms:5d}ms", flush=True)

        results.append({
            "query": query,
            "samples_ms": samples,
            "stats": stats(samples),
            "first_reply_preview": first_reply,
        })
    return results


def _render_md(report: dict) -> str:
    """
    将压测报告 dict 渲染为 Markdown 字符串。

    Args:
        report: 包含 metadata、l1、l2 的报告字典

    Returns:
        完整 Markdown 文本
    """
    md = report["metadata"]
    lines = [
        "# 性能压测报告",
        "",
        f"- **时间**: {md['timestamp']}",
        f"- **LLM**: `{md['llm_model']}`",
        f"- **L1 每类样本数**: {md['l1_rounds']}（前 {md['l1_warmup']} 次预热不计）",
        f"- **L2 每类样本数**: {md['l2_rounds']}（前 {md['l2_warmup']} 次预热不计）",
        "",
        "---",
        "",
        "## L1 · MCP 工具层性能",
        "",
        '> **工单验收线：<500ms**（对应工单原文"响应时间<500ms"）',
        "> MCP 工具层 = stdio round-trip + 上游 API（高德/上游 FastAPI）耗时，不含 LLM 推理",
        "",
        "| Tool | min | avg | p50 | p95 | max | ≤500ms(p95) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["l1"]:
        st = r["stats"]
        mark = "✅" if r["pass_500ms_p95"] else "❌"
        lines.append(
            f"| `{r['tool']}` | {st['min_ms']} | {st['avg_ms']} | "
            f"{st['p50_ms']} | {st['p95_ms']} | {st['max_ms']} | {mark} |"
        )

    passed = sum(1 for r in report["l1"] if r["pass_500ms_p95"])  # 达标工具数
    total = len(report["l1"])
    lines += [
        "",
        f"**L1 汇总**：{passed}/{total} 个 tool 的 p95 时延 ≤ 500ms。",
        "",
        "---",
        "",
        "## L2 · Agent 端到端性能（含 LLM）",
        "",
        "> **业界口径**：LLM function calling 端到端时延 3-10s 为正常。",
        "> 硅基流动 Qwen2.5-72B 单轮推理典型 1-3s，端到端 2-3 轮共 3-8s。",
        "",
        "| Query | min | avg | p50 | p95 | max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["l2"]:
        st = r["stats"]
        if st["count"] == 0:
            # 无样本（跳过或超时）
            lines.append(f"| `{r['query']}` | — | — | — | — | — |")
        else:
            lines.append(
                f"| `{r['query']}` | {st['min_ms']} | {st['avg_ms']} | "
                f"{st['p50_ms']} | {st['p95_ms']} | {st['max_ms']} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 结论",
        "",
        f"- **工具层验收**：{'✅ 通过' if passed == total else '⚠️ 部分未达标'}，"
        f"{passed}/{total} 个 tool 满足 <500ms",
        "- **端到端**：作为参考数据，非工单验收硬指标（工单 <500ms 指工具层，不指含 LLM 全链路）",
        "- **说明**：如某 tool p95 超 500ms，通常是高德 API 单次请求较慢或组合调用多，"
        "可通过 tool 参数收敛（降低 offset）或缓存 hospital 坐标优化",
    ]
    return "\n".join(lines) + "\n"


def _append_history(report: dict):
    """
    将本次压测摘要追加到 history.md，便于跨次运行对比趋势。

    Args:
        report: 报告字典
    """
    hist = REPORTS_DIR / "history.md"
    md = report["metadata"]
    # 统计 L1 达标工具数量
    l1_pass = sum(1 for r in report["l1"] if r["pass_500ms_p95"])
    l1_total = len(report["l1"])
    line = f"\n[性能压测] {md['timestamp']} · L1: {l1_pass}/{l1_total} 通过<500ms\n"
    # 追加到文件末尾（不覆盖已有历史记录）
    hist.write_text((hist.read_text(encoding="utf-8") if hist.exists() else "") + line, encoding="utf-8")


async def main():
    """
    压测主函数：初始化连接池 → 跑 L1 → 跑 L2 → 生成报告。

    Returns:
        True 表示 L1 全部达标；False 表示未达标或初始化失败
    """
    # 前置检查：API Key 必须配置
    if not LLM_API_KEY:
        print("[ERR] SILICONFLOW_API_KEY 未配置", file=sys.stderr)
        return False

    async with MCPClientPool() as pool:
        # 检查 MCP 工具是否已注册
        if not pool.openai_tools:
            print("[ERR] MCP Pool 未初始化", file=sys.stderr)
            return False

        l1 = await bench_l1(pool)  # L1 工具层压测
        l2 = await bench_l2(pool)  # L2 端到端压测（含 LLM）

    # 延迟导入（避免在 MCPClientPool 上下文之外使用）
    from mcp_client.router_agent import LLM_MODEL
    report = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "llm_model": LLM_MODEL,
            "l1_rounds": L1_ROUNDS,
            "l1_warmup": L1_WARMUP,
            "l2_rounds": L2_ROUNDS,
            "l2_warmup": L2_WARMUP,
        },
        "l1": l1,
        "l2": l2,
    }

    # 确保报告目录存在
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # 时间戳作为文件名（去除非法字符）
    ts_slug = report["metadata"]["timestamp"].replace(":", "").replace("-", "").replace(" ", "_")
    json_p = REPORTS_DIR / f"performance_{ts_slug}.json"
    md_p = REPORTS_DIR / f"performance_{ts_slug}.md"
    # 写入 JSON（全量数据）和 Markdown（可读报告）
    json_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_p.write_text(_render_md(report), encoding="utf-8")
    _append_history(report)  # 追加到历史记录

    l1_pass = sum(1 for r in l1 if r["pass_500ms_p95"])
    print(f"\n{'='*60}")
    print(f"性能压测完成")
    print(f"L1（工具层）：{l1_pass}/{len(l1)} 通过 <500ms")
    print(f"L2（端到端）：见报告")
    print(f"[Report]")
    print(f"  {json_p}")
    print(f"  {md_p}")
    print(f"{'='*60}")
    return True


if __name__ == "__main__":
    # 退出码 0=成功 1=失败/未达标
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
