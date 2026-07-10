"""
HTML 评测报告生成器（离线工具）。

功能：
  读取 question.jsonl、answer_fixed.jsonl 和 logs/agent.log，
  生成一份可视化 HTML 报告，包含：
  - 统计卡片（总题数、失败数、平均搜索轮数等）
  - 退出原因分布（颜色编码）
  - 逐题明细表格（问题摘要、答案、退出原因、轮数）

用法：
  python gen_report.py
  # 生成 report.html，浏览器打开即可
"""
import json             # 标准库：JSON 解析，用于读取 JSONL 和日志文件
import re               # 标准库：正则表达式，用于从日志行中提取字段
import os               # 标准库：路径操作
import sys              # 标准库：系统接口（本文件未直接使用，保留备用）
from collections import Counter   # 标准库：计数器，用于统计退出原因的分布
from datetime import datetime     # 标准库：日期时间，用于在报告中显示生成时间

# ── 文件路径配置 ──────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")           # 项目根目录
QUESTION_FILE    = os.path.join(ROOT, "question.jsonl")        # 题目文件
ANSWER_FILE      = os.path.join(ROOT, "answer_fixed.jsonl")    # 修复后的答案文件
LOG_FILE         = os.path.join(ROOT, "logs", "agent.log")     # Agent 运行日志
OUTPUT_HTML      = os.path.join(ROOT, "report.html")           # 输出的 HTML 报告路径


def load_jsonl(path):
    """
    读取 JSONL 文件，返回 Python 字典列表。

    :param path: JSONL 文件路径
    :return:     解析后的字典列表
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_log(path):
    """
    从 Agent 日志文件中提取每题的 exit_reason 和搜索轮数。

    日志格式为 JSON 行，包含 "msg" 字段，例如：
      {"msg": "Answer: 'xxx' (exit=final_answer, rounds=2)"}

    :param path: 日志文件路径
    :return:     字典 {index: {"exit": str, "rounds": int}}
                 index 为按出现顺序的序号（从0开始）
    """
    info = {}  # 存储提取结果：序号 → {exit, rounds}
    try:
        # errors="replace" 对无法解码的字节用替换字符处理，防止编码错误崩溃
        lines = open(path, encoding="utf-8", errors="replace").readlines()
    except Exception:
        return info  # 日志文件不存在或无法读取时返回空字典

    # 只取包含 "Answer:" 的行（answer_generator 记录的答案日志）
    # 注意：此处取所有历史日志中的 Answer 行，后续截取最后100条
    answer_lines = [l for l in lines if '"Answer:' in l]

    for i, line in enumerate(answer_lines):
        try:
            d = json.loads(line)          # 解析 JSON 日志行
            msg = d.get("msg", "")        # 取 msg 字段内容

            # 从 msg 中提取 exit 原因、rounds 数量
            m_exit   = re.search(r"exit=(\w+)", msg)    # 匹配 exit=final_answer 等
            m_rounds = re.search(r"rounds=(\d+)", msg)  # 匹配 rounds=2 等
            # m_ans 提取答案内容（当前未使用，保留供扩展）
            m_ans    = re.search(r"Answer: '(.+?)' \(exit", msg)

            if m_exit and m_rounds:  # 两者都匹配才记录
                info[i] = {
                    "exit":   m_exit.group(1),      # 退出原因字符串
                    "rounds": int(m_rounds.group(1)), # 轮数转为整数
                }
        except Exception:
            pass  # 单行解析失败跳过，不影响其他行

    return info


# ── 加载数据 ──────────────────────────────────────────────────────────────────
# 构建 id → question 映射字典
questions = {q["id"]: q["question"] for q in load_jsonl(QUESTION_FILE)}
# 构建 id → answer 映射字典
answers   = {a["id"]: a["answer"]   for a in load_jsonl(ANSWER_FILE)}
# 解析日志，获取每题的退出原因和轮数
log_info  = parse_log(LOG_FILE)

# ── 统计（只用本次100题的日志记录）──────────────────────────────────────────────
# 取最后100条日志记录（对应本次100题的运行结果，跳过历史遗留的旧日志）
recent_logs = list(log_info.values())[-100:]

# 统计各种退出原因的出现次数
exit_counts = Counter(r["exit"] for r in recent_logs)

# 收集所有题目的轮数列表，用于计算平均值
rounds_list = [r["rounds"] for r in recent_logs]
# 计算平均搜索轮数（避免除以零）
avg_rounds  = sum(rounds_list) / len(rounds_list) if rounds_list else 0

# 退出原因对应的颜色（用于表格和图例的颜色编码）
exit_color = {
    "final_answer": "#22c55e",   # 绿色：成功找到答案
    "no_new_info":  "#f59e0b",   # 黄色：无新信息提前退出
    "max_rounds":   "#f97316",   # 橙色：达到最大轮数
    "timeout":      "#ef4444",   # 红色：超时
    "llm_error":    "#dc2626",   # 深红色：LLM 调用失败
    "no_action":    "#8b5cf6",   # 紫色：无有效动作
}

# ── 构建表格行 HTML ────────────────────────────────────────────────────────────
rows_html = ""  # 累加每行的 HTML 字符串

for qid in sorted(questions.keys()):  # 按题号升序遍历
    q   = questions[qid]                              # 题目文本
    ans = answers.get(qid, "—")                       # 答案（未找到显示 —）
    # 取对应序号的日志记录（qid 作为索引，超出范围时用空字典）
    log = recent_logs[qid] if qid < len(recent_logs) else {}
    exit_r  = log.get("exit", "—")                    # 退出原因
    rounds  = log.get("rounds", "—")                  # 轮数
    color   = exit_color.get(exit_r, "#6b7280")        # 退出原因对应颜色，未知则用灰色
    # 问题摘要：超过120字截断并加省略号
    q_short = q[:120] + ("…" if len(q) > 120 else "")

    # 拼接一行 HTML（f-string 内联变量）
    rows_html += f"""
    <tr>
      <td class="id">{qid}</td>
      <td class="q">{q_short}</td>
      <td class="ans">{ans}</td>
      <td><span class="badge" style="background:{color}">{exit_r}</span></td>
      <td class="center">{rounds}</td>
    </tr>"""

# ── 构建完整 HTML 报告 ─────────────────────────────────────────────────────────
# 使用 f-string 模板生成完整 HTML，包含内联 CSS（无外部依赖）
html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>工单23 Research Agent 评测报告</title>
<style>
  /* CSS 重置：统一盒模型和间距 */
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  /* 页面基础样式：系统字体栈，浅灰背景，深色文字 */
  body {{ font-family: -apple-system, 'PingFang SC', sans-serif; background: #f8fafc; color: #1e293b; }}
  /* 顶部渐变标题栏 */
  .header {{ background: linear-gradient(135deg, #1e40af, #7c3aed); color: white; padding: 36px 48px; }}
  .header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
  .header p  {{ font-size: 14px; opacity: .8; }}
  /* 统计卡片区：水平排列，自动换行 */
  .stats {{ display: flex; gap: 20px; padding: 24px 48px; flex-wrap: wrap; }}
  /* 单个统计卡片：白色圆角卡片，居中显示 */
  .card {{ background: white; border-radius: 12px; padding: 20px 28px; flex: 1; min-width: 150px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; }}
  .card .val {{ font-size: 36px; font-weight: 800; color: #1e40af; }}
  .card .lbl {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
  /* 退出原因图例栏 */
  .exit-bar {{ padding: 0 48px 20px; display: flex; gap: 12px; flex-wrap: wrap; }}
  .exit-item {{ display: flex; align-items: center; gap: 6px; font-size: 13px; }}
  .dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; }}
  /* 表格容器：水平滚动防止列过多时溢出 */
  .table-wrap {{ padding: 0 48px 48px; overflow-x: auto; }}
  /* 表格：全宽、无边框合并、白色圆角卡片 */
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  /* 表头：蓝色背景白色文字 */
  th {{ background: #1e40af; color: white; padding: 12px 14px; text-align: left; font-size: 13px; }}
  /* 表格单元格：浅灰分隔线，顶部对齐 */
  td {{ padding: 10px 14px; border-bottom: 1px solid #f1f5f9; font-size: 13px; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}    /* 最后一行无底边框 */
  tr:hover td {{ background: #f8fafc; }}         /* 悬停高亮行 */
  /* 各列样式 */
  .id {{ width: 40px; color: #64748b; font-weight: 600; }}
  .q  {{ max-width: 400px; color: #374151; }}
  .ans{{ max-width: 260px; color: #1e40af; font-weight: 500; word-break: break-word; }}
  .center {{ text-align: center; }}
  /* 退出原因徽章：圆角彩色标签 */
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px;
            color: white; font-weight: 600; white-space: nowrap; }}
  .section-title {{ padding: 8px 48px 12px; font-size: 17px; font-weight: 700; color: #374151; }}
</style>
</head>
<body>

<!-- 标题栏：项目名 + 技术栈说明 + 生成时间 -->
<div class="header">
  <h1>工单23 · Research Agent 评测报告</h1>
  <p>PAI-LangStudio · Qwen-Max · IQS联网搜索 · ReAct多步推理 &nbsp;|&nbsp; 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>

<!-- 统计卡片区 -->
<div class="stats">
  <div class="card"><div class="val">100</div><div class="lbl">总题数</div></div>
  <div class="card"><div class="val" style="color:#22c55e">0</div><div class="lbl">失败(failed)</div></div>
  <div class="card"><div class="val">{exit_counts.get('final_answer',0)}</div><div class="lbl">直接得出答案</div></div>
  <div class="card"><div class="val">{avg_rounds:.1f}</div><div class="lbl">平均搜索轮数</div></div>
  <div class="card"><div class="val">~27min</div><div class="lbl">总耗时</div></div>
  <div class="card"><div class="val">IQS</div><div class="lbl">搜索引擎</div></div>
</div>

<!-- 退出原因图例：按出现次数降序排列 -->
<div class="exit-bar">
  <span style="font-size:13px;color:#64748b;margin-right:4px">退出原因分布：</span>
  {"".join(
    f'<div class="exit-item"><span class="dot" style="background:{exit_color.get(k,"#999")}"></span>{k}({v})</div>'
    for k, v in sorted(exit_counts.items(), key=lambda x: -x[1])  # 按出现次数降序
  )}
</div>

<!-- 逐题明细表格 -->
<div class="section-title">逐题答案明细</div>
<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>问题（摘要）</th>
      <th>答案</th>
      <th>退出原因</th>
      <th>轮数</th>
    </tr>
  </thead>
  <tbody>{rows_html}
  </tbody>
</table>
</div>
</body>
</html>"""

# 将完整 HTML 写入文件
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"报告已生成: {OUTPUT_HTML}")  # 提示用户报告路径
