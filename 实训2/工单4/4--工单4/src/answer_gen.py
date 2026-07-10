"""
answer_gen.py - 将 SQL 查询结果转换为自然语言回答
"""
import re
from openai import OpenAI
from typing import Any

from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, ANSWER_MODEL

_client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
)

_SYSTEM_PROMPT = """你是一个专业的金融数据分析助手。
用户提出了一个关于基金/股票/债券数据的问题，系统已通过 SQL 查询到原始数据。
请根据原始查询结果，用简洁、准确的中文回答用户问题。

要求：
1. 直接给出答案，不要解释 SQL 或技术细节
2. 数字保留题目要求的小数位数（如无要求则保留2位小数）
3. 涉及百分数时加上 % 符号
4. 如果查询结果为空，回答"未查询到相关数据"
5. 回答简洁，不超过200字
"""

_USER_TEMPLATE = """问题：{question}

SQL查询结果：
{data}

请根据以上数据，直接回答问题："""


def _try_direct_format(rows: list[dict[str, Any]]) -> str | None:
    """
    对于能确定性格式化的结果直接输出，跳过 LLM 避免幻觉。
    覆盖：单行多列、多行单列（列表）、COUNT聚合。
    """
    if not rows:
        return None

    cols = list(rows[0].keys())

    # 单行结果：直接列出所有字段值
    if len(rows) == 1:
        parts = []
        for key, val in rows[0].items():
            if re.search(r'count|数量|数目|只数|条数', str(key), re.I):
                parts.append(f"共 {val} 条")
            else:
                parts.append(f"{key}：{val}")
        if len(parts) <= 6:
            return "，".join(parts)

    # 多行单列：格式化为编号列表
    if len(cols) == 1:
        col = cols[0]
        items = [str(r[col]) for r in rows[:20]]
        return f"{col}：" + "、".join(items)

    # 多行两列（如排名+名称）：格式化为列表
    if len(cols) == 2 and len(rows) <= 10:
        lines = []
        for r in rows:
            vals = list(r.values())
            lines.append(f"{vals[0]}：{vals[1]}")
        return "\n".join(lines)

    return None


def _format_data(rows: list[dict[str, Any]]) -> str:
    """将查询结果格式化为便于 LLM 理解的文本"""
    if not rows:
        return "（无数据）"
    if len(rows) == 1:
        # 单行结果，展开为 key: value 格式
        return "\n".join(f"{k}: {v}" for k, v in rows[0].items())
    # 多行结果，表格格式
    cols = list(rows[0].keys())
    lines = [" | ".join(cols)]
    lines.append("-" * len(lines[0]))
    for row in rows[:50]:  # 最多展示50行，避免 token 超限
        lines.append(" | ".join(str(row.get(c, "")) for c in cols))
    if len(rows) > 50:
        lines.append(f"... 共 {len(rows)} 行，仅展示前50行")
    return "\n".join(lines)


def generate_answer(
    question: str,
    rows: list[dict[str, Any]],
    sql: str = "",
) -> str:
    """
    根据 SQL 查询结果生成自然语言回答

    Args:
        question: 用户原始问题
        rows: SQL 查询结果
        sql: 执行的 SQL（可选，用于 debug）

    Returns:
        自然语言回答字符串
    """
    if not rows:
        return "未查询到相关数据，请确认问题中的条件是否正确。"

    # 方案2：极简结果直接格式化，跳过 LLM
    fast = _try_direct_format(rows)
    if fast:
        return fast

    data_text = _format_data(rows)
    user_content = _USER_TEMPLATE.format(question=question, data=data_text)

    response = _client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.1,
        max_tokens=256,
    )

    return (response.choices[0].message.content or "").strip()
