"""
nl2sql.py - 调用硅基流动 API，将自然语言问题转换为 SQL 语句
核心模块：NL2SQL
"""
import re
import json
from openai import OpenAI

from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, NL2SQL_MODEL
from db_utils import get_full_schema

# 硅基流动兼容 OpenAI SDK
_client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
)

# ── 系统提示词 ────────────────────────────────────────────────
_SYSTEM_PROMPT = """你是一个专业的金融数据库 SQL 专家，负责将用户的自然语言问题转换为 SQLite SQL 查询语句。

## 数据库结构
{schema}

## 转换规则
1. 只输出 SQL 语句，不要解释，不要 markdown 代码块，直接输出纯 SQL
2. 使用标准 SQLite 语法
3. 日期字段通常为文本格式 YYYYMMDD（如 20210331），直接用字符串比较，不要用 DATE() 函数
4. 股票代码、基金代码等字段注意精确匹配
5. 【重要】A股涨跌幅计算公式（固定写法，不允许偏差）：("收盘价(元)" - "昨收盘(元)") / "昨收盘(元)" * 100。"昨收盘(元)"列就是前一日收盘价，计算涨跌幅时禁止再做自连接查前一日数据。若题目公式括号有歧义，一律以本规则为准。
6. 如果问题涉及排序，默认 DESC 降序
7. 字符串匹配使用 LIKE，数字比较直接用 = / > / < 等
8. 只输出 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DROP
9. 表名和列名必须用双引号括起来，特别注意含括号的列名如 "收盘价(元)"、"昨收盘(元)"、"今开盘(元)" 等
10. 若问题需要多表联查，使用 JOIN，注意关联字段
11. A股票日行情表的列名完整示例："股票代码","交易日","昨收盘(元)","今开盘(元)","最高价(元)","最低价(元)","收盘价(元)","成交量(股)","成交金额(元)"
11b.【重要】基金可转债持仓明细的列名：该表没有"股票代码"和"股票名称"列！正确列名为："对应股票代码"（对应A股代码）、"债券名称"（可转债名称）、"基金代码"、"持仓日期"、"报告类型"、"第N大重仓股"
12. 行业查询时注意 A股公司行业划分表 的 交易日期 字段要与行情表的 交易日 字段 JOIN
13. 【重要】查询基金相关数据时，禁止猜测或硬编码基金代码！必须通过基金名称 JOIN 基金基本信息表来获取基金代码，例如：
    SELECT b.债券名称 FROM 基金债券持仓明细 b
    JOIN 基金基本信息 i ON b.基金代码 = i.基金代码
    WHERE i.基金简称 LIKE '%景顺长城中短债债券C%' AND b.持仓日期='20210331'
14. 基金基本信息表中 基金简称 字段存储基金名称，用 LIKE '%基金名%' 模糊匹配
15. "基金债券持仓明细"和"基金股票持仓明细"都有 "第N大重仓股" 字段，按该字段 ASC 升序排序可得前N大持仓（值为1=第一大，值越小排名越靠前）
16. 行业划分标准字段精确匹配：中信行业分类题目用 "行业划分标准" = '中信行业分类'（不要用 LIKE，会导致全表扫描）；一级行业名称举例："综合金融"、"建筑材料"、"非银金融"、"消费者服务" 等（注意：问题中"综合金融行业"对应数据库中的"综合金融"，不要加"行业"二字匹配）
17. 【重要】如果问题问的是"原因"、"背景"、"战略"、"风险"、"主要业务"、"发展历程"等定性描述类问题，这类信息不在结构化数据库中，请输出：SELECT '该问题涉及招股书文本内容，不在结构化数据库中' AS 提示
18. 股票代码是纯数字字符串（如 '600120'），绝对不能用公司名称去匹配股票代码字段

## 输出格式
只输出一条 SQL 语句，末尾不加分号。
"""

_USER_PROMPT_TEMPLATE = """请将以下问题转换为 SQLite SQL 查询语句：

问题：{question}

SQL："""


def _extract_sql(raw: str) -> str:
    """从模型输出中提取纯 SQL 语句（去掉可能的 markdown 包装）"""
    # 去掉 ```sql ... ``` 包装
    match = re.search(r"```(?:sql)?\s*([\s\S]+?)```", raw, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # 去掉行首的 "SQL：" 等前缀
    raw = re.sub(r"^(SQL[：:]\s*|```\s*)", "", raw.strip(), flags=re.IGNORECASE)
    # 只保留 SELECT 开头的部分
    select_match = re.search(r"(SELECT[\s\S]+)", raw, re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip().rstrip(";")
    return raw.strip().rstrip(";")


def generate_sql(question: str, schema: str | None = None) -> str:
    """
    将自然语言问题转换为 SQL 语句

    Args:
        question: 用户的自然语言问题
        schema: 可选，手动传入 Schema；默认自动从数据库读取

    Returns:
        SQL 字符串（不含末尾分号）
    """
    if schema is None:
        schema = get_full_schema()

    system_content = _SYSTEM_PROMPT.format(schema=schema)
    user_content = _USER_PROMPT_TEMPLATE.format(question=question)

    response = _client.chat.completions.create(
        model=NL2SQL_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.0,
        max_tokens=512,
        extra_body={"enable_thinking": False},  # Qwen3 关闭思考模式，直接生成 SQL
    )

    raw_sql = response.choices[0].message.content or ""
    return _extract_sql(raw_sql)


def generate_sql_with_error_hint(
    question: str,
    failed_sql: str,
    error_msg: str,
    schema: str | None = None,
) -> str:
    """
    SQL 执行失败时，带错误信息重新生成 SQL（重试专用）

    Args:
        question: 原始问题
        failed_sql: 上次失败的 SQL
        error_msg: SQLite 报错信息
        schema: 可选 Schema
    """
    if schema is None:
        schema = get_full_schema()

    system_content = _SYSTEM_PROMPT.format(schema=schema)
    user_content = (
        f"问题：{question}\n\n"
        f"上次生成的 SQL 执行报错：\n"
        f"SQL: {failed_sql}\n"
        f"错误: {error_msg}\n\n"
        f"请根据错误信息修正 SQL，重新输出正确的 SQL：\n"
        f"SQL："
    )

    response = _client.chat.completions.create(
        model=NL2SQL_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.0,
        max_tokens=512,
        extra_body={"enable_thinking": False},
    )

    raw_sql = response.choices[0].message.content or ""
    return _extract_sql(raw_sql)


if __name__ == "__main__":
    # 快速测试
    q = "在20210105，中信行业分类划分的一级行业为综合金融行业中，涨跌幅最大股票的股票代码是？"
    print("问题：", q)
    sql = generate_sql(q)
    print("生成SQL：", sql)
