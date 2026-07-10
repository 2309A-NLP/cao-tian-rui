"""
fund_agent.py - 基金数据问答主 Agent
对外暴露两种接口：
  1. FundAgent().query(question) -> str  （直接调用）
  2. fund_db_tool                        （LangChain Tool，供工单6使用）
"""
import logging
import asyncio
import sys
import os

# 确保 src 目录在 Python 路径中（工单6跨目录导入时也能工作）
sys.path.insert(0, os.path.dirname(__file__))

from db_utils import get_full_schema
from nl2sql import generate_sql
from sql_executor import run_sql_with_retry, SQLExecutionError
from answer_gen import generate_answer

logger = logging.getLogger(__name__)

# ── 结果缓存（进程内，重启清空，最多500条）──────────────────
_answer_cache: dict[str, str] = {}
_CACHE_MAX = 500


def _cache_get(q: str) -> "str | None":
    return _answer_cache.get(q)


def _cache_set(q: str, ans: str) -> None:
    if len(_answer_cache) >= _CACHE_MAX:
        _answer_cache.pop(next(iter(_answer_cache)))
    _answer_cache[q] = ans


# ── 问题预分类（零API成本）────────────────────────────────────
# 招股书/公司定性问题的关键词 —— 这类问题不在SQLite里，直接跳过
_PROSPECTUS_KEYWORDS = [
    "招股", "招股说明书", "招股意见书", "招股书",
    "竞争优势", "核心竞争", "主要原因", "主要业务", "主营业务",
    "发展历程", "经营模式", "商业模式", "盈利模式",
    "实际控制人", "控股股东", "股权结构", "发起人",
    "专利", "知识产权", "研发投入", "技术优势",
    "员工人数", "在册员工", "正式员工",
    "利润率", "毛利率", "净利润率",
    "首发", "首次公开", "战略配售", "网上发行", "网下发行", "超额配售",
    "募投项目", "募集资金用途", "超募",
    "风险因素", "经营风险", "市场风险",
    "存货", "应收账款", "流动资产", "固定资产",
    "营业收入", "营业成本", "期间费用",
    "报告期末", "各报告期",
    "变更设立", "改制", "设立时",
    "主要客户", "主要供应商",
    "生产能力", "产能", "产量",
    "销售价格", "价格波动", "定价",
]

# 数据库问题的强信号词 —— 有这些词基本一定是结构化查询
_DB_KEYWORDS = [
    "涨跌幅", "涨幅", "跌幅", "涨停", "跌停",
    "收盘价", "开盘价", "最高价", "最低价", "昨收盘",
    "成交量", "成交金额", "换手率",
    "净值", "单位净值", "资产净值", "累计净值",
    "持仓", "重仓股", "重仓债", "持债", "持股",
    "基金规模", "份额", "申购", "赎回", "净申购",
    "基金日行情", "股票日行情",
    "一级行业", "二级行业", "行业分类",
    "机构投资者", "个人投资者", "持有人结构",
    "季报", "年报", "半年报", "中期报告",
]


def classify_question(question: str) -> str:
    """
    零成本规则分类器，判断问题类型。

    Returns:
        'db'          - 数据库结构化查询，走NL2SQL流程
        'prospectus'  - 招股书/公司定性问题，工单5处理，直接跳过
        'unknown'     - 无法判断，保守处理走NL2SQL
    """
    # 1. 有强数据库信号 → 直接判定为db（即使同时含招股书词）
    for kw in _DB_KEYWORDS:
        if kw in question:
            return "db"

    # 2. 有招股书信号 → 判定为prospectus
    for kw in _PROSPECTUS_KEYWORDS:
        if kw in question:
            return "prospectus"

    # 3. 含8位纯数字日期（如20210105）→ 大概率是数据库查询
    import re
    if re.search(r'\b2\d{7}\b', question):
        return "db"

    return "unknown"


class FundAgent:
    """
    基金数据问答 Agent
    流程：NL2SQL → SQLite执行 → 自然语言回答
    """

    def __init__(self):
        # 预加载 Schema（第一次调用较慢，之后命中缓存）
        self._schema: str | None = None

    def _get_schema(self) -> str:
        if self._schema is None:
            logger.info("正在加载数据库 Schema...")
            self._schema = get_full_schema()
            logger.info("Schema 加载完成")
        return self._schema

    def query(self, question: str) -> str:
        """
        对外主接口：输入自然语言问题，返回自然语言回答

        Args:
            question: 用户的自然语言问题

        Returns:
            自然语言回答字符串；出错时返回友好错误提示
        """
        # 缓存命中直接返回
        cached = _cache_get(question)
        if cached is not None:
            logger.info("[FundAgent] 缓存命中")
            return cached

        logger.info(f"[FundAgent] 处理问题: {question[:60]}...")

        # Step 0: 规则预分类（零API成本）
        q_type = classify_question(question)
        if q_type == "prospectus":
            logger.info(f"[FundAgent] 预分类=招股书，跳过API直接返回")
            _cache_set(question, "__PROSPECTUS__")
            return "__PROSPECTUS__"  # 特殊标记，供工单6识别

        # Step 1: NL2SQL
        try:
            schema = self._get_schema()
            sql = self._call_with_429_retry(generate_sql, question, schema=schema)
            logger.info(f"[FundAgent] 生成SQL: {sql[:100]}...")
        except Exception as e:
            logger.error(f"[FundAgent] SQL生成失败: {e}")
            return f"抱歉，无法理解该问题，请尝试重新描述。（错误：{e}）"

        # Step 2: 执行SQL（含重试）
        try:
            rows, final_sql = run_sql_with_retry(question, sql)
            logger.info(f"[FundAgent] 查询成功，返回 {len(rows)} 行")
        except SQLExecutionError as e:
            logger.error(f"[FundAgent] SQL执行失败: {e}")
            return "抱歉，数据查询失败，该问题可能超出了数据库数据范围。"
        except Exception as e:
            logger.error(f"[FundAgent] 未知错误: {e}")
            return f"抱歉，查询过程中发生未知错误：{e}"

        # Step 3: 生成自然语言回答
        try:
            answer = self._call_with_429_retry(generate_answer, question, rows, sql=final_sql)
            logger.info(f"[FundAgent] 回答生成成功")
            _cache_set(question, answer)
            return answer
        except Exception as e:
            logger.error(f"[FundAgent] 回答生成失败: {e}")
            return f"查询到以下数据：\n{rows[:10]}"

    async def query_async(self, question: str) -> str:
        """异步版本，供工单6并发调用多工具时使用"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, question)

    def query_stream(self, question: str):
        """流式问答生成器，产出 (event_type, content) 元组。
        event_type: "status" | "sql" | "answer" | "error"
        """
        cached = _cache_get(question)
        if cached is not None:
            yield "answer", cached
            return

        q_type = classify_question(question)
        if q_type == "prospectus":
            _cache_set(question, "__PROSPECTUS__")
            yield "answer", "__PROSPECTUS__"
            return

        yield "status", "正在生成 SQL…"
        try:
            schema = self._get_schema()
            sql = self._call_with_429_retry(generate_sql, question, schema=schema)
        except Exception as e:
            yield "error", f"SQL生成失败：{e}"
            return

        yield "sql", sql
        yield "status", "正在执行查询…"
        try:
            rows, final_sql = run_sql_with_retry(question, sql)
        except SQLExecutionError as e:
            yield "error", f"数据查询失败：{e}"
            return
        except Exception as e:
            yield "error", f"查询出错：{e}"
            return

        try:
            answer = self._call_with_429_retry(generate_answer, question, rows, sql=final_sql)
            _cache_set(question, answer)
            yield "answer", answer
        except Exception as e:
            yield "error", f"回答生成失败：{e}"

    @staticmethod
    def _call_with_429_retry(func, *args, **kwargs):
        """遇到429限流时等待60秒后重试，最多3次"""
        import time
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 60 * (attempt + 1)
                    logger.warning(f"触发限流(429)，等待 {wait} 秒后重试...")
                    time.sleep(wait)
                else:
                    raise


# ── 单例（避免每次导入都重新加载Schema）──────────────────────
_agent_instance: FundAgent | None = None


def get_agent() -> FundAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FundAgent()
    return _agent_instance


def query_stream(question: str):
    """模块级流式接口，供 app.py 直接调用"""
    return get_agent().query_stream(question)


async def query_async(question: str) -> str:
    """模块级异步接口，供工单6并发调用"""
    return await get_agent().query_async(question)


# ── 工单6 LangChain Tool 接口 ────────────────────────────────
def _fund_query_func(question: str) -> str:
    """供 LangChain Tool 调用的函数"""
    return get_agent().query(question)


# 懒导入：没装 langchain 时不报错
try:
    from langchain.tools import Tool as LangChainTool

    fund_db_tool = LangChainTool(
        name="基金数据问答",
        description=(
            "查询基金、股票、债券的结构化历史数据（2019-2021年）。"
            "支持查询：基金基本信息、股票/债券持仓明细、日行情、"
            "基金规模、份额持有人结构、行业分类等。"
            "输入：自然语言问题。输出：文字回答。"
        ),
        func=_fund_query_func,
    )
except ImportError:
    fund_db_tool = None  # type: ignore
    logger.debug("langchain 未安装，fund_db_tool 不可用")


# ── 命令行交互测试 ────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    agent = get_agent()
    print("=== 基金数据问答 Agent ===")
    print("输入问题后按回车，输入 'exit' 退出\n")

    if len(sys.argv) > 1:
        # 命令行传参模式：python fund_agent.py "你的问题"
        q = " ".join(sys.argv[1:])
        print(f"问题：{q}")
        print(f"回答：{agent.query(q)}")
    else:
        # 交互模式
        while True:
            try:
                q = input("问题> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n退出")
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            print(f"回答：{agent.query(q)}\n")
