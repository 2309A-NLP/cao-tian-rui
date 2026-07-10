"""
sql_executor.py - 执行 SQL，自动重试（最多 MAX_RETRY 次）
失败时调用 nl2sql 重新生成修正后的 SQL
"""
import logging
from typing import Any

from config import MAX_RETRY
from db_utils import execute_query

logger = logging.getLogger(__name__)


class SQLExecutionError(Exception):
    """SQL 执行失败且重试耗尽时抛出"""
    pass


def run_sql_with_retry(
    question: str,
    initial_sql: str,
) -> tuple[list[dict[str, Any]], str]:
    """
    执行 SQL，失败时自动修正重试。

    Args:
        question: 原始自然语言问题（用于重新生成修正SQL）
        initial_sql: 第一次生成的 SQL

    Returns:
        (rows, final_sql)
        - rows: 查询结果，list[dict]
        - final_sql: 最终成功执行的 SQL（方便调试）

    Raises:
        SQLExecutionError: 重试 MAX_RETRY 次后仍失败
    """
    # 延迟导入，避免循环依赖
    from nl2sql import generate_sql_with_error_hint

    current_sql = initial_sql
    last_error = ""

    for attempt in range(MAX_RETRY + 1):
        try:
            rows = execute_query(current_sql)
            if attempt > 0:
                logger.info(f"SQL 第 {attempt} 次重试成功")
            return rows, current_sql

        except Exception as e:
            last_error = str(e)
            logger.warning(f"SQL 执行失败（第{attempt+1}次）: {last_error}")
            logger.warning(f"失败SQL: {current_sql}")

            if attempt < MAX_RETRY:
                # 带错误信息重新生成 SQL
                logger.info("正在重新生成修正后的 SQL...")
                try:
                    current_sql = generate_sql_with_error_hint(
                        question=question,
                        failed_sql=current_sql,
                        error_msg=last_error,
                    )
                    logger.info(f"修正SQL: {current_sql}")
                except Exception as gen_e:
                    logger.error(f"重新生成SQL失败: {gen_e}")
                    break

    raise SQLExecutionError(
        f"SQL 执行失败，已重试 {MAX_RETRY} 次\n"
        f"最后错误：{last_error}\n"
        f"最后SQL：{current_sql}"
    )
