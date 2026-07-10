"""
db_utils.py - SQLite 连接管理 + Schema 提取与缓存
提供给 nl2sql.py 使用的表结构信息
"""
import sqlite3
import functools
from pathlib import Path
from typing import Any

from config import DB_PATH


# ── 连接 ────────────────────────────────────────────────────
def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """返回只读模式的 SQLite 连接（URI模式）"""
    if not db_path.exists():
        raise FileNotFoundError(
            f"数据库文件不存在：{db_path}\n"
            "请先下载博金杯比赛数据.db 并放到 data/ 目录"
        )
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    # 超过30秒的查询自动中断
    conn.set_progress_handler(_timeout_handler(30), 1000000)
    return conn


def _timeout_handler(max_seconds: int):
    """返回一个进度回调，超时后抛出异常中断查询"""
    import time
    start = [time.time()]

    def handler():
        if time.time() - start[0] > max_seconds:
            raise TimeoutError(f"SQL 查询超时（>{max_seconds}s），已自动中断")
        return None

    return handler


# ── 表名列表 ────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_table_names() -> list[str]:
    """获取数据库中所有用户表名（缓存，只查一次）"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [r["name"] for r in rows]


# ── 单张表 Schema ────────────────────────────────────────────
@functools.lru_cache(maxsize=20)
def get_table_schema(table_name: str) -> str:
    """
    返回单张表的 CREATE TABLE 语句 + 前3行示例数据
    格式供 LLM 理解表结构
    """
    with get_connection() as conn:
        # CREATE TABLE 语句
        ddl_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        ).fetchone()
        ddl = ddl_row["sql"] if ddl_row else f"-- 表 {table_name} 不存在"

        # 前3行示例（帮助LLM理解字段内容/格式）
        try:
            rows = conn.execute(
                f'SELECT * FROM "{table_name}" LIMIT 3'
            ).fetchall()
            if rows:
                cols = rows[0].keys()
                sample_lines = ["-- 示例数据（前3行）："]
                sample_lines.append("-- " + " | ".join(cols))
                for r in rows:
                    sample_lines.append("-- " + " | ".join(str(v) for v in r))
                sample = "\n".join(sample_lines)
            else:
                sample = "-- 该表暂无数据"
        except Exception as e:
            sample = f"-- 示例数据读取失败: {e}"

    return f"{ddl}\n{sample}"


# ── 全量 Schema（供 prompt 使用）─────────────────────────────
@functools.lru_cache(maxsize=1)
def get_full_schema() -> str:
    """
    返回所有表的 Schema 拼接字符串
    第一次调用较慢（读数据库），之后命中缓存
    """
    tables = get_table_names()
    parts = [f"数据库共 {len(tables)} 张表，表名如下：{', '.join(tables)}\n"]
    for t in tables:
        parts.append(f"\n{'='*60}")
        parts.append(f"表名：{t}")
        parts.append(get_table_schema(t))
    return "\n".join(parts)


# ── 工具函数 ────────────────────────────────────────────────
def execute_query(sql: str, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """
    执行 SQL，返回 list[dict] 格式结果
    不暴露给外部直接调用，由 sql_executor.py 包一层重试逻辑
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        if not rows:
            return []
        cols = rows[0].keys()
        return [dict(zip(cols, row)) for row in rows]


if __name__ == "__main__":
    # 快速验证：打印所有表名
    print("数据库表：", get_table_names())
    print("\n=== Schema 预览（前500字）===")
    print(get_full_schema()[:500])
