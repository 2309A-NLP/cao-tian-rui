"""
工单11 医疗挂号Agent - 数据库连接管理 (MySQL 版)

连接池：DBUtils.PooledDB
  - maxconnections=20  应对 50+ 并发请求时的连接复用
  - DictCursor          保证 row["字段名"] 用法与 SQLite 版一致，tools.py 无需改访问方式
事务：db_transaction() 上下文管理器，自动 commit/rollback
"""
# contextlib.contextmanager：将生成器函数装饰为上下文管理器（支持 with 语句）
from contextlib import contextmanager

# pymysql：纯 Python 实现的 MySQL 客户端驱动
import pymysql
import pymysql.cursors   # 提供 DictCursor 游标类

# DBUtils（又名 dbutils）：数据库连接池工具库
# PooledDB：线程安全的连接池，管理多个预创建的 MySQL 连接，避免每次请求都新建连接
from dbutils.pooled_db import PooledDB

# 从配置模块读取数据库连接参数
from src.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

# 模块级单例连接池变量，初始为 None，首次调用 _get_pool() 时创建
_pool: PooledDB | None = None


def _get_pool() -> PooledDB:
    """
    获取（或创建）全局单例连接池。

    使用懒初始化模式（Lazy Initialization）：
      首次调用时创建连接池，之后直接复用同一个池实例。
      避免在模块 import 时就建立数据库连接（此时 .env 可能尚未加载）。
    """
    global _pool               # 声明修改模块级变量（而非创建局部变量）
    if _pool is None:          # 若连接池尚未创建，则初始化
        _pool = PooledDB(
            creator=pymysql,       # 告诉 PooledDB 使用 pymysql 驱动创建底层连接
            maxconnections=20,     # 最大连接数（超过此数的请求排队等待）
            mincached=2,           # 启动时预创建的空闲连接数
            maxcached=10,          # 池中保持的最大空闲连接数（多余的会被关闭）
            blocking=True,         # 连接耗尽时阻塞等待，而不是抛 TooManyConnections 异常
            host=DB_HOST,          # MySQL 主机地址
            port=DB_PORT,          # MySQL 端口
            user=DB_USER,          # MySQL 用户名
            password=DB_PASSWORD,  # MySQL 密码
            database=DB_NAME,      # 使用的数据库名
            charset="utf8mb4",     # 字符集（支持完整 Unicode）
            cursorclass=pymysql.cursors.DictCursor,  # 查询结果以 dict 返回，row["列名"] 访问
            autocommit=False,      # 关闭自动提交，由应用层手动控制事务
        )
    return _pool


def get_connection():
    """
    从连接池获取一个可用连接。

    返回的连接对象使用完后必须调用 .close()，
    这会将连接归还到池中（而非真正关闭 TCP 连接），供下次复用。
    """
    return _get_pool().connection()


@contextmanager
def db_transaction():
    """
    数据库事务上下文管理器，用法：

      with db_transaction() as conn:
          cur = conn.cursor()
          cur.execute(...)   # 执行多条 SQL
          # 正常退出 with 块 → 自动 commit
          # 抛出异常 → 自动 rollback，异常原样上抛

    设计原因：
      book_appointment / cancel_appointment 需要原子性地执行多条 SQL
      （如 UPDATE schedule + INSERT register），用此上下文管理器可以
      确保"要么全成功要么全回滚"，避免数据不一致。
    """
    conn = get_connection()    # 从连接池取一个连接
    try:
        yield conn             # 将连接交给 with 块体使用
        conn.commit()          # with 块正常退出（无异常）时提交事务
    except Exception:
        conn.rollback()        # with 块抛出异常时回滚，撤销所有未提交的修改
        raise                  # 异常原样向上抛出，不吞掉（调用方可以 catch 并返回错误）
    finally:
        conn.close()           # 无论成功还是失败，都归还连接到池中
