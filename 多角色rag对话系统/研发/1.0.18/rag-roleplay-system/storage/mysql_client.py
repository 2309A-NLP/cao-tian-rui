# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
MySQL 客户端封装

负责：
- 连接/断开 MySQL
- 执行 SQL 查询
- 用户注册/登录
- 对话记录保存/查询

⚠️ 常改动的地方：
1. 会话列表的默认 limit（20）和消息列表的默认 limit（50）可根据前端需求调整
2. 表结构变更时（如增加字段）需要同步修改对应的 SQL 语句

⚠️ 注意事项：
1. 数据库名、表名与之前保持一致，请勿随意更改
2. 使用 autocommit=True，无需手动提交事务
3. 所有查询使用参数化，防止 SQL 注入
4. 单例模式确保全局只有一个数据库连接池
"""

import pymysql
import logging
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE

logger = logging.getLogger(__name__)


class MySQLClient:
    """MySQL 客户端单例类"""

    _instance = None

    def __new__(cls):
        """单例模式：保证全局只有一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._conn = None
        return cls._instance

    def connect(self):
        """连接数据库，使用 config 中的参数"""
        if self._conn:
            return
        try:
            self._conn = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                charset="utf8mb4",                     # 支持完整的 UTF-8（包括表情符号）
                cursorclass=pymysql.cursors.DictCursor, # 返回字典格式记录
                autocommit=True                        # 自动提交，无需手动 commit
            )
            logger.info(f"MySQL 连接成功: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
        except Exception as e:
            logger.error(f"MySQL 连接失败: {e}")
            raise

    def disconnect(self):
        """断开数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def cursor(self):
        """
        获取游标的上下文管理器，自动清理
        使用方式：
            with self.cursor() as cursor:
                cursor.execute(...)
        """
        if not self._conn:
            self.connect()
        cursor = self._conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def execute(self, sql: str, params: tuple = None) -> int:
        """执行 DML 语句（INSERT/UPDATE/DELETE），返回影响的行数"""
        with self.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.rowcount

    def fetch_one(self, sql: str, params: tuple = None) -> Optional[Dict]:
        """查询单条记录，返回字典或 None"""
        with self.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()

    def fetch_all(self, sql: str, params: tuple = None) -> List[Dict]:
        """查询多条记录，返回字典列表"""
        with self.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()

    def insert(self, sql: str, params: tuple = None) -> int:
        """
        插入数据并返回自增主键 ID（通常用于 users 表）
        注意：仅对 AUTO_INCREMENT 字段有效
        """
        with self.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.lastrowid

    # ==================== 用户相关 ====================

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """根据用户名获取用户信息（用于登录验证）"""
        sql = "SELECT id, username, nickname, email, status FROM users WHERE username = %s"
        return self.fetch_one(sql, (username,))

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """根据用户ID获取用户信息"""
        sql = "SELECT id, username, nickname, email, status FROM users WHERE id = %s"
        return self.fetch_one(sql, (user_id,))

    def create_user(self, username: str, password_hash: str, email: str = None, nickname: str = None) -> int:
        """创建新用户，返回用户ID（自增主键）"""
        sql = """
            INSERT INTO users (username, password_hash, email, nickname) 
            VALUES (%s, %s, %s, %s)
        """
        return self.insert(sql, (username, password_hash, email, nickname or username))

    def update_last_login(self, user_id: int):
        """更新用户的最后登录时间（用于统计活跃度）"""
        from datetime import datetime
        sql = "UPDATE users SET last_login = %s WHERE id = %s"
        self.execute(sql, (datetime.now(), user_id))

    # ==================== 对话相关 ====================

    def save_conversation(self, conversation_id: str, user_id: int, role_type: str, title: str) -> bool:
        """
        保存对话记录（如已存在则跳过，防止重复插入）
        ⚠️ 常改动：如果需要在对话表中增加字段，请同步修改此处的 INSERT 语句
        """
        # 先检查是否存在（基于 conversation_id 唯一性）
        existing = self.fetch_one(
            "SELECT id FROM conversations WHERE conversation_id = %s",
            (conversation_id,)
        )
        if existing:
            return True

        sql = """
            INSERT INTO conversations (conversation_id, user_id, role_type, title) 
            VALUES (%s, %s, %s, %s)
        """
        self.insert(sql, (conversation_id, user_id, role_type, title))
        return True

    def save_message(self, conversation_id: str, user_id: int, role: str, content: str, sources: str = None):
        """
        保存单条消息，并更新对话的消息计数和更新时间
        ⚠️ 常改动：如果消息表增加了来源结构字段（如 JSON），sources 参数可传 JSON 字符串
        """
        from datetime import datetime
        sql = """
            INSERT INTO messages (conversation_id, user_id, role, content, sources, created_at) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.insert(sql, (conversation_id, user_id, role, content, sources, datetime.now()))

        # 更新对话的消息计数和最后更新时间
        self.execute(
            "UPDATE conversations SET message_count = message_count + 1, updated_at = %s WHERE conversation_id = %s",
            (datetime.now(), conversation_id)
        )

    def get_conversations(self, user_id: int, role_type: str = None, limit: int = 20) -> List[Dict]:
        """
        获取用户的对话列表（按更新时间倒序）
        ⚠️ 常改动：limit 默认 20，若需显示更多可调整；亦可支持分页参数（offset）
        """
        if role_type:
            sql = """
                SELECT conversation_id, title, role_type, created_at, updated_at, message_count 
                FROM conversations 
                WHERE user_id = %s AND role_type = %s
                ORDER BY updated_at DESC LIMIT %s
            """
            return self.fetch_all(sql, (user_id, role_type, limit))
        else:
            sql = """
                SELECT conversation_id, title, role_type, created_at, updated_at, message_count 
                FROM conversations 
                WHERE user_id = %s 
                ORDER BY updated_at DESC LIMIT %s
            """
            return self.fetch_all(sql, (user_id, limit))

    def get_conversation_messages(self, conversation_id: str, user_id: int, limit: int = 50) -> List[Dict]:
        """
        获取对话中的所有消息（按时间正序）
        ⚠️ 常改动：limit 默认 50，可增大或改为分页查询以提高长对话性能
        """
        sql = """
            SELECT role, content, sources, created_at 
            FROM messages 
            WHERE conversation_id = %s AND user_id = %s
            ORDER BY created_at ASC LIMIT %s
        """
        return self.fetch_all(sql, (conversation_id, user_id, limit))

    def delete_conversation(self, conversation_id: str, user_id: int) -> bool:
        """
        删除对话（级联删除消息）
        先验证归属，防止跨用户误删
        """
        # 先验证对话是否属于该用户
        conv = self.fetch_one(
            "SELECT id FROM conversations WHERE conversation_id = %s AND user_id = %s",
            (conversation_id, user_id)
        )
        if not conv:
            return False

        # 删除消息（外键约束未设置级联时需手动删除）
        self.execute("DELETE FROM messages WHERE conversation_id = %s", (conversation_id,))
        # 删除对话
        self.execute("DELETE FROM conversations WHERE conversation_id = %s", (conversation_id,))
        return True

    # 注意：get_user_by_username 方法在文件中重复定义了一次（第 181 行附近）
    # 为避免重复，保留后一个定义；前一个定义（第 96 行）已存在，但 Python 会覆盖。
    # 实际运行时以最后定义为准，两个完全相同，无副作用。
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """根据用户名获取用户（重复定义，保留原样）"""
        sql = "SELECT id, username, nickname, email, status FROM users WHERE username = %s"
        return self.fetch_one(sql, (username,))


# 全局单例实例
mysql_client = MySQLClient()