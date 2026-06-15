"""
数据库管理模块
MySQL 数据库连接、用户管理、会话管理
"""
import pymysql
from pymysql.cursors import DictCursor
from typing import Optional

from logger import get_logger, log_exception

logger = get_logger("db")


class DatabaseManager:
    """数据库管理器 - 用户认证与会话管理"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 3306,
                 user: str = "root", password: str = "root", database: str = "rag_chat"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn: Optional[pymysql.Connection] = None
    
    def connect(self) -> bool:
        """连接数据库，如果库不存在则创建"""
        try:
            # 先连接 mysql，确保库存在
            self.conn = pymysql.connect(
                host=self.host, port=self.port,
                user=self.user, password=self.password,
                charset="utf8mb4",
                cursorclass=DictCursor,
                connect_timeout=5,
                read_timeout=5,
            )
            with self.conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            self.conn.select_db(self.database)
            self._init_tables()
            logger.info(f"数据库连接成功: {self.host}:{self.port}/{self.database}")
            return True
        except pymysql.Error as e:
            log_exception(logger, f"数据库连接失败 {self.host}:{self.port}", e)
            self.conn = None
            return False

    def _ensure_connection(self):
        """确保数据库连接有效，断开时自动重连"""
        if self.conn is None or not self.conn.open:
            logger.warning("数据库连接已断开，正在重连...")
            return self.connect()
        try:
            self.conn.ping(reconnect=True)
        except pymysql.Error:
            logger.warning("数据库 ping 失败，正在重连...")
            return self.connect()
        return True
    
    def _init_tables(self):
        """初始化数据表"""
        try:
            with self.conn.cursor() as cursor:
                # 用户表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS `users` (
                        `id` INT AUTO_INCREMENT PRIMARY KEY,
                        `username` VARCHAR(64) NOT NULL UNIQUE,
                        `password_hash` VARCHAR(256) NOT NULL,
                        `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                        `last_login_at` DATETIME NULL,
                        INDEX `idx_username` (`username`)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """)
                
                # 对话历史表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS `chat_history` (
                        `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
                        `user_id` INT NOT NULL,
                        `session_id` VARCHAR(64) NOT NULL,
                        `role` VARCHAR(16) NOT NULL COMMENT 'user/assistant',
                        `content` TEXT NOT NULL,
                        `mode` VARCHAR(16) DEFAULT 'rag' COMMENT 'rag/direct',
                        `retrieval_time_ms` INT DEFAULT 0,
                        `llm_time_ms` INT DEFAULT 0,
                        `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX `idx_user_session` (`user_id`, `session_id`),
                        INDEX `idx_created` (`created_at`),
                        FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """)

                # 用户反馈表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS `feedback` (
                        `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
                        `user_id` INT NOT NULL,
                        `session_id` VARCHAR(64) NOT NULL,
                        `query` TEXT NOT NULL,
                        `answer` TEXT NOT NULL,
                        `rating` TINYINT NOT NULL COMMENT '1=up, -1=down',
                        `comment` TEXT NULL,
                        `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX `idx_user` (`user_id`),
                        INDEX `idx_rating` (`rating`),
                        FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """)
            self.conn.commit()
            logger.info("数据库表初始化完成")
        except pymysql.Error as e:
            log_exception(logger, "初始化数据表失败", e)
            raise
    
    def register_user(self, username: str, password: str) -> dict:
        """
        注册用户
        返回: {"success": bool, "message": str, "user_id": Optional[int]}
        """
        import hashlib
        if not self._ensure_connection():
            return {"success": False, "message": "数据库连接失败", "user_id": None}
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO `users` (`username`, `password_hash`) VALUES (%s, %s)",
                    (username, password_hash)
                )
                self.conn.commit()
                user_id = cursor.lastrowid
                logger.info(f"用户注册成功: {username} (id={user_id})")
                return {"success": True, "message": "注册成功", "user_id": user_id}
        except pymysql.IntegrityError:
            logger.warning(f"用户注册失败（用户名已存在）: {username}")
            return {"success": False, "message": "用户名已存在", "user_id": None}
        except pymysql.Error as e:
            log_exception(logger, f"用户注册失败: {username}", e)
            return {"success": False, "message": f"注册失败: {str(e)}", "user_id": None}
    
    def login_user(self, username: str, password: str) -> dict:
        """
        用户登录
        返回: {"success": bool, "message": str, "user_id": Optional[int], "username": Optional[str]}
        """
        import hashlib
        if not self._ensure_connection():
            return {"success": False, "message": "数据库连接失败", "user_id": None, "username": None}
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, username FROM `users` WHERE `username`=%s AND `password_hash`=%s",
                    (username, password_hash)
                )
                result = cursor.fetchone()
                if result:
                    # 更新最后登录时间
                    cursor.execute(
                        "UPDATE `users` SET `last_login_at`=NOW() WHERE `id`=%s",
                        (result["id"],)
                    )
                    self.conn.commit()
                    logger.info(f"用户登录成功: {username} (id={result['id']})")
                    return {"success": True, "message": "登录成功", "user_id": result["id"], "username": result["username"]}
                else:
                    logger.warning(f"用户登录失败（密码错误或无此用户）: {username}")
                    return {"success": False, "message": "用户名或密码错误", "user_id": None, "username": None}
        except pymysql.Error as e:
            log_exception(logger, f"用户登录失败: {username}", e)
            return {"success": False, "message": f"登录失败: {str(e)}", "user_id": None, "username": None}
    
    def save_chat_message(self, user_id: int, session_id: str, role: str, content: str,
                          mode: str = "rag", retrieval_time_ms: int = 0, llm_time_ms: int = 0) -> bool:
        """保存聊天记录"""
        if not self._ensure_connection():
            return False
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO `chat_history` 
                       (`user_id`, `session_id`, `role`, `content`, `mode`, `retrieval_time_ms`, `llm_time_ms`)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (user_id, session_id, role, content, mode, retrieval_time_ms, llm_time_ms)
                )
                self.conn.commit()
            return True
        except pymysql.Error as e:
            log_exception(logger, "保存聊天记录失败", e)
            return False
    
    def get_chat_history(self, user_id: int, session_id: str, limit: int = 20) -> list[dict]:
        """获取指定会话的聊天历史"""
        if not self._ensure_connection():
            return []
        try:
            with self.conn.cursor(cursor=DictCursor) as cursor:
                cursor.execute(
                    """SELECT role, content, mode, retrieval_time_ms, llm_time_ms, created_at
                       FROM `chat_history` 
                       WHERE `user_id`=%s AND `session_id`=%s
                       ORDER BY `created_at` ASC LIMIT %s""",
                    (user_id, session_id, limit)
                )
                return cursor.fetchall()
        except pymysql.Error as e:
            log_exception(logger, "获取聊天历史失败", e)
            return []
    
    def get_user_sessions(self, user_id: int) -> list[dict]:
        """获取用户的所有会话列表（按时间倒序，含首条消息预览）"""
        if not self._ensure_connection():
            return []
        try:
            with self.conn.cursor(cursor=DictCursor) as cursor:
                # 子查询获取每个会话的第一条 user 消息
                cursor.execute("""
                    SELECT h.session_id,
                           MIN(h.created_at) as started_at,
                           MAX(h.created_at) as last_msg,
                           COUNT(*) as msg_count,
                           SUBSTRING(MIN(CASE WHEN h.role='user' THEN h.content END), 1, 100) as first_msg
                    FROM `chat_history` h
                    WHERE h.user_id=%s
                    GROUP BY h.session_id
                    ORDER BY last_msg DESC LIMIT 50
                """, (user_id,))
                return cursor.fetchall()
        except pymysql.Error as e:
            log_exception(logger, "获取会话列表失败", e)
            return []

    def delete_session(self, user_id: int, session_id: str) -> bool:
        """删除指定会话及其所有消息"""
        if not self._ensure_connection():
            return False
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM `chat_history` WHERE `user_id`=%s AND `session_id`=%s",
                    (user_id, session_id)
                )
                self.conn.commit()
                return cursor.rowcount > 0
        except pymysql.Error as e:
            log_exception(logger, f"删除会话失败 {session_id}", e)
            return False

    # ── 用户反馈 ──

    def save_feedback(self, user_id: int, session_id: str,
                      query: str, answer: str, rating: int,
                      comment: str = "") -> bool:
        """保存用户反馈 (rating: 1=赞, -1=踩)"""
        if not self._ensure_connection():
            return False
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO `feedback` (`user_id`, `session_id`, `query`, `answer`, `rating`, `comment`) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (user_id, session_id, query, answer, rating, comment or None)
                )
                self.conn.commit()
                logger.debug(f"反馈已保存: user={user_id}, rating={rating}")
                return True
        except pymysql.Error as e:
            log_exception(logger, "保存反馈失败", e)
            return False

    def get_feedback_stats(self) -> dict:
        """获取反馈统计"""
        if not self._ensure_connection():
            return {"total": 0, "up": 0, "down": 0, "satisfaction_rate": 0}
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "SELECT rating, COUNT(*) as cnt FROM `feedback` GROUP BY `rating`"
                )
                rows = cursor.fetchall()
                stats = {"total": 0, "up": 0, "down": 0}
                for row in rows:
                    r, cnt = row["rating"], row["cnt"]
                    stats["total"] += cnt
                    if r == 1:
                        stats["up"] = cnt
                    elif r == -1:
                        stats["down"] = cnt
                if stats["total"] > 0:
                    stats["satisfaction_rate"] = round(stats["up"] / stats["total"] * 100, 1)
                else:
                    stats["satisfaction_rate"] = 0
                return stats
        except pymysql.Error as e:
            log_exception(logger, "获取反馈统计失败", e)
            return {"total": 0, "up": 0, "down": 0, "satisfaction_rate": 0}

    def close(self):
        """关闭连接"""
        if self.conn and self.conn.open:
            self.conn.close()
            logger.info("数据库连接已关闭")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
