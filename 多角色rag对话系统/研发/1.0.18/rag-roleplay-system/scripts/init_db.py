# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
初始化数据库工具
功能：创建 MySQL 数据库和表结构
使用场景：首次部署时初始化数据库
运行方式：python scripts/init_db.py

⚠️ 常改动的地方：
1. 如果表结构需要变更（如增加字段），请同步修改对应的 CREATE TABLE 语句
2. 默认 AI 角色列表（default_roles）可根据需要增删或修改提示词
3. 测试用户的用户名、密码、邮箱等（在 create_test_user 中）
4. 数据库连接配置从 config.py 读取，无需修改此处

⚠️ 注意事项：
1. 脚本会删除已有数据库吗？不会，使用 `IF NOT EXISTS` 创建库和表，不影响已有数据
2. 外键约束：conversations 表引用 users(id)，messages 表引用 conversations(conversation_id)，级联删除
3. 知识库文件表（knowledge_files）当前仅记录元信息，不自动同步 Milvus
4. 运行前请确保 MySQL 服务已启动，并且 config.py 中的数据库配置正确
"""

import sys
import os
import pymysql

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE


def init_database():
    """初始化数据库和所有表结构（幂等操作，可重复执行）"""
    print("=" * 50)
    print("MySQL 数据库初始化工具")
    print("=" * 50)

    # 先不指定数据库，连接 MySQL 实例
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            charset="utf8mb4"
        )
        cursor = conn.cursor()

        # 创建数据库（如果不存在）
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"✓ 数据库已创建/存在: {MYSQL_DATABASE}")

        # 切换到目标数据库
        cursor.execute(f"USE {MYSQL_DATABASE}")

        # 创建用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT PRIMARY KEY AUTO_INCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                email VARCHAR(100),
                nickname VARCHAR(50),
                avatar VARCHAR(255),
                status INT DEFAULT 1,
                role VARCHAR(20) DEFAULT 'user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                INDEX idx_username (username)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("✓ 用户表已创建: users")

        # 创建 AI 角色表（存储各角色的系统提示词）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_roles (
                id INT PRIMARY KEY AUTO_INCREMENT,
                role_id VARCHAR(50) UNIQUE NOT NULL,
                role_name VARCHAR(50) NOT NULL,
                role_type VARCHAR(30),
                system_prompt TEXT,
                description VARCHAR(500),
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_role_id (role_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("✓ AI角色表已创建: ai_roles")

        # 创建对话记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INT PRIMARY KEY AUTO_INCREMENT,
                conversation_id VARCHAR(100) UNIQUE NOT NULL,
                user_id INT NOT NULL,
                role_type VARCHAR(30) NOT NULL,
                title VARCHAR(200),
                message_count INT DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_user_id (user_id),
                INDEX idx_conversation_id (conversation_id),
                INDEX idx_role_type (role_type),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("✓ 对话记录表已创建: conversations")

        # 创建消息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INT PRIMARY KEY AUTO_INCREMENT,
                conversation_id VARCHAR(100) NOT NULL,
                user_id INT NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT,
                sources JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_conversation_id (conversation_id),
                INDEX idx_user_id (user_id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("✓ 消息表已创建: messages")

        # 创建知识库文件记录表（仅记录元信息，不存储向量）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_files (
                id INT PRIMARY KEY AUTO_INCREMENT,
                source_file VARCHAR(255) UNIQUE NOT NULL,
                chunk_count INT DEFAULT 0,
                file_size INT DEFAULT 0,
                file_type VARCHAR(50),
                status ENUM('active', 'deleted', 'processing') DEFAULT 'active',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                deleted_at DATETIME NULL,
                INDEX idx_source_file (source_file),
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("✓ 知识库文件表已创建: knowledge_files")

        # 插入默认 AI 角色（如果不存在则插入，存在则更新提示词）
        # ⚠️ 常改动：可增删角色或修改系统提示词
        default_roles = [
            ("friend", "知心朋友", "friend", "你是温柔友善的虚拟朋友，聊天自然亲切。不要使用emoji。", "日常聊天，温暖陪伴"),
            ("doctor", "专业医生", "doctor", "你是专业医生，科学回答，不夸大。请明确引用医学指南来源。不要使用emoji。",
             "健康咨询，医疗建议"),
            ("psychologist", "心理咨询师", "psychologist", "你是心理咨询师，温和疏导情绪。不要使用emoji。",
             "情绪疏导，心理支持"),
            ("lawyer", "法律顾问", "lawyer", "你是法律咨询师，严谨普法。只能回答知识库中明确包含的内容。",
             "法律咨询，条文解读"),
            ("finance", "理财顾问", "finance", "你是理财师，科普理财，不推荐具体产品。不要使用emoji。", "理财建议，财务规划"),
            ("tcm", "中医师", "tcm", "你是中医师，用中医思路回答。请引用中医典籍或理论。不要使用emoji。",
             "中医调理，养生建议"),
        ]

        for role_id, name, role_type, prompt, desc in default_roles:
            cursor.execute("""
                INSERT INTO ai_roles (role_id, role_name, role_type, system_prompt, description) 
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE role_name = VALUES(role_name), system_prompt = VALUES(system_prompt)
            """, (role_id, name, role_type, prompt, desc))

        print("✓ 默认 AI 角色已插入")

        conn.commit()
        cursor.close()
        conn.close()

        print("\n" + "=" * 50)
        print("数据库初始化完成！")
        print("=" * 50)

    except Exception as e:
        print(f"初始化失败: {e}")
        return False

    return True


def create_test_user():
    """创建测试用户（用于开发调试）"""
    import hashlib

    def hash_password(pwd: str) -> str:
        return hashlib.sha256(pwd.encode()).hexdigest()

    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4"
        )
        cursor = conn.cursor()

        # 检查是否已有 testuser
        cursor.execute("SELECT id FROM users WHERE username = 'testuser'")
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO users (username, password_hash, nickname, email) 
                VALUES (%s, %s, %s, %s)
            """, ("testuser", hash_password("123456"), "测试用户", "test@example.com"))
            conn.commit()
            print("✓ 测试用户已创建: testuser / 123456")
        else:
            print("测试用户已存在")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"创建测试用户失败: {e}")


if __name__ == "__main__":
    if init_database():
        create_test_user()