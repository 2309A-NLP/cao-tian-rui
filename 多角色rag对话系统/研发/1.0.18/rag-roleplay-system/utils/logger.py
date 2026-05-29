# -*- coding: utf-8 -*-
"""
日志配置模块 - 支持多级别日志输出和文件存储

负责：
- 配置全局根日志器（控制台 + 文件轮转）
- 创建专用日志器：rag（RAG检索）、memory（记忆操作）、conversation（对话记录）
- 提供统一的 get_logger 接口

⚠️ 常改动的地方：
1. 日志文件的大小限制（maxBytes，当前 app.log 10MB，error.log 5MB）
2. 备份文件数量（backupCount，app.log 10个，error.log 5个）
3. 日志格式（formatter）可根据需要调整
4. 如果新增其他专用日志器（如 api.log），可在此添加对应方法
5. 轮转策略：当前使用 RotatingFileHandler，可改为 TimedRotatingFileHandler（按时间）

⚠️ 注意事项：
1. 使用单例模式确保日志目录和处理器只初始化一次
2. 专用日志器设置了 propagate=False，避免日志重复输出到根日志器
3. 所有日志文件以 UTF-8 编码存储，支持中文
4. 日志目录默认为项目根目录下的 logs/，如不存在会自动创建
5. 根日志器的控制台输出级别为 INFO，文件输出级别为 DEBUG
"""

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path


class LoggerManager:
    """日志管理器（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 创建日志目录（相对路径，项目根目录下）
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # 日志文件路径定义
        # ⚠️ 常改动：可添加新的日志文件（如 api.log, db.log）
        self.app_log = self.log_dir / "app.log"          # 应用通用日志（DEBUG+）
        self.rag_log = self.log_dir / "rag.log"          # RAG 检索详细日志
        self.memory_log = self.log_dir / "memory.log"    # 记忆操作日志
        self.error_log = self.log_dir / "error.log"      # 仅 ERROR 级别日志
        self.conversation_log = self.log_dir / "conversation.log"  # 对话内容日志

        # 配置根日志器
        self._setup_root_logger()

        self._initialized = True

    def _setup_root_logger(self):
        """
        配置根日志器
        - 控制台输出：INFO 级别
        - app.log：DEBUG 级别，轮转
        - error.log：ERROR 级别，轮转
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # 清除已有的处理器（避免重复添加）
        root_logger.handlers.clear()

        # 控制台处理器（INFO 及以上）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)

        # 应用日志文件处理器（DEBUG 级别，包含详细信息）
        # ⚠️ 常改动：maxBytes=10MB, backupCount=10 可根据磁盘空间调整
        app_handler = logging.handlers.RotatingFileHandler(
            self.app_log, maxBytes=10 * 1024 * 1024, backupCount=10, encoding='utf-8'
        )
        app_handler.setLevel(logging.DEBUG)
        # 详细格式：时间 - 名称 - 级别 - 文件名:行号 - 消息
        app_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        app_handler.setFormatter(app_format)
        root_logger.addHandler(app_handler)

        # 错误日志文件处理器（仅 ERROR 级别）
        # ⚠️ 常改动：maxBytes=5MB, backupCount=5
        error_handler = logging.handlers.RotatingFileHandler(
            self.error_log, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(app_format)
        root_logger.addHandler(error_handler)

    def get_rag_logger(self):
        """获取 RAG 检索专用日志器（不传播到根日志器）"""
        logger = logging.getLogger("rag")

        # 避免重复添加处理器
        if not logger.handlers:
            rag_handler = logging.handlers.RotatingFileHandler(
                self.rag_log, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
            )
            rag_handler.setLevel(logging.DEBUG)
            # 简洁格式，专注于 RAG 事件
            rag_format = logging.Formatter(
                '%(asctime)s - [RAG] - %(message)s'
            )
            rag_handler.setFormatter(rag_format)
            logger.addHandler(rag_handler)
            # 禁止传播到根日志器，避免重复输出
            logger.propagate = False

        return logger

    def get_memory_logger(self):
        """获取记忆操作专用日志器（不传播到根日志器）"""
        logger = logging.getLogger("memory")

        if not logger.handlers:
            memory_handler = logging.handlers.RotatingFileHandler(
                self.memory_log, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
            )
            memory_handler.setLevel(logging.DEBUG)
            memory_format = logging.Formatter(
                '%(asctime)s - [MEMORY] - %(message)s'
            )
            memory_handler.setFormatter(memory_format)
            logger.addHandler(memory_handler)
            logger.propagate = False

        return logger

    def get_conversation_logger(self):
        """获取对话日志器（记录用户-助手完整对话，不传播）"""
        logger = logging.getLogger("conversation")

        if not logger.handlers:
            conv_handler = logging.handlers.RotatingFileHandler(
                self.conversation_log, maxBytes=10 * 1024 * 1024, backupCount=10, encoding='utf-8'
            )
            conv_handler.setLevel(logging.INFO)   # 仅记录一般信息，避免过度
            conv_format = logging.Formatter(
                '%(asctime)s - %(message)s'
            )
            conv_handler.setFormatter(conv_format)
            logger.addHandler(conv_handler)
            logger.propagate = False

        return logger


# 全局实例
logger_manager = LoggerManager()


def get_logger(name: str = None):
    """
    统一的日志获取接口
    ⚠️ 常改动：如果新增专用日志器，需要在此添加对应的 name 映射
    """
    if name == "rag":
        return logger_manager.get_rag_logger()
    elif name == "memory":
        return logger_manager.get_memory_logger()
    elif name == "conversation":
        return logger_manager.get_conversation_logger()
    else:
        # 返回普通 logger，通常使用调用模块的 __name__
        return logging.getLogger(name or __name__)