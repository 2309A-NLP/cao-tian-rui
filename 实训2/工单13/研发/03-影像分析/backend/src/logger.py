"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
结构化 JSON 日志（含工单编号），与规格书 §6 对齐。

本模块实现了以下功能：
1. JsonFormatter：将日志记录格式化为 JSON 字符串，便于日志采集系统（如 ELK）解析
2. get_logger：创建并配置日志记录器（同时输出到控制台和文件）
3. log_request：快捷方法，用于记录 API 请求的结构化事件日志
"""

# json：Python 内置模块，将 Python 对象序列化为 JSON 字符串
import json

# logging：Python 内置的标准日志模块
import logging

# sys：Python 内置模块，此处用于获取标准输出流（sys.stdout）
import sys

# datetime：Python 内置模块，用于获取当前时间；timezone 用于处理时区
from datetime import datetime, timezone

# RotatingFileHandler：Python 内置 logging 模块的文件处理器
# 支持按文件大小自动轮转，避免单个日志文件过大
from logging.handlers import RotatingFileHandler

# Any：类型提示，表示任意类型
from typing import Any

# 从配置模块导入日志相关配置
from .config import LOG_DIR, LOG_LEVEL, WORKORDER_ID


class JsonFormatter(logging.Formatter):
    """
    自定义日志格式化器：将日志记录序列化为 JSON 格式字符串。

    继承自 logging.Formatter，重写 format 方法。
    输出格式示例：
    {"timestamp": "2025-01-01T00:00:00.000Z", "level": "INFO", "message": "...", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        将单条日志记录格式化为 JSON 字符串。

        参数：
            record (logging.LogRecord)：Python logging 系统传入的日志记录对象

        返回值：
            str：JSON 格式的日志字符串
        """
        # 构建基础 JSON 字段字典
        payload: dict[str, Any] = {
            # 当前 UTC 时间，格式为 ISO 8601（精确到毫秒），末尾 Z 表示 UTC 时区
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            # 工单标识，便于在多服务日志中过滤本工单的日志
            "workorder_id": WORKORDER_ID,
            # 日志级别名称（如 INFO、ERROR）
            "level": record.levelname,
            # 日志记录器名称（如 wt13.api、wt13.vlm）
            "logger": record.name,
            # 日志消息正文
            "message": record.getMessage(),
        }

        # 若日志记录携带了额外的结构化字段（通过 extra={"payload": {...}} 传入），则合并进来
        if hasattr(record, "payload") and isinstance(record.payload, dict):
            payload.update(record.payload)

        # 若有异常信息，格式化后追加到 JSON 中
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # 将字典序列化为 JSON 字符串，ensure_ascii=False 保留中文字符（不转义为 \uXXXX）
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "wt13") -> logging.Logger:
    """
    获取或创建一个配置好的 JSON 格式日志记录器。

    日志同时输出到：
    1. 控制台（stdout）
    2. backend/logs/app.log（所有级别，按 50MB 轮转，保留 7 个备份）
    3. backend/logs/error.log（仅 ERROR 及以上，按 50MB 轮转，保留 7 个备份）

    参数：
        name (str)：日志记录器的名称，建议使用模块层级命名（如 "wt13.api"）

    返回值：
        logging.Logger：配置好的日志记录器对象
    """
    # 获取（或创建）指定名称的日志记录器
    logger = logging.getLogger(name)

    # 若已有处理器（handlers），说明已初始化，直接返回避免重复添加
    if logger.handlers:
        return logger

    # 设置最低日志级别，低于此级别的日志会被忽略
    logger.setLevel(LOG_LEVEL.upper())

    # 创建 JSON 格式化器实例（所有处理器共用）
    formatter = JsonFormatter()

    # ── 控制台处理器：将日志输出到标准输出（终端）──
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)          # 使用 JSON 格式化器
    logger.addHandler(stream)               # 注册到 logger

    # ── app.log 处理器：记录所有级别的日志 ──
    # maxBytes=50MB：单个文件超过 50MB 时自动创建新文件
    # backupCount=7：最多保留 7 个历史备份文件
    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=50 * 1024 * 1024, backupCount=7, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)   # 使用 JSON 格式化器
    logger.addHandler(file_handler)        # 注册到 logger

    # ── error.log 处理器：只记录 ERROR 及以上级别的日志 ──
    error_handler = RotatingFileHandler(
        LOG_DIR / "error.log", maxBytes=50 * 1024 * 1024, backupCount=7, encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)  # 此处理器的最低级别为 ERROR
    error_handler.setFormatter(formatter)  # 使用 JSON 格式化器
    logger.addHandler(error_handler)       # 注册到 logger

    # 禁止日志向上传播到根 logger，避免日志被打印两次
    logger.propagate = False
    return logger


def log_request(logger: logging.Logger, level: str, **fields: Any) -> None:
    """
    快捷方法：以 extra=payload 方式记录一条结构化事件日志。

    将任意关键字参数作为结构化字段写入 JSON 日志，便于后续分析。

    参数：
        logger (logging.Logger)：要写入的日志记录器
        level (str)：日志级别字符串，如 "info"、"warning"、"error"
        **fields：任意关键字参数，作为结构化字段写入日志（如 request_id、latency_ms 等）

    使用示例：
        log_request(logger, "info", request_id="abc", task="vqa", latency_ms=200)
    """
    # 根据 level 字符串动态获取对应的日志方法（如 logger.info、logger.error）
    # 若 level 不存在则默认使用 logger.info
    log_fn = getattr(logger, level.lower(), logger.info)

    # 从 fields 中弹出 "message" 字段作为日志消息正文，若没有则默认为 "request"
    msg = fields.pop("message", "request")

    # 调用日志方法，将剩余字段通过 extra={"payload": fields} 传给 JsonFormatter
    log_fn(msg, extra={"payload": fields})
