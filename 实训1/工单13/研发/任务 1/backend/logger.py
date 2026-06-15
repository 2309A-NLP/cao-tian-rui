"""
日志管理模块
统一的日志记录系统，写入 logs/ 目录

日志文件（共5个）：
  rag_app.log      — 所有模块的通用日志（DEBUG及以上）
  rag_error.log    — WARNING/ERROR 汇总（方便快速定位问题）
  rag_qa.log       — 问答质量日志：每次问答的查询、检索来源、LLM回答、幻觉检查
  rag_perf.log     — 性能日志：检索耗时、LLM耗时、token数、各阶段耗时
  rag_audit.log    — 审计日志：谁、什么时候、问了什么、用了哪些来源、是否引用
"""

import os
import sys
import logging
import traceback
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional


def _resolve_log_dir(log_dir: str = "logs") -> str:
    """将相对路径的 log_dir 解析为绝对路径：优先项目根下的 logs/"""
    p = Path(log_dir)
    if p.is_absolute():
        return log_dir
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    project_log = project_root / log_dir
    if project_log.exists() or not (backend_dir / log_dir).exists():
        return str(project_log)
    return log_dir


class _LogManager:
    """
    统一日志管理器

    维护 5 个日志文件，全部按天轮转，保留 30 天。
    每个 logger 共享同一个文件句柄（按名称复用）。
    """

    _instances: dict[str, logging.Logger] = {}

    # 日志文件定义: {handlername: (filename, level)}
    _LOG_FILES = {
        "app": ("rag_app.log", logging.DEBUG),
        "error": ("rag_error.log", logging.WARNING),
        "qa": ("rag_qa.log", logging.INFO),
        "perf": ("rag_perf.log", logging.INFO),
        "audit": ("rag_audit.log", logging.INFO),
    }

    @classmethod
    def _setup_handlers(cls, log_dir: str) -> dict[str, logging.Handler]:
        """创建或获取所有日志文件的处理器（按日志目录缓存）"""
        if not hasattr(cls, "_handler_cache"):
            cls._handler_cache: dict[str, dict[str, logging.Handler]] = {}

        if log_dir in cls._handler_cache:
            return cls._handler_cache[log_dir]

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        handlers = {}
        for name, (filename, level) in cls._LOG_FILES.items():
            handler = TimedRotatingFileHandler(
                filename=str(log_path / filename),
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8",
            )
            handler.setLevel(level)
            # 通用格式（所有文件对齐）
            fmt = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(fmt)
            handlers[name] = handler

        cls._handler_cache[log_dir] = handlers
        return handlers

    @classmethod
    def get_logger(cls, name: str = "rag_app", log_dir: str = "logs",
                   level: str = "DEBUG", console_output: bool = True) -> logging.Logger:
        """获取或创建日志器"""
        if name in cls._instances:
            return cls._instances[name]

        log_dir = _resolve_log_dir(log_dir)
        handlers = cls._setup_handlers(log_dir)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        # 所有 handler 以 _LOG_FILES 的定义为准，按 level 过滤
        for handler in handlers.values():
            logger.addHandler(handler)

        # 控制台输出：只输出 INFO+，避免大量 DEBUG 刷屏
        if console_output:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(getattr(logging, level.upper(), logging.DEBUG))
            console.setFormatter(logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(console)

        cls._instances[name] = logger
        return logger


def get_logger(name: str = "rag_app") -> logging.Logger:
    """快捷获取日志器"""
    return _LogManager.get_logger(name)


def get_qa_logger() -> logging.Logger:
    """获取问答质量日志器（写 rag_qa.log）"""
    return _LogManager.get_logger("rag_qa")


def get_perf_logger() -> logging.Logger:
    """获取性能日志器（写 rag_perf.log）"""
    return _LogManager.get_logger("rag_perf")


def get_audit_logger() -> logging.Logger:
    """获取审计日志器（写 rag_audit.log）"""
    return _LogManager.get_logger("rag_audit")


def log_exception(logger: logging.Logger, msg: str = "发生异常", exc: Optional[Exception] = None):
    """
    记录异常并保留完整栈信息
    不吞掉异常，回滚记录所有错误信息
    """
    if exc is None:
        exc = sys.exc_info()[1]
    if exc is None:
        logger.error(f"{msg} - 无异常信息")
        return

    tb = traceback.format_exc() if sys.exc_info()[0] else "".join(traceback.format_tb(exc.__traceback__))

    # 获取异常链中的所有异常
    chain = []
    current = exc
    while current:
        chain.append(f"{type(current).__name__}: {str(current)}")
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    logger.error(
        f"{msg}\n"
        f"  异常类型: {type(exc).__name__}\n"
        f"  异常信息: {str(exc)}\n"
        f"  异常链: {' -> '.join(chain)}\n"
        f"  调用栈:\n{tb}"
    )
