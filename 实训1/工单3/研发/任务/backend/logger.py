"""
日志管理模块
统一的日志记录系统，写入 logs/ 目录，支持分级日志、异常追踪
"""
import os
import sys
import logging
import traceback
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional


class LoggerManager:
    """
    日志管理器
    - 按大小轮转（10MB），避免跨进程重命名冲突
    - 同时输出到控制台和文件
    - 自动创建日志目录
    - 支持异常栈追踪
    """
    
    _instances: dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(
        cls,
        name: str = "rag_app",
        log_dir: str = "logs",
        level: str = "DEBUG",
        retention_days: int = 30,
        console_output: bool = True,
    ) -> logging.Logger:
        """获取或创建日志器"""
        if name in cls._instances:
            return cls._instances[name]
        
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 创建 logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        logger.handlers.clear()
        
        # 格式化
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        # 文件处理器（按大小轮转 10MB，避免 Windows 跨进程重命名冲突）
        log_file = log_path / f"{name}.log"
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=retention_days,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 错误日志单独文件（按大小轮转）
        error_log_file = log_path / f"{name}_error.log"
        error_handler = RotatingFileHandler(
            filename=str(error_log_file),
            maxBytes=10 * 1024 * 1024,
            backupCount=retention_days,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        cls._instances[name] = logger
        return logger


def get_logger(name: str = "rag_app") -> logging.Logger:
    """快捷获取日志器"""
    return LoggerManager.get_logger(name)


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
    
    # 获取完整异常链
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
