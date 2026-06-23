"""
日志模块 — 简洁实用
"""
import sys
import logging
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler


def get_logger(name: str = "agent", log_dir: str = "logs") -> logging.Logger:
    """获取日志器"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 文件输出（按天轮转，保留30天）
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    fh = TimedRotatingFileHandler(
        filename=str(log_path / "agent.log"),
        when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    return logger
