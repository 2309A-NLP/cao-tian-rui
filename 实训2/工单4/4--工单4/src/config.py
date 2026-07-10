"""
config.py - 从 .env 加载配置，统一暴露给其他模块
所有敏感信息只在 .env 里，代码里不出现明文密钥
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")


def _require(key: str) -> str:
    """读取必填环境变量，缺失时给出明确提示"""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"缺少必填配置：{key}\n"
            f"请复制 .env.example 为 .env 并填入对应值"
        )
    return val


# ── API 配置 ──────────────────────────────────────────────
SILICONFLOW_API_KEY: str = _require("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL: str = os.getenv(
    "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
)

# NL2SQL 使用大模型，准确率更高
NL2SQL_MODEL: str = os.getenv("NL2SQL_MODEL", "Qwen/Qwen2.5-72B-Instruct")
# 生成自然语言回答
ANSWER_MODEL: str = os.getenv("ANSWER_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# ── 数据路径 ──────────────────────────────────────────────
def _resolve_path(env_key: str, default: Path) -> Path:
    """读取路径配置，相对路径以项目根目录为基准转换为绝对路径"""
    raw = os.getenv(env_key)
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else (_BASE_DIR / p).resolve()

DB_PATH: Path = _resolve_path("DB_PATH", _BASE_DIR / "data" / "博金杯比赛数据.db")
QUESTION_FILE: Path = _resolve_path("QUESTION_FILE", _BASE_DIR / "question.jsonl")
OUTPUT_FILE: Path = _resolve_path("OUTPUT_FILE", _BASE_DIR / "outputs" / "answers.jsonl")

# ── 执行参数 ──────────────────────────────────────────────
MAX_RETRY: int = int(os.getenv("MAX_RETRY", "3"))
REQUEST_INTERVAL: float = float(os.getenv("REQUEST_INTERVAL", "0.5"))
