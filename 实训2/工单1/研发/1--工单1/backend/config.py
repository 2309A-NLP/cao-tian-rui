"""
配置模块 — 从 config.json 加载
"""
import json
import os
from pathlib import Path


class Config:
    """全局配置单例"""

    def __init__(self):
        # LLM
        self.llm_api_key: str = ""
        self.llm_base_url: str = "https://api.deepseek.com"
        self.llm_model: str = "deepseek-chat"
        self.llm_temperature: float = 0.3

        # Database
        self.db_host: str = "127.0.0.1"
        self.db_port: int = 3307
        self.db_user: str = "root"
        self.db_password: str = "root"
        self.db_name: str = "agent_money_book"

        # Others
        self.log_dir: str = "logs"
        self.max_tool_call_rounds: int = 5

    @classmethod
    def load(cls, path: str = None) -> "Config":
        """从 JSON 加载配置"""
        if path is None:
            # 默认在项目根目录找 config.json
            path = Path(__file__).resolve().parent.parent / "config.json"
        cfg = cls()
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        # 环境变量覆盖
        env_map = {
            "AGENT_LLM_KEY": "llm_api_key",
            "AGENT_LLM_URL": "llm_base_url",
            "AGENT_LLM_MODEL": "llm_model",
            "AGENT_DB_HOST": "db_host",
            "AGENT_DB_PORT": "db_port",
            "AGENT_DB_USER": "db_user",
            "AGENT_DB_PASS": "db_password",
            "AGENT_DB_NAME": "db_name",
        }
        for env_k, attr in env_map.items():
            val = os.environ.get(env_k)
            if val:
                if attr.endswith("port"):
                    val = int(val)
                setattr(cfg, attr, val)
        return cfg


# 快捷获取
def get_config() -> Config:
    return Config.load()
