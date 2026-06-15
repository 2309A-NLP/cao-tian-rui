"""
base_config.py — 配置管理基类

核心思想：
  将所有配置集中到一个 dataclass 中，支持：
    - 直接属性访问（config.llm_model）
    - JSON 持久化（save / load）
    - 环境变量覆盖（from_env）
  
复用方式：
  1. 继承 BaseConfig，定义你自己的配置字段
  2. 添加 @classmethod from_dict 解析你的业务字段
  3. 调用 load_or_create() 自动恢复或创建默认配置
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _env(var: str, default: Any = None) -> Any:
    """获取环境变量，返回字符串或默认值"""
    return os.environ.get(var, default)


def _typed_env(var: str, typ: type, default: Any = None) -> Any:
    """获取环境变量并转换类型"""
    val = os.environ.get(var)
    if val is None:
        return default
    try:
        return typ(val)
    except (ValueError, TypeError):
        return default


# ──────────────────────────────────────────────
# 配置基类
# ──────────────────────────────────────────────

@dataclass
class BaseConfig:
    """
    配置管理基类。
    
    子类示例:
        @dataclass
        class MyConfig(BaseConfig):
            # 定义业务字段
            db_url: str = "sqlite:///local.db"
            debug: bool = False
            max_retries: int = 3
    
    使用:
        cfg = MyConfig.load_or_create("config.json")
        print(cfg.db_url)
        cfg.db_url = "postgres://..."
        cfg.save()
    """

    # ── 框架通用字段（子类可覆盖） ──
    config_path: str = "config.json"           # 配置文件路径
    verbose: bool = False                      # 是否输出详细日志

    # ──────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """将配置序列化为 JSON 并保存到文件"""
        path = path or self.config_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "BaseConfig":
        """从 JSON 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "BaseConfig":
        """从字典创建配置实例（子类应覆盖以处理业务字段）"""
        field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)

    @classmethod
    def load_or_create(cls, path: str, **defaults) -> "BaseConfig":
        """
        尝试从 path 加载配置，文件不存在则创建默认配置并保存。
        
        用法:
            cfg = MyConfig.load_or_create("my_config.json", debug=True)
        """
        if os.path.exists(path):
            return cls.load(path)
        instance = cls(**defaults)
        instance.config_path = path
        instance.save()
        return instance

    # ──────────────────────────────────────────
    # 环境变量覆盖
    # ──────────────────────────────────────────

    @classmethod
    def from_env(cls, prefix: str = "") -> "BaseConfig":
        """
        从环境变量构造配置。
        环境变量命名规则: {PREFIX}{FIELD_NAME}，全大写。
        例如 prefix="LLM_" 会读取 LLM_MODEL、LLM_API_KEY 等。
        
        类型自动推断（支持 str / int / float / bool）。
        """
        fields = cls.__dataclass_fields__  # type: ignore
        data = {}
        for fname, fdef in fields.items():
            env_key = f"{prefix}{fname.upper()}"
            raw = os.environ.get(env_key)
            if raw is not None:
                # 类型转换
                if fdef.type is bool or fdef.type == bool:
                    data[fname] = raw.lower() in ("1", "true", "yes", "on")
                elif fdef.type is int or fdef.type == int:
                    data[fname] = int(raw)
                elif fdef.type is float or fdef.type == float:
                    data[fname] = float(raw)
                else:
                    data[fname] = raw
        return cls(**data)

    # ──────────────────────────────────────────
    # 实用方法
    # ──────────────────────────────────────────

    def to_dict(self) -> dict:
        """转为普通字典"""
        return asdict(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({asdict(self)})"
