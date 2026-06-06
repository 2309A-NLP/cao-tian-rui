"""
配置管理模块
集中管理所有配置项，支持环境变量覆盖
"""
import os
import json
import platform
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _windows_f_path() -> str:
    """返回 Windows F: 路径格式的 BGE-M3 模型路径"""
    return r"F:\4--专业所有安装的软件及改动设置\2-6--专高6\bge-m3"


def _wsl_f_path() -> str:
    """返回 WSL /mnt/f/ 路径格式的 BGE-M3 模型路径"""
    return "/mnt/f/4--专业所有安装的软件及改动设置/2-6--专高6/bge-m3"


def _default_model_path() -> str:
    """自动检测可用的模型路径，先试 WSL /mnt/ 路径，再试 Windows 路径"""
    if os.path.exists(_wsl_f_path()):
        return _wsl_f_path()
    if os.path.exists(_windows_f_path()):
        return _windows_f_path()
    return _wsl_f_path()  # fallback


def _default_project_root() -> str:
    """根据操作系统自动返回项目根路径"""
    cwd = os.getcwd()
    if platform.system() == "Windows":
        return cwd
    return "/mnt/e/10--agent--任务/任务 1"


@dataclass
class AppConfig:
    # 项目根路径（自动检测）
    project_root: str = ""  # 由 __post_init__ 填充
    
    # 日志配置
    log_dir: str = "logs"
    log_level: str = "DEBUG"
    log_retention_days: int = 30
    
    # 数据库配置
    db_host: str = "127.0.0.1"
    db_port: int = 3307
    db_user: str = "root"
    db_password: str = "root"
    db_name: str = "rag_chat"
    
    # Milvus 向量数据库配置
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection: str = "doc_chunks"
    milvus_dim: int = 1024
    
    # 向量模型配置
    embedding_model_path: str = ""
    embedding_device: str = "auto"
    embedding_batch_size: int = 32
    embedding_max_length: int = 8192
    
    # RAG 配置
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 10
    similarity_threshold: float = 0.0

    # LLM 配置（Mock模式用于无API时）
    llm_provider: str = "deepseek"
    llm_api_key: str = "sk-2c7be8b46e41438fa0547fd5736a9636"
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    
    # PDF 路径
    pdf_path: str = ""  # 由 __post_init__ 填充
    
    # 中间产物目录
    output_dir: str = "output"
    
    # 知识库目录（放 PDF 的地方）
    knowledge_base_dir: str = "knowledge_base"
    
    # 会话相关
    session_timeout_minutes: int = 30
    
    def __post_init__(self):
        """初始化后处理：填充默认值 + 解析环境变量覆盖"""
        if not self.project_root:
            self.project_root = _default_project_root()
        if not self.embedding_model_path:
            self.embedding_model_path = _default_model_path()
        if not self.pdf_path:
            root = Path(self.project_root)
            self.pdf_path = str(root.parent / "招股说明书1-无水印.pdf")
        self._apply_env_overrides()
        self._resolve_paths()
    
    def _apply_env_overrides(self):
        """环境变量覆盖配置，命名规则 APP_xxx_yyy -> xxx.yyy"""
        env_map = {
            "APP_DB_HOST": "db_host",
            "APP_DB_PORT": "db_port",
            "APP_DB_USER": "db_user",
            "APP_DB_PASSWORD": "db_password",
            "APP_DB_NAME": "db_name",
            "APP_LLM_PROVIDER": "llm_provider",
            "APP_LLM_API_KEY": "llm_api_key",
            "APP_LLM_BASE_URL": "llm_base_url",
            "APP_LLM_MODEL": "llm_model",
            "APP_LOG_LEVEL": "log_level",
            "APP_MILVUS_HOST": "milvus_host",
            "APP_MILVUS_PORT": "milvus_port",
        }
        _INT_FIELDS = {"db_port", "milvus_port", "chunk_size", "top_k",
                        "embedding_batch_size", "embedding_max_length",
                        "log_retention_days", "session_timeout_minutes", "milvus_dim"}
        for env_key, attr_name in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                if attr_name in _INT_FIELDS:
                    val = int(val)
                setattr(self, attr_name, val)
    
    def _resolve_paths(self):
        """将相对路径解析为绝对路径"""
        root = Path(self.project_root)
        if not Path(self.log_dir).is_absolute():
            self.log_dir = str(root / self.log_dir)
        if not Path(self.output_dir).is_absolute():
            self.output_dir = str(root / self.output_dir)
        if not Path(self.knowledge_base_dir).is_absolute():
            self.knowledge_base_dir = str(root / self.knowledge_base_dir)
    
    def to_dict(self):
        return asdict(self)
    
    def save(self, path: str = None):
        """保存配置到 JSON 文件"""
        if path is None:
            path = Path(self.project_root) / "config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str = None) -> "AppConfig":
        """从 JSON 文件加载配置"""
        if path is None:
            path = Path(os.getcwd()) / "config.json"
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# 全局单例
config = AppConfig.load()
