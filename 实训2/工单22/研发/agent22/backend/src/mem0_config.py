"""
mem0 配置构造器

踩坑经验（来自原项目 E:/16---实训2/22--工单22）：
- embedder.config 不传 embedding_dims：硅基流动 bge 不支持 matryoshka 维度参数
- vector_store.config 不传 embedding_model_dims：Chroma 不接受该字段
- LLM/Embedder 统一走 openai provider + openai_base_url 指向硅基流动

本模块负责构造 mem0 Memory 所需的配置字典，集中管理所有与 mem0 相关的配置，
避免配置散落在各处，便于统一维护和调整。
"""
# __future__.annotations：延迟类型注解求值
from __future__ import annotations

# os：标准库，读取环境变量（SILICONFLOW_* 和 CHROMA_PATH）
import os
# Path：pathlib 标准库，跨平台路径操作，用于构造 ChromaDB 持久化路径
from pathlib import Path

# dotenv：第三方包，从 .env 文件加载环境变量
from dotenv import load_dotenv

# 加载 .env 文件（若此模块被单独 import 时环境变量可能尚未初始化）
load_dotenv()

# ── 从环境变量读取所有配置项（均有合理的默认值，方便本地开发） ─────────
# 硅基流动 API 密钥（必填，无默认值）
API_KEY   = os.getenv("SILICONFLOW_API_KEY")
# 硅基流动 OpenAI 兼容 API 端点
BASE_URL  = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
# mem0 用于提取/推理记忆的 LLM 模型（约 20s/次，Qwen2.5-72B 综合效果较好）
LLM_MODEL = os.getenv("SILICONFLOW_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
# mem0 用于向量化（embedding）记忆文本的嵌入模型
# bge-large-zh-v1.5：北京智源研究院的中文 embedding 模型，768 维，中文语义能力强
EMB_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

# ── ChromaDB 持久化路径 ────────────────────────────────────────────────
# Path(__file__).parent.parent：从 src/ 上溯两级到 backend/ 目录
_here = Path(__file__).parent.parent          # backend/ 目录

# CHROMA_PATH：向量数据库持久化目录，优先从环境变量读取，默认 backend/data/chroma/
# .resolve()：将相对路径转换为绝对路径，防止工作目录变化导致路径错误
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(_here / "data" / "chroma"))).resolve()

# ── 必要性检查：API Key 为空时立即报错，避免等到实际调用时才发现配置错误 ──
if not API_KEY:
    # RuntimeError：程序运行时配置错误，属于不可继续运行的严重错误
    raise RuntimeError("缺少 SILICONFLOW_API_KEY，请检查 .env")


def build_mem0_config() -> dict:
    """构造并返回 mem0.Memory.from_config() 所需的配置字典。

    mem0 支持多种 LLM、Embedder 和向量数据库的组合。
    本函数固定使用：
        LLM:          硅基流动 Qwen（通过 openai provider 的 openai_base_url 切换）
        Embedder:     硅基流动 bge-large-zh-v1.5（通过 openai provider 的 openai_base_url 切换）
        Vector Store: ChromaDB 本地持久化（不需要独立服务，轻量部署）

    踩坑说明：
        - embedder.config 不传 embedding_dims：硅基流动 bge 不支持 Matryoshka 降维参数，
          传入会导致 API 报错
        - vector_store.config 不传 embedding_model_dims：ChromaDB provider 不接受此字段，
          传入会导致 mem0 初始化异常

    Returns:
        dict: mem0 配置字典，可直接传入 Memory.from_config(config)。
    """
    # 确保 ChromaDB 持久化目录存在（parents=True：自动创建父目录；exist_ok=True：目录已存在不报错）
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    return {
        # ── LLM 配置：mem0 用此模型从对话中提取/推理记忆条目 ──────────────
        "llm": {
            "provider": "openai",   # 使用 openai 兼容的调用方式
            "config": {
                "model": LLM_MODEL,           # Qwen 模型（硅基流动平台）
                "api_key": API_KEY,           # 硅基流动 API 密钥
                "openai_base_url": BASE_URL,  # 关键：将请求路由到硅基流动而非 OpenAI
                "temperature": 0.1,           # 记忆提取任务需要低随机性（高度确定性）
                "max_tokens": 2000,           # 单次提取的最大输出 token 数
            },
        },
        # ── Embedder 配置：mem0 用此模型将记忆文本向量化，用于语义检索 ──────
        "embedder": {
            "provider": "openai",   # 复用 openai provider（支持自定义 base_url）
            "config": {
                "model": EMB_MODEL,           # bge-large-zh-v1.5（中文 embedding 模型）
                "api_key": API_KEY,           # 硅基流动 API 密钥
                "openai_base_url": BASE_URL,  # 指向硅基流动 embedding API
                # 注意：不传 embedding_dims，硅基流动 bge 不支持 matryoshka 维度参数
            },
        },
        # ── 向量数据库配置：存储和检索记忆向量 ────────────────────────────
        "vector_store": {
            "provider": "chroma",   # ChromaDB：轻量级本地向量数据库，无需独立服务
            "config": {
                "collection_name": "agent22_memory",  # 集合名称（类似数据库中的表名）
                "path": str(CHROMA_PATH),             # ChromaDB 数据文件持久化路径
                # 注意：不传 embedding_model_dims，Chroma provider 不接受此字段
            },
        },
        # mem0 配置协议版本（v1.1 对应 mem0 2.0 的配置格式）
        "version": "v1.1",
    }
