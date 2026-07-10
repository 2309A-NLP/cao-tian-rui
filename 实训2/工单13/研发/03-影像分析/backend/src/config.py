"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
全局配置：从 .env 读取，避免硬编码

本模块集中管理所有可配置项，通过读取项目根目录下的 .env 文件
将配置注入环境变量，再通过 os.getenv() 读取。
这样修改配置只需改 .env 文件，无需修改代码。
"""

# os：Python 内置模块，用于读取操作系统环境变量
import os

# pathlib.Path：面向对象的文件路径操作，比 os.path 更简洁
from pathlib import Path

# python-dotenv 库提供的 load_dotenv 函数：
# 读取 .env 文件并将其中的键值对加载到操作系统环境变量中
# 安装方式：pip install python-dotenv
from dotenv import load_dotenv

# 获取当前文件（config.py）的绝对路径，向上两级得到项目根目录（backend/）
_ROOT = Path(__file__).resolve().parent.parent

# 加载 backend/.env 文件中的环境变量（若文件不存在则静默跳过）
load_dotenv(_ROOT / ".env")

# ══════════════════════════════════════════════════════
# LLM / VLM 相关配置（大语言模型 / 视觉语言模型）
# ══════════════════════════════════════════════════════

# 硅基流动（SiliconFlow）平台的 API 密钥，用于调用在线 VLM/LLM 服务
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")

# 硅基流动 API 的基础 URL，使用 OpenAI 兼容协议
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# 视觉语言模型（VLM）名称：Qwen3-VL-32B-Instruct（阿里通义千问系列，支持图文输入）
# 相比 2.5-VL-72B 架构更新，成本更合理
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct")

# 纯文本大语言模型（LLM）名称：Qwen2.5-72B-Instruct（用于纯文本任务）
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# VLM 后端标识（目前仅支持 siliconflow，后续可扩展其他后端）
VLM_BACKEND = os.getenv("VLM_BACKEND", "siliconflow")

# VLM 调用超时时间（秒），防止单次请求无限等待
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT_SECONDS", "60"))

# VLM 调用失败时的最大重试次数（含第一次），采用指数退避策略
VLM_MAX_RETRIES = int(os.getenv("VLM_MAX_RETRIES", "3"))

# ══════════════════════════════════════════════════════
# 向量数据库（ChromaDB）相关配置
# ══════════════════════════════════════════════════════

# ChromaDB 持久化存储目录，默认在 backend/knowledge_base/chroma_db/
# ChromaDB：一个轻量级本地向量数据库，用于存储和检索文本向量
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(_ROOT / "knowledge_base" / "chroma_db")))

# ChromaDB 中的集合（Collection）名称，类似关系数据库中的"表名"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_docs")

# 文本嵌入模型名称：paraphrase-multilingual-MiniLM-L12-v2
# 来自 sentence-transformers 库，支持多语言（含中文），输出 384 维向量
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# RAG 检索时返回的最多文档数量
TOP_K = int(os.getenv("TOP_K", "5"))

# RAG 检索结果的最低相似度阈值，低于此分值的文档将被过滤掉
# 分值范围 0~1，越高表示越相似；0.3 是较宽松的阈值
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))

# ══════════════════════════════════════════════════════
# 业务限制（图像校验规则）
# ══════════════════════════════════════════════════════

# 允许上传的最大图片大小（MB）
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))

# 换算为字节数，用于实际比较
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

# 允许上传的最小图片分辨率（宽和高都不能低于此像素数）
MIN_IMAGE_RESOLUTION = int(os.getenv("MIN_IMAGE_RESOLUTION", "32"))

# 支持的图片格式集合（大写），用于校验 PIL 解析出的图片格式
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "JPG"}

# 用户问题（query）的最大字符数，超出则截断
MAX_QUERY_LENGTH = 500

# ══════════════════════════════════════════════════════
# 服务配置
# ══════════════════════════════════════════════════════

# HTTP 服务监听的 IP 地址，"0.0.0.0" 表示监听所有网络接口
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# HTTP 服务监听的端口号
API_PORT = int(os.getenv("API_PORT", "8013"))

# 日志级别（DEBUG / INFO / WARNING / ERROR / CRITICAL）
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 日志文件存储目录，默认在 backend/logs/
LOG_DIR = Path(os.getenv("LOG_DIR", str(_ROOT / "logs")))

# 若日志目录不存在则自动创建（parents=True 会同时创建中间目录，exist_ok=True 避免目录已存在时报错）
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════
# 合规声明
# ══════════════════════════════════════════════════════

# 免责声明文本，所有响应中自动附带，提醒用户本系统不构成医疗诊断建议
DISCLAIMER = os.getenv("DISCLAIMER", "本系统仅供参考，不构成医疗诊断建议，请遵医嘱")

# ══════════════════════════════════════════════════════
# 工单标识
# ══════════════════════════════════════════════════════

# 工单唯一标识符，用于日志追踪和 /health 接口返回
WORKORDER_ID = "NLP-Agent-医疗智能体-影像分析"
