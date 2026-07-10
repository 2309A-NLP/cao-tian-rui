"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-健康咨询

全局配置模块：从 .env 文件读取所有外部服务连接参数。
统一在此处定义，避免其他模块散落硬编码的密钥和地址。
"""
import os  # 标准库：通过 os.getenv() 读取环境变量

# python-dotenv 包：将 .env 文件中的键值对加载到 os.environ 中
# 必须在 os.getenv() 之前调用，否则 .env 文件中的变量尚未注入到环境
from dotenv import load_dotenv

load_dotenv()  # 加载项目根目录下的 .env 文件（当前工作目录查找，不报错即找到）

# ── LLM 配置（硅基流动 OpenAI 兼容 API）──
# SILICONFLOW_API_KEY：硅基流动平台的 API 密钥，必须在 .env 中配置
LLM_API_KEY  = os.getenv("SILICONFLOW_API_KEY", "")
# SILICONFLOW_BASE_URL：API 基础地址，默认为硅基流动官方地址
LLM_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
# LLM_MODEL：使用的模型名称，默认 Qwen2.5-72B-Instruct（高性能中文模型）
LLM_MODEL    = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# ── Neo4j 图数据库配置 ──
# NEO4J_URI：Bolt 协议连接地址（默认本地 Docker 容器）
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
# NEO4J_USER：数据库用户名（默认 neo4j）
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
# NEO4J_PASSWORD：数据库密码，必须在 .env 中配置，不提供默认值（强制要求）
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# 密码未配置时在导入阶段立即抛出异常，防止服务启动后才发现连接问题
if not NEO4J_PASSWORD:
    raise RuntimeError("NEO4J_PASSWORD 未配置，请在 .env 中设置")
