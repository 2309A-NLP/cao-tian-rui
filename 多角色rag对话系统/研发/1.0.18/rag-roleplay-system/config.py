# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
全局配置文件

⚠️ 常要改动的地方：
1. EMBEDDING_MODEL_PATH - 向量模型路径（根据你的实际路径修改）
2. BGE_RERANKER_PATH - 重排模型路径（根据你的实际路径修改）
3. MYSQL_PASSWORD - 数据库密码
4. OLLAMA_MODEL - 大模型名称

⚠️ 注意事项：
1. 所有集合名、表名与之前代码保持一致，无需重建数据库
2. Milvus 维度和模型必须匹配（bge-m3 是 1024 维，注意与 m3e-base 768 维区别）
"""
# -*- coding: utf-8 -*-
"""
全局配置文件
"""

import os
import torch
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# ==================== LLM 模式切换 ====================
# 选择模式: "ollama" 或 "deepseek"
# ⚠️ 常改动：切换本地/在线模型时修改此项
# LLM_MODE = "deepseek"  # "ollama"=本地 | "deepseek"=在线API
LLM_MODE = "ollama"  # "ollama"=本地 | "deepseek"=在线API

# ==================== 本地 Ollama 配置 ====================
# ⚠️ 常改动：更换模型名称时修改 OLLAMA_MODEL
OLLAMA_MODEL = "deepseek-r1:7b"   # 本地 Ollama 中的模型名，例如 llama3, qwen2.5:7b
OLLAMA_BASE_URL = "http://localhost:11434"   # Ollama 默认端口，一般无需改动

# ==================== 在线 DeepSeek API 配置 ====================
# ⚠️ 常改动：使用在线模式时，必须替换 API 密钥
DEEPSEEK_API_KEY = "sk-你的DeepSeek密钥"  # 替换为你的真实密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ==================== 通用 LLM 参数 ====================
# ⚠️ 常改动：可根据需要调整温度（创造性）和输出长度
LLM_TEMPERATURE = 0.3          # 0~1，越低回答越确定，越高越有创造性
LLM_NUM_PREDICT = 1024         # 最大生成 token 数

# ==================== Embedding 模型配置 ====================
def get_optimal_device():
    """自动选择最佳计算设备（GPU优先）"""
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            print(f"[设备检测] 使用 GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        except:
            return "cpu"
    return "cpu"

# ⚠️ 常改动：向量模型路径，根据你的实际存放位置修改
EMBEDDING_MODEL_PATH = r"F:\4--专业所有安装的软件及改动设置\2-6--专高6\bge-m3"
# 注意：bge-m3 模型输出维度为 1024，Milvus 集合的维度必须与此一致
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", get_optimal_device())

# ==================== Milvus 配置 ====================
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
# ⚠️ 集合名：如果更改，需要重新创建集合并重建索引
MILVUS_COLLECTION_NAME = "rag_chatbot_v2"
MILVUS_LONG_TERM_MEMORY_COLLECTION = "long_term_memory"

# Milvus 索引参数（影响检索性能）
MILVUS_INDEX_PARAMS = {
    "metric_type": "COSINE",      # 相似度计算方式：余弦相似度
    "index_type": "IVF_FLAT",     # 索引类型
    "params": {"nlist": 128}      # 聚类中心数
}

# Milvus 搜索参数（影响检索召回率）
MILVUS_SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"nprobe": 8}       # 搜索时访问的聚类数，越大召回越高但越慢
}

# ==================== MySQL 配置 ====================
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
# ⚠️ 常改动：数据库密码，根据你的实际密码修改
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_chatbot")

# ==================== Redis 配置 ====================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
# 短期记忆/缓存的过期时间（秒），默认 1 天
REDIS_SHORT_TERM_TTL = 86400

# ==================== BGE 重排模型配置 ====================
# ⚠️ 常改动：重排模型路径，根据你的实际位置修改
BGE_RERANKER_PATH = r"F:\4--专业所有安装的软件及改动设置\2-3--专高3\3-bge-reranker-base\bge-reranker-base"

# ==================== 知识库配置 ====================
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge_base"

# ==================== 缓存配置 ====================
CACHE_MAX_SIZE = 100   # 内存中缓存的最大问答对数量

# ==================== 法律关键词 ====================
# ⚠️ 常改动：根据业务需要增删关键词，用于识别法律相关问题
# 注意：列表中的正则表达式如 '第.*条' 会匹配任何包含"第"加任意字符加"条"的内容
LEGAL_KEYWORDS = [
    '法律', '民法典', '刑法', '规定', '合法', '违法', '权利', '义务',
    '合同', '侵权', '遗失物', '捡到', '失物', '归还', '赔偿', '责任',
    '诉讼', '法院', '判决', '条款', '第.*条', '物权', '债权'
]

# ==================== 角色提示词 ====================
# ⚠️ 常改动：可修改各个角色的系统提示词，调整回答风格或增加约束
# 注意：当前 lawyer 角色有特殊要求（必须引用来源、禁止编造），修改时请保留关键规则
ROLE_PROMPTS = {
    "friend": "你是温柔友善的虚拟朋友，聊天自然亲切。不要使用emoji。",
    "doctor": "你是专业医生，科学回答，不夸大。请明确引用医学指南来源。不要使用emoji。",
    "tcm": "你是中医师，用中医思路回答。请引用中医典籍或理论。不要使用emoji。",
    "psychologist": "你是心理咨询师，温和疏导情绪。不要使用emoji。",
    "lawyer": """你是法律咨询师，严谨普法。

【重要规则】
1. 只能回答知识库中明确包含的内容，不得编造任何法律条文
2. 回答时必须明确引用来源，格式如：《民法典》第X条
3. 如果知识库中没有相关信息，告知用户："知识库中未找到相关信息，建议咨询专业律师"
4. 禁止使用emoji

【回答格式】
📚 参考来源：《文档名称》""",
    "finance": "你是理财师，科普理财，不推荐具体产品。不要使用emoji。"
}