"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-健康咨询

记忆系统桥接模块（自包含版本）。
- 数据存储在本项目 data/chroma 目录，与其他工单完全独立
- 所有操作失败均静默返回（不抛异常），保证主流程不受影响
- recall()  : 每次对话前调用，检索患者历史记忆
- remember(): fire-and-forget 后台写入，不阻塞主线程

mem0 + chromadb 为可选依赖：未安装时记忆功能自动禁用，其余功能正常运行。

依赖说明：
  mem0 包：长期记忆管理库（https://mem0.ai），支持向量存储、LLM 摘要、语义检索
  chromadb 包：轻量级本地向量数据库，mem0 的后端存储之一
"""
import os          # 标准库：读取环境变量（LLM API Key 等）
import threading   # 标准库：创建后台线程（异步写入记忆，不阻塞请求）
import logging     # 标准库：记录调试和警告日志
from pathlib import Path       # 标准库：跨平台路径操作
from dotenv import load_dotenv  # python-dotenv 包：加载 .env 环境变量

# 初始化模块级日志记录器（日志名为模块路径，方便过滤）
logger = logging.getLogger(__name__)

# 加载 .env 文件（确保环境变量在 os.getenv() 前就绪）
load_dotenv()

# ── 从环境变量读取 LLM 和向量模型配置 ──
API_KEY   = os.getenv("SILICONFLOW_API_KEY", "")                           # 硅基流动 API Key
BASE_URL  = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")  # 兼容 OpenAI 的 API 地址
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")           # 用于记忆摘要的 LLM
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")        # 中文向量化模型（BAAI 出品）

# 本项目自己的 ChromaDB 数据目录（独立于其他工单，避免数据污染）
_CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"

# 患者 user_id 前缀，区分不同领域的记忆（隔离 wt12 和其他工单数据）
_DOMAIN_PREFIX = "patient"
# 记忆内容最大字符数（防止过长文本占用向量空间和 token）
_MAX_MSG_CHARS = 400
# ChromaDB 集合名称（本工单专用，不与 wt11/wt13 等冲突）
_COLLECTION    = "wt12_memory"

# ── 单例模式：mem0.Memory 实例 ──
_mem_instance = None
_init_lock    = threading.Lock()  # 线程锁，防止并发初始化创建多个实例


def _build_config() -> dict:
    """
    构建 mem0.Memory.from_config() 所需的配置字典。

    配置分三部分：
      - llm: 用于将对话摘要提炼为结构化记忆的语言模型
      - embedder: 将记忆文本转为向量的嵌入模型
      - vector_store: 向量的持久化存储（使用本地 ChromaDB）

    返回:
        dict: mem0 配置字典
    """
    return {
        "llm": {
            "provider": "openai",        # 使用 OpenAI 兼容接口
            "config": {
                "model": LLM_MODEL,
                "api_key": API_KEY,
                "openai_base_url": BASE_URL,  # 指向硅基流动
                "temperature": 0.1,       # 低温度：摘要生成要准确，不要创意
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",        # 使用 OpenAI 兼容嵌入接口
            "config": {
                "model": EMB_MODEL,       # BAAI/bge-large-zh-v1.5：中文优化嵌入模型
                "api_key": API_KEY,
                "openai_base_url": BASE_URL,
            },
        },
        "vector_store": {
            "provider": "chroma",         # 向量存储后端：ChromaDB（本地文件存储）
            "config": {
                "collection_name": _COLLECTION,  # 集合名（命名空间隔离）
                "path": str(_CHROMA_DIR),         # 数据目录（持久化到磁盘）
            },
        },
        "version": "v1.1",               # mem0 配置版本号
    }


def _get_mem():
    """
    获取 mem0.Memory 单例（双重检查锁定，线程安全的懒加载）。

    首次调用时尝试导入 mem0 包并初始化 Memory 实例。
    如果 mem0 未安装或初始化失败，_mem_instance 保持 None，
    记忆功能静默禁用，不影响主流程。

    返回:
        mem0.Memory 实例，或 None（记忆功能禁用时）
    """
    global _mem_instance
    if _mem_instance is not None:
        return _mem_instance  # 快速路径：已初始化，直接返回

    # 加锁避免多线程同时初始化（双重检查锁定模式）
    with _init_lock:
        if _mem_instance is not None:
            return _mem_instance  # 锁内再检查一次，防止重复初始化
        try:
            # mem0 包：长期记忆管理库，支持多种 LLM 和向量存储后端
            from mem0 import Memory
            _mem_instance = Memory.from_config(_build_config())  # 按配置初始化
            logger.info("[memory_bridge] mem0 初始化成功，chroma=%s", _CHROMA_DIR)
        except Exception as exc:
            # 捕获所有异常（ImportError/连接失败/配置错误），降级为禁用状态
            logger.debug("[memory_bridge] 初始化失败，记忆功能禁用: %s", exc)
            _mem_instance = None  # 保持 None，后续调用不再重试

    return _mem_instance


def _truncate(text: str) -> str:
    """
    截断过长文本，防止超出向量化和 LLM 的处理限制。

    参数:
        text (str): 原始文本

    返回:
        str: 截断后的文本（超长时追加"...(截断)"提示）
    """
    return text[:_MAX_MSG_CHARS] + "...(截断)" if len(text) > _MAX_MSG_CHARS else text


def recall(patient_id: str, query: str, limit: int = 5) -> list:
    """
    语义检索患者历史记忆（每次对话前调用）。

    通过向量相似度找出与当前问题最相关的历史记忆条目。
    失败时静默返回空列表，不影响主流程。

    参数:
        patient_id (str): 患者唯一标识（来自请求的 patient_id 字段）
        query (str): 当前问题（用于计算与历史记忆的相似度）
        limit (int): 最多返回的记忆条目数，默认 5

    返回:
        list: 记忆条目字典列表，每条包含 "memory" 字段；失败时返回 []
    """
    try:
        mem = _get_mem()
        if mem is None:
            return []  # 记忆功能禁用，返回空列表

        # 构造用户 ID：加上领域前缀，隔离不同工单的患者记忆
        user_id = f"{_DOMAIN_PREFIX}_{patient_id}"

        # mem.search()：按 user_id 过滤后，对 query 做语义相似度检索
        # filters 参数限制只搜索该患者的记忆
        results = mem.search(query, filters={"user_id": user_id}, limit=limit)

        # mem0 不同版本的返回格式不一致：新版返回 dict（含 "results" key），旧版返回 list
        if isinstance(results, dict):
            return results.get("results", [])
        return results or []

    except Exception as exc:
        # 任何异常（网络/向量库/解析错误）都静默处理
        logger.warning("[memory_bridge.recall] 失败: %s", exc)
        return []


def remember(patient_id: str, query: str, reply: str) -> None:
    """
    将本轮对话（问 + 答）异步写入患者记忆（fire-and-forget 模式）。

    在后台守护线程中执行，不阻塞 HTTP 响应的返回。
    失败时只打印警告日志，不影响主流程。

    参数:
        patient_id (str): 患者唯一标识
        query (str): 用户问题
        reply (str): Agent 回答
    """
    def _do():
        """后台线程实际执行的写入函数。"""
        try:
            mem = _get_mem()
            if mem is None:
                return  # 记忆功能禁用，直接退出

            user_id = f"{_DOMAIN_PREFIX}_{patient_id}"

            # mem.add()：将对话消息列表处理为结构化记忆并写入向量存储
            # mem0 内部会调用 LLM 将对话提炼为关键信息（摘要式记忆）
            mem.add(
                messages=[
                    {"role": "user",      "content": _truncate(query)},  # 截断防超限
                    {"role": "assistant", "content": _truncate(reply)},  # 截断防超限
                ],
                user_id=user_id,
                metadata={
                    "domain": "medical",  # 领域标签，便于后期过滤查询
                    "source": "wt12",     # 来源标记，标识本工单写入的记忆
                },
            )
        except Exception as exc:
            logger.warning("[memory_bridge.remember] 写入失败: %s", exc)

    # 创建守护线程（daemon=True：主进程退出时自动销毁，不阻塞关闭）
    threading.Thread(target=_do, name=f"mem_{patient_id}", daemon=True).start()


def format_for_prompt(memories: list) -> str:
    """
    将召回的记忆条目列表格式化为可插入 LLM prompt 的文本块。

    参数:
        memories (list): recall() 返回的记忆条目列表

    返回:
        str: 格式化的记忆文本（空列表时返回空字符串）
    """
    if not memories:
        return ""  # 无记忆时返回空字符串，调用方条件包裹记忆块

    lines = []
    for i, m in enumerate(memories, 1):
        # 兼容 mem0 不同版本的字段名："memory"（新版）/ "content"（旧版）/ 兜底 str()
        content = m.get("memory") or m.get("content") or str(m)
        lines.append(f"  {i}. {content}")  # 编号格式，清晰易读

    return "\n".join(lines)  # 多条记忆换行拼接
