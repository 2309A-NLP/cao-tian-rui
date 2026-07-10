"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
ChromaDB 向量库封装：build/query/count。

本模块封装了与 ChromaDB 向量数据库的所有交互：
- 文档写入（upsert，存在则更新，不存在则插入）
- 相似度检索（cosine 余弦相似度）
- 文档计数

技术细节：
- 嵌入模型：paraphrase-multilingual-MiniLM-L12-v2（384维，中英文皆可）
- 持久化：knowledge_base/chroma_db/
- 相似度度量：cosine（余弦相似度），取值 0~1，越高越相似
"""

# os：Python 内置模块，用于设置环境变量
import os

# Optional：类型提示，表示可以为 None
from typing import Optional

# 设置环境变量，告知 HuggingFace transformers 不联网下载模型（离线模式）
# 必须在导入 transformers/sentence_transformers 之前设置，否则无效
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# chromadb：轻量级开源向量数据库，支持本地持久化
# 用于存储文本向量并执行近似最近邻（ANN）搜索
# 安装方式：pip install chromadb
import chromadb

# Settings：ChromaDB 的配置类，用于控制客户端行为
from chromadb.config import Settings

# 从配置模块导入向量库相关参数
from .config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, EMBED_MODEL, RAG_SCORE_THRESHOLD, TOP_K

# 导入日志记录器
from .logger import get_logger

# 导入参考文档数据模型
from .models import RefDoc

# 获取本模块专用的日志记录器
logger = get_logger("wt13.rag_store")

# ── 全局单例缓存（延迟初始化，避免启动时立即加载大型模型）──
_embedder = None    # sentence-transformers 嵌入模型实例
_client = None      # ChromaDB 客户端实例
_collection = None  # ChromaDB 集合实例


def get_embedder():
    """
    获取（或延迟创建）文本嵌入模型单例。

    使用延迟初始化模式：首次调用时加载模型，后续调用直接返回缓存实例。
    SentenceTransformer 模型文件约几百 MB，加载一次缓存复用。

    返回值：
        SentenceTransformer：已加载的嵌入模型实例
    """
    global _embedder
    if _embedder is None:
        # sentence_transformers：Hugging Face 出品的句子嵌入库
        # 将文本转换为固定维度的向量，支持多语言语义相似度计算
        # 安装方式：pip install sentence-transformers
        from sentence_transformers import SentenceTransformer
        logger.info("加载嵌入模型", extra={"payload": {"model": EMBED_MODEL}})
        # 加载指定的多语言嵌入模型（从本地缓存或在线下载）
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection():
    """
    获取（或延迟创建）ChromaDB 集合单例。

    首次调用时创建持久化客户端和集合，后续调用直接返回缓存实例。

    返回值：
        chromadb.Collection：ChromaDB 集合对象，支持 upsert/query/count 操作
    """
    global _client, _collection
    if _collection is None:
        # 确保持久化目录存在（parents=True 创建中间目录，exist_ok=True 不报重复创建错误）
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        # 创建 PersistentClient：数据持久化到磁盘（重启后数据不丢失）
        # anonymized_telemetry=False：关闭匿名遥测数据上传（保护隐私）
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建指定名称的集合
        # hnsw:space="cosine"：使用余弦相似度作为距离度量
        # HNSW（Hierarchical Navigable Small World）是 ChromaDB 内置的近似最近邻索引算法
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},  # 指定向量空间为余弦相似度
        )
        logger.info(
            "ChromaDB 就绪",
            extra={"payload": {"collection": CHROMA_COLLECTION, "count": _collection.count()}},
        )
    return _collection


def count_docs() -> int:
    """
    查询向量知识库中的文档总数。

    返回值：
        int：文档总数，若向量库不可用则返回 0
    """
    try:
        return get_collection().count()   # 调用 ChromaDB 集合的 count 方法
    except Exception:
        return 0  # 任何异常（如数据库文件不存在）都静默返回 0


def add_documents(docs: list[dict]) -> int:
    """
    批量写入文档到向量库（upsert 语义：存在则更新，不存在则插入）。

    参数：
        docs (list[dict])：文档列表，每个文档格式为：
            {
                'id': str,         # 文档唯一标识符
                'text': str,       # 要嵌入的文本内容
                'metadata': dict   # 附加元数据（可选）
            }

    返回值：
        int：实际写入的文档数量
    """
    if not docs:
        return 0  # 空列表直接返回 0，不执行任何操作

    col = get_collection()     # 获取 ChromaDB 集合
    embedder = get_embedder()  # 获取嵌入模型

    # 提取各字段列表（向量化操作批量处理，比逐条处理更高效）
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metas = [d.get("metadata", {}) for d in docs]

    # 批量编码文本为向量
    # normalize_embeddings=True：将向量归一化为单位向量（L2 范数为 1）
    # 归一化后余弦相似度 = 点积，计算更高效
    # show_progress_bar=False：不显示进度条（批量脚本中设为 True 更好）
    # .tolist()：将 numpy 数组转换为 Python 列表（ChromaDB 要求）
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    # upsert：有则更新，无则插入（幂等操作，可安全重复执行）
    col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
    return len(docs)


def query(
    text: str,
    top_k: int = TOP_K,
    score_threshold: float = RAG_SCORE_THRESHOLD,
    where: Optional[dict] = None,
) -> list[RefDoc]:
    """
    根据输入文本执行语义相似度检索，返回最相关的文档列表。

    检索流程：
    1. 将查询文本编码为向量
    2. 在 ChromaDB 中执行 ANN 搜索
    3. 将余弦距离转换为相似度分值
    4. 过滤低于阈值的结果

    参数：
        text (str)：查询文本（用户问题或影像描述）
        top_k (int)：返回的最大文档数，默认来自配置
        score_threshold (float)：最低相似度阈值（0~1），低于此值的结果被过滤
        where (Optional[dict])：ChromaDB 的元数据过滤条件（如 {"modality": "CT"}）

    返回值：
        list[RefDoc]：相似度从高到低排序的参考文档列表（已过滤低分项）
    """
    # 查询文本为空则直接返回空列表
    if not text.strip():
        return []

    col = get_collection()  # 获取 ChromaDB 集合

    # 若知识库为空（尚未建立）则直接返回空列表
    if col.count() == 0:
        return []

    embedder = get_embedder()  # 获取嵌入模型

    # 将查询文本编码为归一化向量（[text] 是长度为 1 的列表，输出也是列表）
    q_emb = embedder.encode([text], normalize_embeddings=True, show_progress_bar=False).tolist()

    # 执行向量相似度检索
    # query_embeddings：查询向量列表
    # n_results：返回的最大结果数
    # where：可选的元数据过滤条件（None 表示不过滤）
    res = col.query(query_embeddings=q_emb, n_results=top_k, where=where)

    # 提取检索结果（[0] 是因为 query 支持批量查询，结果是二维列表，取第一条）
    docs = res.get("documents", [[]])[0]       # 文档文本列表
    metas = res.get("metadatas", [[]])[0]      # 元数据列表
    ids = res.get("ids", [[]])[0]              # 文档 ID 列表
    dists = res.get("distances", [[]])[0]      # 距离列表（余弦距离，范围 0~2，0 最相似）

    refs: list[RefDoc] = []
    # 遍历所有检索结果，将余弦距离转为相似度分值，过滤低分
    for doc, meta, doc_id, dist in zip(docs, metas, ids, dists):
        # 余弦距离 → 余弦相似度：score = 1 - distance
        # max(0.0, ...) 防止因浮点误差出现负数
        score = max(0.0, 1.0 - float(dist))

        # 低于阈值的文档直接跳过（不纳入结果）
        if score < score_threshold:
            continue

        # 构建参考文档对象并加入结果列表
        refs.append(
            RefDoc(
                doc_id=str(doc_id),                          # 文档 ID
                title=(meta or {}).get("title", ""),          # 从元数据取标题
                snippet=doc,                                   # 文档文本片段
                score=round(score, 4),                        # 保留 4 位小数的相似度分值
            )
        )
    return refs
