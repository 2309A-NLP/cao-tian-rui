"""
base_vector_store.py — 向量存储抽象层

核心思想：
  定义向量数据库的统一接口。
  无论底层是 FAISS / Chroma / Milvus / 纯 numpy，
  业务代码都只操作 add_documents() / search() / save() / load()。

复用方式：
  1. 继承 BaseVectorStore，实现四个核心方法
  2. 注册到 VectorStoreFactory
  3. 你的 RAG 引擎拿到 BaseVectorStore 接口，不知道底层是什么
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class Document:
    """
    单个文档片段。
    
    属性:
        id: 唯一标识（如 "doc_0001"）
        text: 原始文本
        metadata: 额外信息（页码、来源文件名、章节标题等）
        embedding: 向量（可选，某些场景不需要显式存储）
    """
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class SearchResult:
    """
    检索结果。
    
    属性:
        document: 匹配的文档
        score: 相似度分数（越大越相似）
    """
    document: Document
    score: float


# ──────────────────────────────────────────────
# 向量存储抽象接口
# ──────────────────────────────────────────────

class BaseVectorStore(ABC):
    """
    向量存储抽象基类。
    
    所有向量数据库（FAISS、Chroma、Milvus、Weaviate 等）都实现这四个方法：
      add_documents(docs)  — 批量添加文档
      search(query, k)      — 语义搜索，返回 Top-K
      save(path)            — 持久化到磁盘
      load(path)            — 从磁盘恢复
    """

    # ── 生命期 ──

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> int:
        """
        批量添加文档到向量库。
        
        参数:
            documents: Document 列表（每个 Document 必须有 text，embedding 可选）
        
        返回:
            实际添加的文档数量
        """
        ...

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """
        语义搜索。
        
        参数:
            query: 查询文本（非向量，由内部自动编码）
            k: 返回 Top-K 结果
        
        返回:
            SearchResult 列表，按 score 降序排列
        """
        ...

    @abstractmethod
    def save(self, path: str) -> str:
        """
        将向量库持久化到磁盘。
        
        参数:
            path: 保存路径（可以是文件或目录）
        
        返回:
            实际保存的路径
        """
        ...

    @abstractmethod
    def load(self, path: str) -> int:
        """
        从磁盘恢复向量库。
        
        参数:
            path: 之前 save 的路径
        
        返回:
            加载的文档数量
        """
        ...

    # ── 辅助 ──

    @property
    @abstractmethod
    def size(self) -> int:
        """当前存储的文档数量"""
        ...

    def clear(self) -> None:
        """清空所有数据（可选覆写）"""
        raise NotImplementedError

    # ── 批处理 ──

    def add_documents_batched(
        self,
        documents: list[Document],
        batch_size: int = 64,
    ) -> int:
        """
        分批添加文档（避免大列表一次性吃满内存）。
        
        默认实现：按 batch_size 分批调用 add_documents。
        如果子类的 add_documents 已有分页逻辑，可覆盖此方法。
        """
        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            total += self.add_documents(batch)
        return total


# ──────────────────────────────────────────────
# 工厂
# ──────────────────────────────────────────────

class VectorStoreFactory:
    """
    向量存储工厂。
    
    用法:
        @VectorStoreFactory.register("faiss")
        class FAISSStore(BaseVectorStore):
            ...
        
        store = VectorStoreFactory.create("faiss", dim=768)
    """

    _registry: dict[str, type[BaseVectorStore]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(store_cls: type[BaseVectorStore]):
            cls._registry[name] = store_cls
            return store_cls
        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseVectorStore:
        if name not in cls._registry:
            raise ValueError(
                f"未知的向量存储: {name}。"
                f" 已注册: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_stores(cls) -> list[str]:
        return list(cls._registry.keys())
