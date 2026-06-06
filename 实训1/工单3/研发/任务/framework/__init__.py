"""
Framework __init__ — 统一导出所有基类，方便 import
"""

from .base_config import BaseConfig
from .base_llm import BaseLLMProvider, LLMFactory, LLMError, LLMConfigError, LLMRateLimitError
from .base_vector_store import BaseVectorStore, VectorStoreFactory, Document, SearchResult
from .base_chunker import BaseChunker, ChunkerFactory, FixedSizeChunker, SentenceChunker, Chunk
from .base_rag import BaseRAGEngine, ChatSession, ChatMessage, RAGResult

__all__ = [
    # 配置
    "BaseConfig",
    # LLM
    "BaseLLMProvider",
    "LLMFactory",
    "LLMError",
    "LLMConfigError",
    "LLMRateLimitError",
    # 向量存储
    "BaseVectorStore",
    "VectorStoreFactory",
    "Document",
    "SearchResult",
    # 分块
    "BaseChunker",
    "ChunkerFactory",
    "FixedSizeChunker",
    "SentenceChunker",
    "Chunk",
    # RAG
    "BaseRAGEngine",
    "ChatSession",
    "ChatMessage",
    "RAGResult",
]
