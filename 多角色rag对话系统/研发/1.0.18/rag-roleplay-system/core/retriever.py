# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
多路召回器

负责：
- Milvus 向量检索（语义相似度）
- MySQL 关键词检索（对话标题匹配）
- Redis 历史对话检索（短期记忆）
- 结果融合与去重

⚠️ 常改动的地方：
1. 各召回源的权重（weights 字典中的 milvus/mysql/redis 权重，当前分别为 0.6/0.2/0.2）
2. 检索时的 top_k 值（召回阶段各自可能取不同倍数，如 milvus 取 top_k*2）
3. MySQL 关键词提取方式（当前取 query[:20]）
4. Redis 键匹配模式（当前 pattern = f"*{query[:10]}*"）
5. 融合算法中的排名分数计算公式（当前 rank_score = weight / (rank + 1)）

⚠️ 注意事项：
1. 权重可调整，milvus 权重最高，融合后取前 5 条结果
2. 融合算法使用倒数排名融合（Reciprocal Rank Fusion）
3. 各召回源失败时会自动降级为空结果，不影响整体流程
4. Milvus 分数会进行归一化（除以最大分数），使不同检索源的分数可比
5. 文档去重基于 page_content[:100] 作为唯一标识
"""

import logging
from typing import List, Tuple, Dict
from collections import defaultdict

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MultiPathRetriever:
    """多路召回器：整合 Milvus、MySQL、Redis 三种检索方式并融合结果"""

    def __init__(self, vector_store=None, mysql_conn=None, redis_client=None):
        """
        初始化多路召回器

        参数:
            vector_store: LangChain 向量存储（Milvus）
            mysql_conn: MySQL 连接对象（需支持 cursor 上下文）
            redis_client: Redis 客户端实例
        """
        self.vector_store = vector_store
        self.mysql_conn = mysql_conn
        self.redis_client = redis_client

        # 召回权重（融合时使用）
        # ⚠️ 常改动：可调整各路的权重比例，总和无要求（融合时会分别使用）
        self.weights = {
            "milvus": 0.6,
            "mysql": 0.2,
            "redis": 0.2
        }

    def recall_from_milvus(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Milvus 向量检索，返回 (Document, 归一化分数) 列表
        ⚠️ 常改动：top_k 默认 10，可根据需要调整候选数量
        """
        if self.vector_store is None:
            return []
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            # 归一化分数：将原始相似度归一化到 [0,1] 区间（除以最大分数）
            max_score = max([s for _, s in results]) if results else 1.0
            normalized = [(doc, s / max_score) for doc, s in results]
            logger.debug(f"[召回-Milvus] {len(normalized)} 个文档")
            return normalized
        except Exception as e:
            logger.warning(f"[召回-Milvus] 失败: {e}")
            return []

    def recall_from_mysql(self, query: str, user_id: int = None, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        MySQL 关键词检索（基于对话标题的模糊匹配）
        ⚠️ 常改动：
            1. 关键词提取方式（当前取 query[:20]）
            2. 权重分数硬编码为 0.7（也可改为可配置）
        """
        if self.mysql_conn is None:
            return []

        documents = []
        # 提取关键词（前20个字符作为匹配词）
        # ⚠️ 常改动：可改用分词或更精细的关键词提取
        keywords = query[:20]

        try:
            with self.mysql_conn.cursor() as cursor:
                like = f"%{keywords}%"
                cursor.execute(
                    "SELECT conversation_id, title, role_type FROM conversations "
                    "WHERE title LIKE %s ORDER BY updated_at DESC LIMIT %s",
                    (like, top_k)
                )
                for row in cursor.fetchall():
                    content = f"对话: {row['title']} (角色: {row['role_type']})"
                    doc = Document(page_content=content, metadata={"source": "MySQL"})
                    # 固定分数 0.7（后续融合时还会乘以权重）
                    documents.append((doc, 0.7))
            logger.debug(f"[召回-MySQL] {len(documents)} 个文档")
        except Exception as e:
            logger.warning(f"[召回-MySQL] 失败: {e}")

        return documents

    def recall_from_redis(self, query: str, user_id: str = None, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Redis 历史对话检索（基于键模式匹配）
        ⚠️ 常改动：
            1. 键匹配模式（当前 pattern = f"*{query[:10]}*"）
            2. 固定分数硬编码为 0.6
        """
        if self.redis_client is None:
            return []

        documents = []
        try:
            # 取查询的前10个字符作为键匹配模式
            # ⚠️ 常改动：可改用更合适的检索策略（如对内容进行分词）
            pattern = f"*{query[:10]}*"
            for key in self.redis_client.keys(pattern)[:top_k]:
                value = self.redis_client.get(key)
                if value:
                    content = value[:300]   # 限制长度
                    doc = Document(page_content=content, metadata={"source": "Redis"})
                    documents.append((doc, 0.6))
            logger.debug(f"[召回-Redis] {len(documents)} 个文档")
        except Exception as e:
            logger.warning(f"[召回-Redis] 失败: {e}")

        return documents

    def _fusion(self, results_list: List[List[Tuple[Document, float]]]) -> List[Tuple[Document, float]]:
        """
        融合多路结果（倒数排名融合法）
        算法：对于每个文档，计算 score = sum(weight_i / (rank_i + 1))
        最终按总分降序排序，返回 top 5

        ⚠️ 常改动：
            1. 融合后的最终结果数量（当前硬编码 5）
            2. 排名分数计算公式（可改为 1/(rank+常数) 或其他函数）
        """
        doc_map = defaultdict(lambda: {"score": 0, "doc": None})
        source_weights = [self.weights["milvus"], self.weights["mysql"], self.weights["redis"]]

        # 遍历各路召回结果
        for source_idx, results in enumerate(results_list):
            for rank, (doc, _) in enumerate(results):
                # 使用文档内容的前100字符作为去重标识（避免长文本重复）
                doc_id = doc.page_content[:100]
                # 排名分数 = 权重 / (排名位置 + 1)
                rank_score = source_weights[source_idx] / (rank + 1)
                doc_map[doc_id]["score"] += rank_score
                if doc_map[doc_id]["doc"] is None:
                    doc_map[doc_id]["doc"] = doc

        # 按总分降序排序，取前5条非空文档
        sorted_results = sorted(doc_map.values(), key=lambda x: x["score"], reverse=True)
        return [(item["doc"], item["score"]) for item in sorted_results[:5] if item["doc"]]

    def retrieve(self, query: str, user_id: str = None, top_k: int = 5) -> List[Document]:
        """
        执行多路召回，返回融合后的最终文档列表（仅文档，不含分数）
        ⚠️ 常改动：top_k 参数当前未实际使用，内部融合固定取前5，可调整
        """
        # 三路召回（Milvus 取 2*top_k 候选，其他取 top_k 候选）
        milvus = self.recall_from_milvus(query, top_k=top_k * 2)
        mysql = self.recall_from_mysql(query, user_id=user_id, top_k=top_k)
        redis = self.recall_from_redis(query, user_id=user_id, top_k=top_k)

        # 融合
        fused = self._fusion([milvus, mysql, redis])
        return [doc for doc, _ in fused]


def create_retriever(vector_store, mysql_conn, redis_client) -> MultiPathRetriever:
    """工厂方法：创建多路召回器实例"""
    return MultiPathRetriever(vector_store, mysql_conn, redis_client)