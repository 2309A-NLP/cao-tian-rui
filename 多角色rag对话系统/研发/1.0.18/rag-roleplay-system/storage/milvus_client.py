# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
Milvus 客户端封装

负责：
- 连接/断开 Milvus
- 集合管理（创建、删除、检查）
- 提供索引和搜索参数

⚠️ 常改动的地方：
1. 集合名（如 rag_chatbot_v2）在 config.py 中定义
2. 索引参数（MILVUS_INDEX_PARAMS）和搜索参数（MILVUS_SEARCH_PARAMS）在 config.py 中调整
3. 如需更换 embedding 模型，必须确保向量维度与 Milvus 集合维度一致（当前 bge-m3 是 1024 维）

⚠️ 注意事项：
1. 集合名在 config.py 中定义，与之前保持一致，否则会重建索引
2. 向量维度和 embedding 模型强相关：bge-m3 → 1024 维，m3e-base → 768 维
3. 长期记忆集合（long_term_memory）维度同上，需保持一致
4. 连接别名固定为 "default"，多处依赖该别名
"""

import logging
from typing import Optional, Dict
from pymilvus import connections, utility, Collection

from config import (
    MILVUS_HOST, MILVUS_PORT,
    MILVUS_INDEX_PARAMS, MILVUS_SEARCH_PARAMS
)

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus 客户端单例类"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connected = False
        return cls._instance

    def connect(self):
        """
        连接 Milvus（使用 config 中的 host 和 port）
        ⚠️ 注意：连接别名 hardcode 为 "default"
        """
        if self._connected:
            return
        try:
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
            self._connected = True
            logger.info(f"Milvus 连接成功: {MILVUS_HOST}:{MILVUS_PORT}")
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise

    def disconnect(self):
        """断开与 Milvus 的连接"""
        if self._connected:
            connections.disconnect("default")
            self._connected = False

    def has_collection(self, collection_name: str) -> bool:
        """检查指定集合是否已在 Milvus 中存在"""
        self.connect()
        return utility.has_collection(collection_name)

    def drop_collection(self, collection_name: str):
        """
        删除集合（数据将永久丢失！）
        ⚠️ 谨慎使用：通常在重置知识库时调用
        """
        self.connect()
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.warning(f"已删除集合: {collection_name}")

    def get_collection(self, collection_name: str) -> Collection:
        """获取集合对象，用于直接操作 Milvus（如统计数量）"""
        self.connect()
        return Collection(collection_name)

    def get_index_params(self) -> Dict:
        """
        获取索引参数（副本，避免外部修改原字典）
        ⚠️ 常改动：可在 config.py 中调整索引类型（如 IVF_FLAT, HNSW）
        """
        return MILVUS_INDEX_PARAMS.copy()

    def get_search_params(self) -> Dict:
        """
        获取搜索参数（副本，避免外部修改原字典）
        ⚠️ 常改动：可调整 nprobe（搜索聚类数）以平衡速度/召回率
        """
        return MILVUS_SEARCH_PARAMS.copy()


# 全局单例实例
milvus_client = MilvusClient()