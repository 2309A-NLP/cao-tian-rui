# -*- coding: utf-8 -*-
"""
记忆管理器

负责：
- 短期记忆（Redis）：保存最近多轮对话，用于上下文连贯
- 长期记忆（Milvus）：保存用户的身份、偏好等关键信息，支持向量检索
- 信息提取：从对话中识别用户身份、职业、偏好等并保存到长期记忆

⚠️ 常改动的地方：
1. 短期记忆保留条数（通过 save_short_term 中的 LTRIM 控制，在 redis_client 中）
2. 长期记忆检索时的 top_k 和 min_score（当前 top_k=5, min_score=0.0，但在 routes 中调用时传入了 min_score=0.3）
3. 长期记忆集合名称（MILVUS_LONG_TERM_MEMORY_COLLECTION 在 config.py 中定义）
4. 信息提取（extract_important_info）中的正则表达式模式和关键词列表
5. 长期记忆保存时的内容长度限制（当前 300 字符）

⚠️ 注意事项：
1. 长期记忆只保存用户信息（identity, occupation, preference, health, name），不保存 advice 类型（助手的回答）
2. 长期记忆集合会自动创建，需要 embedding_func 来获取向量维度
3. 短期记忆基于 Redis，如果 Redis 不可用则短期记忆功能失效
4. 信息提取中的正则表达式可能误匹配，需要根据实际数据调整
"""

import re
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime

from storage.redis_client import redis_client
from storage.milvus_client import milvus_client
from config import MILVUS_LONG_TERM_MEMORY_COLLECTION, MILVUS_INDEX_PARAMS
from utils.logger import get_logger

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器：管理短期记忆（Redis）和长期记忆（Milvus）"""

    def __init__(self, embedding_func=None):
        """
        初始化记忆管理器

        参数:
            embedding_func: Embedding 包装类，用于生成长期记忆的向量
        """
        self.embedding_func = embedding_func
        self.long_term_collection = MILVUS_LONG_TERM_MEMORY_COLLECTION
        self.memory_logger = get_logger("memory")   # 专用的记忆日志记录器

    def _check_and_create_collection(self):
        """
        检查长期记忆集合是否存在，不存在则自动创建
        ⚠️ 注意事项：需要 embedding_func 已初始化，以获取向量维度
        """
        if milvus_client.has_collection(self.long_term_collection):
            return True

        try:
            from pymilvus import Collection, CollectionSchema, FieldSchema, DataType

            if self.embedding_func is None:
                logger.error("embedding_func 未设置，无法获取维度")
                return False

            # 测试嵌入获取维度
            test_embedding = self.embedding_func.embed_query("test")
            dim = len(test_embedding)
            logger.info(f"长期记忆集合向量维度: {dim}")

            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="memory_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="ai_role_id", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="conversation_record_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=30),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="importance", dtype=DataType.FLOAT),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]

            schema = CollectionSchema(fields, description="长期记忆存储", enable_dynamic_field=True)
            collection = Collection(name=self.long_term_collection, schema=schema)

            # 创建索引并加载
            collection.create_index(field_name="embedding", index_params=MILVUS_INDEX_PARAMS)
            collection.load()

            logger.info(f"长期记忆集合已创建: {self.long_term_collection}, 维度: {dim}")
            return True

        except Exception as e:
            logger.error(f"创建长期记忆集合失败: {e}")
            return False

    # ==================== 短期记忆（Redis） ====================

    def save_short_term(self, user_id: str, role: str, session_id: str,
                        question: str, answer: str):
        """保存短期记忆（最近对话）到 Redis"""
        redis_client.save_short_term_memory(user_id, role, session_id, question, answer)
        logger.debug(f"[短期记忆] 已保存: user={user_id}, role={role}")

    def get_short_term(self, user_id: str, role: str, session_id: str, limit: int = 10) -> List[Dict]:
        """
        获取短期记忆，默认返回最近 10 条
        ⚠️ 常改动：limit 默认值可根据对话长度调整
        """
        memories = redis_client.get_short_term_memory(user_id, role, session_id, limit)
        if memories:
            logger.debug(f"[短期记忆] 获取到 {len(memories)} 条")
        return memories

    def format_short_term_text(self, memories: List[Dict]) -> str:
        """将短期记忆格式化为提示词中的文本块"""
        if not memories:
            return ""
        lines = ["【历史对话】"]
        for mem in memories:
            lines.append(f"用户: {mem['q']}")
            lines.append(f"助手: {mem['a']}")
        return "\n".join(lines)

    # ==================== 长期记忆（Milvus） - 只保存用户信息 ====================

    def save_long_term(self, user_id: str, role: str, conversation_id: str,
                       content: str, memory_type: str, importance: float = 0.5):
        """
        保存长期记忆（仅保存用户身份、偏好等重要信息）
        ⚠️ 禁止保存 advice 类型（助手的回答），已在调用前过滤
        """
        # 禁止保存 advice 类型（安全防护）
        if memory_type == "advice":
            print(f"[记忆保存] 跳过 advice 类型，不保存到长期记忆")
            return

        if not self._check_and_create_collection():
            logger.warning("长期记忆集合不可用，跳过保存")
            return

        if self.embedding_func is None:
            logger.warning("embedding_func 未设置，跳过长期记忆保存")
            return

        # 限制保存的内容长度（重要信息通常较短）
        # ⚠️ 常改动：可调整长度阈值
        if len(content) > 300:
            logger.debug(f"[长期记忆] 内容过长({len(content)}字符)，已截断")
            content = content[:300]

        try:
            embedding = self.embedding_func.embed_query(content[:500])

            from pymilvus import Collection
            collection = milvus_client.get_collection(self.long_term_collection)

            # 生成唯一 memory_id
            memory_id = f"mem_{int(time.time())}_{user_id}_{memory_type}"
            data = [{
                "memory_id": memory_id,
                "user_id": str(user_id),
                "ai_role_id": role,
                "conversation_record_id": conversation_id,
                "memory_type": memory_type,
                "content": content[:5000],
                "importance": importance,
                "created_at": datetime.now().isoformat(),
                "embedding": embedding
            }]

            collection.insert(data)
            collection.flush()

            print(f"[记忆保存] 类型={memory_type}, 重要性={importance}, 内容={content[:80]}...")
            self.memory_logger.info(f"保存 {memory_type}: user={user_id}, role={role}, content={content[:50]}...")

        except Exception as e:
            logger.error(f"[长期记忆] 保存失败: {e}")

    def retrieve_long_term(self, user_id: str, role: str, query: str,
                           top_k: int = 5, memory_types: List[str] = None,
                           min_score: float = 0.0) -> List[Dict]:
        """
        检索长期记忆，返回相关记忆列表（按相似度得分降序）
        ⚠️ 常改动：top_k（默认5）和 min_score（默认0.0）可根据需要调整
        ⚠️ 注意事项：返回的记忆中包含 score（相似度）、importance（重要性）等字段
        """
        if not milvus_client.has_collection(self.long_term_collection):
            return []

        if self.embedding_func is None:
            return []

        try:
            query_embedding = self.embedding_func.embed_query(query)

            # 构造过滤表达式：user_id 和 ai_role_id 必须匹配
            expr = f'user_id == "{user_id}" and ai_role_id == "{role}"'
            if memory_types:
                type_filter = " or ".join([f'memory_type == "{t}"' for t in memory_types])
                expr = f"{expr} and ({type_filter})"

            from pymilvus import Collection
            collection = milvus_client.get_collection(self.long_term_collection)
            collection.load()

            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 8}},
                limit=top_k,
                expr=expr,
                output_fields=["memory_id", "memory_type", "importance", "created_at", "content"]
            )

            memories = []
            for hits in results:
                for hit in hits:
                    content = ""
                    try:
                        if hasattr(hit.entity, 'content'):
                            content = hit.entity.content
                        elif hasattr(hit.entity, '__getitem__'):
                            content = hit.entity['content']
                    except:
                        pass

                    if content:
                        score = hit.score
                        mem_type = getattr(hit.entity, 'memory_type', 'unknown')
                        print(f"[记忆检索] 得分={score:.4f} | 类型={mem_type} | 内容={content[:60]}...")

                        if score >= min_score:
                            memories.append({
                                "content": content[:500],
                                "score": score,
                                "type": mem_type,
                                "importance": getattr(hit.entity, 'importance', 0),
                                "created_at": getattr(hit.entity, 'created_at', '')
                            })

            if memories:
                print(f"[记忆检索汇总] 共检索到 {len(memories)} 条记忆，最高得分={memories[0]['score']:.4f}")
                self.memory_logger.info(
                    f"检索到 {len(memories)} 条记忆: user={user_id}, role={role}, query={query[:50]}...")
            return memories

        except Exception as e:
            logger.error(f"[长期记忆] 检索失败: {e}")
            return []

    # ==================== 提取重要信息（只提取用户信息） ====================

    def extract_important_info(self, question: str, answer: str) -> List[tuple]:
        """
        从对话中提取重要信息（仅记录用户身份、偏好等关键事实）
        不保存助手的回答内容（advice）
        返回: [(content, memory_type, importance), ...]

        ⚠️ 常改动：
            1. 身份信息正则表达式模式
            2. 偏好关键词列表
            3. 健康关键词列表
            4. 姓名/昵称正则模式
        """
        important_info = []

        # 1. 身份信息（最高优先级 importance=0.9）
        identity_patterns = [
            (r"我是([^，,。！？；：\n]{1,30})(?:[，,。！？；：\n]|$)", "identity"),
            (r"我是一名([^，,。！？；：\n]{1,30})(?:[，,。！？；：\n]|$)", "identity"),
            (r"我从事([^，,。！？；：\n]{1,30})(?:工作|职业)", "identity"),
            (r"我的([^，,。！？；：\n]{0,5})是([^，,。！？；：\n]{1,30})", "identity"),
            (r"外号([^，,。！？；：\n]{1,20})", "identity"),
            (r"叫我([^，,。！？；：\n]{1,20})(?:吧|就好|,|，)", "identity"),
            (r"我今年(\d{1,3})岁", "identity"),
            (r"我(\d{1,3})岁了", "identity"),
        ]

        for pattern, mem_type in identity_patterns:
            try:
                match = re.search(pattern, question)
                if match:
                    content = match.group(0).strip()
                    important_info.append((content, mem_type, 0.9))
                    print(f"[信息提取] 身份信息: {content}")
                    break
            except re.error as e:
                print(f"[正则错误] pattern={pattern}, error={e}")
                continue

        # 2. 职业信息（单独处理，importance=0.88）
        job_patterns = [
            r"我是([^，,。！？；：\n]{1,20})(?:老师|医生|律师|工程师|学生|程序员|经理|总监)",
            r"我当([^，,。！？；：\n]{1,20})(?:老师|医生|律师|工程师|学生|程序员)",
            r"我是个([^，,。！？；：\n]{1,20})(?:老师|医生|律师|工程师|学生)",
        ]
        for pattern in job_patterns:
            try:
                match = re.search(pattern, question)
                if match:
                    content = match.group(0).strip()
                    existing = [c for c, _, _ in important_info if c == content]
                    if not existing:
                        important_info.append((content, "occupation", 0.88))
                        print(f"[信息提取] 职业信息: {content}")
                    break
            except re.error as e:
                print(f"[正则错误] pattern={pattern}, error={e}")
                continue

        # 3. 偏好信息 (importance=0.7)
        preference_keywords = ["我喜欢", "我讨厌", "我爱", "我恨", "我习惯", "我偏好", "我经常"]
        for kw in preference_keywords:
            if kw in question:
                sentences = re.split(r'[。！？]', question)
                for sent in sentences:
                    if kw in sent:
                        content = sent.strip()
                        if len(content) < 100:
                            important_info.append((content, "preference", 0.7))
                            print(f"[信息提取] 偏好信息: {content}")
                            break
                break

        # 4. 健康状况 (importance=0.8)
        health_keywords = ["我患有", "我得了", "我生病", "我不舒服", "我的症状", "我过敏"]
        for kw in health_keywords:
            if kw in question:
                important_info.append((question[:100], "health", 0.8))
                print(f"[信息提取] 健康信息: {question[:80]}...")
                break

        # 5. 姓名/昵称 (importance=0.85)
        name_patterns = [
            r"我叫([^，,。！？；：\n]{1,20})",
            r"我的名字叫([^，,。！？；：\n]{1,20})",
            r"可以叫我([^，,。！？；：\n]{1,20})",
        ]
        for pattern in name_patterns:
            try:
                match = re.search(pattern, question)
                if match:
                    content = match.group(0).strip()
                    existing = [c for c, _, _ in important_info if c == content]
                    if not existing:
                        important_info.append((content, "name", 0.85))
                        print(f"[信息提取] 姓名信息: {content}")
                    break
            except re.error as e:
                print(f"[正则错误] pattern={pattern}, error={e}")
                continue

        # ❌ 不再保存 advice 类型（助手的回答）
        # 原来的 if "建议" in answer 逻辑已移除

        # 去重（基于内容前50字符）
        seen = set()
        unique_info = []
        for content, mem_type, importance in important_info:
            key = content[:50]
            if key not in seen:
                seen.add(key)
                unique_info.append((content, mem_type, importance))

        return unique_info


# 全局变量
_memory_manager = None


def init_memory_manager(embedding_func):
    """初始化记忆管理器（在 main.py 中调用）"""
    global _memory_manager
    _memory_manager = MemoryManager(embedding_func)
    return _memory_manager


def get_memory_manager():
    """获取记忆管理器实例（单例）"""
    return _memory_manager