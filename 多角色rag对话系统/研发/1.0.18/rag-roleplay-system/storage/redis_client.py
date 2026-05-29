# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
Redis 客户端封装

负责：
- 短期记忆存储（最近20条对话）
- 查询缓存
- 会话状态管理

⚠️ 常改动的地方：
1. 短期记忆保留条数（当前硬编码 19，即最多 20 条）可调整
2. 缓存默认 TTL（3600 秒）可根据业务需求修改
3. 短期记忆 TTL（REDIS_SHORT_TERM_TTL）在 config.py 中定义，默认 86400 秒（24小时）

⚠️ 注意事项：
1. Key 格式：short_term:user:{user_id}:role:{role}:session:{session_id}
2. 使用 List 结构，LPUSH 新消息，LTRIM 保留最近 20 条（索引 0-19）
3. TTL 24小时，过期自动删除
4. Redis 连接失败时不会干扰主流程，仅记录警告并降级（短期记忆功能不可用）
"""

import redis
import json
import logging
from typing import List, Dict, Optional

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_SHORT_TERM_TTL

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 客户端单例"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
        return cls._instance

    def connect(self):
        """连接 Redis，失败时降级为 None，不影响主流程"""
        if self._client:
            return
        try:
            self._client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,          # 自动解码为字符串
                socket_connect_timeout=2        # 连接超时 2 秒
            )
            self._client.ping()
            logger.info(f"Redis 连接成功: {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            # 警告而非错误，允许系统在没有 Redis 时继续运行（但短期记忆失效）
            logger.warning(f"Redis 连接失败: {e}（短期记忆功能不可用）")
            self._client = None

    @property
    def client(self):
        """获取原始 Redis 客户端，自动连接"""
        if self._client is None:
            self.connect()
        return self._client

    # ==================== 短期记忆 ====================

    def _get_memory_key(self, user_id: str, role: str, session_id: str) -> str:
        """
        生成短期记忆的 Redis Key

        ⚠️ 注意事项：session_id 必须与对话绑定的会话一致，否则记忆无法关联
        """
        return f"short_term:user:{user_id}:role:{role}:session:{session_id}"

    def save_short_term_memory(self, user_id: str, role: str, session_id: str,
                               question: str, answer: str):
        """
        保存短期记忆（最近20条对话）

        ⚠️ 常改动：如果想保留更多或更少对话，可修改 ltrim 的第二个参数（当前 19 意味着保留索引 0-19，共20条）
        ⚠️ 易错点：user_id、role、session_id 必须一致，否则记忆无法正确聚合
        """
        if self.client is None:
            return

        key = self._get_memory_key(user_id, role, session_id)
        # 将问题和答案打包为 JSON，ensure_ascii=False 保留中文
        data = json.dumps({"q": question, "a": answer}, ensure_ascii=False)

        self.client.lpush(key, data)
        # 只保留最近 20 条（索引 0 ~ 19）
        self.client.ltrim(key, 0, 19)
        # 设置过期时间（默认 24 小时）
        self.client.expire(key, REDIS_SHORT_TERM_TTL)

        logger.debug(f"[Redis] 保存短期记忆: {key}, 当前长度: {self.client.llen(key)}")

    def get_short_term_memory(self, user_id: str, role: str, session_id: str, limit: int = 10) -> List[Dict]:
        """
        获取短期记忆，返回最近的 N 条对话（默认 10）

        Returns:
            List[Dict]: [{"q": "问题", "a": "答案"}, ...]
        """
        if self.client is None:
            return []

        key = self._get_memory_key(user_id, role, session_id)

        if not self.client.exists(key):
            logger.debug(f"[Redis] 短期记忆不存在: {key}")
            return []

        # lrange 返回索引 0 到 limit-1 的元素（最新的在索引 0）
        history = self.client.lrange(key, 0, limit - 1)
        result = []
        for h in history:
            if h:
                try:
                    result.append(json.loads(h))
                except json.JSONDecodeError:
                    # 忽略损坏的记录
                    pass

        logger.debug(f"[Redis] 获取短期记忆: {key}, 数量: {len(result)}")
        return result

    def delete_short_term_memory(self, user_id: str, role: str, session_id: str):
        """删除短期记忆（通常在切换会话或重置对话时调用）"""
        if self.client is None:
            return
        key = self._get_memory_key(user_id, role, session_id)
        self.client.delete(key)

    # ==================== 查询缓存 ====================

    def cache_get(self, key: str) -> Optional[str]:
        """从缓存中获取值，不存在返回 None"""
        if self.client is None:
            return None
        return self.client.get(key)

    def cache_set(self, key: str, value: str, ttl: int = 3600):
        """
        设置缓存，默认 TTL 为 3600 秒（1小时）
        ⚠️ 常改动：可根据业务调整默认 TTL，或调用时传入自定义 ttl
        """
        if self.client is None:
            return
        self.client.setex(key, ttl, value)

    def cache_delete(self, key: str):
        """删除缓存"""
        if self.client is None:
            return
        self.client.delete(key)


# 全局单例实例
redis_client = RedisClient()