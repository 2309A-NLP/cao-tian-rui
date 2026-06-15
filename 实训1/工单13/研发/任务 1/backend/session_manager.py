"""
Redis 对话会话管理器
支持多用户隔离、自动过期、多轮记忆
"""
import json
import time
import uuid
import re
from typing import Optional

from logger import get_logger, log_exception

# ChatSession 定义 — 简化版，用于 rag_engine.py 的 import
class ChatSession:
    """简化版对话会话，适配 rag_engine.py 接口"""
    def __init__(self, session_id: str = "", system_prompt: str = "", user_id: int = 0):
        self.session_id = session_id or f"ses_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.system_prompt = system_prompt
        self.messages: list[dict] = []
        self.user_id = user_id
        self.mode = ""
        self.history = self.messages

    def get_context(self) -> str:
        messages = self.messages[-6:]  # 最近 6 轮
        return "\n".join(
            f"{'用户' if m['role']=='user' else '助手'}: {m['content']}"
            for m in messages
        )

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def set_mode(self, mode: str):
        self.mode = mode


logger = get_logger("session")


class SessionManager:
    """基于 Redis 的会话管理器"""

    def __init__(self, host="127.0.0.1", port=6379, db=0, password="",
                 ttl_hours=24, max_history=10):
        import redis
        self.redis = redis.Redis(
            host=host, port=port, db=db, password=password or None,
            decode_responses=True,
        )
        self.ttl = ttl_hours * 3600
        self.max_history = max_history
        self._connected = False
        logger.info(f"SessionManager 初始化: Redis {host}:{port} db={db}")

    def _key_history(self, session_id: str) -> str:
        return f"session:{session_id}:history"

    def _key_meta(self, session_id: str) -> str:
        return f"session:{session_id}:meta"

    def _key_user_sessions(self, user_id: int) -> str:
        return f"user:{user_id}:sessions"

    def ping(self) -> bool:
        """检查 Redis 连接"""
        try:
            self.redis.ping()
            self._connected = True
            return True
        except Exception as e:
            log_exception(logger, "Redis ping 失败", e)
            self._connected = False
            return False

    def create_session(self, session_id: Optional[str] = None,
                       user_id: int = 0) -> str:
        """创建新会话，返回 session_id"""
        sid = session_id or f"ses_{uuid.uuid4().hex[:8]}"
        meta_key = self._key_meta(sid)
        now = int(time.time())

        # 元信息
        self.redis.hset(meta_key, mapping={
            "user_id": str(user_id),
            "created_at": str(now),
            "mode": "rag",
        })
        self.redis.expire(meta_key, self.ttl)

        # 用户->会话 索引
        if user_id:
            self.redis.sadd(self._key_user_sessions(user_id), sid)
            self.redis.expire(self._key_user_sessions(user_id), self.ttl)

        logger.debug(f"创建会话: {sid} (user={user_id})")
        return sid

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return self.redis.exists(self._key_meta(session_id)) > 0

    def get_session_ids(self, user_id: int) -> list[str]:
        """获取用户的所有会话ID"""
        return list(self.redis.smembers(self._key_user_sessions(user_id)))

    def set_mode(self, session_id: str, mode: str):
        """设置会话模式"""
        if mode in ("rag", "direct"):
            self.redis.hset(self._key_meta(session_id), "mode", mode)

    def get_mode(self, session_id: str) -> str:
        """获取会话模式"""
        mode = self.redis.hget(self._key_meta(session_id), "mode")
        return mode if mode else "rag"

    # ── 历史消息 ──────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str,
                    mode: str = "rag", sources: Optional[list] = None,
                    resolved: Optional[str] = None):
        """添加一条消息到历史
        
        Args:
            resolved: 指代消解后的问题（仅 role=user 时有）
        """
        key = self._key_history(session_id)
        msg = {
            "role": role,
            "content": content,
            "ts": int(time.time()),
        }
        if resolved:
            msg["resolved"] = resolved
        if sources:
            msg["sources"] = sources

        self.redis.rpush(key, json.dumps(msg, ensure_ascii=False))
        # 限制历史长度（保留最近 N*2 条，因为 user+assistant 算一轮）
        max_len = self.max_history * 2 + 2
        current_len = self.redis.llen(key)
        if current_len > max_len:
            self.redis.ltrim(key, current_len - max_len, -1)

        # 续期
        self.redis.expire(key, self.ttl)
        self.redis.expire(self._key_meta(session_id), self.ttl)

    def get_history(self, session_id: str,
                    last_n: Optional[int] = None) -> list[dict]:
        """获取会话历史
        
        Args:
            last_n: 只取最近 N 条（None 表示全部）
        Returns:
            [{"role": str, "content": str, "resolved": Optional[str]}, ...]
        """
        key = self._key_history(session_id)
        if last_n:
            msgs = self.redis.lrange(key, -last_n, -1)
        else:
            msgs = self.redis.lrange(key, 0, -1)

        result = []
        for m in msgs:
            try:
                result.append(json.loads(m))
            except json.JSONDecodeError:
                continue
        return result

    def get_recent_context(self, session_id: str,
                           max_rounds: int = 3) -> str:
        """获取最近几轮对话摘要，用于指代消解
        
        返回格式：
          用户：xxx
          助手：xxx
          用户：xxx
        """
        history = self.get_history(session_id, last_n=max_rounds * 2)
        lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            # 如果是消解后的问题，用消解后的版本
            content = msg.get("resolved", msg["content"])
            # 只保留前200字符（避免太长）
            if len(content) > 200:
                content = content[:200] + "…"
            lines.append(f"{role}：{content}")
        return "\n".join(lines)

    def clear_session(self, session_id: str):
        """清除会话数据"""
        meta_key = self._key_meta(session_id)
        user_id = self.redis.hget(meta_key, "user_id")
        if user_id:
            self.redis.srem(self._key_user_sessions(int(user_id)), session_id)
        self.redis.delete(self._key_history(session_id))
        self.redis.delete(meta_key)

    def close(self):
        """关闭连接"""
        if self._connected:
            try:
                self.redis.close()
            except Exception:
                pass


# ── 指代消解 ──────────────────────────────────────────────

class CoreferenceResolver:
    """指代消解器：用 LLM 将代词替换为具体实体"""

    @staticmethod
    def resolve(question: str, context: str, llm_provider=None) -> str:
        """对问题做指代消解，返回替换后的完整问题
        
        Args:
            question: 当前用户问题（可能含代词）
            context: 对话历史上下文（最近几轮）
            llm_provider: LLM 调用实例，None 时返回原问题
        Returns:
            替换后的完整问题
        """
        if not llm_provider:
            return question

        # 如果问题中不含常见的指代词，直接返回
        if not CoreferenceResolver._has_reference(question):
            return question

        # 如果没历史，也没法消解
        if not context or not context.strip():
            return question

        # 调用 LLM 做一步消解
        prompt = (
            "你是一个指代消解助手。根据对话历史，将当前用户问题中的代词 "
            "（它、他、她、这、那、它们等）替换为具体的实体名称。\n\n"
            "规则：\n"
            "1. 只输出替换后的问题，不要任何额外内容。\n"
            "2. 如果没有代词需要替换，原样输出用户问题。\n"
            "3. 如果实体名称不明确，使用最可能的那个。\n\n"
            f"对话历史：\n{context}\n\n"
            f"当前问题：{question}\n"
            "输出："
        )

        try:
            answer = llm_provider.ask(prompt, system_prompt="你是一个指代消解助手。", max_tokens=128)
            resolved = answer.strip().strip('"').strip("'")
            if resolved and resolved != question:
                logger.debug(f"指代消解: '{question[:60]}' → '{resolved[:80]}'")
                return resolved
        except Exception as e:
            log_exception(logger, f"指代消解失败: {question}", e)

        return question

    @staticmethod
    def _has_reference(text: str) -> bool:
        """检测文本中是否含指代词"""
        ref_words = {"它", "他", "她", "它们", "他们", "她们",
                     "这", "这个", "那个", "该", "这些", "那些",
                     "其", "该等", "上述", "以上", "该些"}
        for word in ref_words:
            if word in text:
                return True
        return False
