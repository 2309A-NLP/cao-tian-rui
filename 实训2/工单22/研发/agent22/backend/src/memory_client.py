"""
MemoryClient —— mem0 薄封装，三领域共用单例

关键设计：
- recall()    : 对话前调用，语义检索历史记忆，失败降级为空列表
- remember()  : 对话后调用，默认 fire-and-forget（Qwen 抽取约 20s，不能阻塞主链路）
- list_all()  : 调试/前端"记忆面板"调用
- clear()     : 清除测试数据

mem0 2.0 语法注意：search/get_all 必须用 filters={"user_id": ...}，不能顶层传 user_id
bge-large-zh-v1.5 中文字符上限约 420 字，remember() 截断至 400 字/条

本模块是整个系统的记忆核心：
    - mem0（Memory Zero）是一个专为 AI Agent 设计的长期记忆框架，
      可从对话中自动提取关键信息，压缩存储为语义摘要，支持跨对话的向量语义检索
    - 底层使用 ChromaDB 存储向量，Qwen LLM 提取记忆摘要，bge embedding 向量化
"""
# __future__.annotations：延迟类型注解求值
from __future__ import annotations

# logging：标准库，用于记录记忆读写操作的详细日志
import logging
# threading：标准库，提供线程类（Thread）和线程锁（Lock），
# 用于实现 fire-and-forget 异步写入和单例线程安全初始化
import threading
# Optional：typing 标准库，声明可为 None 的类型
from typing import Optional

# mem0：第三方长期记忆框架（Memory Zero），专为 LLM Agent 设计，
# 核心功能：从对话中自动提取关键信息（用 LLM 提炼）、向量化存储（用 embedding 模型）、
# 语义检索（用向量相似度匹配），实现跨对话的"记住用户"能力
# Memory：mem0 的核心类，封装了 add（写入）/ search（检索）/ get_all（列表）/ delete（删除）操作
from mem0 import Memory

# 导入 mem0 配置构造函数
from src.mem0_config import build_mem0_config

# 当前模块专属 logger
logger = logging.getLogger("agent22.memory")

# ── fire-and-forget 线程管理 ────────────────────────────────────────
# 收集所有尚未完成的 remember 后台线程，供 lifespan shutdown 时等待
_pending_threads: list[threading.Thread] = []  # 未完成的后台写入线程列表
# 线程锁：保护 _pending_threads 列表的并发读写（多线程环境下防止竞态条件）
_pending_lock = threading.Lock()


def wait_pending_writes(timeout: float = 60.0) -> None:
    """阻塞等待所有 fire-and-forget 后台写入线程完成。

    在 FastAPI lifespan shutdown 钩子中调用，确保应用关闭时不丢失正在进行的记忆写入。

    Args:
        timeout: 每个线程的最长等待时间（秒），超时后放弃等待继续下一个线程，默认 60s。
    """
    # 在锁保护下拷贝当前待完成线程列表（避免等待过程中列表被修改）
    with _pending_lock:
        threads = list(_pending_threads)  # 浅拷贝，拿到当前快照

    # 逐个等待线程完成（join 会阻塞直到线程结束或超时）
    for t in threads:
        t.join(timeout=timeout)  # timeout 参数：单个线程最长等待时间

    logger.info("[memory] 所有后台写入线程已完成（或超时）")


class MemoryClient:
    """mem0 长期记忆客户端的薄封装，全局单例模式。

    设计要点：
        - 单例：三个领域的 Agent 共享同一个 MemoryClient 实例，避免重复初始化 mem0
          （mem0 初始化会加载 embedding 模型、连接 ChromaDB，耗时约 2-5s）
        - 懒加载：首次调用 instance() 时才初始化，而非模块导入时立即初始化
        - 失败快速报错：初始化失败后标记 _init_failed，后续请求立即抛出 RuntimeError，
          而非每次都重试（防止无限重试 API）
        - 线程安全：使用 _lock 保证多线程环境下单例初始化只执行一次（双重检查锁定模式）
    """

    _instance: Optional["MemoryClient"] = None  # 全局单例实例（初始为 None）
    _init_failed: bool = False                   # 初始化失败标记（防止无限重试）
    _lock = threading.Lock()                     # 类级别线程锁（保护单例初始化）

    def __init__(self) -> None:
        """初始化 mem0 Memory 实例。

        调用 build_mem0_config() 获取配置，然后调用 Memory.from_config() 创建
        带有 LLM 提取器、embedding 向量化器和 ChromaDB 存储后端的 mem0 实例。
        """
        cfg = build_mem0_config()  # 构造 mem0 配置字典（LLM + Embedder + ChromaDB）

        # 打印初始化日志，显示使用的 LLM 和 embedding 模型名称
        logger.info("初始化 mem0.Memory（LLM=%s, Emb=%s）",
                    cfg["llm"]["config"]["model"],       # LLM 模型（用于提取记忆）
                    cfg["embedder"]["config"]["model"])  # Embedding 模型（用于向量化）

        # Memory.from_config：根据配置字典创建 mem0 实例
        # 内部会初始化 LLM 客户端、embedding 客户端和向量数据库连接
        self._memory = Memory.from_config(cfg)

    @classmethod
    def instance(cls) -> "MemoryClient":
        """获取 MemoryClient 全局单例（线程安全的懒加载）。

        使用双重检查锁定模式（Double-Checked Locking）：
            1. 第一次检查（无锁）：避免每次调用都加锁（性能优化）
            2. 加锁
            3. 第二次检查（有锁内）：防止在第一次检查到加锁之间其他线程已完成初始化

        Returns:
            MemoryClient: 全局单例实例。

        Raises:
            RuntimeError: 初始化失败（API Key 无效、ChromaDB 无法写入等），
                          且后续所有调用都会立即抛出此错误（快速失败策略）。
        """
        # 第一次检查（无锁，避免正常运行路径的锁开销）
        if cls._instance is None and not cls._init_failed:
            with cls._lock:  # 加锁，防止多线程同时初始化
                # 第二次检查（有锁内，防止竞态）：双重检查锁定模式的关键
                if cls._instance is None and not cls._init_failed:
                    try:
                        cls._instance = cls()  # 调用 __init__，初始化 mem0
                    except Exception as e:
                        cls._init_failed = True  # 标记初始化失败，后续不再重试
                        logger.error("MemoryClient 初始化失败，后续请求将快速失败: %s", e)
                        raise  # 重新抛出原始异常，让调用方感知到初始化错误

        # 若初始化曾经失败，直接抛出 RuntimeError（快速失败，不重试）
        if cls._init_failed:
            raise RuntimeError("MemoryClient 初始化失败，请检查 API key 和 ChromaDB 配置")

        return cls._instance  # 返回已初始化的单例实例

    def recall(self, user_id: str, query: str, *, limit: int = 5) -> list[dict]:
        """语义检索该用户的历史记忆（对话前调用）。

        使用 bge embedding 模型将 query 向量化，在 ChromaDB 中做余弦相似度检索，
        返回最相关的历史记忆条目。

        失败时优雅降级为空列表（不阻断对话流程），并记录 WARNING 日志。

        Args:
            user_id: 完整的 user_id（含领域前缀，如 "patient_u001"）。
            query:   本次对话的问题文本，用于语义相似度匹配。
            limit:   最多返回的记忆条目数，默认 5 条。

        Returns:
            list[dict]: 每项为 {"memory": str, "score": float}，
                        score 为余弦相似度 [0, 1]，越高越相关。
                        若检索失败则返回 []。
        """
        try:
            # mem0 2.0 语法：必须用 filters={"user_id": ...}，不能作为顶层参数传 user_id
            # （mem0 1.x 的 user_id 参数在 2.0 中被废弃，改为 filters 字典）
            raw = self._memory.search(
                query=query,                     # 检索用的查询文本
                filters={"user_id": user_id},    # 按 user_id 过滤（用户隔离）
                limit=limit,                     # 最多返回条数
            )
        except Exception as e:
            # 检索失败（网络问题、ChromaDB 文件损坏等）：优雅降级，不影响对话流程
            logger.warning("[recall] 检索失败 user_id=%s err=%s（返回空列表降级）", user_id, e)
            return []  # 降级返回空列表，对话继续但无历史记忆注入

        # mem0 search 返回格式有两种可能：
        # - dict 格式：{"results": [...]}（mem0 2.0 标准格式）
        # - list 格式：[...]（旧版本或某些配置下直接返回列表）
        # 这里做兼容处理，统一取出实际的结果列表
        hits = raw.get("results", []) if isinstance(raw, dict) else raw

        # 标准化每条结果为 {"memory": str, "score": float} 格式
        return [
            {
                "memory": h.get("memory", ""),           # 记忆文本内容
                "score": round(h.get("score", 0.0), 4)  # 相似度分数，保留 4 位小数
            }
            for h in hits
        ]

    # bge-large-zh-v1.5 中文字符安全上限（实测约 420 字符，保留 20 字符 buffer）
    _MAX_MSG_CHARS = 400

    @classmethod
    def _truncate(cls, messages: list[dict]) -> list[dict]:
        """将消息列表中超过字符上限的内容截断。

        bge-large-zh-v1.5 对输入长度有限制（约 420 中文字符），
        超过上限会导致 embedding API 报错。截断为 400 字符 + 说明尾缀。

        Args:
            messages: 原始消息列表，每项为 {"role": str, "content": str}。

        Returns:
            list[dict]: 内容已截断的消息列表（原对象未修改，返回新列表）。
        """
        result = []
        for m in messages:
            content = m.get("content", "")  # 获取消息内容（安全取值）

            # 若内容超过最大字符数，截断并附加说明
            if len(content) > cls._MAX_MSG_CHARS:
                # content[:400]：取前 400 字符；"…（已截断）"：提示 LLM 内容不完整
                content = content[:cls._MAX_MSG_CHARS] + "…（已截断）"

            # {**m, "content": content}：展开原消息 dict 的所有字段，并用新 content 覆盖
            result.append({**m, "content": content})
        return result

    def remember(
        self,
        user_id: str,
        messages: list[dict],
        *,
        metadata: Optional[dict] = None,
        blocking: bool = False,
    ) -> None:
        """将本轮对话写入 mem0 记忆存储（对话后调用）。

        mem0 的 add() 内部会：
            1. 调用 LLM（Qwen）从消息中提取关键信息（约 20s）
            2. 调用 embedding 模型将提取的摘要向量化
            3. 将向量和元数据写入 ChromaDB

        默认使用 fire-and-forget（后台线程），不阻塞当前请求响应。
        测试场景下传 blocking=True 可同步等待写入完成。

        Args:
            user_id:  完整 user_id（含领域前缀）。
            messages: 本轮对话消息列表，格式 [{"role": "user", "content": "..."}, ...]。
            metadata: 附加元数据（如 domain、session_id），存入 ChromaDB，便于过滤查询。
            blocking: True=同步等待写入完成；False（默认）=后台线程异步写入。
        """
        # 先截断超长消息内容，避免 embedding API 报错
        safe = self._truncate(messages)

        def _do() -> None:
            """实际执行 mem0 写入操作的内部函数（同步执行）。"""
            try:
                # self._memory.add：mem0 核心写入方法
                # messages：对话消息（mem0 用 LLM 从中提取记忆摘要）
                # user_id：记忆归属标识（用于隔离不同用户的记忆）
                # metadata：附加元数据（存入向量数据库，可用于过滤查询）
                self._memory.add(
                    messages=safe,               # 截断后的对话消息
                    user_id=user_id,             # 记忆归属用户
                    metadata=metadata or {},     # 附加元数据（默认空字典）
                )
                logger.info("[remember] 完成 user_id=%s msgs=%d", user_id, len(safe))
            except Exception as e:
                # 写入失败（API 超时、ChromaDB 写入错误等）：仅记录 WARNING，不影响对话
                logger.warning("[remember] 失败 user_id=%s err=%s", user_id, e)

        def _do_async() -> None:
            """后台线程执行函数：调用 _do() 后从待完成列表中移除自身。"""
            try:
                _do()  # 执行实际写入
            finally:
                # finally 块确保无论成功/失败都从待完成列表中移除当前线程
                cur = threading.current_thread()  # 获取当前线程对象
                with _pending_lock:               # 加锁保护列表操作
                    try:
                        _pending_threads.remove(cur)  # 从待完成列表移除自身
                    except ValueError:
                        pass  # 若已被移除（理论上不会），忽略 ValueError

        # 根据 blocking 参数决定同步还是异步执行
        if blocking:
            # 同步模式：直接在当前线程执行，调用方等待完成后才继续（用于测试场景）
            _do()
        else:
            # 异步模式：创建后台线程执行写入
            # daemon=False：非 daemon 线程，进程退出前 Python 会等待其完成
            #               （配合 lifespan shutdown 的 wait_pending_writes 确保不丢数据）
            t = threading.Thread(target=_do_async, name=f"mem_{user_id}", daemon=False)
            with _pending_lock:             # 加锁，线程安全地添加到待完成列表
                _pending_threads.append(t)  # 注册到待完成列表，供 shutdown 等待
            t.start()                       # 启动后台线程

    def list_all(self, user_id: str) -> list[dict]:
        """获取该用户的全部记忆条目。

        主要用于前端"记忆面板"展示和调试。

        Args:
            user_id: 完整 user_id（含领域前缀）。

        Returns:
            list[dict]: 每项为 {"id": str, "memory": str, "metadata": dict}。
                        失败时返回 []（优雅降级）。
        """
        try:
            # mem0 2.0：get_all 也必须用 filters={"user_id": ...} 过滤
            raw = self._memory.get_all(filters={"user_id": user_id})
        except Exception as e:
            logger.warning("[list_all] 失败 user_id=%s err=%s", user_id, e)
            return []  # 优雅降级，返回空列表

        # 兼容 mem0 返回格式（dict 或 list）
        results = raw.get("results", []) if isinstance(raw, dict) else raw

        # 标准化为 {"id", "memory", "metadata"} 格式
        return [
            {
                "id": m.get("id", ""),           # mem0 自动分配的 UUID（删除时需要）
                "memory": m.get("memory", ""),   # 记忆文本内容（LLM 提取的摘要）
                "metadata": m.get("metadata", {})  # 写入时附加的元数据
            }
            for m in results
        ]

    def clear(self, user_id: str) -> int:
        """清空该用户的全部记忆，返回实际成功删除的条数。

        先用 list_all() 获取所有记忆 ID，再逐条调用 memory.delete()。
        若个别删除失败，仅记录日志，不影响其他条目的删除。

        Args:
            user_id: 完整 user_id（含领域前缀）。

        Returns:
            int: 实际成功删除的记忆条数（可能小于总数，若部分删除失败）。
        """
        all_mem = self.list_all(user_id)  # 获取该用户所有记忆条目
        deleted = 0  # 成功删除计数器

        for m in all_mem:
            try:
                # memory.delete：按 memory_id（UUID）删除单条记忆
                self._memory.delete(memory_id=m["id"])
                deleted += 1  # 删除成功，计数加一
            except Exception as e:
                # 单条删除失败（如 ID 已不存在）：记录警告，继续删除其他条目
                logger.warning("[clear] 删除失败 id=%s err=%s", m["id"], e)

        # 记录清空结果日志（成功条数/总条数）
        logger.info("[clear] 清空 user_id=%s 成功 %d/%d 条", user_id, deleted, len(all_mem))
        return deleted  # 返回实际删除条数


def make_user_id(domain: str, raw_id: str) -> str:
    """构造带领域前缀的完整 mem0 user_id。

    不同领域使用不同前缀，实现记忆的领域级隔离：
        - medical   → patient_{raw_id}
        - travel    → traveler_{raw_id}
        - education → student_{raw_id}

    同一个原始 user_id（如 "u001"）在不同领域产生不同的 user_id，
    确保各领域的记忆互不干扰。

    Args:
        domain: 领域名称（medical/travel/education）。
        raw_id: 原始用户 ID（由前端传入，不含前缀）。

    Returns:
        str: 完整 user_id，如 "patient_u001"。

    Raises:
        ValueError: domain 不在合法范围内。
    """
    # 领域名称 → user_id 前缀的映射字典
    prefixes = {"medical": "patient_", "travel": "traveler_", "education": "student_"}

    # 若 domain 不合法，抛出 ValueError（调用方应在此之前完成校验）
    if domain not in prefixes:
        raise ValueError(f"未知 domain: {domain}")

    # 拼接前缀和原始 ID，如 "patient_" + "u001" → "patient_u001"
    return f"{prefixes[domain]}{raw_id}"
