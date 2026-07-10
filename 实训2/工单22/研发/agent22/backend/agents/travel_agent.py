"""
文旅 Mock Agent

流程：recall → LLM（注入偏好/历史）→ remember（fire-and-forget）

本模块实现智慧文旅领域的智能体：每次对话前先召回旅行者的历史偏好和行程记录，
将其注入提示词后调用 LLM 生成个性化旅行推荐，对话后异步写入新的记忆条目。
逻辑结构与 education_agent 完全对称，领域差异仅在 system prompt 和 user_id 前缀上。
"""
# __future__.annotations：延迟类型注解求值，允许注解中使用尚未定义的类名
from __future__ import annotations

# logging：标准库，提供模块级日志记录
import logging
# time：标准库，time.perf_counter() 用于高精度计时
import time
# Optional：typing 标准库，Optional[str] = Union[str, None]，声明可空参数
from typing import Optional

# 导入本项目 LLM 封装（调用硅基流动 Qwen API）
from src.llm_client import LLMClient
# 导入记忆客户端单例和 user_id 构造工具函数
from src.memory_client import MemoryClient, make_user_id

# 当前模块专属 logger，日志前缀 "agent22.agent.travel"
logger = logging.getLogger("agent22.agent.travel")

# ── System Prompt：文旅助手角色定义和行为规则 ────────────────────────────
# 告知 LLM 必须先读取旅行者历史记忆，根据偏好个性化推荐，避免重复推荐不喜欢的内容
SYSTEM_PROMPT = """你是一名专业的智慧文旅助手。请遵循以下原则：

1. 回答旅行规划问题时，必须先阅读【旅行者历史记忆】，根据其偏好、预算、禁忌等信息个性化推荐。
2. 若旅行者曾明确表达不喜欢某类活动或地点，不要重复推荐。
3. 使用简体中文，语气活泼友好，适当使用具体数字（价格、距离、时间）使建议更实用。
4. 推荐内容涵盖：目的地、住宿档次、必体验活动、饮食推荐。
5. 如不确定，提醒旅行者结合实际情况再确认。
"""

# ── User Prompt 模板：将历史记忆和本轮问题拼接为完整用户侧输入 ──────────
# {memory_block}：运行时替换为格式化后的记忆文本块
# {query}：运行时替换为旅行者本次输入的问题
USER_TEMPLATE = """【旅行者历史记忆】
{memory_block}

【本轮问题】
{query}
"""


def _fmt(memories: list[dict]) -> str:
    """将 recall 返回的记忆列表格式化为多行可读字符串。

    Args:
        memories: recall() 返回的列表，每项格式为 {"memory": str, "score": float}。

    Returns:
        str: 格式化的记忆文本（"序号. 内容" 多行），或首次咨询的说明文字。
    """
    # 若记忆为空（旅行者首次咨询或记忆已清空），告知 LLM 这是首次对话
    if not memories:
        return "（暂无历史记录，为首次咨询）"

    # 列表推导：生成"1. 内容\n2. 内容\n..."格式的记忆条目列表
    # 过滤掉 memory 字段为空或 None 的无效条目
    lines = [f"{i}. {m['memory'].strip()}" for i, m in enumerate(memories, 1) if m.get("memory")]

    # 用换行符拼接成多行文本；若所有条目均无效则返回默认文字
    return "\n".join(lines) or "（暂无历史记录）"


class TravelAgent:
    """智慧文旅智能体。

    负责：召回旅行者历史偏好/行程记录 → 注入提示词调用 LLM 生成个性化推荐
    → 异步写入本轮对话记忆，供下次对话召回。
    """

    domain = "travel"  # 领域标识，用于构造 user_id 前缀（traveler_xxx）

    def __init__(self) -> None:
        """初始化 LLM 客户端和 MemoryClient 单例。"""
        self.llm = LLMClient()              # LLM 客户端（调用硅基流动 Qwen）
        self.memory = MemoryClient.instance()  # 全局记忆客户端单例（懒加载，线程安全）

    def chat(self, traveler_id: str, query: str, *, session_id: Optional[str] = None) -> dict:
        """处理一次旅行者对话请求。

        Args:
            traveler_id: 原始旅行者 ID（如 "user001"），不含领域前缀。
            query:       旅行者本次输入的问题（如"帮我规划三亚五日游"）。
            session_id:  可选的会话 ID，写入 mem0 metadata，便于按会话过滤记忆。

        Returns:
            dict: {
                "reply":      str,        # LLM 生成的旅行建议文本
                "recalled":   list[dict], # 本轮召回的历史记忆条目（含相似度分数）
                "elapsed_ms": int,        # recall 检索耗时（毫秒）
                "source":     str,        # 固定 "mock_llm"（与医疗 Agent 的 wt12_neo4j 区分）
            }
        """
        # 构造带领域前缀的 user_id，如 "traveler_user001"
        user_id = make_user_id(self.domain, traveler_id)

        # —— 步骤 1：召回旅行者历史偏好/行程记忆 ——
        t0 = time.perf_counter()  # 开始高精度计时
        recalled = self.memory.recall(user_id, query, limit=5)  # 语义检索最相关的 5 条记忆
        # 计算 recall 耗时并转换为毫秒整数
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        # 结构化日志：记录旅行者 ID、召回条数和检索耗时
        logger.info("[travel] traveler=%s 召回 %d 条 %dms", traveler_id, len(recalled), elapsed_ms)

        # —— 步骤 2：将记忆注入提示词，调用 LLM 生成个性化回复 ——
        reply = self.llm.chat(
            system=SYSTEM_PROMPT,  # 文旅助手角色定义
            # 将格式化的记忆块和本轮问题填入模板，构成完整用户侧输入
            user=USER_TEMPLATE.format(memory_block=_fmt(recalled), query=query),
        )

        # —— 步骤 3：异步（fire-and-forget）写入本轮对话记忆 ——
        # blocking=False（默认）：在后台线程中写入，不阻塞当前响应，延迟约 20s（mem0 提取）
        self.memory.remember(
            user_id=user_id,
            # 将用户原始问题和助手回复作为消息对存入 mem0
            messages=[{"role": "user", "content": query}, {"role": "assistant", "content": reply}],
            # metadata 附加领域标识和会话 ID，供后续查询过滤
            metadata={"domain": self.domain, "session_id": session_id or "web"},
        )

        # 返回标准结果字典，由 API 路由层封装为 ChatResponse
        return {"reply": reply, "recalled": recalled, "elapsed_ms": elapsed_ms, "source": "mock_llm"}
