"""
教育 Mock Agent

流程：recall → LLM（注入学习记录）→ remember（fire-and-forget）

本模块实现教育辅导领域的智能体：每次对话前先从 mem0 召回该学生的历史学习记录，
将记录注入提示词后调用 LLM 生成个性化回复，对话结束后异步写入新的记忆条目。
"""
# __future__.annotations：延迟类型注解求值，允许在注解中使用尚未定义的类型（如字符串形式的类名）
from __future__ import annotations

# logging：Python 标准库，用于模块级日志记录
import logging
# time：Python 标准库，time.perf_counter() 提供高精度计时（微秒级），用于统计 recall 耗时
import time
# Optional：typing 标准库，Optional[str] 等价于 Union[str, None]，声明参数可以为 None
from typing import Optional

# 导入本项目的 LLM 客户端封装（调用硅基流动 Qwen API 的单轮对话补全）
from src.llm_client import LLMClient
# 导入记忆客户端单例和用户 ID 构造工具函数
from src.memory_client import MemoryClient, make_user_id

# 获取当前模块专属 logger，日志前缀为 "agent22.agent.education"
logger = logging.getLogger("agent22.agent.education")

# ── System Prompt：定义 LLM 扮演的角色和行为准则 ────────────────────────
# 这段字符串作为 messages 中 role="system" 的内容传给 LLM，
# 指导模型在回答前必须先阅读学生历史记忆，并根据学习进度调整讲解深度
SYSTEM_PROMPT = """你是一名耐心细致的智能教育辅导老师。请遵循以下原则：

1. 回答问题前，必须先阅读【学生历史记忆】，了解其学习进度、已掌握概念、薄弱点和学习习惯。
2. 根据学生水平调整讲解深度：已掌握的概念不重复解释，针对薄弱点重点强化。
3. 使用简体中文，语气鼓励积极，适当举例子或类比帮助理解。
4. 每次回复末尾给出 1-2 个针对性的练习题或下一步学习建议。
5. 如学生有错误认知，温和纠正并说明原因。
"""

# ── User Prompt 模板：将召回的记忆块和本轮问题拼接成完整的用户侧输入 ──────
# {memory_block}：格式化占位符，运行时替换为召回记忆的多行文本
# {query}：格式化占位符，运行时替换为学生本次输入的问题
USER_TEMPLATE = """【学生历史记忆】
{memory_block}

【本轮问题】
{query}
"""


def _fmt(memories: list[dict]) -> str:
    """将 recall 返回的记忆列表格式化为可读的多行字符串。

    Args:
        memories: recall() 返回的列表，每项为 {"memory": str, "score": float}。

    Returns:
        str: 格式化后的记忆文本，供注入提示词使用。
             若无记忆则返回提示性说明文字。
    """
    # 若记忆列表为空（首次对话或记忆被清空），返回说明文字告知 LLM 这是首次辅导
    if not memories:
        return "（暂无学习记录，为首次辅导）"

    # 列表推导式：逐条格式化为"序号. 记忆内容"的形式
    # enumerate(memories, 1)：从 1 开始编号
    # m.get("memory")：安全取值，若该键不存在则为 None（被 if 过滤掉）
    # .strip()：去除首尾空白字符
    lines = [f"{i}. {m['memory'].strip()}" for i, m in enumerate(memories, 1) if m.get("memory")]

    # "\n".join(lines)：将所有行拼接为多行字符串；若所有条目都无 memory 字段则返回默认文字
    return "\n".join(lines) or "（暂无学习记录）"


class EducationAgent:
    """教育辅导智能体。

    负责：召回学生历史学习记录 → 注入提示词调用 LLM → 异步写入新记忆。
    每个实例持有一个 LLMClient 和单例 MemoryClient。
    """

    domain = "education"  # 领域标识符，用于构造 user_id 前缀（student_xxx）

    def __init__(self) -> None:
        """初始化 LLM 客户端和记忆客户端。"""
        self.llm = LLMClient()              # 实例化 LLM 客户端（内部读取 SILICONFLOW_* 环境变量）
        self.memory = MemoryClient.instance()  # 获取 MemoryClient 全局单例（懒加载，首次调用时初始化 mem0）

    def chat(self, student_id: str, query: str, *, session_id: Optional[str] = None) -> dict:
        """处理一次学生对话请求。

        流程：
            1. 将 student_id 加前缀构造唯一 user_id（student_{student_id}）。
            2. 用 memory.recall() 检索与本次问题语义相关的历史学习记录（最多 5 条）。
            3. 将记忆块格式化后注入 USER_TEMPLATE，连同 SYSTEM_PROMPT 调用 LLM 生成回复。
            4. 异步（fire-and-forget）将本轮对话写入 mem0，不阻塞当前响应。

        Args:
            student_id: 原始学生 ID（如 "user001"），不含前缀。
            query:      学生本次输入的问题文本。
            session_id: 可选的会话 ID，写入记忆的 metadata 中，便于后续按会话过滤。

        Returns:
            dict: {
                "reply":      str,       # LLM 生成的回复文本
                "recalled":   list[dict],# 本轮召回的记忆条目（含 score 相似度）
                "elapsed_ms": int,       # recall 检索耗时（毫秒）
                "source":     str,       # 固定为 "mock_llm"（区别于医疗 Agent 的 wt12_neo4j）
            }
        """
        # 构造带领域前缀的 user_id，例如 "student_user001"
        # make_user_id 会校验 domain 是否合法（medical/travel/education）
        user_id = make_user_id(self.domain, student_id)

        # —— 步骤 1：召回历史记忆 ——
        t0 = time.perf_counter()  # 记录 recall 开始时间（高精度计时）
        recalled = self.memory.recall(user_id, query, limit=5)  # 语义检索最多 5 条相关记忆
        # (time.perf_counter() - t0) * 1000：将秒转为毫秒；int() 取整
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        # 打印结构化日志：学生 ID、召回条数、检索耗时
        logger.info("[education] student=%s 召回 %d 条 %dms", student_id, len(recalled), elapsed_ms)

        # —— 步骤 2：调用 LLM 生成回复 ——
        reply = self.llm.chat(
            system=SYSTEM_PROMPT,  # 角色定义和行为准则
            # USER_TEMPLATE.format(...)：将记忆块和问题填入模板，生成完整用户侧提示
            user=USER_TEMPLATE.format(memory_block=_fmt(recalled), query=query),
        )

        # —— 步骤 3：异步写入本轮对话记忆（fire-and-forget，不阻塞响应） ——
        self.memory.remember(
            user_id=user_id,
            # 将本轮完整的用户问题和助手回复作为消息列表写入 mem0
            # mem0 会用 LLM 从这段对话中提取关键信息，压缩存储为记忆条目
            messages=[{"role": "user", "content": query}, {"role": "assistant", "content": reply}],
            # metadata：附加元数据，存入向量数据库，可用于后续过滤查询
            metadata={"domain": self.domain, "session_id": session_id or "web"},
        )

        # 返回结果字典，供 API 路由层封装为标准响应体
        return {"reply": reply, "recalled": recalled, "elapsed_ms": elapsed_ms, "source": "mock_llm"}
