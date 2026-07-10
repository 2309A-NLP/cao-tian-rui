"""
API 请求 / 响应 Pydantic 模型

本模块定义所有 HTTP 接口的请求体和响应体数据结构。
使用 Pydantic 进行自动校验和序列化，FastAPI 会根据这些模型自动生成 OpenAPI 文档。
"""
# __future__.annotations：延迟类型注解求值，兼容 Python 3.9 以下版本的 list[...] 语法
from __future__ import annotations

# Optional：typing 标准库，Optional[X] = Union[X, None]，用于声明可选字段
from typing import Optional

# pydantic：第三方数据验证库，通过类型注解自动校验数据合法性，
# 并提供序列化（.dict()/.json()）和反序列化（parse_obj()）功能
# BaseModel：所有 Pydantic 模型的基类，定义字段后自动获得校验、序列化能力
# Field：用于为字段添加额外约束（如 min_length、description），同时生成 OpenAPI schema 文档
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════
# 请求体模型
# ══════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """对话接口（POST /api/chat/{domain}）的请求体。

    Fields:
        user_id:    用户标识符，由前端自定义（如 "user001"）。
                    后端会加领域前缀转为完整 user_id（如 "patient_user001"）。
        query:      用户输入的问题文本，最短 1 个字符（防止空字符串请求）。
        session_id: 可选的会话 ID，用于区分同一用户的不同对话场景。
    """
    # Field(...)：第一个参数 ... 表示必填字段（无默认值）；description 会出现在 /docs 文档中
    user_id: str = Field(..., description="用户 ID（前端可自定义，如 user001）")
    # min_length=1：验证 query 不能为空字符串，FastAPI 会自动返回 422 错误给不合规请求
    query: str   = Field(..., min_length=1, description="用户输入的问题")
    # Optional[str] + 默认值 None：此字段不传时默认为 None，不影响接口调用
    session_id: Optional[str] = Field(None, description="会话 ID，可不传")


# ══════════════════════════════════════════════════════════════════════
# 响应体模型
# ══════════════════════════════════════════════════════════════════════

class MemoryItem(BaseModel):
    """单条召回记忆的数据结构。

    Fields:
        memory: 记忆文本内容（由 mem0 从历史对话中提取的语义摘要）。
        score:  向量相似度分数，范围 [0, 1]，越高表示与当前查询越相关。
    """
    memory: str           # 记忆文本内容
    score: float = 0.0    # 相似度分数，默认 0.0（无分数时的占位值）


class ChatResponse(BaseModel):
    """对话接口的响应体。

    Fields:
        reply:      LLM 生成的回复文本（最终呈现给用户的内容）。
        recalled:   本轮从 mem0 召回的历史记忆列表（含相似度分数），供前端"记忆面板"展示。
        domain:     本次请求的领域（medical/travel/education），前端用于区分显示样式。
        elapsed_ms: recall 检索耗时（毫秒），用于前端性能监控和日志。
        source:     回复的来源标识，区分三种路径：
                    - "wt12_neo4j"：来自工单12的 Neo4j 知识图谱增强回复
                    - "fallback_llm"：工单12不可用时的本地 LLM 降级回复
                    - "mock_llm"：文旅/教育领域直接调用本地 LLM 的正常回复
    """
    reply: str                            # 最终回复文本
    recalled: list[MemoryItem]            # 本轮召回的历史记忆列表（含相似度分数）
    domain: str                           # 领域标识（medical/travel/education）
    elapsed_ms: int = 0                   # recall 检索耗时（毫秒）
    source: str = "mock_llm"             # 回复来源标识


class MemoryListItem(BaseModel):
    """记忆列表接口中单条记忆的数据结构。

    Fields:
        id:       mem0 为每条记忆分配的唯一 UUID，用于删除特定记忆。
        memory:   记忆文本内容。
        metadata: 写入时附加的元数据字典（如 domain、session_id）。
    """
    id: str                # mem0 记忆条目的唯一标识符（UUID 字符串）
    memory: str            # 记忆文本内容
    metadata: dict = {}    # 附加元数据，默认空字典


class MemoryListResponse(BaseModel):
    """记忆列表接口（GET /api/memory/{domain}/{user_id}）的响应体。

    Fields:
        user_id:   完整的 user_id（含领域前缀，如 "patient_user001"）。
        domain:    领域标识。
        memories:  该用户在该领域的全部记忆条目列表。
        total:     记忆条目总数（等于 len(memories)）。
    """
    user_id: str                        # 完整 user_id（含前缀）
    domain: str                         # 领域标识
    memories: list[MemoryListItem]      # 全部记忆条目
    total: int                          # 记忆总条数


class ClearResponse(BaseModel):
    """记忆清空接口（DELETE /api/memory/{domain}/{user_id}）的响应体。

    Fields:
        user_id:  被清空的完整 user_id。
        deleted:  实际成功删除的记忆条数（可能小于总数，若个别删除失败）。
        message:  人类可读的操作结果描述（如 "已清空 3 条记忆"）。
    """
    user_id: str    # 被清空的用户 ID（含领域前缀）
    deleted: int    # 实际删除的记忆条数
    message: str    # 操作结果描述文字
