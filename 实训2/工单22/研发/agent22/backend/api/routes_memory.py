"""
记忆管理路由

GET    /api/memory/{domain}/{user_id}   -> 查看该用户全部记忆
DELETE /api/memory/{domain}/{user_id}   -> 清空该用户记忆

本模块提供"记忆面板"所需的查看和清空接口，供前端展示历史记忆条目，
以及 demo 演示时重置用户记忆使用。
"""
# __future__.annotations：延迟类型注解求值
from __future__ import annotations

# logging：标准库，用于记录接口调用日志
import logging

# APIRouter：FastAPI 路由分组组件，设置公共前缀 /api/memory
# HTTPException：HTTP 错误响应，非法 domain 时返回 404
from fastapi import APIRouter, HTTPException

# 导入记忆客户端单例和 user_id 构造函数
from src.memory_client import MemoryClient, make_user_id
# 导入响应体 Pydantic 模型
from api.models import MemoryListResponse, MemoryListItem, ClearResponse

# 当前模块专属 logger
logger = logging.getLogger("agent22.api.memory")

# 创建路由器：所有端点 URL 共享前缀 /api/memory，在 /docs 中归入 "memory" 分组
router = APIRouter(prefix="/api/memory", tags=["memory"])

# 合法领域集合（用于入参校验，防止无效 domain 查询 mem0 数据库）
VALID_DOMAINS = {"medical", "travel", "education"}


def _resolve(domain: str, user_id: str) -> str:
    """校验 domain 合法性并构造完整的 mem0 user_id。

    Args:
        domain:  领域名称（由 URL 路径参数传入）。
        user_id: 原始用户 ID（不含领域前缀）。

    Returns:
        str: 完整的 mem0 user_id，如 "patient_user001"。

    Raises:
        HTTPException(404): domain 不在 VALID_DOMAINS 中时抛出。
    """
    # 若 domain 不在合法集合中，立即返回 404 错误，不继续查询
    if domain not in VALID_DOMAINS:
        raise HTTPException(status_code=404, detail=f"未知 domain: {domain}")
    # make_user_id：加领域前缀，如 make_user_id("medical", "u1") → "patient_u1"
    return make_user_id(domain, user_id)


# ── 记忆查询端点 ─────────────────────────────────────────────────────
@router.get("/{domain}/{user_id}", response_model=MemoryListResponse)
async def list_memories(domain: str, user_id: str):
    """获取指定用户在指定领域的全部记忆条目。

    主要用途：
        - 前端"记忆面板"展示用户的历史记忆摘要
        - 调试时检查 mem0 中存储了哪些内容

    Args:
        domain:  路径参数，领域名称（medical/travel/education）。
        user_id: 路径参数，原始用户 ID（不含领域前缀）。

    Returns:
        MemoryListResponse: 包含完整 user_id、领域、记忆条目列表和总条数。
    """
    # 校验 domain 并构造完整 user_id（非法 domain 会在此抛出 404）
    full_uid = _resolve(domain, user_id)

    # 调用 MemoryClient 单例的 list_all() 方法，从 ChromaDB 查询该用户所有记忆
    raw = MemoryClient.instance().list_all(full_uid)

    # 将原始 dict 列表转换为 Pydantic 模型列表，并封装为标准响应结构
    return MemoryListResponse(
        user_id=full_uid,                                  # 完整 user_id（含领域前缀）
        domain=domain,                                     # 领域名称
        memories=[MemoryListItem(**m) for m in raw],       # 每条原始 dict 解包为 MemoryListItem
        total=len(raw),                                    # 记忆总条数
    )


# ── 记忆清空端点 ─────────────────────────────────────────────────────
@router.delete("/{domain}/{user_id}", response_model=ClearResponse)
async def clear_memories(domain: str, user_id: str):
    """清空指定用户在指定领域的全部记忆。

    主要用途：
        - demo 演示时重置用户数据，便于重新体验记忆功能
        - 测试结束后清理测试数据

    Args:
        domain:  路径参数，领域名称（medical/travel/education）。
        user_id: 路径参数，原始用户 ID（不含领域前缀）。

    Returns:
        ClearResponse: 包含被清空的 user_id、实际删除条数和操作结果描述。
    """
    # 校验 domain 并构造完整 user_id
    full_uid = _resolve(domain, user_id)

    # 调用 MemoryClient.clear()：逐条删除该用户的所有记忆，返回成功删除的条数
    deleted = MemoryClient.instance().clear(full_uid)

    # 返回操作结果
    return ClearResponse(
        user_id=full_uid,                        # 被清空的完整 user_id
        deleted=deleted,                         # 实际删除条数（失败的条目不计入）
        message=f"已清空 {deleted} 条记忆",     # 人类可读的操作结果描述
    )
