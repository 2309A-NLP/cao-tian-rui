"""
对话路由

POST /api/chat/medical      -> MedicalAgent.chat()
POST /api/chat/travel       -> TravelAgent.chat()
POST /api/chat/education    -> EducationAgent.chat()

本模块将三个领域的 Agent 统一暴露为同一路径模式 /api/chat/{domain}，
通过路径参数 domain 动态路由到对应的 Agent 实例。
Agent 实例使用懒加载单例模式，避免重复初始化的开销。
"""
# __future__.annotations：延迟类型注解求值
from __future__ import annotations

# asyncio：Python 标准库，异步 I/O 框架
# asyncio.to_thread：将同步阻塞函数包装为协程，在线程池中运行，不阻塞 asyncio 事件循环
import asyncio
# logging：标准库，模块级日志
import logging
# time：标准库，高精度计时
import time

# APIRouter：FastAPI 的路由组件，类似 Flask Blueprint，可分组管理端点并设置公共前缀/标签
# HTTPException：FastAPI 内置的 HTTP 异常类，raise 后自动返回对应状态码的 JSON 错误响应
from fastapi import APIRouter, HTTPException

# 三个领域的智能体类
from agents.medical_agent import MedicalAgent
from agents.travel_agent import TravelAgent
from agents.education_agent import EducationAgent

# 导入请求/响应 Pydantic 模型
from api.models import ChatRequest, ChatResponse, MemoryItem

# 获取当前模块专属 logger
logger = logging.getLogger("agent22.api.chat")

# 创建路由器：prefix="/api/chat" 表示本路由器下所有端点的 URL 前缀
# tags=["chat"] 用于在 /docs Swagger UI 中对端点进行分组显示
router = APIRouter(prefix="/api/chat", tags=["chat"])

# ── Agent 单例缓存字典 ────────────────────────────────────────────────
# 三个 Agent 各自惰性初始化并缓存为单例，避免每次请求都重建（Agent 初始化会连接 MemoryClient）
# dict[str, object]：键为领域名称，值为 Agent 实例
_agents: dict[str, object] = {}


def _get_agent(domain: str):
    """按领域名称获取（或惰性初始化）对应的 Agent 单例。

    使用 Python 3.10+ 的 match/case 语法（类似 switch）匹配领域名称。
    首次调用时实例化 Agent，后续调用直接从 _agents 缓存返回。

    Args:
        domain: 领域名称，合法值为 "medical"/"travel"/"education"。

    Returns:
        对应的 Agent 实例（MedicalAgent/TravelAgent/EducationAgent）。

    Raises:
        HTTPException(404): domain 不在合法范围内时抛出。
    """
    # 若缓存中已有此 domain 的 Agent 实例，直接返回（避免重复初始化）
    if domain not in _agents:
        # match/case：Python 3.10+ 结构化模式匹配，根据 domain 值创建对应 Agent
        match domain:
            case "medical":
                _agents[domain] = MedicalAgent()    # 医疗 Agent（含工单12调用逻辑）
            case "travel":
                _agents[domain] = TravelAgent()     # 文旅 Agent（直接调用本地 LLM）
            case "education":
                _agents[domain] = EducationAgent()  # 教育 Agent（直接调用本地 LLM）
            case _:
                # 通配符 _：任何其他值都走此分支，返回 404 错误
                raise HTTPException(status_code=404, detail=f"未知 domain: {domain}")
    return _agents[domain]  # 返回缓存的 Agent 实例


# ── 对话端点 ─────────────────────────────────────────────────────────
@router.post("/{domain}", response_model=ChatResponse)
async def chat(domain: str, req: ChatRequest):
    """与指定领域智能体进行一轮对话。

    FastAPI 自动从 URL 路径提取 domain 参数，从请求体解析并校验 ChatRequest。
    Agent.chat() 是同步阻塞方法（内部有 HTTP 请求和向量检索），
    使用 asyncio.to_thread 包装，在线程池中运行，不阻塞 asyncio 事件循环。

    Args:
        domain: 路径参数，领域名称：medical | travel | education。
        req:    请求体，由 FastAPI 自动从 JSON body 解析并校验（Pydantic）。

    Returns:
        ChatResponse: 包含回复文本、召回记忆、来源标识等字段的响应体。

    Raises:
        HTTPException(404): domain 非法。
        HTTPException(500): Agent 内部发生未捕获异常。
    """
    # 获取（或惰性初始化）对应领域的 Agent 单例
    agent = _get_agent(domain)

    try:
        t0 = time.perf_counter()  # 记录整体请求开始时间（含 LLM 调用）

        # asyncio.to_thread：在线程池中运行同步函数，避免阻塞事件循环
        # agent.chat(**kwargs)：根据 domain 传入不同的 ID 参数名（patient_id/traveler_id/student_id）
        result = await asyncio.to_thread(
            agent.chat,
            **{_id_param(domain): req.user_id},  # 动态关键字参数：如 patient_id="user001"
            query=req.query,                      # 用户问题
            session_id=req.session_id,            # 可选会话 ID
        )
        # 计算整体处理耗时（毫秒）
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

    except Exception as e:
        # 记录详细错误日志（含 traceback）
        logger.exception("[chat] domain=%s user=%s 出错", domain, req.user_id)
        # 向客户端返回 500 错误，detail 包含异常信息（生产环境可隐藏细节）
        raise HTTPException(status_code=500, detail=str(e))

    # 将 Agent 返回的原始 dict 转换为类型安全的 Pydantic 响应模型
    return ChatResponse(
        reply=result["reply"],                              # LLM 回复文本
        # 将 recalled 列表中每项 dict 转换为 MemoryItem Pydantic 对象（校验字段类型）
        recalled=[MemoryItem(**m) for m in result["recalled"]],
        domain=domain,                                      # 回填领域名称
        # 优先使用 Agent 内部统计的 recall 耗时，次选整体请求耗时
        elapsed_ms=result.get("elapsed_ms", elapsed_ms),
        source=result.get("source", "mock_llm"),           # 回复来源标识
    )


def _id_param(domain: str) -> str:
    """将领域名称映射为 Agent.chat() 方法的用户 ID 参数名。

    不同领域的 Agent.chat() 方法使用不同的 ID 参数名以提高可读性
    （patient_id vs traveler_id vs student_id）。
    此函数作为映射层，让路由层的代码保持统一。

    Args:
        domain: 领域名称（medical/travel/education）。

    Returns:
        str: 对应 Agent.chat() 方法的 ID 参数名。

    Raises:
        KeyError: domain 不在映射表中（理论上不会发生，已在 _get_agent 中校验）。
    """
    # dict 字面量映射：领域名 → 对应 Agent.chat() 的参数名
    return {"medical": "patient_id", "travel": "traveler_id", "education": "student_id"}[domain]
