"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
FastAPI 入口：/health、/api/analyze，以及根路径提供前端页面。

本文件是整个后端的核心路由层，负责：
1. 接收 HTTP 请求，进行输入校验（task枚举、MIME类型、图片校验、query长度）
2. 将请求分发到对应的 Handler（VQAHandler / MRGHandler / RAGHandler）
3. 处理 Handler 返回值并构造标准 JSON 响应
4. 统一错误处理（VLMError → 5xx，校验错误 → 4xx）
5. 结构化请求日志（含请求ID、耗时分解）
"""

# os：Python 内置模块，用于读取环境变量（CORS_ORIGINS）
import os

# time：Python 内置模块，用于计算各阶段耗时（perf_counter 精度高于 time.time）
import time

# pathlib.Path：面向对象的文件路径操作
from pathlib import Path

# Optional：类型提示，表示可以为 None
from typing import Optional

# FastAPI 相关导入
# Depends：依赖注入装饰器，用于声明请求需要满足的依赖（如鉴权）
# FastAPI：应用类，创建 ASGI 应用实例
# File/Form/UploadFile：用于处理 multipart/form-data 上传的文件和表单字段
# HTTPException：FastAPI 的 HTTP 异常类，抛出后自动返回对应状态码
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

# CORSMiddleware：跨域资源共享（CORS）中间件
# 允许浏览器前端从不同域名访问此 API
from fastapi.middleware.cors import CORSMiddleware

# FileResponse：返回静态文件（如 HTML 页面）
# JSONResponse：返回 JSON 格式响应（手动控制状态码时使用）
from fastapi.responses import FileResponse, JSONResponse

# 从本包各模块导入所需内容（使用相对导入）
from .auth import verify_api_key        # API Key 鉴权函数
from .config import (
    CHROMA_COLLECTION,     # ChromaDB 集合名（健康检查时使用）
    DISCLAIMER,            # 医疗免责声明文本
    MAX_QUERY_LENGTH,      # query 最大长度（超出截断）
    VLM_BACKEND,           # VLM 后端标识
    VLM_MODEL,             # VLM 模型名称
    WORKORDER_ID,          # 工单标识
)
from .handlers.mrg import MRGHandler    # 报告生成处理器
from .handlers.rag import RAGHandler    # 检索增强生成处理器
from .handlers.vqa import VQAHandler    # 视觉问答处理器
from .logger import get_logger, log_request  # 日志工具
from .models import (
    AnalyzeResponse,   # 成功响应数据模型
    ErrorResponse,     # 错误响应数据模型
    HealthResponse,    # 健康检查响应模型
    TaskType,          # 任务类型枚举
)
from .utils import ImageValidationError, new_request_id, validate_image  # 图像校验工具
from .vlm_client import VLMError, image_bytes_to_b64  # VLM 客户端工具

# 获取本模块专用的日志记录器
logger = get_logger("wt13.api")

# 静态文件目录（backend/static/），用于提供前端 HTML 页面
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# 创建 FastAPI 应用实例，配置 OpenAPI 文档元信息
app = FastAPI(
    title="医疗影像分析 API（工单13）",
    version="0.2.0",
    description="医疗智能体影像分析：VQA / MRG / RAG（NLP方向）",
)

# ── CORS 配置 ──
# 读取环境变量 CORS_ORIGINS（逗号分隔的允许源列表）
# 若未配置则使用默认值（允许 8011-8015 和 8000 端口的 localhost）
_origins = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8011,http://127.0.0.1:8011,"
    "http://localhost:8012,http://127.0.0.1:8012,"
    "http://localhost:8013,http://127.0.0.1:8013,"
    "http://localhost:8014,http://127.0.0.1:8014,"
    "http://localhost:8015,http://127.0.0.1:8015,"
    "http://localhost:8000,http://127.0.0.1:8000",
).split(",")]  # split(",") 按逗号分割为列表，strip() 去掉每个元素的空白

# 注册 CORS 中间件（必须在定义路由之前注册）
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,          # 允许跨域请求的来源列表
    allow_methods=["GET", "POST"],   # 允许的 HTTP 方法（只开放必要方法）
    allow_headers=["Content-Type", "X-API-Key"],  # 允许的请求头
)

# ── Handler 单例缓存（延迟初始化）──
# 第一次请求时才创建 Handler 实例（避免启动时加载模型导致延迟）
_handlers: dict[str, object] = {}


def get_handler(task: TaskType):
    """
    根据任务类型获取（或延迟创建）对应的 Handler 单例。

    参数：
        task (TaskType)：任务类型枚举值

    返回值：
        VQAHandler | MRGHandler | RAGHandler：对应的处理器实例

    异常：
        HTTPException(501)：CV 预留任务（classification/detection/segmentation）抛出
    """
    # VQA Handler：首次调用时创建，后续复用
    if task == TaskType.VQA:
        if "vqa" not in _handlers:
            _handlers["vqa"] = VQAHandler()  # 创建 VQAHandler 实例并缓存
        return _handlers["vqa"]

    # MRG Handler：首次调用时创建，后续复用
    if task == TaskType.MRG:
        if "mrg" not in _handlers:
            _handlers["mrg"] = MRGHandler()
        return _handlers["mrg"]

    # RAG Handler：首次调用时创建，后续复用
    if task == TaskType.RAG:
        if "rag" not in _handlers:
            _handlers["rag"] = RAGHandler()
        return _handlers["rag"]

    # CV 预留任务：本工单（NLP 方向）不实现，返回 501 Not Implemented
    raise HTTPException(
        status_code=501,
        detail={"error_code": "NOT_IMPLEMENTED", "message": f"任务 {task.value} 属于 CV 方向，本工单（NLP）不实现"},
    )


# ── 前端页面路由 ──

@app.get("/", include_in_schema=False)  # include_in_schema=False：此路由不出现在 OpenAPI 文档中
async def serve_index():
    """根路径返回前端 HTML 页面（如 static/index.html）。"""
    index_path = _STATIC_DIR / "index.html"

    # 若前端文件不存在，返回 404 提示
    if not index_path.exists():
        return JSONResponse(status_code=404, content={"message": "前端文件不存在，请检查 static/index.html"})

    # FileResponse：直接将文件内容作为 HTTP 响应返回（浏览器会渲染 HTML）
    return FileResponse(index_path)


# ── 健康检查端点 ──

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    健康检查接口，返回服务状态和知识库信息。

    此接口不需要鉴权，供监控系统和负载均衡器探活使用。
    即使 ChromaDB 不可用，也始终返回 200（kb_docs 默认 0）。
    """
    try:
        # 延迟导入避免循环依赖，同时 try/except 保证 ChromaDB 不可用时不崩溃
        from .rag_store import count_docs
        kb = count_docs()  # 查询知识库文档数
    except Exception:
        kb = 0  # 任何异常都降级为 0（服务本身仍然正常）

    return HealthResponse(
        status="ok",
        workorder_id=WORKORDER_ID,
        vlm_backend=VLM_BACKEND,
        vlm_model=VLM_MODEL,
        kb_docs=kb,
    )


# ── 核心分析接口 ──

@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,              # 成功时的响应模型（用于 OpenAPI 文档生成）
    dependencies=[Depends(verify_api_key)],      # 注入鉴权依赖（API Key 校验）
    responses={                                  # 文档中展示的可能错误响应
        400: {"model": ErrorResponse},           # 缺少必填字段
        413: {"model": ErrorResponse},           # 图片过大
        415: {"model": ErrorResponse},           # 不支持的文件格式
        422: {"model": ErrorResponse},           # 参数校验失败
        503: {"model": ErrorResponse},           # VLM 服务不可用
        504: {"model": ErrorResponse},           # VLM 调用超时
    },
)
async def analyze(
    task: str = Form(..., description="vqa | mrg | rag"),  # ... 表示必填
    image: UploadFile = File(...),                          # 必填：上传的图片文件
    query: Optional[str] = Form(default=None, description="用户问题（vqa/rag 必填）"),  # 可选
    session_id: Optional[str] = Form(default=None),        # 可选：会话 ID（用于日志）
):
    """
    医疗影像分析主接口。

    支持三种任务类型：
    - vqa：视觉问答，image + query → answer
    - mrg：报告生成，image [+ query 作为临床背景] → report + answer
    - rag：检索增强问答，image + query → answer + references

    请求格式：multipart/form-data
    - task：字符串，"vqa"/"mrg"/"rag"
    - image：图片文件（JPEG/PNG，最大 20MB，最小 32×32 像素）
    - query：问题字符串（VQA/RAG 必填，MRG 可选）
    - session_id：可选，用于日志关联

    参数：
        task (str)：任务类型字符串（来自 Form 表单）
        image (UploadFile)：上传的图片文件对象
        query (Optional[str])：用户问题，可为 None
        session_id (Optional[str])：会话 ID，可为 None

    返回值：
        AnalyzeResponse | JSONResponse（错误时）
    """
    # 生成本次请求的唯一 ID（用于全链路日志追踪）
    request_id = new_request_id()
    t0 = time.perf_counter()  # 记录请求开始时间（用于计算各阶段耗时）

    # ── 步骤1：task 枚举校验 ──
    # 将字符串转换为 TaskType 枚举，不在枚举中则返回 422
    try:
        task_enum = TaskType(task.lower())  # lower()：大小写不敏感
    except ValueError:
        # task 值不在 TaskType 枚举中（如 "invalid_task"）
        api_ms = int((time.perf_counter() - t0) * 1000)
        log_request(
            logger, "warning", request_id=request_id, task=task, status="rejected",
            error="INVALID_TASK", latency_ms={"total": api_ms, "api_layer": api_ms},
        )
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                request_id=request_id,
                error_code="INVALID_TASK",
                message=f"不支持的任务类型: {task}，仅支持 vqa/mrg/rag",
                latency_ms=api_ms,
            ).model_dump(),  # model_dump()：Pydantic v2 中将模型转为字典
        )

    # ── 步骤2：图像 content-type 前置检查（在读取文件内容之前，快速拒绝明显非法格式）──
    allowed_mimes = {"image/jpeg", "image/jpg", "image/png"}
    if image.content_type and image.content_type.lower() not in allowed_mimes:
        # MIME 类型不在白名单中（如 image/gif、application/pdf）
        api_ms = int((time.perf_counter() - t0) * 1000)
        return JSONResponse(
            status_code=415,  # 415 Unsupported Media Type
            content=ErrorResponse(
                request_id=request_id,
                error_code="UNSUPPORTED_FORMAT",
                message=f"不支持的文件类型: {image.content_type}，仅支持 image/jpeg、image/png",
                latency_ms=api_ms,
            ).model_dump(),
        )

    # ── 步骤3：图像读取 & 深度校验（使用 PIL 校验格式/分辨率/完整性）──
    try:
        data = await image.read()  # 异步读取上传文件的全部字节数据
        fmt, (w, h) = validate_image(data, filename=image.filename)  # 调用校验函数
    except ImageValidationError as e:
        # 校验失败（如文件过大、格式错误、分辨率过低）
        api_ms = int((time.perf_counter() - t0) * 1000)
        log_request(
            logger, "warning", request_id=request_id, task=task_enum.value,
            status="rejected", error=e.code,
            image_size_kb=len(data) // 1024 if "data" in locals() else 0,  # 若 data 已赋值则记录大小
            latency_ms={"total": api_ms, "api_layer": api_ms},
        )
        return JSONResponse(
            status_code=e.status,  # 从异常对象取 HTTP 状态码（400/413/415/422）
            content=ErrorResponse(
                request_id=request_id, error_code=e.code, message=str(e), latency_ms=api_ms,
            ).model_dump(),
        )

    # 将图片字节数据转换为 base64 字符串（VLM API 所需格式）
    image_b64 = image_bytes_to_b64(data)
    # 记录 API 层（校验层）耗时（不含 Handler 执行耗时）
    api_layer_ms = int((time.perf_counter() - t0) * 1000)

    # ── 步骤4：query 校验（VQA/RAG 任务的 query 必填）──
    if task_enum in (TaskType.VQA, TaskType.RAG):
        if not query or not query.strip():
            # query 为 None 或纯空白字符串
            api_ms = int((time.perf_counter() - t0) * 1000)
            log_request(
                logger, "warning", request_id=request_id, task=task_enum.value,
                status="rejected", error="MISSING_QUERY",
                latency_ms={"total": api_ms, "api_layer": api_ms},
            )
            return JSONResponse(
                status_code=400,  # 400 Bad Request
                content=ErrorResponse(
                    request_id=request_id,
                    error_code="MISSING_QUERY",
                    message=f"{task_enum.value.upper()} 任务需要提供 query",
                    latency_ms=api_ms,
                ).model_dump(),
            )

        # query 超长时截断（避免 VLM 超 token 限制）
        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]  # 截断到最大长度
            log_request(logger, "warning", request_id=request_id, task=task_enum.value,
                        message="query 被截断至 500 字")

    # ── 步骤5：获取对应 Handler（延迟初始化）──
    try:
        handler = get_handler(task_enum)
    except HTTPException as e:
        # CV 预留任务抛出 501 Not Implemented
        return JSONResponse(
            status_code=e.status_code,
            content=ErrorResponse(
                request_id=request_id,
                error_code=e.detail.get("error_code", "NOT_IMPLEMENTED"),
                message=e.detail.get("message", str(e.detail)),
                latency_ms=int((time.perf_counter() - t0) * 1000),
            ).model_dump(),
        )

    # ── 步骤6：执行推理（调用 Handler）──
    try:
        # VQA 任务处理分支
        if task_enum == TaskType.VQA:
            vlm_t0 = time.perf_counter()  # 记录 VLM 调用开始时间
            answer = handler.run(image_b64=image_b64, query=query, image_format=fmt)
            vlm_ms = int((time.perf_counter() - vlm_t0) * 1000)  # VLM 调用耗时
            total_ms = int((time.perf_counter() - t0) * 1000)    # 总耗时

            # 记录成功日志（含详细耗时分解）
            log_request(
                logger, "info", request_id=request_id, task="vqa",
                session_id=session_id, image_size_kb=len(data) // 1024,
                image_resolution=f"{w}x{h}", query_length=len(query) if query else 0,
                vlm_backend=VLM_BACKEND, vlm_model=VLM_MODEL,
                latency_ms={"total": total_ms, "api_layer": api_layer_ms, "vlm_call": vlm_ms},
                status="success",
            )
            return AnalyzeResponse(
                request_id=request_id, task=task_enum, answer=answer,
                latency_ms=total_ms, model=VLM_MODEL, disclaimer=DISCLAIMER,
            )

        # MRG 任务处理分支
        if task_enum == TaskType.MRG:
            vlm_t0 = time.perf_counter()
            # MRG handler.run 返回 (ReportSchema, 格式化文本) 元组
            report, text = handler.run(image_b64=image_b64, extra_context=query or "", image_format=fmt)
            vlm_ms = int((time.perf_counter() - vlm_t0) * 1000)
            total_ms = int((time.perf_counter() - t0) * 1000)

            log_request(
                logger, "info", request_id=request_id, task="mrg",
                session_id=session_id, image_size_kb=len(data) // 1024,
                image_resolution=f"{w}x{h}",
                vlm_backend=VLM_BACKEND, vlm_model=VLM_MODEL,
                latency_ms={"total": total_ms, "api_layer": api_layer_ms, "vlm_call": vlm_ms},
                status="success",
            )
            return AnalyzeResponse(
                request_id=request_id, task=task_enum, answer=text, report=report,
                latency_ms=total_ms, model=VLM_MODEL, disclaimer=DISCLAIMER,
            )

        # RAG 任务处理分支
        if task_enum == TaskType.RAG:
            vlm_t0 = time.perf_counter()
            # RAG handler.run 返回 (回答文本, [RefDoc列表]) 元组
            answer, refs = handler.run(image_b64=image_b64, query=query, image_format=fmt)
            vlm_ms = int((time.perf_counter() - vlm_t0) * 1000)
            total_ms = int((time.perf_counter() - t0) * 1000)

            log_request(
                logger, "info", request_id=request_id, task="rag",
                session_id=session_id, image_size_kb=len(data) // 1024,
                image_resolution=f"{w}x{h}", query_length=len(query) if query else 0,
                retrieved_docs=len(refs),          # 记录检索到的文档数
                top_score=refs[0].score if refs else 0.0,  # 记录最高相似度分值
                vlm_backend=VLM_BACKEND, vlm_model=VLM_MODEL,
                latency_ms={"total": total_ms, "api_layer": api_layer_ms, "vlm_call": vlm_ms},
                status="success",
            )
            return AnalyzeResponse(
                request_id=request_id, task=task_enum, answer=answer, references=refs,
                latency_ms=total_ms, model=VLM_MODEL, disclaimer=DISCLAIMER,
            )

        # 理论上不会到这里（所有 TaskType 都已处理）
        raise HTTPException(status_code=501, detail={"error_code": "NOT_IMPLEMENTED", "message": f"{task_enum.value} 尚未启用"})

    # ── 异常处理层 ──

    except VLMError as e:
        # VLM 调用失败（超时/限流/API错误）
        api_ms = int((time.perf_counter() - t0) * 1000)
        log_request(
            logger, "error", request_id=request_id, task=task_enum.value,
            status="error", error=str(e), error_code=e.code,
            latency_ms={"total": api_ms, "api_layer": api_layer_ms},
        )
        # VLM_TIMEOUT → 504 Gateway Timeout（上游超时）
        # 其他 VLM 错误 → 503 Service Unavailable（上游不可用）
        status = 504 if e.code == "VLM_TIMEOUT" else 503
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(
                request_id=request_id, error_code=e.code, message=str(e), latency_ms=api_ms,
            ).model_dump(),
        )

    except HTTPException:
        # FastAPI 的 HTTPException 直接向上抛出（不捕获，由 FastAPI 框架处理）
        raise

    except Exception as e:  # noqa: BLE001
        # 未预料的异常（如 Handler 内部 bug）→ 500 Internal Server Error
        api_ms = int((time.perf_counter() - t0) * 1000)
        log_request(
            logger, "error", request_id=request_id, task=task_enum.value,
            status="error", error=repr(e),  # repr(e) 包含异常类型和消息
            latency_ms={"total": api_ms, "api_layer": api_layer_ms},
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                request_id=request_id,
                error_code="INTERNAL_ERROR",
                message="服务内部错误，请稍后重试",  # 不暴露内部错误详情给客户端
                latency_ms=api_ms,
            ).model_dump(),
        )


def main():
    """
    本地运行入口：python -m src.api

    与 run.py 的区别：run.py 是项目级启动脚本（直接 python run.py），
    此函数用于通过 python -m src.api 方式运行（模块化运行）。
    """
    import uvicorn
    from .config import API_HOST, API_PORT
    uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=False)


# 当此模块被直接运行时（而非被 import）执行 main()
if __name__ == "__main__":
    main()
