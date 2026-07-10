"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-健康咨询

FastAPI HTTP 接口，端口 8012（医疗智能体体系：wt11=8011, wt12=8012, wt13=8013...）

端点：
  GET  /           → 返回 chat.html 前端页面
  GET  /health     → 健康检查
  GET  /stats      → 知识图谱节点统计（前端侧边栏使用）
  POST /chat       → 主要问答接口（一次性返回）
  POST /chat/stream → 流式问答接口（SSE，首字节 <500ms）
  POST /cache/clear → 清空缓存
  GET  /cache/stats → 缓存状态
"""
import json   # 标准库：序列化 SSE 事件为 JSON 字符串
import os     # 标准库：读取环境变量（CORS_ORIGINS）
import re     # 标准库：正则校验 patient_id 合法字符
from pathlib import Path  # 标准库：路径操作，定位 chat.html 文件

# fastapi 包：高性能异步 Web 框架，自动生成 OpenAPI 文档
from fastapi import Depends, FastAPI, HTTPException
# CORSMiddleware：跨域资源共享中间件，允许前端页面跨端口调用 API
from fastapi.middleware.cors import CORSMiddleware
# HTMLResponse：返回 HTML 内容；StreamingResponse：SSE 流式返回
from fastapi.responses import HTMLResponse, StreamingResponse
# pydantic 包：数据验证框架，BaseModel 定义请求/响应结构
from pydantic import BaseModel, Field, field_validator

# 导入 Agent 核心功能
from src.agent import run_agent, run_agent_stream, cache_stats, clear_cache, validate_query
from src.auth import verify_api_key  # API Key 鉴权依赖项

# 创建 FastAPI 应用实例，title/version 会显示在自动生成的 /docs 文档中
app = FastAPI(title="医疗健康咨询Agent", version="2.0")

# ── 解析允许的跨域来源列表（从环境变量读取，逗号分隔）──
# 默认允许同体系其他微服务端口（8011/8012/8013）和本地开发端口（8000）
_origins = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8011,http://127.0.0.1:8011,"
    "http://localhost:8012,http://127.0.0.1:8012,"
    "http://localhost:8013,http://127.0.0.1:8013,"
    "http://localhost:8000,http://127.0.0.1:8000",
).split(",")]

# 注册 CORS 中间件，配置允许的来源、方法和请求头
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,       # 允许的来源列表
    allow_methods=["GET", "POST"], # 只允许 GET 和 POST
    allow_headers=["*"],           # 允许所有请求头（包括 X-API-Key）
)

# 前端 HTML 文件路径（相对于本文件往上一级目录）
_HTML = Path(__file__).parent.parent / "chat.html"


# ── 请求/响应数据模型 ──

class ChatRequest(BaseModel):
    """
    /chat 和 /chat/stream 接口的请求体模型。
    Pydantic 会自动校验字段类型和约束，不符合时返回 422 错误。
    """
    # query 为必填字段，长度限制 1-500 字符
    query: str = Field(..., min_length=1, max_length=500, description="用户提问")
    # patient_id 可选，默认"default_patient"，最长 64 字符
    patient_id: str = Field("default_patient", max_length=64, description="患者ID")

    @field_validator("query")
    @classmethod
    def _check_query(cls, v: str) -> str:
        """
        额外校验 query 内容（Pydantic 字段级别验证器）。
        调用 Agent 的 validate_query 函数，确保与内部逻辑一致。
        """
        err = validate_query(v)
        if err:
            raise ValueError(err)  # Pydantic 会将其转换为 422 响应
        return v.strip()  # 返回去空白版本

    @field_validator("patient_id")
    @classmethod
    def _check_patient_id(cls, v: str) -> str:
        """
        校验 patient_id 只含安全字符（防 NoSQL 注入/路径注入）。
        只允许字母、数字、下划线、中划线。
        """
        v = v.strip() or "default_patient"  # 空白值回退到默认
        # 正则白名单：只允许 [A-Za-z0-9_-]
        if not re.match(r"^[A-Za-z0-9_\-]+$", v):
            raise ValueError("patient_id 只允许字母、数字、下划线、中划线")
        return v


class ChatResponse(BaseModel):
    """
    /chat 接口的响应体模型。
    Pydantic 负责序列化为 JSON 并过滤多余字段。
    """
    reply: str                        # 回答文本
    intent: str                       # 意图分类结果
    entity: str                       # 提取的医学实体
    graph_data: list                  # 图谱查询原始数据
    elapsed_ms: int = 0               # 总耗时毫秒
    recalled_memories: list = []      # 召回的历史记忆
    cache_hit: bool = False           # 是否命中缓存
    intent_source: str = "llm"        # 意图来源（rule/llm）
    nl2cypher_used: bool = False      # 是否走了 NL2Cypher 兜底路径


# ── HTTP 端点定义 ──

@app.get("/", response_class=HTMLResponse)
def index():
    """返回前端聊天页面（chat.html）。"""
    return _HTML.read_text(encoding="utf-8")  # 读取 HTML 文件内容返回给浏览器


@app.get("/health")
def health():
    """健康检查端点，供 Docker/K8s 探针使用。"""
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """
    返回知识图谱节点统计数字，供前端侧边栏展示。
    Neo4j 连接失败时返回已知的静态数据（服务可用性不受影响）。
    """
    from src.graph import get_driver  # 延迟导入，避免启动时 Neo4j 未就绪导致崩溃
    try:
        result = {}
        with get_driver().session() as s:
            # 逐标签查询节点总数
            for label in ["Disease", "Symptom", "Drug", "Department", "Food", "Transmission"]:
                c = s.run(f"MATCH (n:{label}) RETURN count(n) as c").single()["c"]
                result[label] = c
            # 查询全图关系总数
            rel_c = s.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
            result["relations"] = rel_c
        return result
    except Exception:
        # Neo4j 不可用时返回静态兜底数据（数字来自已知的完整导入结果）
        return {
            "Disease": 6720, "Symptom": 17924, "Drug": 3828,
            "Department": 54, "Food": 4754, "Transmission": 351,
            "relations": 129986,
        }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(req: ChatRequest):
    """
    非流式问答接口：一次性返回完整回答。

    依赖项 verify_api_key 在请求进入此函数前自动执行鉴权检查。
    Depends() 是 FastAPI 的依赖注入机制。
    """
    result = run_agent(req.query, patient_id=req.patient_id)
    return ChatResponse(**result)  # 将 dict 展开赋值给 Pydantic 模型


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_stream(req: ChatRequest):
    """
    SSE 流式问答接口。

    返回 StreamingResponse，媒体类型为 text/event-stream。
    事件格式：`data: {json}\\n\\n`（标准 SSE 格式）
    """
    def event_gen():
        """生成器函数：逐事件产出 SSE 格式字符串。"""
        for evt in run_agent_stream(req.query, patient_id=req.patient_id):
            # 每个事件序列化为 JSON 并包裹为 SSE data 行
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",        # 禁止浏览器缓存 SSE 响应
            "X-Accel-Buffering": "no",           # 关闭 Nginx 缓冲，确保逐块下发（关键！）
        },
    )


@app.get("/cache/stats", dependencies=[Depends(verify_api_key)])
def cache_status():
    """返回当前 LRU 缓存状态（条目数 / 最大容量）。"""
    return cache_stats()


@app.post("/cache/clear", dependencies=[Depends(verify_api_key)])
def cache_clear():
    """清空 LRU 缓存，返回清空的条目数。用于测试前重置缓存状态。"""
    n = clear_cache()
    return {"cleared": n}
