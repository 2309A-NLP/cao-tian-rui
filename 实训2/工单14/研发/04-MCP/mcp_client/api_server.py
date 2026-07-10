"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-MCP

MCP 网关（FastAPI :8014）
------------------------
对外暴露：
  GET  /                         → 返回 frontend/map.html
  POST /chat                     → Router Agent 主入口（LLM 路由 + 多轮 tool）
  GET  /api/amap/js-config       → 前端加载高德 JS 需要的 key/安全码（后端注入，避免泄露）
  GET  /api/amap/hospital        → 直接调 amap_mcp.hospital_search
  GET  /api/amap/route           → 直接调 amap_mcp.route_planning
  GET  /api/amap/nearby          → 直接调 amap_mcp.nearby_hotels / restaurants / pharmacies / parking
  GET  /health                   → 健康检查
  GET  /servers                  → MCP 各 Server 的连接状态与工具列表

MCPClientPool 的生命周期通过 lifespan 事件管理，全局单例。
"""
import json  # 标准库：JSON 序列化/反序列化，用于解析工具返回值
import os  # 标准库：读取环境变量（AMAP_JS_KEY、PORT 等）
from contextlib import asynccontextmanager  # 标准库：提供异步上下文管理器装饰器，用于 FastAPI lifespan
from pathlib import Path  # 标准库：面向对象的文件路径操作
from typing import Optional  # 标准库：类型注解，Optional[X] 表示 X 或 None

from dotenv import load_dotenv  # python-dotenv 包：从 .env 文件加载环境变量到 os.environ
from fastapi import Depends, FastAPI, HTTPException, Query  # fastapi 包：Web 框架核心组件
# - Depends: 依赖注入（如鉴权中间件）
# - FastAPI: 主应用类
# - HTTPException: HTTP 错误响应
# - Query: 声明查询参数及其描述/校验规则
from fastapi.middleware.cors import CORSMiddleware  # fastapi 内置 CORS 跨域中间件
from fastapi.responses import HTMLResponse  # fastapi：返回 HTML 字符串的响应类
from pydantic import BaseModel  # pydantic 包：基于类型注解的数据校验/序列化，FastAPI 请求/响应体基类

from mcp_client.auth import verify_api_key  # 本项目：API Key 鉴权中间件函数
from mcp_client.router_agent import MCPClientPool, agent_answer  # 本项目：MCP 连接池 + 路由 Agent 主函数

# 解析出 04-MCP/ 根目录的绝对路径（__file__ 是本文件，.parent.parent 上跳两级）
_ROOT = Path(__file__).resolve().parent.parent
# 加载 .env 配置文件（覆盖之前的 os.environ）
load_dotenv(_ROOT / ".env")

# 从环境变量读取高德 JS SDK 所需的 key 和安全码（前端不直接存这两个值，防止泄露）
AMAP_JS_KEY = os.getenv("AMAP_JS_KEY", "")
AMAP_JS_SECURITY_CODE = os.getenv("AMAP_JS_SECURITY_CODE", "")
# 网关监听端口，默认 8014
PORT = int(os.getenv("MCP_GATEWAY_PORT", "8014"))
# 前端地图页面的路径
_HTML = _ROOT / "frontend" / "map.html"


# ────────── Lifespan：全局 MCPClientPool ──────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 事件：应用启动时建立 5 个 MCP Server 的长连接，关闭时释放。

    使用 async with MCPClientPool() 保证进入时初始化、退出时清理（AsyncExitStack 负责子进程回收）。
    pool 挂载到 app.state.pool，供所有路由函数读取。
    """
    print(f"[api_server] 正在初始化 MCP Client Pool ...", flush=True)
    # 进入 MCPClientPool 异步上下文：启动 5 个 stdio 子进程并建立 MCP session
    async with MCPClientPool() as pool:
        app.state.pool = pool  # 将连接池注入 app 全局状态，供各路由 handler 访问
        print(f"[api_server] Pool ready. 已注册 tool: {list(pool.tool_to_server.keys())}", flush=True)
        yield  # yield 之前 = 启动逻辑；yield 之后 = 关闭逻辑
    # 退出 async with 后，MCPClientPool.__aexit__ 会关闭所有子进程
    print(f"[api_server] Pool closed.", flush=True)


# 创建 FastAPI 主应用，title/version 显示在自动生成的 /docs 页
app = FastAPI(title="医疗智能体 MCP 网关", version="1.0", lifespan=lifespan)

# 从环境变量解析 CORS 允许的来源列表（逗号分隔），默认允许本机四个常用端口
_origins = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8000,http://localhost:8011,http://localhost:8012,http://localhost:8014",
).split(",")]

# 注册 CORS 中间件：允许指定来源的 GET/POST/OPTIONS 请求跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,       # 允许的来源列表
    allow_methods=["GET", "POST", "OPTIONS"],  # 允许的 HTTP 方法
    allow_headers=["*"],          # 允许所有请求头（含 X-API-Key 鉴权头）
)


# ────────── 请求/响应模型 ──────────

class ChatRequest(BaseModel):
    """POST /chat 的请求体模型。"""
    query: str       # 用户自然语言输入
    user_id: int = 1  # 用户 ID，默认 1（挂号等工具需要）


class ChatResponse(BaseModel):
    """POST /chat 的响应体模型。"""
    reply: str                # LLM 最终自然语言回复
    intent_hits: list[str] = []  # 本轮命中的 tool 名称列表（意图轨迹）
    tool_calls: list[dict] = []  # 详细工具调用记录（name/args/result）
    turns: int = 0            # 实际多轮次数


# ────────── 前端页面 ──────────

@app.get("/", response_class=HTMLResponse)
def index():
    """
    返回高德地图前端页面 (frontend/map.html)。
    文件不存在时返回 404。
    """
    # 检查 map.html 是否存在
    if not _HTML.exists():
        return HTMLResponse("<h3>frontend/map.html 未找到</h3>", status_code=404)
    # 读取并以 HTML 类型返回
    return _HTML.read_text(encoding="utf-8")


# ────────── 健康检查 & 状态 ──────────

@app.get("/health")
def health():
    """简单健康检查，返回服务名称和端口，供 Docker/k8s 探活使用。"""
    return {"status": "ok", "service": "mcp-gateway", "port": PORT}


@app.get("/servers")
def servers(request_app: FastAPI = Depends(lambda: app)):
    """
    返回 5 个 MCP Server 的连接状态和已注册工具列表。
    Pool 未初始化时返回 not_ready。
    """
    # 从 app.state 获取连接池（若 lifespan 尚未就绪则为 None）
    pool: Optional[MCPClientPool] = getattr(app.state, "pool", None)
    if not pool:
        # Pool 还没初始化完成，返回未就绪状态
        return {"status": "not_ready", "servers": []}
    result = []
    # 遍历所有 MCP Server 名称，逐个检查是否存活及已注册哪些工具
    for name in ["registration", "knowledge", "amap", "imaging", "voice"]:
        alive = name in pool.sessions  # sessions 字典中有该 key 说明子进程已建立
        # 从 tool_to_server 反查属于该 server 的所有工具名
        tools = [n for n, s in pool.tool_to_server.items() if s == name]
        result.append({"server": name, "alive": alive, "tools": tools})
    return {"status": "ready", "servers": result}


# ────────── 主 Chat 接口 ──────────

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    """
    主对话接口，带 API Key 鉴权（verify_api_key 作为依赖注入）。
    将请求转发给 agent_answer，返回最终回复和工具调用轨迹。
    """
    # 从 app.state 取出连接池
    pool: Optional[MCPClientPool] = getattr(app.state, "pool", None)
    if not pool:
        # 连接池未就绪，返回 500 错误
        raise HTTPException(500, "MCP Pool 未初始化")
    # 调用路由 Agent 主逻辑（多轮 LLM + tool 调用）
    result = await agent_answer(pool, req.query, user_id=req.user_id)
    # 将结果字典解包为 ChatResponse 对象返回
    return ChatResponse(**result)


# ────────── 高德前端配置 ──────────

@app.get("/api/amap/js-config")
def amap_js_config():
    """
    前端加载高德 JS SDK 前需要拿到 key 和 securityJsCode。
    通过后端注入避免明文写在 HTML 里（防止 key 被扒取滥用）。
    """
    # key 未配置时返回错误提示
    if not AMAP_JS_KEY:
        return {"ok": False, "error": "AMAP_JS_KEY 未配置"}
    return {
        "ok": True,
        "js_key": AMAP_JS_KEY,
        "security_js_code": AMAP_JS_SECURITY_CODE,
    }


# ────────── 高德 REST 快捷代理（前端表单直接调） ──────────

async def _call_amap_tool(request_app: FastAPI, name: str, args: dict) -> dict:
    """
    内部辅助函数：通过连接池调用指定高德工具，解析返回的 JSON。

    Args:
        request_app: FastAPI 应用实例（用于访问 app.state.pool）
        name: MCP 工具名，如 "hospital_search"
        args: 工具参数字典

    Returns:
        工具返回的 JSON dict；解析失败时返回错误 dict
    """
    pool: Optional[MCPClientPool] = getattr(request_app.state, "pool", None)
    if not pool:
        raise HTTPException(500, "MCP Pool 未初始化")
    # 调用 MCP 工具，返回原始字符串
    raw = await pool.call(name, args)
    try:
        # 尝试解析 JSON 字符串
        return json.loads(raw)
    except json.JSONDecodeError:
        # 上游返回非 JSON（如错误信息），截取前 200 字符作为提示
        return {"ok": False, "error": "上游返回非 JSON", "raw": raw[:200]}


@app.get("/api/amap/hospital")
async def api_hospital(name: str = Query(...), city: str = "北京"):
    """
    医院搜索快捷接口，转发给 amap_mcp 的 hospital_search 工具。

    Args:
        name: 医院名称（必填，Query(...)表示必需参数）
        city: 城市，默认"北京"
    """
    return await _call_amap_tool(app, "hospital_search", {"name": name, "city": city})


@app.get("/api/amap/route")
async def api_route(
    origin: str = Query(...),        # 起点，必填
    destination: str = Query(...),   # 终点，必填
    mode: str = "driving",           # 出行方式，默认驾车
    city: str = "北京",              # 城市，公交查询时必需
):
    """路线规划快捷接口，转发给 amap_mcp 的 route_planning 工具。"""
    return await _call_amap_tool(app, "route_planning", {
        "origin": origin, "destination": destination, "mode": mode, "city": city,
    })


@app.get("/api/amap/nearby")
async def api_nearby(
    hospital: str = Query(...),  # 医院名称，必填
    category: str = Query("hotels", description="hotels|restaurants|pharmacies|parking"),  # 周边类别
    radius: Optional[int] = None,  # 搜索半径（米），可选
    city: str = "北京",
):
    """
    医院周边搜索快捷接口，根据 category 分发到不同工具。

    category 映射：
      hotels       -> nearby_hotels
      restaurants  -> nearby_restaurants
      pharmacies   -> nearby_pharmacies
      parking      -> nearby_parking
    """
    # category 到工具名的映射表
    tool_map = {
        "hotels": "nearby_hotels",
        "restaurants": "nearby_restaurants",
        "pharmacies": "nearby_pharmacies",
        "parking": "nearby_parking",
    }
    # category 不在允许值中，返回 400 错误
    if category not in tool_map:
        raise HTTPException(400, f"category 无效，可选: {list(tool_map)}")
    args = {"hospital": hospital, "city": city}
    # radius 参数可选，有值才传
    if radius:
        args["radius"] = radius
    return await _call_amap_tool(app, tool_map[category], args)


# ────────── 直接启动 ──────────

if __name__ == "__main__":
    import uvicorn  # uvicorn：ASGI 服务器，用于运行 FastAPI 应用
    # 以非热重载模式启动（生产环境），监听所有网卡
    uvicorn.run("mcp_client.api_server:app", host="0.0.0.0", port=PORT, reload=False)
