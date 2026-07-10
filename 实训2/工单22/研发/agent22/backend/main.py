"""
FastAPI 主入口

- 挂载对话路由（/api/chat）和记忆路由（/api/memory）
- 静态文件托管前端（/  →  ../frontend/）
- 允许前端跨域访问（CORS）
- lifespan shutdown：等待所有 fire-and-forget 记忆写入线程完成

启动：
    cd "F:/kimi  project/agent22/backend"
    venv/Scripts/activate
    uvicorn main:app --host 0.0.0.0 --port 8022 --reload
"""
# __future__.annotations：延迟类型注解求值，允许在类型提示中引用尚未定义的类名（避免循环引用报错）
from __future__ import annotations

# logging：Python 标准库，提供多级别日志记录（DEBUG/INFO/WARNING/ERROR），输出到控制台或文件
import logging
# os：Python 标准库，用于读取操作系统环境变量（os.getenv）、路径拼接等
import os
# sys：Python 标准库，sys.path 是 Python 模块搜索路径列表，可动态添加自定义目录
import sys
# asynccontextmanager：将普通异步生成器函数装饰为异步上下文管理器，用于定义 FastAPI lifespan 钩子
from contextlib import asynccontextmanager
# Path：pathlib 标准库，面向对象的跨平台路径操作类，比 os.path 更简洁直观
from pathlib import Path

# dotenv（python-dotenv）：第三方包，从 .env 文件读取键值对并写入 os.environ，
# 方便本地开发时管理 API key 等敏感配置，不硬编码在代码里
from dotenv import load_dotenv

# ── 加载环境变量（必须在导入 agents 前完成，否则 agents 里的 os.getenv 读不到值）──────────────────────────
# Path(__file__).parent：main.py 所在目录（即 backend/）
# / ".env"：Path 对象的 / 运算符等价于 os.path.join，拼接出 backend/.env 的绝对路径
load_dotenv(Path(__file__).parent / ".env")  # 将 backend/.env 中的配置项加载到环境变量

# ── 将 backend/ 加入 Python 模块搜索路径，使 `from src.xxx` / `from agents.xxx` 等导入生效 ──
# str(Path(__file__).parent)：得到 backend/ 目录的字符串形式绝对路径
# sys.path.insert(0, ...)：插到搜索列表最前面，确保优先从 backend/ 查找模块
sys.path.insert(0, str(Path(__file__).parent))  # 注册 backend/ 为顶级包搜索路径

# FastAPI：第三方 Web 框架，基于 Starlette + Pydantic，支持异步、自动生成 OpenAPI 文档
from fastapi import FastAPI
# CORSMiddleware：Starlette 内置中间件，向响应头注入 Access-Control-Allow-* 字段，
# 使浏览器允许前端页面向不同端口/域名的后端发起请求
from fastapi.middleware.cors import CORSMiddleware
# StaticFiles：将本地目录挂载为静态资源服务，前端的 CSS/JS/图片通过此方式暴露
from fastapi.staticfiles import StaticFiles
# FileResponse：以流式方式返回磁盘文件（如 index.html），支持 Content-Type 自动推断
from fastapi.responses import FileResponse

# 导入对话路由模块（包含 POST /api/chat/medical 等端点）
from api.routes_chat import router as chat_router
# 导入记忆管理路由模块（包含 GET/DELETE /api/memory/* 端点）
from api.routes_memory import router as memory_router
# 导入后台写入等待函数：应用关闭时调用，确保所有异步记忆写入线程都已完成
from src.memory_client import wait_pending_writes

# ── 日志配置 ─────────────────────────────────────────────────────
# basicConfig：一次性配置根 logger，所有子模块的 logger 默认继承此配置
logging.basicConfig(
    level=logging.INFO,                                       # 只输出 INFO 及以上级别（过滤 DEBUG 噪音）
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",   # 格式：时间 [级别] 模块名 - 消息内容
    datefmt="%H:%M:%S",                                       # 时间戳格式：时:分:秒
)

# 创建当前模块专属的 logger，名称 "agent22.main" 会出现在每条日志前缀中
logger = logging.getLogger("agent22.main")


# ── Lifespan：FastAPI 应用生命周期管理（替代旧版 @app.on_event）────────
# @asynccontextmanager：将此函数变成异步上下文管理器
# yield 之前 = startup 逻辑；yield 之后 = shutdown 逻辑
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 应用生命周期钩子。

    startup：直接 yield，无需额外初始化（MemoryClient 是懒加载单例）。
    shutdown：阻塞等待所有后台记忆写入线程完成，防止进程退出时丢失未写入的记忆。

    Args:
        app: FastAPI 应用实例（由框架自动传入，此处未直接使用）。
    """
    yield  # ← 应用正常运行阶段；yield 前后分别对应 startup / shutdown

    # —— 以下代码在应用接收到关闭信号（Ctrl-C / SIGTERM）后执行 ——
    logger.info("shutdown: 等待后台记忆写入线程...")
    # asyncio.to_thread：把同步阻塞函数 wait_pending_writes 放到线程池执行，
    # 避免阻塞 asyncio 事件循环导致其他清理任务无法运行
    # __import__("asyncio")：动态导入，避免在模块顶部出现循环引用（此处可直接 import asyncio）
    # wait_pending_writes(60.0)：最多等待 60 秒，超时后强制退出
    await __import__("asyncio").to_thread(wait_pending_writes, 60.0)
    logger.info("shutdown: 完成")


# ── FastAPI 应用实例 ──────────────────────────────────────────────────
# FastAPI()：创建 ASGI 应用对象，uvicorn 会加载此对象并启动 HTTP 服务
app = FastAPI(
    title="Agent22 - mem0 多领域长期记忆系统",           # 显示在自动生成的 /docs（Swagger UI）页面标题
    description="医疗 / 文旅 / 教育三领域智能体，基于 mem0 实现跨对话长期记忆",  # /docs 页面描述文字
    version="1.0.0",    # API 版本号，显示在 /docs 和 /openapi.json 中
    lifespan=lifespan,  # 指定生命周期管理器（FastAPI 0.93+ 推荐方式）
)

# ── CORS 中间件 ──────────────────────────────────────────────────────
# 浏览器的同源策略会阻止前端 JS 向不同端口的后端发请求；CORS 中间件通过响应头告知浏览器允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 允许所有来源（生产环境建议改为 ["http://localhost:3000"] 等具体域名）
    allow_methods=["*"],   # 允许所有 HTTP 方法：GET, POST, PUT, DELETE, OPTIONS 等
    allow_headers=["*"],   # 允许所有请求头字段（Content-Type, Authorization 等）
)

# ── 路由注册 ─────────────────────────────────────────────────────
# include_router：将子路由模块合并到主应用，等价于"挂载路由蓝图"
app.include_router(chat_router)    # 挂载对话路由：POST /api/chat/{domain}
app.include_router(memory_router)  # 挂载记忆路由：GET/DELETE /api/memory/{domain}/{user_id}


# ── 健康检查端点 ─────────────────────────────────────────────────────
# GET /api/health：供 Kubernetes 存活探针、前端启动检测使用
@app.get("/api/health")
async def health():
    """返回服务健康状态和版本号。

    Returns:
        dict: {"status": "ok", "version": "1.0.0"}
    """
    return {"status": "ok", "version": "1.0.0"}  # 返回 JSON，前端可据此判断后端是否已就绪


# ── 静态前端资源托管 ──────────────────────────────────────────────────
# 计算前端目录路径：backend/ 的父目录（agent22/）下的 frontend/ 子目录
_frontend = Path(__file__).parent.parent / "frontend"

# 仅当 frontend/ 目录实际存在时才挂载（开发环境可能只跑后端，无前端目录）
if _frontend.exists():
    # StaticFiles：将 frontend/ 目录下的所有文件通过 /static/ URL 前缀暴露
    # 前端的 index.html 中引用的 /static/style.css 等资源都由此提供
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

    # GET /：访问根路径时返回前端 SPA 入口页面
    # include_in_schema=False：不在 /docs Swagger 文档中显示此端点（属于前端，非 API）
    @app.get("/", include_in_schema=False)
    async def index():
        """返回前端 SPA（单页应用）入口 index.html。"""
        # FileResponse：以流的方式传输 index.html 文件内容，浏览器收到后渲染页面
        return FileResponse(str(_frontend / "index.html"))
