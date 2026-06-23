"""
FastAPI 后端 — 记账 Agent 智能体
工单编号：人工智能 NLP-Agent 数字人项目-记账本任务
启动：python run.py
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel

# 把项目根目录加入 sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from backend.config import get_config
from backend.logger import get_logger
from backend.database import DatabaseManager
from backend.llm_provider import LLMProvider
from backend.agent_engine import AgentEngine

logger = get_logger("api")

# ═══════════════════════════════════════════════
#  全局资源
# ═══════════════════════════════════════════════
config = get_config()
db: Optional[DatabaseManager] = None
llm: Optional[LLMProvider] = None
agent: Optional[AgentEngine] = None


def _init_resources():
    """初始化数据库、LLM、Agent"""
    global db, llm, agent

    # 数据库
    db = DatabaseManager(
        host=config.db_host, port=config.db_port,
        user=config.db_user, password=config.db_password,
        database=config.db_name,
    )
    if not db.connect():
        logger.error("数据库连接失败！请检查 MySQL 是否启动")
        db = None
    else:
        logger.info("数据库就绪")

    # LLM
    llm = LLMProvider(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        temperature=config.llm_temperature,
    )
    logger.info(f"LLM 就绪: {config.llm_model} @ {config.llm_base_url}")

    # Agent
    if llm and db:
        agent = AgentEngine(llm, db, max_rounds=config.max_tool_call_rounds)
        logger.info("Agent 引擎就绪")
    else:
        logger.error("Agent 引擎初始化失败：缺少 LLM 或 DB")
        agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_resources()
    yield
    if db:
        db.close()


app = FastAPI(
    title="家庭记账 Agent",
    description="基于 Function Calling 的智能记账助手",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════
#  静态文件
# ═══════════════════════════════════════════════
_static_dir = _project_root / "api" / "static"


@app.get("/")
def index():
    """前端页面"""
    idx = _static_dir / "index.html"
    if idx.exists():
        return FileResponse(str(idx), headers={"Cache-Control": "no-cache"})
    return JSONResponse({"message": "前端页面未就绪"}, status_code=404)


# ═══════════════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


# ═══════════════════════════════════════════════
#  API 端点
# ═══════════════════════════════════════════════

@app.get("/api/health")
def health():
    """健康检查"""
    return {
        "status": "ok",
        "database": db is not None and (db.conn is not None and db.conn.open),
        "llm": config.llm_model,
        "agent_ready": agent is not None,
    }


@app.get("/api/welcome")
def welcome():
    """页面加载时调用，返回开场白并创建一个新会话"""
    if not agent:
        raise HTTPException(503, "Agent 引擎未就绪")
    from backend.agent_engine import OPENING_MESSAGE
    sess = agent.get_session()
    sess.opening_sent = True  # 标记已发，chat() 不再重复
    return {"reply": OPENING_MESSAGE, "session_id": sess.session_id}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    同步对话接口。
    """
    if not agent:
        raise HTTPException(503, "Agent 引擎未就绪，请检查数据库和 LLM 配置")

    if not req.message or not req.message.strip():
        raise HTTPException(400, "消息不能为空")

    logger.info(f"收到消息: {req.message[:80]}...")

    try:
        result = agent.chat(req.message.strip(), session_id=req.session_id)
        logger.info(f"回复: {result.get('reply','')[:80]}...")
        return {
            "reply": result["reply"],
            "session_id": result["session_id"],
            "tool_calls_made": result.get("tool_calls_made", []),
        }
    except Exception as e:
        import traceback
        logger.error(f"Agent 异常: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Agent 异常: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口（SSE）。前端通过 EventSource/fetch 接收逐字推送。
    """
    if not agent:
        raise HTTPException(503, "Agent 引擎未就绪")

    if not req.message or not req.message.strip():
        raise HTTPException(400, "消息不能为空")

    logger.info(f"收到流式消息: {req.message[:80]}...")

    def generate():
        import json
        for event in agent.chat_stream(req.message.strip(), session_id=req.session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ═══════════════════════════════════════════════
#  调试端点
# ═══════════════════════════════════════════════

@app.get("/api/records")
def list_records(member: Optional[str] = Query(None), limit: int = Query(50)):
    """查看数据库中的记录（调试用）"""
    if not db:
        raise HTTPException(503, "数据库未连接")
    result = db.query_records(member=member)
    return result


@app.get("/api/sessions")
def list_sessions():
    """查看活跃会话数（调试用）"""
    if not agent:
        return {"sessions": 0}
    return {"sessions": len(agent.sessions), "ids": list(agent.sessions.keys())}
