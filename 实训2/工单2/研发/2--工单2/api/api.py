"""
FastAPI 后端 — 日程提醒智能体
工单编号：人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
启动：python run.py
"""
import sys
import json
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
from backend.reminder import ReminderThread, pop_pending_reminders, last_reminders

logger = get_logger("api")

# ═══════════════════════════════════════════════
#  全局资源
# ═══════════════════════════════════════════════
config = get_config()
db: Optional[DatabaseManager] = None
llm: Optional[LLMProvider] = None
agent: Optional[AgentEngine] = None
reminder_thread: Optional[ReminderThread] = None


def _init_resources():
    """初始化数据库、LLM、Agent、提醒线程"""
    global db, llm, agent, reminder_thread

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

    # 后台提醒线程（独立连接，避免线程共享 socket 导致 WinError 10038）
    if db:
        _reminder_db = DatabaseManager(
            host=config.db_host, port=config.db_port,
            user=config.db_user, password=config.db_password,
            database=config.db_name,
        )
        _reminder_db.connect()
        reminder_thread = ReminderThread(db_getter=lambda: _reminder_db, interval=30)
        reminder_thread.start()
        logger.info("后台提醒线程已启动")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_resources()
    yield
    if reminder_thread:
        reminder_thread.stop()
    if db:
        db.close()


app = FastAPI(
    title="日程提醒智能体 - 小暖",
    description="基于 Function Calling 的智能日程提醒助手",
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
    return JSONResponse({"message": "前端页面未就绪，请访问 API 端点"}, status_code=404)


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
        "reminder_running": reminder_thread is not None and reminder_thread.is_alive(),
    }


@app.get("/api/welcome")
def welcome():
    """获取开场白 + 创建新会话"""
    if not agent:
        raise HTTPException(503, "Agent 引擎未就绪")
    from backend.agent_engine import OPENING_MESSAGE
    sess = agent.get_session()
    sess.opening_sent = True
    return {"reply": OPENING_MESSAGE, "session_id": sess.session_id}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """同步对话接口"""
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
    """流式对话接口（SSE）"""
    if not agent:
        raise HTTPException(503, "Agent 引擎未就绪")

    if not req.message or not req.message.strip():
        raise HTTPException(400, "消息不能为空")

    logger.info(f"收到流式消息: {req.message[:80]}...")

    def generate():
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
#  提醒相关端点
# ═══════════════════════════════════════════════

@app.get("/api/reminders/pending")
def pending_reminders():
    """
    前端轮询此端点获取新提醒。
    返回后即从队列中移除，保证不重复推送。
    """
    reminders = pop_pending_reminders()
    return {"reminders": reminders, "count": len(reminders)}


@app.get("/api/reminders/history")
def reminder_history():
    """查看最近的提醒历史（调试用）"""
    return {"reminders": last_reminders[-20:], "count": len(last_reminders)}


# ═══════════════════════════════════════════════
#  调试端点
# ═══════════════════════════════════════════════

@app.get("/api/records")
def list_records(date: Optional[str] = Query(None), limit: int = Query(50)):
    """
    查看数据库中的日程记录（调试用）。
    验收标准要求的「已存数据表」查验入口。
    """
    if not db:
        raise HTTPException(503, "数据库未连接")
    result = db.query_schedules(schedule_date=date)
    return result


@app.get("/api/sessions")
def list_sessions():
    """查看活跃会话数（调试用）"""
    if not agent:
        return {"sessions": 0}
    return {"sessions": len(agent.sessions), "ids": list(agent.sessions.keys())}
