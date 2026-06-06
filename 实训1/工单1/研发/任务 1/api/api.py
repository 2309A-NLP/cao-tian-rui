"""
RAG 问答系统 - FastAPI 后端
前后端分离架构，提供 REST API

启动：uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import json
import hashlib
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── 路径 ──
_project_root = Path(__file__).resolve().parent.parent  # 项目根目录
_backend_dir = _project_root / "backend"
os.chdir(str(_project_root))
for _p in [str(_project_root), str(_backend_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 导入业务模块（绝对导入，避免子进程 sys.path 丢失） ──
from backend.config import AppConfig
from backend.logger import get_logger
from backend.database import DatabaseManager
from backend.llm_provider import LLMFactory
from backend.vector_store import VectorStore
from backend.pdf_processor import PDFProcessor
from backend.rag_engine import RAGEngine
from backend.rag_engine import ChatSession

logger = get_logger("api")

# ── 加载配置 ──
config_path = _project_root / "config.json"
config = AppConfig.load(str(config_path))

# ── 全局资源 ──
db: Optional[DatabaseManager] = None
vs: Optional[VectorStore] = None
llm = None
rag: Optional[RAGEngine] = None
vec_count = 0
_sessions: dict[str, ChatSession] = {}
KNOWLEDGE_BASE_DIR = _project_root / "knowledge_base"


def _init_resources():
    """初始化全局资源（单次）"""
    global db, vs, llm, rag, vec_count, KNOWLEDGE_BASE_DIR
    os.makedirs(config.log_dir, exist_ok=True)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 数据库
    db = DatabaseManager(
        host=config.db_host, port=config.db_port,
        user=config.db_user, password=config.db_password,
        database=config.db_name,
    )
    if not db.connect():
        logger.warning("数据库连接失败")
        db = None

    # LLM
    try:
        llm = LLMFactory.create(
            provider=config.llm_provider,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
        )
        logger.info(f"LLM 已加载: {llm.name}")
    except Exception as e:
        logger.warning(f"LLM 加载失败: {e}")
        llm = None

    # 向量存储
    try:
        vs = VectorStore(
            model_path=config.embedding_model_path,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            collection_name=config.milvus_collection,
        )
        if vs.load_model() and vs.connect_milvus():
            vec_count = vs.count()
            logger.info(f"向量存储已加载 (向量数: {vec_count})")
        else:
            vs = None
            vec_count = 0
    except Exception as e:
        logger.warning(f"向量存储加载失败: {e}")
        vs = None
        vec_count = 0

    # RAG 引擎
    if vs and llm:
        rag = RAGEngine(
            vector_store=vs,
            llm_provider=llm,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
        )
    else:
        rag = None

    logger.info("API 资源初始化完成")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_resources()
    yield
    # 清理
    if vs:
        vs.disconnect()
    if db:
        db.close()


app = FastAPI(
    title="RAG 文档问答 API",
    description="基于招股说明书的知识问答系统 - 前后端分离",
    version="2.0.0",
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
#  模型
# ═══════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class AskRequest(BaseModel):
    query: str
    mode: str = "rag"           # rag / direct / both
    language: str = "中文"      # 中文 / English
    user_id: Optional[int] = None
    session_id: Optional[str] = None

class IndexRequest(BaseModel):
    filenames: list[str] = []   # 知识库里已有的文件名列表


# ═══════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _get_or_create_session(user_id: int, session_id: Optional[str] = None) -> ChatSession:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    s = ChatSession(user_id=user_id, session_id=session_id or f"ses_{int(time.time())}")
    if s.session_id:
        _sessions[s.session_id] = s
    return s


# ═══════════════════════════════════════════════
#  静态文件 & 前端页面
# ═══════════════════════════════════════════════

_static_dir = _project_root / "api" / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/")
def index():
    """返回前端首页"""
    idx = _static_dir / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "前端页面未生成，请先构建 index.html"}, status_code=404)


# ═══════════════════════════════════════════════
#  认证 API
# ═══════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm": llm.name if llm else "unavailable",
        "vector_count": vec_count,
        "database": db is not None,
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/api/auth/login")
def login(req: LoginRequest):
    global db
    if db is None:
        raise HTTPException(503, "数据库未连接")
    result = db.login_user(req.username, _hash_password(req.password))
    if result["success"]:
        return result
    raise HTTPException(401, result.get("message", "登录失败"))

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    global db
    if db is None:
        raise HTTPException(503, "数据库未连接")
    if len(req.username) < 2:
        raise HTTPException(400, "用户名至少 2 个字符")
    if len(req.password) < 4:
        raise HTTPException(400, "密码至少 4 个字符")
    result = db.register_user(req.username, _hash_password(req.password))
    if result["success"]:
        return result
    raise HTTPException(400, result.get("message", "注册失败"))


# ═══════════════════════════════════════════════
#  知识库 API
# ═══════════════════════════════════════════════

@app.get("/api/knowledge")
def get_knowledge():
    """获取知识库状态"""
    if not vs:
        return {"documents": [], "vector_count": 0, "connected": False}
    try:
        docs = vs.list_documents()
        cnt = vs.count()
        return {"documents": docs, "vector_count": cnt, "connected": True}
    except Exception as e:
        return {"documents": [], "vector_count": 0, "connected": False, "error": str(e)}

@app.post("/api/knowledge/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """上传 PDF 到 knowledge_base（判重 + 自动索引）"""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "只支持 PDF 文件")
    
    # 判重
    if vs and vs.is_document_indexed(file.filename):
        return {"filename": file.filename, "status": "exists", "message": "知识库已存在该文档"}
    
    save_path = KNOWLEDGE_BASE_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)
    
    # 自动索引
    indexed = False
    chunks = 0
    if vs:
        from main import index_pdf
        try:
            chunks = index_pdf(str(save_path), config, vs, drop_existing=False, force=False)
            indexed = True
        except Exception as e:
            logger.warning(f"自动索引失败: {e}")
    
    return {
        "filename": file.filename,
        "size": len(content),
        "status": "indexed" if indexed else "uploaded",
        "chunks": chunks,
        "message": f"上传成功，已索引 {chunks} 个块" if indexed else "上传成功（索引失败）"
    }

@app.post("/api/knowledge/index")
def index_pdf_route(req: IndexRequest):
    """索引未入库的 PDF（跳过已索引的）"""
    from main import index_pdf

    if not vs:
        raise HTTPException(503, "向量存储未初始化")

    # 获取已索引的文档列表
    indexed_docs = set(vs.list_documents())
    
    # 扫描 knowledge_base 目录
    pdf_files = sorted([f for f in os.listdir(str(KNOWLEDGE_BASE_DIR)) if f.lower().endswith(".pdf")])
    
    if req.filenames:
        # 只索引指定的文件
        pdf_files = [f for f in pdf_files if f in req.filenames]
    
    results = []
    total = 0
    for fname in pdf_files:
        if fname in indexed_docs:
            results.append({"filename": fname, "status": "exists", "chunks": 0})
            continue
        pdf_path = str(KNOWLEDGE_BASE_DIR / fname)
        chunks = index_pdf(pdf_path, config, vs, drop_existing=False, force=False)
        total += chunks
        results.append({"filename": fname, "status": "indexed", "chunks": chunks})
    
    return {"total_chunks": total, "results": results}

@app.get("/api/knowledge/reindex")
def reindex_all():
    """清空并重新索引所有知识库文件"""
    if not vs:
        raise HTTPException(503, "向量存储未初始化")
    from main import index_pdf

    pdf_files = sorted([f for f in os.listdir(str(KNOWLEDGE_BASE_DIR)) if f.lower().endswith(".pdf")])
    results = []
    total = 0
    for i, fname in enumerate(pdf_files):
        pdf_path = str(KNOWLEDGE_BASE_DIR / fname)
        chunks = index_pdf(pdf_path, config, vs, drop_existing=(i == 0), force=True)
        total += chunks
        results.append({"filename": fname, "chunks": chunks})
    return {"total_chunks": total, "results": results}


# ═══════════════════════════════════════════════
#  问答 API
# ═══════════════════════════════════════════════

@app.post("/api/ask")
def ask(req: AskRequest):
    """单次问答（同步）"""
    if not rag:
        raise HTTPException(503, "RAG 引擎未初始化")

    session = _get_or_create_session(req.user_id or 0, req.session_id)
    session.set_mode(req.mode)

    try:
        result = rag.answer(req.query, session, language=req.language)
        # 保存到数据库
        if db and req.user_id:
            db.save_chat_message(
                user_id=req.user_id,
                session_id=session.session_id,
                role="user", content=req.query,
                mode=req.mode,
            )
            db.save_chat_message(
                user_id=req.user_id,
                session_id=session.session_id,
                role="assistant", content=result["answer"],
                mode=req.mode,
                retrieval_time_ms=result.get("retrieval_time_ms", 0),
                llm_time_ms=result.get("llm_time_ms", 0),
            )
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "retrieval_time_ms": result.get("retrieval_time_ms", 0),
            "llm_time_ms": result.get("llm_time_ms", 0),
            "total_time_ms": result.get("total_time_ms", 0),
            "mode": result.get("mode", req.mode),
            "session_id": session.session_id,
        }
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """流式问答（Server-Sent Events）"""
    if not rag:
        raise HTTPException(503, "RAG 引擎未初始化")

    session = _get_or_create_session(req.user_id or 0, req.session_id)
    session.set_mode(req.mode)

    async def event_stream():
        try:
            for event in _stream_answer(req, session):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"流式问答出错: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _stream_answer(req: AskRequest, session: ChatSession):
    """流式回答生成器（同步）"""
    retrieval_sources = []
    retrieval_ms = 0
    full_answer = ""
    llm_ms = 0

    for event in rag.answer_stream(req.query, session, language=req.language):
        if event["type"] == "retrieval":
            retrieval_sources = event.get("sources", [])
            retrieval_ms = event.get("time_ms", 0)
            yield {"type": "retrieval", "sources": retrieval_sources, "time_ms": retrieval_ms}
        elif event["type"] == "token":
            full_answer += event["content"]
            yield {"type": "token", "content": event["content"]}
        elif event["type"] == "done":
            llm_ms = event.get("llm_ms", 0)
            total_ms = event.get("total_ms", 0)
            yield {
                "type": "done",
                "answer": full_answer,
                "retrieval_time_ms": retrieval_ms,
                "llm_time_ms": llm_ms,
                "total_time_ms": total_ms,
                "sources": retrieval_sources,
                "session_id": session.session_id,
            }
        elif event["type"] == "error":
            yield {"type": "error", "message": event.get("message", "未知错误")}

    # 保存到数据库
    if db and req.user_id:
        try:
            db.save_chat_message(
                user_id=req.user_id,
                session_id=session.session_id,
                role="user", content=req.query,
                mode=req.mode,
            )
            db.save_chat_message(
                user_id=req.user_id,
                session_id=session.session_id,
                role="assistant", content=full_answer,
                mode=req.mode,
                retrieval_time_ms=retrieval_ms,
                llm_time_ms=llm_ms,
            )
        except Exception as e:
            logger.warning(f"保存聊天记录失败: {e}")


# ═══════════════════════════════════════════════
#  对话历史 API
# ═══════════════════════════════════════════════

@app.get("/api/history/sessions")
def get_sessions(user_id: int = Query(...)):
    """获取用户的会话列表"""
    if not db:
        return {"sessions": []}
    try:
        sessions = db.get_user_sessions(user_id)
        return {"sessions": sessions}
    except Exception as e:
        return {"sessions": [], "error": str(e)}

@app.get("/api/history/messages")
def get_messages(user_id: int = Query(...), session_id: str = Query(...)):
    """获取会话的聊天记录"""
    if not db:
        return {"messages": []}
    try:
        msgs = db.get_chat_history(user_id, session_id)
        return {"messages": msgs}
    except Exception as e:
        return {"messages": [], "error": str(e)}

@app.delete("/api/history/session")
def delete_session(user_id: int = Query(...), session_id: str = Query(...)):
    """删除指定会话"""
    if not db:
        raise HTTPException(503, "数据库未连接")
    ok = db.delete_session(user_id, session_id)
    if ok:
        return {"success": True, "message": "会话已删除"}
    return {"success": False, "message": "会话不存在或已删除"}


# ═══════════════════════════════════════════════
#  配置 API
# ═══════════════════════════════════════════════

@app.get("/api/config")
def get_config():
    """获取当前配置（隐藏敏感字段）"""
    cfg = config.to_dict()
    cfg.pop("llm_api_key", None)
    return cfg


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
