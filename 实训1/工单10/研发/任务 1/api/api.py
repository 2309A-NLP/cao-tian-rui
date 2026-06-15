"""
RAG 问答系统 - FastAPI 后端
前后端分离架构，提供 REST API

启动：uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import json
import re
import hashlib
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response
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
from backend.rag_engine import RAGEngine, ChatSession
from backend.session_manager import SessionManager, CoreferenceResolver

logger = get_logger("api")

# ── 加载配置（支持 RAG_CONFIG_PATH 环境变量覆盖） ──
_env_config = os.environ.get("RAG_CONFIG_PATH", "")
if _env_config and Path(_env_config).exists():
    config_path = Path(_env_config)
else:
    config_path = _project_root / "config.json"
config = AppConfig.load(str(config_path))

# 如果配置中的 knowledge_base_dir 是相对路径，转为绝对
if not os.path.isabs(config.knowledge_base_dir):
    config.knowledge_base_dir = str(config_path.parent / config.knowledge_base_dir)
if not os.path.isabs(config.output_dir):
    config.output_dir = str(config_path.parent / config.output_dir)
if not os.path.isabs(config.log_dir):
    config.log_dir = str(config_path.parent / config.log_dir)

# ── 全局资源 ──
db: Optional[DatabaseManager] = None
vs: Optional[VectorStore] = None
llm = None
rag: Optional[RAGEngine] = None
vec_count = 0
session_mgr: Optional[SessionManager] = None
KNOWLEDGE_BASE_DIR = _project_root / "knowledge_base"


def _init_resources():
    """初始化全局资源（单次）"""
    global db, vs, llm, rag, vec_count, session_mgr, KNOWLEDGE_BASE_DIR
    os.makedirs(config.log_dir, exist_ok=True)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Redis 会话管理器
    try:
        session_mgr = SessionManager(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            ttl_hours=config.session_ttl_hours,
            max_history=config.max_history_rounds,
        )
        if session_mgr.ping():
            logger.info("Redis 会话管理器已连接")
        else:
            logger.warning("Redis 连接失败，使用内存模式备用")
            session_mgr = None
    except Exception as e:
        logger.warning(f"Redis 初始化失败: {e}")
        session_mgr = None

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
            model_type=config.embedding_model_type,
        )
        if vs.load_model() and vs.connect_milvus():
            vec_count = vs.count()
            logger.info(f"向量存储已加载 (向量数: {vec_count})")
            # 尝试构建 BM25 索引（从本地 chunks 文件恢复）
            try:
                import jieba, rank_bm25  # noqa: F401
                chunk_dir = _project_root / "output" / "chunks"
                all_chunks_file = chunk_dir / "all_chunks.json"
                if all_chunks_file.exists():
                    import json
                    with open(all_chunks_file, encoding="utf-8") as f:
                        all_chunks = json.load(f)
                    if all_chunks:
                        logger.info(f"加载 {len(all_chunks)} 个 chunk 构建 BM25...")
                        vs.chunks_metadata = all_chunks
                        vs.build_bm25_index(all_chunks)
            except ImportError:
                logger.info("BM25 依赖未安装，跳过 BM25 索引")
            except Exception as e:
                logger.warning(f"BM25 索引构建失败: {e}")
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
    # ── 检索策略参数 ──
    retrieval_mode: Optional[str] = None    # vector / fulltext / hybrid
    retrieval_alpha: Optional[float] = None # BM25 权重 (0-1)
    rerank_method: Optional[str] = None    # none / reranker / keyword / adaptive
    top_k: Optional[int] = None            # 返回结果条数
    similarity_threshold: Optional[float] = None  # 相似度阈值
    query_rewrite: Optional[bool] = None   # 是否启用查询改写

class IndexRequest(BaseModel):
    filenames: list[str] = []   # 知识库里已有的文件名列表


# ═══════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _get_or_create_session(user_id: int, session_id: Optional[str] = None) -> ChatSession:
    """获取或创建会话（通过 Redis）"""
    if session_id and session_mgr and session_mgr.session_exists(session_id):
        # 已有会话，从 Redis 重建 ChatSession 对象
        mode = session_mgr.get_mode(session_id)
        history = session_mgr.get_history(session_id)
        s = ChatSession(user_id=user_id, session_id=session_id)
        s.mode = mode
        for msg in history:
            s.history.append({"role": msg["role"], "content": msg["content"]})
        return s
    
    # 新建会话
    s = ChatSession(user_id=user_id, session_id=session_id or f"ses_{int(time.time())}")
    if session_mgr:
        session_mgr.create_session(s.session_id, user_id=user_id)
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
        return FileResponse(str(idx), headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        })
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
        "embedding_model": config.embedding_model_type,
        "database": db is not None,
        "redis": session_mgr is not None,
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
        # 从 Milvus 获取文档列表（可能因文件名标记缺失而返回空）
        docs = vs.list_documents()
        cnt = vs.count()
        # 如果 list_documents 返回空但有向量，用 knowledge_base 目录的文件名
        if not docs and cnt > 0:
            import os
            kb_dir = str(KNOWLEDGE_BASE_DIR)
            if os.path.isdir(kb_dir):
                docs = sorted([f for f in os.listdir(kb_dir) if f.lower().endswith(".pdf")])
        return {"documents": docs, "vector_count": cnt, "connected": True, "embedding_model": config.embedding_model_type}
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

def _extract_question(query: str) -> str:
    """从 JSON 格式的 query 中提取真正的 question 字段"""
    match = re.search(r'"question"\s*:\s*"([^"]+)"', query)
    if match:
        extracted = match.group(1)
        logger.debug(f"API层 JSON query 清洗: {query[:60]}... -> {extracted}")
        return extracted
    return query

@app.post("/api/ask")
def ask(req: AskRequest):
    """单次问答（同步）"""
    if not rag:
        raise HTTPException(503, "RAG 引擎未初始化")
    # 清洗 JSON query
    req.query = _extract_question(req.query)

    session = _get_or_create_session(req.user_id or 0, req.session_id)
    session.set_mode(req.mode)

    # 指代消解：将代词替换为具体实体
    resolved_query = req.query
    if session_mgr and llm:
        context = session_mgr.get_recent_context(session.session_id, max_rounds=3)
        resolved_query = CoreferenceResolver.resolve(req.query, context, llm)
    
    final_query = resolved_query  # 消解后的问题用于检索

    try:
        # 构建检索配置（始终构建，使用请求参数或默认值）
        from retrieval_strategy import RetrievalConfig
        rc = RetrievalConfig(
            mode=req.retrieval_mode or config.retrieval_mode,
            top_k=req.top_k or config.top_k,
            similarity_threshold=req.similarity_threshold if req.similarity_threshold is not None else config.similarity_threshold,
            alpha=req.retrieval_alpha if req.retrieval_alpha is not None else config.retrieval_alpha,
            rerank_method=req.rerank_method or config.rerank_method,
            query_rewrite=req.query_rewrite if req.query_rewrite is not None else config.query_rewrite_enabled,
        )

        # ── 日志：本次请求配置 ──
        logger.info(f"━━━ 问答请求 ━━━ query=\"{req.query[:60]}\" mode={req.mode}")
        logger.info(f"  检索: mode={rc.mode} top_k={rc.top_k} threshold={rc.similarity_threshold} alpha={rc.alpha}")
        logger.info(f"  重排={rc.rerank_method} rewrite={rc.query_rewrite}")
        if resolved_query != req.query:
            logger.info(f"  指代消解: \"{req.query}\" → \"{resolved_query}\"")

        result = rag.answer(final_query, session, language=req.language, mode=req.mode, retrieval_config=rc)
        
        # ── 日志：检索结果 ──
        sources = result.get("sources", [])
        logger.info(f"  结果: {len(sources)} 条来源, 模式={result.get('mode','?')}")
        logger.info(f"  耗时: 检索={result.get('retrieval_time_ms',0):.0f}ms LLM={result.get('llm_time_ms',0):.0f}ms 总计={result.get('total_time_ms',0):.0f}ms")
        for i, src in enumerate(sources[:3]):
            logger.info(f"    [{i+1}] score={src.get('score','?'):.3f} page={src.get('page','?')} snippet={src.get('text','')[:80]}...")

        # 保存到 Redis
        if session_mgr:
            session_mgr.add_message(session.session_id, "user", req.query,
                                    mode=req.mode, resolved=resolved_query if resolved_query != req.query else None)
            session_mgr.add_message(session.session_id, "assistant", result["answer"],
                                    mode=req.mode)
        
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
    # 清洗 JSON query
    req.query = _extract_question(req.query)

    session = _get_or_create_session(req.user_id or 0, req.session_id)
    session.set_mode(req.mode)

    # 指代消解
    resolved_query = req.query
    if session_mgr and llm:
        context = session_mgr.get_recent_context(session.session_id, max_rounds=3)
        resolved_query = CoreferenceResolver.resolve(req.query, context, llm)
    final_query = resolved_query

    # 构建检索配置（始终构建）
    from retrieval_strategy import RetrievalConfig
    rc = RetrievalConfig(
        mode=req.retrieval_mode or config.retrieval_mode,
        top_k=req.top_k or config.top_k,
        similarity_threshold=req.similarity_threshold if req.similarity_threshold is not None else config.similarity_threshold,
        alpha=req.retrieval_alpha if req.retrieval_alpha is not None else config.retrieval_alpha,
        rerank_method=req.rerank_method or config.rerank_method,
        query_rewrite=req.query_rewrite if req.query_rewrite is not None else config.query_rewrite_enabled,
    )

    # ── 日志：流式请求配置 ──
    logger.info(f"━━━ 流式问答 ━━━ query=\"{req.query[:60]}\" mode={req.mode}")
    logger.info(f"  检索: mode={rc.mode} top_k={rc.top_k} threshold={rc.similarity_threshold} alpha={rc.alpha}")
    logger.info(f"  重排={rc.rerank_method} rewrite={rc.query_rewrite}")
    if resolved_query != req.query:
        logger.info(f"  指代消解: \"{req.query}\" → \"{resolved_query}\"")

    async def event_stream():
        try:
            retrieval_sources = []
            retrieval_ms = 0
            full_answer = ""
            for event in rag.answer_stream(final_query, session, language=req.language, mode_override=req.mode, retrieval_config=rc):
                if event["type"] == "retrieval":
                    retrieval_sources = event.get("sources", [])
                    retrieval_ms = event.get("time_ms", 0)
                    logger.info(f"  检索完成: {len(retrieval_sources)} 条来源, {retrieval_ms:.0f}ms")
                    for i, src in enumerate(retrieval_sources[:3]):
                        logger.info(f"    [{i+1}] score={src.get('score','?'):.3f} page={src.get('page','?')} snippet={src.get('content','')[:80]}...")
                elif event["type"] == "token":
                    full_answer += event["content"]
                elif event["type"] == "done":
                    logger.info(f"  完成: 回答长度={len(full_answer)}字")
                    # 保存到 Redis
                    if session_mgr:
                        session_mgr.add_message(session.session_id, "user", req.query,
                                                mode=req.mode,
                                                resolved=resolved_query if resolved_query != req.query else None)
                        session_mgr.add_message(session.session_id, "assistant", full_answer,
                                                mode=req.mode)
                    # 保存到数据库（历史记录依赖此表）
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
                                llm_time_ms=event.get("llm_time_ms", 0),
                            )
                            logger.info(f"  已保存到数据库: user={req.user_id} session={session.session_id}")
                        except Exception as db_err:
                            logger.error(f"  数据库保存失败（不影响回答）: {db_err}")
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


def _stream_answer(req: AskRequest, session: ChatSession, retrieval_config=None):
    """流式回答生成器（同步）"""
    retrieval_sources = []
    retrieval_ms = 0
    full_answer = ""
    llm_ms = 0

    for event in rag.answer_stream(req.query, session, language=req.language, mode_override=req.mode, retrieval_config=retrieval_config):
        if event["type"] == "retrieval":
            retrieval_sources = event.get("sources", [])
            retrieval_ms = event.get("time_ms", 0)
            yield {"type": "retrieval", "sources": retrieval_sources, "time_ms": retrieval_ms}
        elif event["type"] == "token":
            full_answer += event["content"]
            yield {"type": "token", "content": event["content"]}
        elif event["type"] == "done":
            llm_ms = event.get("llm_time_ms", 0)
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
        rows = db.get_user_sessions(user_id)
        sessions = []
        for r in rows:
            sessions.append({
                "session_id": r.get("session_id", ""),
                "first_msg": r.get("first_msg", ""),
                "msg_count": r.get("msg_count", 0),
                "started_at": str(r.get("started_at", "")),
                "last_msg": str(r.get("last_msg", "")),
            })
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        return {"sessions": [], "error": str(e)}

@app.get("/api/history/messages")
def get_messages(user_id: int = Query(...), session_id: str = Query(...)):
    """获取会话的聊天记录（优先从 Redis，Redis 没有再从数据库）"""
    # 先试 Redis
    msgs = []
    if session_mgr and session_mgr.session_exists(session_id):
        history = session_mgr.get_history(session_id)
        msgs = []
        for m in history:
            msgs.append({
                "role": m["role"],
                "content": m["content"],
                "retrieval_time_ms": 0,
                "llm_time_ms": 0,
            })
        return {"messages": msgs, "source": "redis"}
    
    # Redis 没有，从数据库读
    if db:
        try:
            rows = db.get_chat_history(user_id, session_id)
            msgs = []
            for r in rows:
                msgs.append({
                    "role": r.get("role", "assistant"),
                    "content": r.get("content", ""),
                    "mode": r.get("mode", ""),
                    "retrieval_time_ms": r.get("retrieval_time_ms", 0),
                    "llm_time_ms": r.get("llm_time_ms", 0),
                    "created_at": str(r.get("created_at", "")),
                })
            return {"messages": msgs, "source": "db"}
        except Exception as e:
            logger.error(f"读取历史消息失败: {e}")
            return {"messages": [], "error": str(e)}
    return {"messages": []}

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
#  反馈 API
# ═══════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    user_id: int
    session_id: str
    query: str
    answer: str
    rating: int           # 1=赞, -1=踩
    comment: str = ""

@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """提交用户反馈"""
    if not db:
        raise HTTPException(503, "数据库未连接")
    if req.rating not in (1, -1):
        raise HTTPException(400, "rating 必须是 1 或 -1")
    ok = db.save_feedback(req.user_id, req.session_id, req.query, req.answer, req.rating, req.comment)
    if not ok:
        raise HTTPException(500, "保存反馈失败")
    return {"status": "ok"}

@app.get("/api/feedback/stats")
def feedback_stats():
    """获取反馈统计"""
    if not db:
        return {"total": 0, "up": 0, "down": 0, "satisfaction_rate": 0}
    return db.get_feedback_stats()


# ═══════════════════════════════════════════════
#  前端日志上报 API
# ═══════════════════════════════════════════════

class FrontendLogRequest(BaseModel):
    level: str = "info"         # info / warn / error
    message: str
    data: Optional[dict] = None
    url: str = ""
    user_agent: str = ""

@app.post("/api/log/frontend")
def frontend_log(req: FrontendLogRequest):
    """接收前端上报的日志（缓存问题、JS错误、API异常等）"""
    level_map = {"info": logger.info, "warn": logger.warning, "error": logger.error}
    log_fn = level_map.get(req.level, logger.info)
    extra = f" | data={json.dumps(req.data, ensure_ascii=False)}" if req.data else ""
    log_fn(
        f"[前端] {req.message}"
        f"{extra}"
        f" | url={req.url[:200]}" if req.url else ""
    )
    return {"status": "ok"}


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
    uvicorn.run("api:app", host="0.0.0.0", port=8010, reload=True)
