# -*- coding: utf-8 -*-
"""
API 路由定义

⚠️ 常改动的地方：
1. 知识库目录（KNOWLEDGE_BASE_DIR）默认路径
2. 短期记忆保留条数（在 memory.get_short_term 中 limit 参数）
3. 缓存的 TTL（默认 86400 秒，可调整）
4. 长期记忆的评分阈值（min_score=0.3）和重要性阈值（0.6）
5. 检索的 top_k 值（当前为 5）
6. 文件上传允许的扩展名列表
7. 法律关键词判断（依赖 prompt_manager.is_law_question）

⚠️ 注意事项：
1. 流式接口使用了全局变量 rag_engine, memory_mgr, retriever，必须在 main.py 中调用 init_rag_components 初始化
2. 长期记忆检索分为两个查询：问题相关 + 身份相关，合并后去重
3. 法律场景下会检查 has_knowledge（是否有知识库内容）并传递到提示词
4. 缓存键包含 role 和 user_id，确保不同用户/角色的答案不串扰
5. 文件上传支持增量插入向量库，但需要确保 vector_store 支持 add_documents 和 ids 参数
"""

import json
import re
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from .models import RegisterRequest, LoginRequest, ChatRequest
from storage.mysql_client import mysql_client
from storage.redis_client import redis_client
from core.prompt_manager import prompt_manager

router = APIRouter()

# ==================== 全局变量 ====================
# 这些变量在 init_rag_components 中被赋值，模块级别全局
rag_engine = None
memory_mgr = None
retriever = None

# 知识库目录（用户上传文件的存放位置）
# ⚠️ 常改动：可改为绝对路径或配置项
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)


# ==================== 初始化函数 ====================
def init_rag_components(vector_store, embedding_func):
    """
    初始化 RAG 组件，在 main.py 中调用
    ⚠️ 常改动：如果新增或替换组件，请在这里修改
    """
    global rag_engine, memory_mgr, retriever

    from core.rag_engine import create_rag_engine
    from core.memory import init_memory_manager
    from core.retriever import create_retriever
    from storage.mysql_client import mysql_client
    from storage.redis_client import redis_client

    # 创建 RAG 引擎（启用重排序器）
    rag_engine = create_rag_engine(vector_store, embedding_func, use_reranker=True)
    # 初始化记忆管理器（长期记忆使用相同 embedding 函数）
    memory_mgr = init_memory_manager(embedding_func)
    # 创建检索器（传入 vector_store、MySQL 连接、Redis 客户端）
    retriever = create_retriever(vector_store, mysql_client._conn, redis_client.client)

    print("[INFO] RAG 组件初始化完成")


# ==================== 辅助函数 ====================

def hash_password(password: str) -> str:
    """密码哈希：SHA256 十六进制字符串"""
    return hashlib.sha256(password.encode()).hexdigest()


def format_sources(sources: list) -> list:
    """
    格式化来源信息，用于前端展示
    ⚠️ 常改动：如果需要调整文件名展示格式（如去除日期、下划线），可修改正则表达式
    """
    result = []
    for s in sources:
        name = s.get("file", "知识库文档")
        # 去除如 _20260101 的日期后缀
        name = re.sub(r'_\d{8}', '', name)
        # 去除 .docx 或 .pdf 扩展名
        name = re.sub(r'\.docx$|\.pdf$', '', name)
        # 去除所有下划线
        name = name.replace('_', '')
        # 如果处理后名字过短或无意义，默认显示为“民法典”
        if not name or len(name) < 2:
            name = "民法典"
        result.append({"file": f"《{name}》", "preview": s.get("preview", "")})
    return result


def get_cache_key(user_id: int, role: str, question: str) -> str:
    """生成 Redis 缓存键，格式：qa_cache:{role}:{user_id}:{question_md5}"""
    question_hash = hashlib.md5(question.encode()).hexdigest()
    return f"qa_cache:{role}:{user_id}:{question_hash}"


def get_cached_answer(user_id: int, role: str, question: str) -> Optional[str]:
    """从 Redis 获取缓存的答案，若不存在返回 None"""
    cache_key = get_cache_key(user_id, role, question)
    try:
        cached = redis_client.cache_get(cache_key)
        if cached:
            print(f"[缓存命中] 问题: {question[:50]}...")
            return cached
    except Exception as e:
        print(f"[缓存错误] get: {e}")
    return None


def save_cached_answer(user_id: int, role: str, question: str, answer: str, ttl: int = 86400):
    """
    保存答案到 Redis 缓存（默认24小时过期）
    ⚠️ 常改动：可根据问题类型调整 TTL，如高频问题可延长
    """
    cache_key = get_cache_key(user_id, role, question)
    try:
        redis_client.cache_set(cache_key, answer, ttl)
        print(f"[缓存保存] 问题: {question[:50]}..., TTL={ttl}s")
    except Exception as e:
        print(f"[缓存错误] set: {e}")


# ==================== 健康检查 ====================

@router.get("/api/health")
async def health():
    """健康检查接口，用于监控和负载均衡"""
    return {"status": "ok"}


# ==================== 用户相关 ====================

@router.post("/api/register")
async def register(req: RegisterRequest):
    """用户注册：检查用户名是否存在，创建新用户"""
    existing = mysql_client.get_user_by_username(req.username)
    if existing:
        return {"code": 400, "message": "用户名已存在"}

    user_id = mysql_client.create_user(
        username=req.username,
        password_hash=hash_password(req.password),
        email=req.email
    )
    return {"code": 200, "message": "注册成功", "user_id": user_id}


@router.post("/api/login")
async def login(req: LoginRequest):
    """用户登录：验证用户名和密码哈希，更新最后登录时间"""
    user = mysql_client.fetch_one(
        "SELECT id, username, nickname, email, status FROM users WHERE username = %s AND password_hash = %s",
        (req.username, hash_password(req.password))
    )
    if not user or user.get('status') != 1:
        return {"code": 401, "message": "用户名或密码错误"}

    mysql_client.update_last_login(user['id'])
    return {"code": 200, "message": "登录成功", "user": user}


# ==================== 对话相关 ====================

@router.post("/api/chat")
async def chat(req: ChatRequest):
    """
    普通对话接口（非流式）
    ⚠️ 常改动：
    1. 检索 top_k 数量（当前 5）
    2. 短期记忆 limit（当前 5）
    3. 长期记忆检索的 top_k 和 min_score
    4. 长期记忆重要性阈值（当前 0.6）
    """
    global rag_engine, memory_mgr, retriever

    if rag_engine is None:
        return {"code": 500, "message": "RAG引擎未初始化", "answer": "系统初始化中，请稍后重试"}

    # 生成或使用 conversation_id（若前端未提供则自动生成）
    conv_id = req.conversation_id or f"conv_{req.user_id}_{req.role}"

    # 0. 检查缓存（相同问题的快速回答）
    cached_answer = get_cached_answer(req.user_id, req.role, req.message)
    if cached_answer:
        return {
            "code": 200,
            "answer": cached_answer,
            "sources": [],
            "conversation_id": conv_id,
            "from_cache": True
        }

    # 1. 判断是否为法律问题（依赖 prompt_manager 中的关键词匹配）
    is_law = prompt_manager.is_law_question(req.message)
    print(f"[问题分析] 角色={req.role}, 是否法律问题={is_law}")

    # 2. 检索长期记忆（同时用当前问题和身份查询）
    memories_by_question = memory_mgr.retrieve_long_term(
        user_id=str(req.user_id),
        role=req.role,
        query=req.message,
        top_k=3,
        min_score=0.3
    )

    # 专门构建身份查询（只查询用户信息类型）
    identity_query = "我的身份 我是什么 我是谁 我的职业 我的名字 外号"
    memories_by_identity = memory_mgr.retrieve_long_term(
        user_id=str(req.user_id),
        role=req.role,
        query=identity_query,
        top_k=3,
        memory_types=['preference', 'fact', 'identity', 'occupation', 'name', 'health'],
        min_score=0.3
    )

    # 打印记忆得分统计（调试用）
    if memories_by_question:
        scores = [m.get('score', 0) for m in memories_by_question]
        print(
            f"[记忆得分统计-问题] 最高={max(scores):.4f}, 最低={min(scores):.4f}, 平均={sum(scores) / len(scores):.4f}")
    if memories_by_identity:
        scores = [m.get('score', 0) for m in memories_by_identity]
        print(
            f"[记忆得分统计-身份] 最高={max(scores):.4f}, 最低={min(scores):.4f}, 平均={sum(scores) / len(scores):.4f}")

    # 合并去重 - 使用 memory_id 作为唯一标识（若无 id 则用内容+得分临时构造）
    all_memories = {}
    for mem in memories_by_question + memories_by_identity:
        content = mem.get('content', '')
        if content and len(content) > 5:
            mem_key = mem.get('memory_id', f"{content[:100]}_{mem.get('score', 0)}")
            if mem_key not in all_memories or mem.get('score', 0) > all_memories[mem_key].get('score', 0):
                all_memories[mem_key] = mem

    long_memories = list(all_memories.values())

    # 只保留重要性 >= 0.6 的记忆（身份、事实类），且内容长度不超过300字符
    long_memories = [m for m in long_memories if m.get('importance', 0) >= 0.6 and len(m.get('content', '')) <= 300]

    print(f"[长期记忆] 检索到 {len(long_memories)} 条相关记忆（已过滤重要性<0.6或内容过长）")
    for mem in long_memories:
        print(
            f"  内容: {mem.get('content', '')[:80]}... (得分={mem.get('score', 0):.4f}, 重要性={mem.get('importance', 0)})")

    # 3. 检索知识库（RAG）
    docs, sources, context = rag_engine.retrieve(req.message, top_k=5)
    print(f"[检索结果] 检索到 {len(docs)} 个文档, context长度: {len(context)}")

    # 4. 获取短期记忆（最近5轮对话）
    short_memories = memory_mgr.get_short_term(
        user_id=str(req.user_id),
        role=req.role,
        session_id=conv_id,
        limit=5
    )
    history_text = memory_mgr.format_short_term_text(short_memories)

    # 5. 构建提示词（根据是否法律问题选择不同构建函数）
    role_prompt = prompt_manager.get_role_prompt(req.role)

    if is_law or req.role == "lawyer":
        has_knowledge = len(docs) > 0 and len(context) > 50
        prompt = prompt_manager.build_law_prompt(
            role_prompt, context, long_memories, history_text, req.message, has_knowledge
        )
    else:
        prompt = prompt_manager.build_normal_prompt(
            role_prompt, context, long_memories, history_text, req.message
        )

    # 打印提示词前500字符（调试用）
    print(f"[提示词预览] {prompt[:500]}...")

    # 6. 调用 LLM 生成答案（非流式）
    answer = rag_engine.chat(prompt)

    # 7. 保存到缓存（相同问题快速回答）
    save_cached_answer(req.user_id, req.role, req.message, answer)

    # 8. 保存短期记忆
    memory_mgr.save_short_term(str(req.user_id), req.role, conv_id, req.message, answer)

    # 9. 提取并保存长期记忆（只保存用户信息，advice 已被过滤）
    important_info = memory_mgr.extract_important_info(req.message, answer)
    for content, mem_type, importance in important_info:
        memory_mgr.save_long_term(
            user_id=str(req.user_id),
            role=req.role,
            conversation_id=conv_id,
            content=content,
            memory_type=mem_type,
            importance=importance
        )

    # 10. 格式化来源（用于前端展示）
    sources_response = rag_engine.format_sources_response(docs) if docs else []

    return {
        "code": 200,
        "answer": answer,
        "sources": format_sources(sources_response),
        "conversation_id": conv_id,
        "from_cache": False
    }


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口（Server-Sent Events）
    ⚠️ 注意事项：
    1. 生成器函数内部使用了相同的检索和记忆逻辑
    2. 流式生成过程中如果发生异常，可能会导致连接中断
    3. 最终会保存缓存、短期记忆和长期记忆
    """
    global rag_engine, memory_mgr, retriever

    conv_id = req.conversation_id or f"conv_{req.user_id}_{req.role}"
    is_law = prompt_manager.is_law_question(req.message)
    print(f"[流式-问题分析] 角色={req.role}, 是否法律问题={is_law}")

    # 检查缓存：如果有缓存，直接以流式形式返回缓存答案
    cached_answer = get_cached_answer(req.user_id, req.role, req.message)
    if cached_answer:
        async def cached_generate():
            yield f"data: {json.dumps({'content': cached_answer, 'done': False})}\n\n"
            yield f"data: {json.dumps({'done': True, 'from_cache': True})}\n\n"

        return StreamingResponse(cached_generate(), media_type="text/event-stream")

    async def generate():
        full_answer = ""

        # 检索长期记忆（与普通接口相同）
        memories_by_question = memory_mgr.retrieve_long_term(
            user_id=str(req.user_id), role=req.role, query=req.message, top_k=3, min_score=0.3
        )
        identity_query = "我的身份 我是什么 我是谁 我的职业 我的名字"
        memories_by_identity = memory_mgr.retrieve_long_term(
            user_id=str(req.user_id), role=req.role, query=identity_query, top_k=3,
            memory_types=['preference', 'fact', 'identity', 'occupation', 'name', 'health'],
            min_score=0.3
        )

        # 打印记忆得分统计
        if memories_by_question:
            scores = [m.get('score', 0) for m in memories_by_question]
            print(f"[流式-记忆得分统计-问题] 最高={max(scores):.4f}, 平均={sum(scores) / len(scores):.4f}")
        if memories_by_identity:
            scores = [m.get('score', 0) for m in memories_by_identity]
            print(f"[流式-记忆得分统计-身份] 最高={max(scores):.4f}, 平均={sum(scores) / len(scores):.4f}")

        # 合并去重
        all_memories = {}
        for mem in memories_by_question + memories_by_identity:
            content = mem.get('content', '')
            if content and len(content) > 5:
                mem_key = mem.get('memory_id', f"{content[:100]}_{mem.get('score', 0)}")
                if mem_key not in all_memories or mem.get('score', 0) > all_memories[mem_key].get('score', 0):
                    all_memories[mem_key] = mem

        long_memories = list(all_memories.values())

        # 只保留重要性 >= 0.6 的记忆
        long_memories = [m for m in long_memories if m.get('importance', 0) >= 0.6 and len(m.get('content', '')) <= 300]

        # 检索知识库
        docs, sources, context = rag_engine.retrieve(req.message, top_k=5)

        # 获取短期记忆
        short_memories = memory_mgr.get_short_term(
            user_id=str(req.user_id), role=req.role, session_id=conv_id, limit=5
        )
        history_text = memory_mgr.format_short_term_text(short_memories)

        # 构建提示词
        role_prompt = prompt_manager.get_role_prompt(req.role)

        if is_law or req.role == "lawyer":
            has_knowledge = len(docs) > 0 and len(context) > 50
            prompt = prompt_manager.build_law_prompt(
                role_prompt, context, long_memories, history_text, req.message, has_knowledge
            )
        else:
            prompt = prompt_manager.build_normal_prompt(
                role_prompt, context, long_memories, history_text, req.message
            )

        # 流式生成
        async for chunk in rag_engine.chat_stream(prompt):
            full_answer += chunk
            yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"

        # 保存到缓存
        save_cached_answer(req.user_id, req.role, req.message, full_answer)

        # 保存短期记忆
        memory_mgr.save_short_term(str(req.user_id), req.role, conv_id, req.message, full_answer)

        # 提取并保存长期记忆（只保存用户信息）
        important_info = memory_mgr.extract_important_info(req.message, full_answer)
        for content, mem_type, importance in important_info:
            memory_mgr.save_long_term(
                user_id=str(req.user_id), role=req.role, conversation_id=conv_id,
                content=content, memory_type=mem_type, importance=importance
            )

        yield f"data: {json.dumps({'done': True, 'from_cache': False})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ==================== 对话历史 ====================

@router.get("/api/conversations")
async def get_conversations(user_id: int, role: str = None):
    """获取用户的对话列表，可按角色过滤"""
    conversations = mysql_client.get_conversations(user_id, role)
    return {"code": 200, "conversations": conversations}


@router.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str, user_id: int):
    """获取某个对话的完整消息列表（按时间正序）"""
    messages = mysql_client.get_conversation_messages(conversation_id, user_id)
    # 尝试将 sources 字段从 JSON 字符串解析为列表
    for msg in messages:
        if msg.get('sources') and isinstance(msg['sources'], str):
            try:
                msg['sources'] = json.loads(msg['sources'])
            except:
                msg['sources'] = []
    return {"code": 200, "messages": messages}


@router.delete("/api/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: int):
    """删除对话及其所有消息"""
    result = mysql_client.delete_conversation(conversation_id, user_id)
    return {"code": 200 if result else 404, "message": "删除成功" if result else "对话不存在"}


@router.get("/api/rag/stats")
async def get_rag_stats():
    """获取 RAG 统计信息（当前仅占位）"""
    return {"code": 200, "stats": {"total_queries": 0, "avg_time_ms": 0}}


# ==================== 知识库动态更新 ====================

@router.post("/api/knowledge/upload")
async def upload_knowledge_file(
        file: UploadFile = File(...),
        auto_process: bool = Form(True)
):
    """
    上传知识库文件（用户主动上传）

    支持格式: PDF, DOCX, TXT, MD, JSON, CSV
    ⚠️ 常改动：如需支持更多文件类型，修改 allowed_extensions 列表
    ⚠️ 注意事项：
    1. 文件保存在 knowledge_base 目录下
    2. 自动处理时会调用 DataProcessor 解析和分块，并增量插入 Milvus
    3. 需要确保 rag_engine.vector_store 支持 add_documents 方法且接受 ids 参数
    """
    global rag_engine

    # 1. 验证文件类型
    allowed_extensions = ['.pdf', '.docx', '.txt', '.md', '.json', '.csv']
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        return {
            "code": 400,
            "message": f"不支持的文件类型: {file_ext}，支持类型: {', '.join(allowed_extensions)}"
        }

    # 2. 保存文件到 knowledge_base 目录
    file_path = KNOWLEDGE_BASE_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return {"code": 500, "message": f"文件保存失败: {str(e)}"}

    # 3. 自动处理（向量化）
    if auto_process and rag_engine:
        try:
            from processor.data_processor import DataProcessor
            from langchain_core.documents import Document
            import uuid

            print(f"[知识库更新] 开始处理: {file.filename}")

            # 加载文档（使用 DataProcessor 的 loader）
            processor = DataProcessor()
            texts = processor.loader.load_document(str(file_path))

            if not texts:
                return {"code": 500, "message": "文档解析失败，无有效内容"}

            # 获取分块策略（根据文件类型自适应）
            strategy = processor.adaptive_splitter.get_splitter_for_file(str(file_path))
            splitter, _ = strategy

            # 创建 LangChain Document 对象列表
            docs = []
            for text in texts:
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file.filename,
                        "source_file": file.filename,
                        "upload_time": datetime.now().isoformat()
                    }
                )
                docs.append(doc)

            # 分块
            split_docs = splitter.split_documents(docs)
            print(f"[知识库更新] 分块完成: {len(split_docs)} 个块")

            # 增量插入 Milvus（使用 vector_store.add_documents，并传入 ids）
            if rag_engine.vector_store:
                ids = [str(uuid.uuid4()) for _ in split_docs]
                rag_engine.vector_store.add_documents(split_docs, ids=ids)
                print(f"[知识库更新] 成功添加 {len(split_docs)} 个向量")

                return {
                    "code": 200,
                    "message": f"文件上传成功，已处理 {len(split_docs)} 个向量块",
                    "data": {
                        "file_name": file.filename,
                        "chunk_count": len(split_docs),
                        "status": "processed"
                    }
                }
            else:
                return {
                    "code": 200,
                    "message": f"文件已保存，但向量库未就绪",
                    "data": {"file_name": file.filename, "status": "saved_only"}
                }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"code": 500, "message": f"文档处理失败: {str(e)}"}

    return {"code": 200, "message": f"文件已保存: {file.filename}"}


@router.get("/api/knowledge/files")
async def get_knowledge_files():
    """获取知识库文件列表（支持多种格式，返回文件元信息）"""
    files_info = []

    for ext in ['*.pdf', '*.docx', '*.txt', '*.md', '*.json', '*.csv']:
        for file_path in KNOWLEDGE_BASE_DIR.glob(ext):
            stat = file_path.stat()
            files_info.append({
                "name": file_path.name,
                "size": stat.st_size,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "type": file_path.suffix[1:].upper()
            })

    files_info.sort(key=lambda x: x["modified_time"], reverse=True)

    return {
        "code": 200,
        "data": {
            "total": len(files_info),
            "files": files_info
        }
    }


@router.delete("/api/knowledge/files/{file_name}")
async def delete_knowledge_file(file_name: str):
    """
    删除知识库文件，并尝试从 Milvus 中删除相关向量
    ⚠️ 注意事项：Milvus 删除基于 source 字段等于文件名，需确保元数据中 source 字段匹配
    """
    global rag_engine

    file_path = KNOWLEDGE_BASE_DIR / file_name

    if not file_path.exists():
        return {"code": 404, "message": f"文件不存在: {file_name}"}

    # 从 Milvus 中删除相关向量（假设 collection 名为 "rag_chatbot_v2"）
    if rag_engine and rag_engine.vector_store:
        try:
            from pymilvus import Collection
            collection = Collection("rag_chatbot_v2")
            collection.delete(f'source == "{file_name}"')
            print(f"[知识库更新] 已从 Milvus 删除: {file_name}")
        except Exception as e:
            print(f"[知识库更新] 从 Milvus 删除失败: {e}")

    # 删除物理文件
    file_path.unlink()

    return {"code": 200, "message": f"文件已删除: {file_name}"}


# ==================== 缓存管理接口 ====================

@router.delete("/api/cache/{user_id}/{role}")
async def clear_user_cache(user_id: int, role: str):
    """
    清除指定用户和角色的所有缓存
    ⚠️ 注意事项：使用 SCAN 命令避免阻塞 Redis，逐个删除匹配的键
    """
    pattern = f"qa_cache:{role}:{user_id}:*"
    try:
        cache_keys = []
        cursor = 0
        while True:
            cursor, keys = redis_client.client.scan(cursor, match=pattern, count=100)
            cache_keys.extend(keys)
            if cursor == 0:
                break
        if cache_keys:
            redis_client.client.delete(*cache_keys)
            return {"code": 200, "message": f"已清除 {len(cache_keys)} 条缓存"}
        return {"code": 200, "message": "无缓存需要清除"}
    except Exception as e:
        print(f"[缓存清理错误] {e}")
        return {"code": 500, "message": f"清理失败: {e}"}


@router.get("/api/cache/stats")
async def get_cache_stats():
    """获取缓存统计信息（总缓存条目数）"""
    try:
        pattern = "qa_cache:*"
        cache_keys = []
        cursor = 0
        while True:
            cursor, keys = redis_client.client.scan(cursor, match=pattern, count=100)
            cache_keys.extend(keys)
            if cursor == 0:
                break
        return {"code": 200, "stats": {"total_cached": len(cache_keys)}}
    except Exception as e:
        return {"code": 500, "stats": {"total_cached": 0, "error": str(e)}}