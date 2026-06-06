"""
Streamlit Web 应用 - RAG 对话系统
登录注册 + 问答界面 + 时间显示
"""
import os
import sys
import random
import time
from pathlib import Path

import streamlit as st

# 将项目根目录加入 path（自动检测，支持 Windows 和 WSL）
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))

from config import AppConfig

config = AppConfig.load(str(_project_root / "config.json"))
from logger import get_logger, LoggerManager
from database import DatabaseManager
from llm_provider import LLMFactory
from vector_store import VectorStore
from rag_engine import RAGEngine, ChatSession
from auth_page import show_auth_page
from sidebar import show_sidebar
from retrieval_strategy import RetrievalConfig

logger = get_logger("web")

# ──────────────────────────────────────────────
#  趣味消息池（思考中时随机展示）
# ──────────────────────────────────────────────
MESSAGES_ZH = [
    "⏳ 正在思考...知识就是力量，慢慢来比较快 💪",
    "⏳ 正在思考...罗马不是一天建成的，答案正在降临 🏛️",
    "⏳ 正在思考...好饭不怕晚，好答案值得等 🍚",
    "⏳ 正在思考...莫急，我正在浩瀚的文档中为你寻宝 🔍",
    "⏳ 正在思考...耐心是智慧的一部分，马上就好 🧘",
    "⏳ 正在思考...山重水复疑无路，柳暗花明又一村 🌸",
    "⏳ 正在思考...学问之道，求其放心而已矣 📖",
    "⏳ 正在思考...君子不器，博学而笃志 🎯",
    "⏳ 正在思考...衣带渐宽终不悔，为伊消得人憔悴 💭",
    "⏳ 正在思考...三个臭皮匠，顶个诸葛亮 🤝",
    "⏳ 正在思考...读万卷书不如行万里路，我正在行路 🚶",
    "⏳ 正在思考...不要急，好东西总是留给有耐心的人 🎁",
    "⏳ 正在思考...学而不思则罔，思而不学则殆 🤔",
    "⏳ 正在思考...博观而约取，厚积而薄发 📚",
    "⏳ 正在思考...千里之行始于足下，答案正在路上 🚀",
    "⏳ 正在思考...世界如此美妙，你却如此急躁，这样不好不好 😌",
    "⏳ 正在思考...滴水穿石，非一日之功 🪨",
    "⏳ 正在思考...聪明在于勤奋，天才在于积累 ✨",
    "⏳ 正在思考...前人栽树后人乘凉，我在文档里耕耘 🌳",
    "⏳ 正在思考...学问常看胜于我者，则德业日进 📈",
    "⏳ 正在思考...知之者不如好之者，好之者不如乐之者 😊",
    "⏳ 正在思考...不积跬步无以至千里，我正一步步找答案 👣",
    "⏳ 正在思考...人生在勤，勤则不匮 ⏰",
    "⏳ 正在思考...书山有路勤为径，学海无涯苦作舟 ⛵",
    "⏳ 正在思考...知之为知之，不知为不知，是知也 ⚖️",
    "⏳ 正在思考...温故而知新，可以为师矣 👨‍🏫",
    "⏳ 正在思考...业精于勤荒于嬉，行成于思毁于随 ✍️",
    "⏳ 正在思考...工欲善其事，必先利其器 — 正磨刀呢 🔪",
    "⏳ 正在思考...欲速则不达，见小利则大事不成 🐢",
    "⏳ 正在思考...君子求诸己，小人求诸人 — 求诸文档 📄",
]

MESSAGES_EN = [
    "⏳ Thinking... great minds think slowly, but surely 💪",
    "⏳ Thinking... patience is the companion of wisdom 🧘",
    "⏳ Thinking... Rome wasn't built in a day 🏛️",
    "⏳ Thinking... good things come to those who wait 🎁",
    "⏳ Thinking... knowledge is power, searching the archives 🔍",
    "⏳ Thinking... the best answers are worth the wait ⏰",
    "⏳ Thinking... patience grasshopper, the answer will come 🦗",
    "⏳ Thinking... a journey of a thousand miles begins with a single search 🚀",
    "⏳ Thinking... seek and ye shall find 📖",
    "⏳ Thinking... genius is 1% inspiration and 99% model inference ✨",
    "⏳ Thinking... keep calm and let the AI do the heavy lifting 😌",
    "⏳ Thinking... the mind is not a vessel to be filled but a fire to be kindled 🔥",
    "⏳ Thinking... the only true wisdom is in knowing you know nothing — Socrates 🤔",
    "⏳ Thinking... in the middle of difficulty lies opportunity — Einstein 🌟",
    "⏳ Thinking... stay hungry, stay foolish — Jobs 🍎",
    "⏳ Thinking... the best time to plant a tree was 20 years ago. The second best time is now 🌳",
    "⏳ Thinking... science is organized knowledge. Wisdom is organized life — Kant 🧠",
    "⏳ Thinking... what you get by achieving your goals is not as important as what you become 🏆",
    "⏳ Thinking... it does not matter how slowly you go, as long as you do not stop 🐢",
    "⏳ Thinking... perseverance is the hard work you do after you get tired of doing the hard work 🗿",
]

# ──────────────────────────────────────────────
#  页面全局配置
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG 文档问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
#  初始化全局资源（只初始化一次）
# ──────────────────────────────────────────────
@st.cache_resource
def init_resources():
    """初始化全局资源：数据库、LLM、向量存储、RAG引擎"""
    resources = {}

    # 1. 日志目录
    os.makedirs(config.log_dir, exist_ok=True)

    # 2. 数据库
    db = DatabaseManager(
        host=config.db_host, port=config.db_port,
        user=config.db_user, password=config.db_password,
        database=config.db_name,
    )
    if db.connect():
        resources["db"] = db
        logger.info("数据库初始化成功")
    else:
        logger.error("数据库初始化失败，部分功能不可用")
        resources["db"] = None

    # 3. LLM
    try:
        llm = LLMFactory.create(
            provider=config.llm_provider,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
        )
        resources["llm"] = llm
        logger.info(f"LLM 初始化成功: {llm.name}")
    except Exception as e:
        logger.error(f"LLM 初始化失败: {e}")
        resources["llm"] = None

    # 4. 向量存储
    try:
        vs = VectorStore(
            model_path=config.embedding_model_path,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            collection_name=config.milvus_collection,
        )
        if vs.load_model() and vs.connect_milvus():
            resources["vector_store"] = vs
            vec_count = vs.count()
            logger.info(f"向量存储初始化成功 (向量数: {vec_count})")
            resources["vector_count"] = vec_count
        else:
            resources["vector_store"] = None
            resources["vector_count"] = 0
            logger.warning("向量存储初始化失败，RAG 模式不可用")
    except Exception as e:
        logger.error(f"向量存储初始化失败: {e}")
        resources["vector_store"] = None
        resources["vector_count"] = 0

    # 5. RAG 引擎
    if resources.get("vector_store") and resources.get("llm"):
        resources["rag"] = RAGEngine(
            vector_store=resources["vector_store"],
            llm_provider=resources["llm"],
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
        )
    else:
        resources["rag"] = None

    logger.info("全局资源初始化完成")
    return resources


resources = init_resources()
db = resources.get("db")
llm = resources.get("llm")
vs = resources.get("vector_store")
rag = resources.get("rag")
vec_count = resources.get("vector_count", 0)

# ──────────────────────────────────────────────
#  Session State 初始化
# ──────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.session_id = None
    st.session_state.mode = "rag"
    st.session_state.chat_messages = []
    st.session_state.current_answer = None
    st.session_state.processing = False
    st.session_state.pending_query = None

# ──────────────────────────────────────────────
#  问答界面
# ──────────────────────────────────────────────
def show_chat_page():
    """问答主界面"""
    # ── 侧边栏 ──
    selected_lang, selected_mode = show_sidebar(vs, vec_count, db, config, llm)

    # 顶部导航栏
    col_header, col_user, col_mode, col_logout = st.columns([4, 2, 2, 1])

    with col_header:
        st.markdown(
            '<div style="font-size:22px;font-weight:700;color:#1a1a2e;">📚 RAG 文档问答</div>',
            unsafe_allow_html=True
        )

    with col_user:
        st.markdown(
            f'<div style="font-size:14px;color:#666;text-align:right;padding:6px 0;">'
            f'👤 {st.session_state.username} (ID: {st.session_state.user_id})</div>',
            unsafe_allow_html=True
        )

    with col_mode:
        current_mode = st.session_state.mode
        mode_options = {
            "rag": "📚 结合知识库作答",
            "direct": "🤖 纯 LLM 作答",
        }
        selected_label = st.selectbox(
            "回答模式",
            options=list(mode_options.keys()),
            format_func=lambda x: mode_options[x],
            label_visibility="collapsed",
            key="mode_selector",
        )
        if selected_label != st.session_state.mode:
            st.session_state.mode = selected_label
            st.rerun()

    with col_logout:
        if st.button("退出", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()

    st.divider()

    # 状态栏
    col_status = st.columns(6)
    with col_status[0]:
        st.markdown(
            f'<span style="font-size:12px;color:#666;">模式: '
            f'{"📚 知识库" if st.session_state.mode == "rag" else "🤖 LLM"}</span>',
            unsafe_allow_html=True
        )
    with col_status[1]:
        st.markdown(
            f'<span style="font-size:12px;color:#666;">知识库: {vec_count} 片段</span>',
            unsafe_allow_html=True
        )
    with col_status[2]:
        model_name = llm.name if llm else "未配置"
        st.markdown(
            f'<span style="font-size:12px;color:#666;">模型: {model_name}</span>',
            unsafe_allow_html=True
        )
    with col_status[5]:
        st.markdown(
            f'<span style="font-size:12px;color:#999;text-align:right;">'
            f'{time.strftime("%Y-%m-%d %H:%M:%S")}</span>',
            unsafe_allow_html=True
        )

    # 聊天记录展示区
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#999;">
                <div style="font-size:48px;margin-bottom:16px;">💬</div>
                <div style="font-size:18px;font-weight:600;color:#666;margin-bottom:8px;">开始提问</div>
                <div style="font-size:14px;">
                    在下方输入框提问，系统将根据当前模式回答<br>
                    📚 知识库模式：从招股说明书中检索相关内容<br>
                    🤖 LLM 模式：仅使用语言模型自身知识
                </div>
                <div style="margin-top:16px;font-size:12px;color:#bbb;">
                    向量模型: BGE-M3 | 向量数据库: Milvus
                </div>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="text-align:right;margin:8px 0;">'
                    f'<span style="background:#e3f2fd;padding:10px 16px;border-radius:18px 18px 4px 18px;'
                    f'display:inline-block;max-width:75%;font-size:14px;line-height:1.5;">'
                    f'{msg["content"]}</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="text-align:left;margin:8px 0;">'
                    f'<span style="background:#f5f5f5;padding:10px 16px;border-radius:18px 18px 18px 4px;'
                    f'display:inline-block;max-width:75%;font-size:14px;line-height:1.5;">'
                    f'{msg["content"]}</span></div>',
                    unsafe_allow_html=True
                )

            if "timing" in msg:
                t = msg["timing"]
                timing_parts = []
                if t.get("retrieval", 0) > 0:
                    timing_parts.append(f"检索: {t['retrieval']}ms")
                timing_parts.append(f"LLM: {t['llm']}ms")
                timing_parts.append(f"总耗时: {t['total']}ms")
                timing_text = " | ".join(timing_parts)
                st.markdown(
                    f'<div style="text-align:{"left" if msg["role"]=="assistant" else "right"};'
                    f'font-size:11px;color:#aaa;margin:2px 0 8px;">⏱️ {timing_text}</div>',
                    unsafe_allow_html=True
                )

    # 正在作答时的提示
    if st.session_state.get("processing"):
        st.markdown(
            '<div style="text-align:left;margin:4px 0;">'
            '<span style="background:#fffbe6;padding:6px 14px;border-radius:12px;'
            'display:inline-block;font-size:13px;color:#b8860b;">'
            '⏳ 正在作答，请稍候...</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # 输入区
    col_input = st.columns([5, 1])

    with col_input[0]:
        query = st.chat_input("输入你的问题...", key="question_input", disabled=st.session_state.get("processing", False))

    with col_input[1]:
        clear_btn = st.button("🔄 清空对话", use_container_width=True)
        if clear_btn:
            st.session_state.chat_messages = []
            st.rerun()

    if query and query.strip():
        # 步骤 1: 立即添加消息 + 占位 → rerun 显示
        st.session_state.chat_messages.append({"role": "user", "content": query.strip()})
        current_lang = st.session_state.get("language", "中文")
        msg_pool = MESSAGES_ZH if current_lang == "中文" else MESSAGES_EN
        placeholder_msg = random.choice(msg_pool) + "  ⏳"
        st.session_state.chat_messages.append({"role": "assistant", "content": placeholder_msg, "placeholder": True})
        st.session_state.pending_query = query.strip()
        st.rerun()

    # 步骤 2: 检测待处理的 query，异步执行 LLM 调用
    if st.session_state.get("pending_query") and not st.session_state.get("processing"):
        query = st.session_state.pending_query
        st.session_state.processing = True
        st.session_state.pending_query = None

        try:
            if st.session_state.session_id is None:
                st.session_state.session_id = f"ses_{int(time.time())}"

            session = ChatSession(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
            )
            session.mode = st.session_state.mode

            for msg in st.session_state.chat_messages:
                if msg["role"] == "user" and msg["content"] == query:
                    session.add_message("user", msg["content"])

            current_mode = st.session_state.mode
            current_lang = st.session_state.get("language", "中文")

            # 从 session_state 读取检索配置
            retrieval_cfg = RetrievalConfig(
                mode=st.session_state.get("retrieval_mode", "hybrid"),
                top_k=st.session_state.get("top_k", config.top_k),
                similarity_threshold=st.session_state.get("similarity_threshold", config.similarity_threshold),
                alpha=st.session_state.get("retrieval_alpha", 0.4),
                rerank_method=st.session_state.get("rerank_method", "adaptive"),
                query_rewrite=st.session_state.get("query_rewrite", True),
            )

            if current_mode == "rag" and rag is not None:
                retrieval_sources = []
                retrieval_ms = 0
                full_answer = ""
                llm_ms = 0
                total_ms = 0

                for event in rag.answer_stream(query, session, language=current_lang, retrieval_config=retrieval_cfg):
                    if event["type"] == "retrieval":
                        retrieval_sources = event["sources"]
                        retrieval_ms = event["time_ms"]
                    elif event["type"] == "token":
                        full_answer += event["content"]
                    elif event["type"] == "done":
                        llm_ms = event["llm_ms"]
                        total_ms = event["total_ms"]
                    elif event["type"] == "error":
                        full_answer = f"【系统错误】{event['message']}"
                        total_ms = event["total_ms"]

                result = {
                    "answer": full_answer, "sources": retrieval_sources,
                    "retrieval_time_ms": retrieval_ms, "llm_time_ms": llm_ms,
                    "total_time_ms": total_ms, "mode": "rag",
                }
            elif current_mode == "rag" and rag is None:
                result = {
                    "answer": "向量存储未初始化，无法使用知识库模式。请先索引 PDF 文档，或切换为纯 LLM 模式。",
                    "sources": [], "retrieval_time_ms": 0,
                    "llm_time_ms": 0, "total_time_ms": 0, "mode": "rag",
                }
            elif current_mode == "direct":
                t0 = time.time()
                full_answer = ""
                if llm:
                    for token in llm.ask_stream(query):
                        full_answer += token
                else:
                    full_answer = "LLM 未初始化，请配置 API Key。"
                llm_ms = int((time.time() - t0) * 1000)
                result = {
                    "answer": full_answer, "sources": [],
                    "retrieval_time_ms": 0, "llm_time_ms": llm_ms,
                    "total_time_ms": llm_ms, "mode": "direct",
                }
            elif current_mode == "both":
                # 左右分栏：RAG + 纯 LLM
                rag_answer = ""
                rag_sources = []
                rag_retrieval_ms = 0
                rag_llm_ms = 0
                if rag is not None:
                    for event in rag.answer_stream(query, session, language=current_lang, retrieval_config=retrieval_cfg):
                        if event["type"] == "retrieval":
                            rag_sources = event["sources"]
                            rag_retrieval_ms = event["time_ms"]
                        elif event["type"] == "token":
                            rag_answer += event["content"]
                        elif event["type"] == "done":
                            rag_llm_ms = event.get("llm_time_ms", event.get("llm_ms", 0))
                    rag_total_ms = rag_retrieval_ms + rag_llm_ms
                else:
                    rag_answer = "知识库未初始化"

                t0 = time.time()
                direct_answer = ""
                if llm:
                    for token in llm.ask_stream(query):
                        direct_answer += token
                else:
                    direct_answer = "LLM 未初始化"
                direct_llm_ms = int((time.time() - t0) * 1000)

                result = {
                    "answer": f"【知识库增强】\n{rag_answer}\n\n【纯 LLM】\n{direct_answer}",
                    "sources": rag_sources,
                    "retrieval_time_ms": rag_retrieval_ms,
                    "llm_time_ms": f"RAG: {rag_llm_ms}ms / Direct: {direct_llm_ms}ms",
                    "total_time_ms": max(rag_retrieval_ms + rag_llm_ms, direct_llm_ms),
                    "mode": "both",
                }

            else:
                # 兜底：走 RAG 模式
                full_answer = ""
                retrieval_sources = []
                retrieval_ms = 0
                llm_ms = 0
                if rag:
                    for event in rag.answer_stream(query, session, language=current_lang, retrieval_config=retrieval_cfg):
                        if event["type"] == "retrieval":
                            retrieval_sources = event["sources"]
                            retrieval_ms = event["time_ms"]
                        elif event["type"] == "token":
                            full_answer += event["content"]
                        elif event["type"] == "done":
                            llm_ms = event["llm_ms"]
                result = {
                    "answer": full_answer, "sources": retrieval_sources,
                    "retrieval_time_ms": retrieval_ms, "llm_time_ms": llm_ms,
                    "total_time_ms": retrieval_ms + llm_ms, "mode": "rag",
                }

            # 替换占位消息为真实回答
            for i, msg in enumerate(st.session_state.chat_messages):
                if msg.get("placeholder"):
                    st.session_state.chat_messages[i] = {
                        "role": "assistant",
                        "content": result["answer"],
                        "timing": {
                            "retrieval": result["retrieval_time_ms"],
                            "llm": result["llm_time_ms"],
                            "total": result["total_time_ms"],
                        }
                    }
                    break

            if db:
                try:
                    db.save_chat_message(
                        user_id=st.session_state.user_id,
                        session_id=st.session_state.session_id,
                        role="user", content=query,
                        mode=st.session_state.mode,
                    )
                    db.save_chat_message(
                        user_id=st.session_state.user_id,
                        session_id=st.session_state.session_id,
                        role="assistant", content=result["answer"],
                        mode=st.session_state.mode,
                        retrieval_time_ms=result["retrieval_time_ms"],
                        llm_time_ms=result["llm_time_ms"],
                    )
                except Exception as e:
                    logger.warning(f"保存聊天记录到数据库失败: {e}")

        except Exception as e:
            logger.error(f"处理提问失败: {e}")
            for i, msg in enumerate(st.session_state.chat_messages):
                if msg.get("placeholder"):
                    st.session_state.chat_messages[i] = {
                        "role": "assistant",
                        "content": f"【系统错误】处理请求时发生异常: {str(e)}",
                        "timing": {"retrieval": 0, "llm": 0, "total": 0},
                    }
                    break

        st.session_state.processing = False
        st.rerun()


# ──────────────────────────────────────────────
#  页面路由
# ──────────────────────────────────────────────
if st.session_state.logged_in:
    show_chat_page()
else:
    show_auth_page(db)
