"""侧边栏组件：模式选择、知识库管理、PDF上传、历史会话"""

import streamlit as st
from pathlib import Path


def show_sidebar(vs, vec_count, db, config, llm):
    """渲染侧边栏，返回 (selected_lang, selected_mode)"""
    _project_root = Path(__file__).resolve().parent

    with st.sidebar:
        st.markdown("### 设置")

        # ── 语言选择 ──
        lang_options = {"中文": "中文", "English": "English"}
        current_lang = st.session_state.get("language", "中文")
        selected_lang = st.selectbox(
            "语言 / Language",
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            index=0 if current_lang == "中文" else 1,
            key="lang_selector",
        )
        if selected_lang != current_lang:
            st.session_state.language = selected_lang
            st.session_state.pending_lang_change = True

        st.divider()

        # ── 模式选择 ──
        mode_label = "回答模式" if selected_lang == "中文" else "Answer Mode"
        st.markdown(f"**{mode_label}**")
        rag_label = "📚 结合知识库作答" if selected_lang == "中文" else "📚 RAG Mode"
        direct_label = "🤖 纯 LLM 作答" if selected_lang == "中文" else "🤖 Direct LLM"
        both_label = "🔀 两者都选" if selected_lang == "中文" else "🔀 Both"

        current_mode = st.session_state.get("mode", "rag")
        selected_mode = st.radio(
            mode_label,
            options=["rag", "direct", "both"],
            format_func=lambda x: {"rag": rag_label, "direct": direct_label, "both": both_label}[x],
            index=0 if current_mode == "rag" else (1 if current_mode == "direct" else 2),
            key="mode_radio",
            label_visibility="collapsed",
        )
        if selected_mode != current_mode:
            st.session_state.mode = selected_mode
            st.rerun()

        # ── 检索策略配置（仅 RAG 模式下有效）──
        retrieval_label = "🔍 检索策略" if selected_lang == "中文" else "🔍 Retrieval Strategy"
        st.markdown(f"**{retrieval_label}**")
        
        # 检索模式
        rm_label = "检索方式" if selected_lang == "中文" else "Retrieval Mode"
        rm_opts = {
            "hybrid": "混合检索 (向量+全文)" if selected_lang == "中文" else "Hybrid (Vector+Fulltext)",
            "vector": "仅向量检索" if selected_lang == "中文" else "Vector Only",
            "fulltext": "仅全文检索" if selected_lang == "中文" else "Fulltext Only",
        }
        current_rm = st.session_state.get("retrieval_mode", "hybrid")
        selected_rm = st.selectbox(
            rm_label,
            options=list(rm_opts.keys()),
            format_func=lambda x: rm_opts[x],
            index=list(rm_opts.keys()).index(current_rm) if current_rm in rm_opts else 0,
            key="retrieval_mode_select",
            label_visibility="collapsed",
        )
        st.session_state.retrieval_mode = selected_rm
        
        # 混合检索的 alpha 权重
        if selected_rm == "hybrid":
            current_alpha = st.session_state.get("retrieval_alpha", 0.4)
            alpha_label = "全文检索权重" if selected_lang == "中文" else "Fulltext Weight"
            alpha = st.slider(
                alpha_label,
                min_value=0.0, max_value=1.0, value=current_alpha, step=0.1,
                key="alpha_slider",
            )
            st.session_state.retrieval_alpha = alpha
        
        # 重排算法
        rerank_label = "重排算法" if selected_lang == "中文" else "Rerank Method"
        rerank_opts = {
            "adaptive": "自适应重排" if selected_lang == "中文" else "Adaptive",
            "reranker": "Reranker 精排" if selected_lang == "中文" else "Reranker",
            "keyword": "关键词重排" if selected_lang == "中文" else "Keyword",
            "none": "不重排" if selected_lang == "中文" else "None",
        }
        current_rr = st.session_state.get("rerank_method", "adaptive")
        selected_rr = st.selectbox(
            rerank_label,
            options=list(rerank_opts.keys()),
            format_func=lambda x: rerank_opts[x],
            index=list(rerank_opts.keys()).index(current_rr) if current_rr in rerank_opts else 0,
            key="rerank_method_select",
            label_visibility="collapsed",
        )
        st.session_state.rerank_method = selected_rr

        # top_k
        tk_label = "返回条数 (top_k)" if selected_lang == "中文" else "Top K"
        current_tk = st.session_state.get("top_k", 10)
        top_k = st.slider(
            tk_label,
            min_value=3, max_value=20, value=current_tk, step=1,
            key="top_k_slider",
        )
        st.session_state.top_k = top_k

        # 相似度阈值
        st_label = "相似度阈值" if selected_lang == "中文" else "Similarity Threshold"
        current_st = st.session_state.get("similarity_threshold", 0.0)
        threshold = st.slider(
            st_label,
            min_value=0.0, max_value=0.8, value=current_st, step=0.05,
            key="threshold_slider",
        )
        st.session_state.similarity_threshold = threshold

        # 查询改写开关
        qr_label = "启用查询改写" if selected_lang == "中文" else "Enable Query Rewrite"
        current_qr = st.session_state.get("query_rewrite", True)
        query_rewrite = st.checkbox(qr_label, value=current_qr, key="query_rewrite_cb")
        st.session_state.query_rewrite = query_rewrite

        st.divider()

        # ── 知识库管理 ──
        kb_label = "📂 知识库管理" if selected_lang == "中文" else "📂 Knowledge Base"
        st.markdown(f"**{kb_label}**")

        if vs:
            try:
                doc_list = vs.list_documents()
            except Exception:
                doc_list = []

            if selected_lang == "中文":
                kb_status = f"向量总数: {vec_count} 条" if doc_list else "知识库为空"
            else:
                kb_status = f"Vectors: {vec_count}" if doc_list else "Knowledge base is empty"
            st.caption(kb_status)

            if doc_list:
                if selected_lang == "中文":
                    st.markdown("**已索引的文档:**")
                    for doc_name in doc_list[:10]:
                        st.markdown(f"- 📄 {doc_name}")
                    if len(doc_list) > 10:
                        st.caption(f"... 还有 {len(doc_list)-10} 个文档")
            else:
                info_text = (
                    "尚无文档被索引。上传 PDF 或运行 `python main.py --scan`"
                    if selected_lang == "中文"
                    else "No documents indexed yet. Upload a PDF or run `python main.py --scan`"
                )
                st.info(info_text)
        else:
            warn_text = "向量存储未初始化" if selected_lang == "中文" else "Vector store not initialized"
            st.warning(warn_text)

        st.divider()

        # ── 上传 PDF ──
        upload_label = "上传新 PDF" if selected_lang == "中文" else "Upload PDF"
        st.markdown(f"**{upload_label}**")
        uploaded_file = st.file_uploader(
            upload_label,
            type=["pdf"],
            key="pdf_uploader",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            kb_dir = _project_root / "knowledge_base"
            kb_dir.mkdir(parents=True, exist_ok=True)
            save_path = kb_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"{uploaded_file.name} 已上传")

            if st.button("📥 索引到知识库", key="index_btn", use_container_width=True):
                spinner_text = "正在解析并索引 PDF..." if selected_lang == "中文" else "Indexing PDF..."
                with st.spinner(spinner_text):
                    try:
                        from main import index_pdf

                        index_pdf(str(save_path), vs, config, llm)
                        success_text = "索引完成！" if selected_lang == "中文" else "Indexing complete!"
                        st.success(success_text)
                        st.rerun()
                    except Exception as e:
                        st.error(f"索引失败: {e}")

        st.divider()

        # ── 历史会话 ──
        hist_label = "📋 历史会话" if selected_lang == "中文" else "📋 History"
        st.markdown(f"**{hist_label}**")
        if db and st.session_state.user_id:
            try:
                sessions = db.get_user_sessions(st.session_state.user_id)
            except Exception:
                sessions = []
            if sessions:
                for ses in sessions:
                    ses_id = ses.get("session_id", "")
                    title_text = f"Session {ses_id[:12]}"
                    btn_label = title_text[:25] + "..." if len(title_text) > 25 else title_text

                    if st.button(btn_label, key=f"ses_{ses_id}", use_container_width=True):
                        st.session_state.session_id = ses_id
                        st.session_state.chat_messages = []
                        try:
                            history = db.get_chat_history(st.session_state.user_id, ses_id)
                            for h in history:
                                st.session_state.chat_messages.append({
                                    "role": h["role"],
                                    "content": h["content"],
                                    "timing": {
                                        "retrieval": h.get("retrieval_time_ms", 0),
                                        "llm": h.get("llm_time_ms", 0),
                                        "total": h.get("retrieval_time_ms", 0) + h.get("llm_time_ms", 0),
                                    } if h["role"] == "assistant" else None,
                                })
                        except Exception:
                            pass
                        st.rerun()
            else:
                no_hist_label = "暂无历史会话" if selected_lang == "中文" else "No history"
                st.caption(no_hist_label)
        else:
            no_hist_label = "暂无历史会话" if selected_lang == "中文" else "No history"
            st.caption(no_hist_label)

    return selected_lang, selected_mode
