"""
RAG 对话引擎
- 负责多轮对话、上下文管理、检索增强生成
- 支持 RAG / Direct / Both 三种模式
"""
import os
import re
import json
import time
from typing import Optional

from retrieval_strategy import RetrievalConfig
from session_manager import ChatSession, SessionManager
from logger import get_logger

logger = get_logger("rag")


class RAGEngine:
    """RAG 引擎：负责检索增强生成的全部逻辑"""

    def __init__(
        self,
        vector_store,
        llm_provider,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
    ):
        self.vector_store = vector_store
        self.llm = llm_provider
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.session_mgr = SessionManager()

        # 延迟导入 RetrievalExecutor，避免启动时循环依赖
        from retrieval_strategy import RetrievalExecutor
        self.executor = RetrievalExecutor(
            vector_store=vector_store,
        )

    # ─── 对话核心 ──────────────────────────────────────────

    def answer(self, query: str, session: Optional[ChatSession] = None,
               language: str = "中文", mode: str = "rag",
               retrieval_config=None) -> dict:
        """api.py 兼容的 answer 接口，返回 dict"""
        if session is None:
            session = self.session_mgr.create_or_get("default")
        if mode == "direct":
            t0 = time.perf_counter()
            response, sources, used_mode = self._direct_answer(query, session)
            retrieval_ms = 0
            llm_ms = int((time.perf_counter() - t0) * 1000)
            total_ms = llm_ms
            mode = used_mode
        else:
            response, sources, mode, retrieval_ms, llm_ms, total_ms = self._rag_answer(
                query,
                session,
                llm_model=None,
                mode_override=mode,
                retrieval_config=retrieval_config,
            )
        if isinstance(response, tuple):
            answer = response[0]
        else:
            answer = response
        return {
            "answer": answer,
            "sources": sources if sources else [],
            "mode": mode or "rag",
            "retrieval_time_ms": retrieval_ms,
            "llm_time_ms": llm_ms,
            "total_time_ms": total_ms,
        }

    def chat(self, query: str, mode: str = "rag",
             session: Optional[ChatSession] = None,
             llm_model: str = None,
             retrieval_config=None) -> tuple[str, list[dict], str]:
        """统一的对话入口"""
        if session is None:
            session = self.session_mgr.create_or_get("default")

        if mode == "direct":
            return self._direct_answer(query, session, llm_model)
        elif mode == "both":
            response, sources, _, _, _, _ = self._rag_answer(
                query,
                session,
                llm_model,
                mode_override="both",
                retrieval_config=retrieval_config,
            )
            return response, sources, "both"
        else:
            response, sources, used_mode, _, _, _ = self._rag_answer(
                query,
                session,
                llm_model,
                mode_override="rag",
                retrieval_config=retrieval_config,
            )
            return response, sources, used_mode

    def _rag_answer(self, query: str, session: ChatSession,
                    llm_model: str = None,
                    mode_override: str = None,
                    retrieval_config=None) -> tuple[str, list[dict], str, int, int, int]:
        """RAG 模式：检索 + 生成"""
        t_start = time.perf_counter()
        sources, prompt, system_prompt, used_mode = self._prepare_rag(
            query,
            session,
            mode_override=mode_override,
            retrieval_config=retrieval_config,
        )
        t_retrieval_end = time.perf_counter()

        response = self.llm.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            session=session,
            model=llm_model,
        )
        t_llm_end = time.perf_counter()

        retrieval_ms = int((t_retrieval_end - t_start) * 1000)
        llm_ms = int((t_llm_end - t_retrieval_end) * 1000)
        total_ms = int((t_llm_end - t_start) * 1000)
        return response, sources, used_mode, retrieval_ms, llm_ms, total_ms

    def _direct_answer(self, query: str, session: ChatSession,
                       llm_model: str = None) -> tuple[str, list[dict], str]:
        """Direct 模式：直接对话"""
        system_prompt = "你是一个专业的 AI 助手。"
        session.add_message("user", query)
        response = self.llm.chat(
            prompt=query,
            system_prompt=system_prompt,
            session=session,
            model=llm_model,
        )
        return response, [], "direct"

    def _both_answer(self, query: str, session: ChatSession,
                     llm_model: str = None) -> tuple[str, list[dict], str]:
        """Both 模式：RAG 检索 + 混合回答"""
        rag_response, sources, _ = self._rag_answer(query, session, llm_model)
        return rag_response, sources, "both"

    # ─── 检索准备 ──────────────────────────────────────────

    def _prepare_rag(self, query: str, session: ChatSession,
                     mode_override: str = None,
                     retrieval_config=None) -> tuple[list[dict], str, str, str]:
        """准备 RAG 所需的数据：检索 + 构建 prompt"""
        mode = mode_override or "hybrid"
        if retrieval_config is not None:
            config = retrieval_config
            # retrieval_config.mode 已由外部请求设置为 vector/fulltext/hybrid
            if not getattr(config, 'top_k', None):
                config.top_k = self.top_k
            if not getattr(config, 'similarity_threshold', None) and config.similarity_threshold != 0:
                config.similarity_threshold = self.similarity_threshold
        else:
            config = RetrievalConfig(
                mode="hybrid",
                top_k=self.top_k,
                similarity_threshold=self.similarity_threshold,
                query_rewrite=True,
                rerank_method="adaptive",
            )

        query_for_search = query
        if config.query_rewrite:
            rewritten = self._rewrite_query(query, session)
            if rewritten and rewritten != query:
                query_for_search = rewritten
                logger.info(f"查询改写: {query[:40]} -> {rewritten[:40]}")

        raw_sources = self.executor.search_with_rerank(query_for_search, config)

        # 带权重的文档块排序（去掉重复页码最近的优先）
        sources = self._deduplicate_sources(raw_sources)
        context_text = self._build_context(sources)
        prompt = self._build_rag_prompt(query, context_text, session)
        lang_inst = "请用中文回答。"

        system_prompt = (
            "你是一个专业的金融文档问答助手。请基于下面的知识库内容回答用户问题。\n"
            "规则：\n"
            "1. 仔细阅读知识库内容。如果包含所需数据，提取出来并引用页码 [第 X 页]。\n"
            "2. 只有在知识库中确实完全找不到任何相关信息时，才说知识库中未包含该信息。不要编造，不要使用可能或许等猜测。\n"
            f"【语言要求】{lang_inst}"
        )

        return sources, prompt, system_prompt, mode

    def _full_search(self, query: str, config: Optional[RetrievalConfig] = None) -> list[dict]:
        cfg = config or self.get_default_config("hybrid")
        cfg.top_k = self.top_k
        cfg.similarity_threshold = self.similarity_threshold
        return self.executor.search_with_rerank(query, cfg)

    @staticmethod
    def _extract_question(query: str) -> str:
        import re
        match = re.search(r'"question"\s*:\s*"([^"]+)"', query)
        if match:
            return match.group(1)
        return query

    def _rewrite_query(self, query: str, session: ChatSession) -> str:
        """查询改写：根据对话历史完善当前问题

        仅在当前问题包含代词（它、他、这、那、其、该、上述等）时执行，
        否则直接返回原问题，节省不必要的 LLM 调用。
        """
        # 代词检测：没有代词就不需要改写
        PRONOUNS = {"它", "他", "她", "这", "那", "其", "该", "上述",
                     "这个", "那个", "这些", "那些", "它们", "他们",
                     "该等", "前述", "上述的", "所述", "上面"}
        if not any(p in query for p in PRONOUNS):
            return query

        history = session.get_context()
        if not history:
            return query

        rewrite_prompt = (
            "基于对话历史，将当前问题改写为一个独立完整的问题。\n"
            f"对话历史：{history}\n"
            f"当前问题：{query}\n"
            "改写后的问题（只输出改写后的内容）："
        )
        try:
            result = self.llm.chat(prompt=rewrite_prompt, max_tokens=200)
            result = result.strip().strip('"').strip("'")
            return result if result else query
        except Exception as e:
            logger.warning(f"查询改写失败: {e}")
            return query

    # ─── 上下文构建 ──────────────────────────────────────────

    def _build_context(self, sources: list[dict]) -> str:
        """将检索结果构造成上下文文本。

        只取前 MAX_CONTEXT_CHUNKS 块喂给 LLM：Hit@5≈0.94 说明答案几乎都在前几块，
        过多上下文只会增大 prefill、拖慢生成、还可能用无关块带偏 LLM。
        注意：这里裁剪的只是喂给 LLM 的上下文，API 仍返回完整 sources（检索指标不受影响）。
        多答案题（如需列举多家公司）若召回分散，可上调此值。
        """
        MAX_CONTEXT_CHUNKS = 8
        if not sources:
            return ""

        parts = []
        seen = set()
        uniq = 0
        for s in sources:
            uid = f"{s.get('page', '?')}-{s.get('chunk_id', '?')}"
            if uid in seen:
                continue
            seen.add(uid)
            uniq += 1
            page = s.get("page", "?")
            score = s.get("score", 0)
            text = s.get("text", "")
            # 截断过长文本
            if len(text) > 1200:
                text = text[:1200] + "..."

            parts.append(f"[第{page}页] (score={score:.4f}) {text}")

            if uniq >= MAX_CONTEXT_CHUNKS:
                break

        return "\n\n".join(parts)

    @staticmethod
    def _deduplicate_sources(sources: list[dict]) -> list[dict]:
        """去重：相同 chunk_id 保留最先出现的"""
        seen = set()
        result = []
        for s in sources:
            uid = s.get("chunk_id", s.get("page"))
            if uid in seen:
                continue
            seen.add(uid)
            result.append(s)
        return result

    def _build_rag_prompt(self, query: str, context: str, session: ChatSession) -> str:
        """构建 RAG 模式下的 prompt"""
        parts = []

        chat_context = session.get_context()
        if chat_context:
            parts.append(chat_context)
            parts.append("")

        if context:
            parts.append("以下是知识库中与问题相关的内容：")
            parts.append("---")
            parts.append(context)
            parts.append("---")

        parts.append(f"用户的问题是：{query}")
        parts.append("请根据上面的知识库内容回答。先仔细阅读参考资料，如果参考资料中有相关信息（包括数据、描述、列表、说明等），请直接回答。只有在确实完全找不到任何相关信息时，才说知识库中未包含该信息。注意：回答中的页码必须与参考资料中标明的页码一致，不要自己猜页码。请仔细阅读参考资料中的表格数据，包括金额和比例等具体数字。")

        return "\n".join(parts)

    def _build_direct_prompt(self, query: str, session: ChatSession) -> str:
        """构建 Direct 模式下的 prompt"""
        parts = []

        chat_context = session.get_context()
        if chat_context:
            parts.append(chat_context)
            parts.append("")

        parts.append(f"用户的问题是：{query}")

        return "\n".join(parts)

    # ─── 流式回答 ──────────────────────────────────────────

    def answer_stream(self, query: str, session: Optional[ChatSession] = None,
                      mode: str = "rag",
                      llm_model: str = None,
                      language: str = "中文", mode_override: str = None,
                      retrieval_config=None):
        """流式回答生成器（兼容 api.py 的事件格式）

        Yields:
            {"type": "retrieval", "sources": [...], "time_ms": int}
            {"type": "token", "content": str}
            {"type": "done", "sources": [...], "mode": str,
             "llm_time_ms": int, "total_ms": int}
        """
        # 兼容 api.py 传参
        if mode_override:
            mode = mode_override

        if session is None:
            session_id = self.session_mgr.create_session("default")
            from session_manager import ChatSession as CS
            session = CS(session_id=session_id)

        import time
        t_start = time.time()

        # ── 检索阶段 ──
        if mode in ("rag", "both", "hybrid"):
            sources, prompt, system_prompt, used_mode = self._prepare_rag(
                query, session, mode_override=mode,
                retrieval_config=retrieval_config
            )
        else:
            sources = []
            prompt = query
            system_prompt = "你是一个专业的 AI 助手。"
            used_mode = "direct"

        t_retrieval = time.time()
        retrieval_ms = int((t_retrieval - t_start) * 1000)

        # yield 检索结果事件
        yield {
            "type": "retrieval",
            "sources": sources,
            "time_ms": retrieval_ms,
        }

        # ── 生成阶段 ──
        response_text = ""
        for chunk in self.llm.ask_stream(
            prompt=prompt,
            system_prompt=system_prompt,
        ):
            response_text += chunk
            yield {"type": "token", "content": chunk}

        t_llm_end = time.time()
        llm_ms = int((t_llm_end - t_retrieval) * 1000)
        total_ms = int((t_llm_end - t_start) * 1000)

        session.add_message("assistant", response_text)

        # yield 完成事件
        yield {
            "type": "done",
            "sources": sources,
            "mode": used_mode,
            "llm_time_ms": llm_ms,
            "total_ms": total_ms,
        }

    def _retrieve_and_build(self, query: str, session: ChatSession,
                            mode: str) -> tuple[str, list[dict], str, str]:
        """共享检索 + prompt 构建逻辑"""
        return self._prepare_rag(query, session, mode_override=mode)

    # ─── 工具方法 ──────────────────────────────────────────

    def get_default_config(self, mode: str = "hybrid") -> RetrievalConfig:
        return RetrievalConfig(
            mode=mode,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            query_rewrite=True,
            rerank_method="adaptive",
        )

    def ask(self, query: str, mode: str = "rag",
            session: Optional[ChatSession] = None) -> str:
        """简易接口：返回答案文本"""
        answer, sources, used_mode = self.chat(query, mode, session)
        return answer

    def rewrite_query(self, query: str) -> str:
        """独立查询改写"""
        rewrite_prompt = (
            "将以下问题改写为更完整的表述：\n"
            f"原始问题：{query}\n"
            "改写后："
        )
        try:
            return self.llm.chat(prompt=rewrite_prompt, max_tokens=200).strip()
        except Exception as e:
            logger.warning(f"查询改写失败: {e}")
            return query

    def classify_query(self, query: str) -> str:
        """判断是否需要 RAG"""
        prompt = (
            "判断以下问题是否需要通过知识库搜索来回答。"
            "只需回答 'yes' 或 'no'。\n"
            f"问题：{query}\n"
        )
        try:
            result = self.llm.chat(prompt=prompt, max_tokens=10).strip().lower()
            return "rag" if result.startswith("y") else "direct"
        except Exception as e:
            logger.warning(f"查询分类失败: {e}")
            return "rag"
