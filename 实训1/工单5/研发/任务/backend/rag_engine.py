"""
RAG 引擎模块
检索增强生成核心逻辑 + 对话管理器
"""
import time
import uuid
from typing import Optional

from logger import get_logger, log_exception
from retrieval_strategy import RetrievalConfig, RetrievalExecutor

logger = get_logger("rag")


class ChatSession:
    """对话会话 - 管理上下文历史"""
    
    def __init__(self, user_id: int, session_id: Optional[str] = None,
                 max_history: int = 20, max_context_length: int = 4000):
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.history: list[dict] = []  # [{"role": str, "content": str}, ...]
        self.max_history = max_history
        self.max_context_length = max_context_length
        self.mode = "rag"  # rag / direct
        logger.debug(f"新建对话会话: user={user_id}, session={self.session_id}")
    
    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self.history.append({"role": role, "content": content})
        # 限制历史长度
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self) -> str:
        """
        构建对话上下文文本
        用于 LLM 的 system prompt 或前缀
        """
        if not self.history:
            return ""
        
        lines = ["以下是历史对话："]
        for msg in self.history[-6:]:  # 保留最近 6 轮
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}: {msg['content'][-200:]}")
        
        return "\n".join(lines)
    
    def set_mode(self, mode: str):
        if mode in ("rag", "direct"):
            self.mode = mode
    
    def clear(self):
        self.history.clear()


class RAGEngine:
    """
    RAG 引擎
    整合 PDF 处理 + 检索策略 + LLM 调用
    内部使用 RetrievalExecutor 管理检索策略
    """
    
    def __init__(self, vector_store, llm_provider,
                 top_k: int = 10, similarity_threshold: float = 0.3,
                 reranker_model_path: str = ""):
        self.vector_store = vector_store
        self.llm = llm_provider
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.reranker_model_path = reranker_model_path
        # 使用 RetrievalExecutor 统一管理检索策略
        self.executor = RetrievalExecutor(vector_store, reranker_model_path)
        if reranker_model_path:
            logger.info(f"Reranker 模型路径已配置: {reranker_model_path}")

    def get_default_config(self, mode: str = "hybrid") -> RetrievalConfig:
        """获取默认检索配置（基于引擎初始化参数）"""
        return RetrievalConfig(
            mode=mode,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            rerank_method="adaptive",
        )

    def _full_search(self, query: str, config: Optional[RetrievalConfig] = None) -> list[dict]:
        """
        完整的检索 + 重排流程
        如果 config 为 None，使用默认配置（hybrid + adaptive）
        """
        cfg = config or self.get_default_config("hybrid")
        cfg.top_k = self.top_k
        cfg.similarity_threshold = self.similarity_threshold
        return self.executor.search_with_rerank(query, cfg)
    
    @staticmethod
    def _extract_question(query: str) -> str:
        """从 JSON 格式的 query 中提取真正的 question 字段"""
        import re
        match = re.search(r'"question"\s*:\s*"([^"]+)"', query)
        if match:
            extracted = match.group(1)
            logger.debug(f"JSON query 清洗: {query[:60]}... -> {extracted}")
            return extracted
        return query

    def answer_stream(self, query: str, session: ChatSession, language: str = "中文",
                      mode_override: Optional[str] = None,
                      retrieval_config: Optional[RetrievalConfig] = None):
        """
        流式 RAG 回答
        先返回检索结果和检索耗时，再流式输出 LLM 回答
        Yields: dict 事件
            {"type": "retrieval", "sources": [...], "time_ms": N}
            {"type": "timing_base", "retrieval_ms": N}
            {"type": "token", "content": "..."}  (逐个 token)
            {"type": "done", "llm_ms": N, "total_ms": N}
            {"type": "error", "message": "..."}
        
        Args:
            retrieval_config: 检索配置（可为 None，使用默认）
        """
        query = self._extract_question(query)
        timings = {"retrieval": 0, "llm": 0}
        start_total = time.time()

        try:
            # 判断模式：mode_override 优先于 session.mode
            mode = mode_override if mode_override else (
                session.mode if hasattr(session, 'mode') and session.mode in ("rag", "direct", "both") else "rag"
            )
            if mode == "rag" or mode == "both":
                # 1. 使用 RetrievalExecutor 统一检索 + 重排
                t0 = time.time()
                cfg = retrieval_config or self.get_default_config("hybrid")
                cfg.top_k = self.top_k
                cfg.similarity_threshold = self.similarity_threshold
                sources_top = self.executor.search_with_rerank(query, cfg)
                timings["retrieval"] = int((time.time() - t0) * 1000)

                if sources_top:
                    pages = sorted(set(s["page"] for s in sources_top))
                    logger.debug(f"[检索] 取前{self.top_k}条, 来自第 {pages} 页:")
                    for i, s in enumerate(sources_top, 1):
                        preview = s["text"][:120].replace("\n", " ")
                        logger.debug(f"  [{i}] 第{s['page']}页 (score={s['score']:.4f}) {preview}...")

                yield {"type": "retrieval", "sources": sources_top, "time_ms": timings["retrieval"]}

                # 2. 构建 prompt
                context_text = self._build_context(sources_top)
                prompt = self._build_rag_prompt(query, context_text, session)
                lang_inst = (
                    "请严格使用中文回答所有问题。答案要准确、简洁、有条理。"
                    if language == "中文"
                    else "Please answer all questions strictly in English. Be accurate, concise, and well-organized."
                )
                system_prompt = (
                    "你是一个专业的金融文档问答助手。请基于下面的知识库内容回答用户问题。\n"
                    "规则：\n"
                    "1. 如果知识库中包含所需数据，提取出来并引用页码 [第 X 页]。\n"
                    "2. 如果知识库中没有与用户问题相关的信息（包括数据、描述、列表等），直接说\"知识库中未包含该信息\"。不要编造，不要使用\"可能\"\"或许\"等猜测。\n"
                    f"【语言要求】{lang_inst}"
                )
            else:

                yield {"type": "retrieval", "sources": [], "time_ms": 0}
                prompt = self._build_direct_prompt(query, session)
                lang_inst = (
                    "请严格使用中文回答所有问题。答案要准确、简洁、有条理。"
                    if language == "中文"
                    else "Please answer all questions strictly in English. Be accurate, concise, and well-organized."
                )
                system_prompt = f"你是一个专业的 AI 助手。{lang_inst}"

            # 3. 流式 LLM 回答
            t0 = time.time()
            full_answer = ""
            for token in self.llm.ask_stream(prompt, system_prompt=system_prompt):
                full_answer += token
                yield {"type": "token", "content": token}

            timings["llm"] = int((time.time() - t0) * 1000)
            total_time = int((time.time() - start_total) * 1000)

            # 保存历史
            session.add_message("user", query)
            session.add_message("assistant", full_answer)

            total_time = int((time.time() - start_total) * 1000)
            logger.info(
                f"回答完成: mode={session.mode}, "
                f"retrieval={timings['retrieval']}ms, "
                f"llm={timings['llm']}ms, "
                f"total={total_time}ms"
            )

            yield {"type": "done", "llm_time_ms": timings["llm"], "total_time_ms": total_time}

        except Exception as e:
            log_exception(logger, "RAG 流式回答生成失败", e)
            total_time = int((time.time() - start_total) * 1000)
            yield {"type": "error", "message": str(e), "total_time_ms": total_time}

    def answer(self, query: str, session: ChatSession, language: str = "中文",
               retrieval_config: Optional[RetrievalConfig] = None) -> dict:
        """
        完整 RAG 回答流程
        返回: {
            "answer": str,
            "sources": list[dict],
            "retrieval_time_ms": int,
            "llm_time_ms": int,
            "total_time_ms": int,
            "mode": str,
        }
        """
        # 清洗 JSON query
        query = self._extract_question(query)

        # ── 记录耗时 ──
        timings = {"retrieval": 0, "llm": 0}
        start_total = time.time()
        
        try:
            if session.mode == "rag":
                # ── 1. 使用 RetrievalExecutor 统一检索 + 重排 ──
                t0 = time.time()
                cfg = retrieval_config or self.get_default_config("hybrid")
                cfg.top_k = self.top_k
                cfg.similarity_threshold = self.similarity_threshold
                sources_top = self.executor.search_with_rerank(query, cfg)
                timings["retrieval"] = int((time.time() - t0) * 1000)

                if sources_top:
                    pages = sorted(set(s["page"] for s in sources_top))
                    logger.debug(f"[检索] 取前{self.top_k}条, 来自第 {pages} 页:")
                    for i, s in enumerate(sources_top, 1):
                        preview = s["text"][:120].replace("\n", " ")
                        logger.debug(f"  [{i}] 第{s['page']}页 (score={s['score']:.4f}) {preview}...")

                # ── 2. 构建上下文 ──
                context_text = self._build_context(sources_top)
                
                # ── 3. 构建 prompt ──
                prompt = self._build_rag_prompt(query, context_text, session)
                lang_inst = (
                    "请严格使用中文回答所有问题。答案要准确、简洁、有条理。"
                    if language == "中文"
                    else "Please answer all questions strictly in English. Be accurate, concise, and well-organized."
                )
                system_prompt = (
                    "你是一个专业的金融文档问答助手。请基于给定的知识库内容回答用户问题。\n"
                    "重要规则（必须严格遵守）：\n"
                    "1. 只从下面\"知识库内容\"中提取答案。如果找不到与用户问题相关的信息（包括数据、描述、列表等），直接说\"知识库中未包含该信息\"。\n"
                    "2. 回答时必须引用信息来源页码，格式为[第 X 页]。例如：'该公司的注册资本为5,325万元[第42页]。'\n"
                    "3. 禁止使用'没有直接提及'、'没有明确提及'、'未明确列出'等回避型措辞。\n"
                    "4. 如果知识库中完全没有相关信息，请如实说'知识库中未包含该信息'，并告诉用户当前知识库的主要内容范围。\n"
                    f"【语言要求】{lang_inst}"
                )
            else:
                # Direct 模式 - 纯 LLM
                sources_top = []
                context_text = session.get_context()
                prompt = self._build_direct_prompt(query, session)
                lang_inst = (
                    "请严格使用中文回答所有问题。答案要准确、简洁、有条理。"
                    if language == "中文"
                    else "Please answer all questions strictly in English. Be accurate, concise, and well-organized."
                )
                system_prompt = f"你是一个专业的 AI 助手。{lang_inst}"
            
            # ── 4. LLM 回答 ──
            t0 = time.time()
            answer = self.llm.ask(prompt, system_prompt=system_prompt)
            timings["llm"] = int((time.time() - t0) * 1000)
            
            # ── 5. 保存到历史 ──
            session.add_message("user", query)
            session.add_message("assistant", answer)
            
            total_time = int((time.time() - start_total) * 1000)
            
            logger.info(
                f"回答完成: mode={session.mode}, "
                f"retrieval={timings['retrieval']}ms, "
                f"llm={timings['llm']}ms, "
                f"total={total_time}ms"
            )
            
            return {
                "answer": answer,
                "sources": sources_top if session.mode == "rag" else [],
                "retrieval_time_ms": timings["retrieval"],
                "llm_time_ms": timings["llm"],
                "total_time_ms": total_time,
                "mode": session.mode,
            }
            
        except Exception as e:
            log_exception(logger, "RAG 回答生成失败", e)
            total_time = int((time.time() - start_total) * 1000)
            return {
                "answer": f"【系统错误】回答生成过程中发生异常: {str(e)}",
                "sources": [],
                "retrieval_time_ms": timings["retrieval"],
                "llm_time_ms": timings["llm"],
                "total_time_ms": total_time,
                "mode": session.mode,
            }
    
    def _build_context(self, sources: list[dict], max_chars: int = 8000) -> str:
        """从检索结果构建上下文文本，去重页眉并截断到 max_chars 字符
        优先保留含表格数据的 source（含\t或【表格】标记的）
        """
        if not sources:
            return ""

        known_headers = {
            "武汉兴图新科电子股份有限公司",
            "武汉兴图新科电子股份有限公司                      招股说明书（申报稿）",
            "武汉力源信息技术股份有限公司",
            "武汉力源信息技术股份有限公司                           招股意向书",
            "招股说明书（申报稿）",
            "招股意向书",
        }
        lines = []
        total_chars = 0

        # 按原始排序合并：遍历 sources，表格源全部保留，非表格只取5个
        text_count = 0
        for src in sources:
            is_table = "\t" in src.get("text", "") or "【表格】" in src.get("text", "")
            if not is_table:
                if text_count >= 5:
                    continue
                text_count += 1
            page = src.get("page", "?")
            text = src.get("text", "")
            score = src.get("score", 0)
            # 去掉已知的重复页眉
            for h in known_headers:
                if text.startswith(h):
                    text = text[len(h):].lstrip("\n ")
                    break
            # 表格类 source 不截断，非表格截断到800
            if "\t" not in text and "【表格】" not in text:
                if len(text) > 800:
                    text = text[:800] + "...(截断)"
            line = f"[参考资料 {len(lines)//2+1}] (第 {page} 页, 相关度: {score:.4f})\n{text}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            lines.append("")
            total_chars += len(line)

        result = "\n".join(lines)
        logger.debug(f"构建上下文: {len(sources)}条 → {len(result)} 字符, 含15,000={',5000' in result or '15,000' in result}")
        return result
    
    def _build_rag_prompt(self, query: str, context: str, session: ChatSession) -> str:
        """构建 RAG 模式下的 prompt"""
        parts = []
        
        # 对话历史
        chat_context = session.get_context()
        if chat_context:
            parts.append(chat_context)
            parts.append("")
        
        # 知识库上下文
        if context:
            parts.append("以下是知识库中与问题相关的内容：")
            parts.append("---")
            parts.append(context)
            parts.append("---")
        
        # 当前问题
        parts.append(f"用户的问题是：{query}")
        parts.append("请结合上面的知识库内容回答。如果参考资料中没有与用户问题相关的信息（包括数据、描述、列表、说明等），直接说\"知识库中未包含该信息\"，不要编造。注意：回答中的页码必须与参考资料中标明的页码一致，不要自己猜页码。请仔细阅读参考资料中的表格数据，包括金额和比例等具体数字。")

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
