"""
RAG 引擎模块
检索增强生成核心逻辑 + 对话管理器
"""
import time
import uuid
from typing import Optional

from logger import get_logger, log_exception

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
    整合 PDF 处理 + 向量检索 + LLM 调用
    """
    
    def __init__(self, vector_store, llm_provider,
                 top_k: int = 10, similarity_threshold: float = 0.3):
        self.vector_store = vector_store
        self.llm = llm_provider
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
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

    def answer_stream(self, query: str, session: ChatSession, language: str = "中文"):
        """
        流式 RAG 回答
        先返回检索结果和检索耗时，再流式输出 LLM 回答
        Yields: dict 事件
            {"type": "retrieval", "sources": [...], "time_ms": N}
            {"type": "timing_base", "retrieval_ms": N}
            {"type": "token", "content": "..."}  (逐个 token)
            {"type": "done", "llm_ms": N, "total_ms": N}
            {"type": "error", "message": "..."}
        """
        query = self._extract_question(query)
        timings = {"retrieval": 0, "llm": 0}
        start_total = time.time()

        try:
            # 判断模式：从 session.mode 读取
            mode = session.mode if hasattr(session, 'mode') and session.mode in ("rag", "direct", "both") else "rag"
            if mode == "rag" or mode == "both":
                # 1. 检索
                t0 = time.time()
                sources = self.vector_store.search(query, top_k=self.top_k)
                timings["retrieval"] = int((time.time() - t0) * 1000)
                sources = [s for s in sources if s["score"] >= self.similarity_threshold]

                if sources:
                    pages = sorted(set(s["page"] for s in sources))
                    print(f"\n[检索] 命中 {len(sources)} 条, 来自第 {pages} 页:")
                    for i, s in enumerate(sources, 1):
                        preview = s["text"][:120].replace("\n", " ")
                        print(f"  [{i}] 第{s['page']}页 (score={s['score']:.4f}) {preview}...")
                    print()

                yield {"type": "retrieval", "sources": sources, "time_ms": timings["retrieval"]}

                # 2. 构建 prompt
                context_text = self._build_context(sources)
                prompt = self._build_rag_prompt(query, context_text, session)
                lang_inst = (
                    "请严格使用中文回答所有问题。答案要准确、简洁、有条理。"
                    if language == "中文"
                    else "Please answer all questions strictly in English. Be accurate, concise, and well-organized."
                )
                system_prompt = (
                    "你是一个专业的金融文档问答助手。请基于给定的知识库内容回答用户问题。\n"
                    "重要规则（必须严格遵守）：\n"
                    "1. 必须从知识库内容中提取答案。即使信息不完整，也要尽最大努力从中提取有用信息来回答，"
                    "不要轻易说'没有找到'。\n"
                    "2. 回答时必须引用信息来源页码，格式为[第 X 页]。例如：'该公司的注册资本为5,325万元[第42页]。'\n"
                    "3. 禁止使用'没有直接提及'、'没有明确提及'、'未明确列出'等回避型措辞。\n"
                    "4. 如果知识库中完全没有相关信息，请如实说'知识库中未包含该信息'，并告诉用户当前知识库的主要内容范围。\n"
                    f"【语言要求】{lang_inst}"
                )
            else:
                sources = []
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

            yield {"type": "done", "llm_ms": timings["llm"], "total_ms": total_time}

        except Exception as e:
            log_exception(logger, "RAG 流式回答生成失败", e)
            total_time = int((time.time() - start_total) * 1000)
            yield {"type": "error", "message": str(e), "total_ms": total_time}

    def answer(self, query: str, session: ChatSession, language: str = "中文") -> dict:
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
                # ── 1. 向量检索 ──
                t0 = time.time()
                sources = self.vector_store.search(query, top_k=self.top_k)
                timings["retrieval"] = int((time.time() - t0) * 1000)

                # 过滤低分结果
                sources = [s for s in sources if s["score"] >= self.similarity_threshold]
                logger.debug(f"检索结果: {len(sources)} 条")

                # ── 打印检索详情到控制台（方便排查）──
                if sources:
                    pages = sorted(set(s["page"] for s in sources))
                    print(f"\n[检索] 命中 {len(sources)} 条, 来自第 {pages} 页:")
                    for i, s in enumerate(sources, 1):
                        preview = s["text"][:120].replace("\n", " ")
                        print(f"  [{i}] 第{s['page']}页 (score={s['score']:.4f}) {preview}...")
                    print()

                # ── 2. 构建上下文 ──
                context_text = self._build_context(sources)
                
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
                    "1. 必须从知识库内容中提取答案。即使信息不完整，也要尽最大努力从中提取有用信息来回答，"
                    "不要轻易说'没有找到'。\n"
                    "2. 回答时必须引用信息来源页码，格式为[第 X 页]。例如：'该公司的注册资本为5,325万元[第42页]。'\n"
                    "3. 禁止使用'没有直接提及'、'没有明确提及'、'未明确列出'等回避型措辞。\n"
                    "4. 如果知识库中完全没有相关信息，请如实说'知识库中未包含该信息'，并告诉用户当前知识库的主要内容范围。\n"
                    f"【语言要求】{lang_inst}"
                )
            else:
                # Direct 模式 - 纯 LLM
                sources = []
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
                "sources": sources,
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
    
    def _build_context(self, sources: list[dict], max_chars: int = 4000) -> str:
        """从检索结果构建上下文文本，去重页眉并截断到 max_chars 字符"""
        if not sources:
            return ""
        
        known_headers = {
            "武汉兴图新科电子股份有限公司",
            "武汉兴图新科电子股份有限公司                      招股说明书（申报稿）",
            "招股说明书（申报稿）",
        }
        lines = []
        total_chars = 0
        for i, src in enumerate(sources, 1):
            page = src.get("page", "?")
            text = src.get("text", "")
            score = src.get("score", 0)
            # 去掉已知的重复页眉
            for h in known_headers:
                if text.startswith(h):
                    text = text[len(h):].lstrip("\n ")
                    break
            # 对每个块也限制长度
            if len(text) > 1000:
                text = text[:1000] + "...(截断)"
            line = f"[参考资料 {i}] (第 {page} 页, 相关度: {score:.4f})\n{text}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            lines.append("")
            total_chars += len(line)
        
        result = "\n".join(lines)
        logger.debug(f"构建上下文: {len(sources)}条 → {len(result)} 字符")
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
        parts.append("请结合知识库内容回答。")
        
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
