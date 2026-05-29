# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
RAG 引擎

负责：
- 知识库检索（多路召回 + 重排序）
- LLM 调用生成回答
- 完整的对话处理流程

⚠️ 常改动的地方：
1. 检索时的 top_k 参数（在 retrieve 方法中，默认 top_k=5，但内部会先取 top_k*2 再重排序）
2. 重排序模型的设备（当前强制 cpu，可改为 cuda 加速）
3. LLM 模式切换（LLM_MODE 在 config.py 中定义）
4. 重排序时截断文档长度（当前 512 字符，可调整）
5. 日志记录的级别和格式

⚠️ 注意事项：
1. 检索失败时自动降级，返回空 context
2. 重排序模型可选（use_reranker），如果加载失败会自动禁用
3. 流式和非流式两种模式，分别调用不同的 LLM 实例
4. 使用 utils.logger.get_logger("rag") 记录专用日志
"""

import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

from config import (
    OLLAMA_MODEL, LLM_TEMPERATURE, LLM_NUM_PREDICT,
    BGE_RERANKER_PATH, LLM_MODE, DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
)
from utils.logger import get_logger

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 引擎：封装检索、重排序、LLM 调用"""

    def __init__(self, vector_store, embedding_func, use_reranker: bool = True):
        """
        初始化 RAG 引擎

        参数:
            vector_store: LangChain 向量存储实例（Milvus）
            embedding_func: Embedding 包装类，提供 embed_query 等方法
            use_reranker: 是否启用 BGE 重排序器，默认 True
        """
        self.vector_store = vector_store
        self.embedding_func = embedding_func
        self.use_reranker = use_reranker
        self.rag_logger = get_logger("rag")   # 专用的 RAG 日志记录器

        self.reranker = None
        if use_reranker:
            self._init_reranker()

    def _init_reranker(self):
        """初始化 BGE 重排序器（CrossEncoder）"""
        try:
            from sentence_transformers import CrossEncoder
            # ⚠️ 常改动：可改为 device="cuda" 使用 GPU 加速重排序
            self.reranker = CrossEncoder(BGE_RERANKER_PATH, device="cpu")
            logger.info(f"[Reranker] BGE 重排序器加载成功 (device=cpu)")
        except Exception as e:
            logger.warning(f"[Reranker] 加载失败: {e}")
            self.use_reranker = False

    def _rerank(self, query: str, docs: List[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        使用 BGE 对检索结果进行重排序

        ⚠️ 常改动：可调整每个文档输入重排序器的最大长度（当前 pairs 中截取前 512 字符）
        """
        if not self.use_reranker or self.reranker is None or not docs:
            return docs[:top_k]

        try:
            # 构造 (query, doc_content) 对，截断长文档以提升性能
            pairs = [(query, doc.page_content[:512]) for doc, _ in docs]
            # 预测相关性分数
            scores = self.reranker.predict(pairs)
            # 重新组合并按分数降序排序
            reranked = [(doc, float(score)) for (doc, _), score in zip(docs, scores)]
            reranked.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"[Reranker] 重排序完成: {len(docs)} -> {top_k}")
            return reranked[:top_k]
        except Exception as e:
            logger.warning(f"[Reranker] 失败: {e}")
            return docs[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Document], List[str], str]:
        """
        从知识库检索相关文档，支持重排序

        参数:
            query: 用户查询文本
            top_k: 最终返回的文档数量（默认 5）

        返回:
            documents: Document 对象列表
            sources: 来源文件名列表
            context: 格式化的上下文字符串（用于提示词）

        ⚠️ 注意事项：
            1. 内部会先检索 top_k * 2 个候选，再通过重排序选出最终 top_k
            2. 重排序失败时会自动降级为原始检索结果
            3. 日志记录检索耗时和来源文件
        """
        start_time = time.time()

        # 先检索 top_k * 2 个候选文档（为重排序留出余地）
        results = self.vector_store.similarity_search_with_score(query, k=top_k * 2)

        if self.use_reranker:
            results = self._rerank(query, results, top_k)
        else:
            results = results[:top_k]

        documents = [doc for doc, _ in results]
        elapsed = (time.time() - start_time) * 1000

        # 记录检索日志（前3个来源）
        sources_list = []
        for doc in documents[:3]:
            src = doc.metadata.get("source_file", "未知")
            sources_list.append(src)

        self.rag_logger.info(
            f"检索: '{query[:80]}' | 结果数: {len(documents)} | 耗时: {elapsed:.0f}ms | 来源: {', '.join(sources_list[:3])}")

        # 调试输出（控制台打印）
        if documents:
            print(f"[检索成功] 找到 {len(documents)} 个文档")
            for i, doc in enumerate(documents[:2]):
                print(f"  文档{i + 1}: {doc.page_content[:100]}...")
        else:
            print(f"[检索失败] 没有找到相关文档")

        # 构建 context（格式化为便于 LLM 阅读的文本）
        if documents:
            context_parts = []
            for i, (doc, score) in enumerate(results):
                content = doc.page_content[:800]   # 每个文档最多取 800 字符
                source = doc.metadata.get("source_file", "未知文档")
                # 清理路径，只保留文件名
                if '/' in source:
                    source = source.split('/')[-1]
                if '\\' in source:
                    source = source.split('\\')[-1]
                context_parts.append(f"[参考{i + 1}] 来自《{source}》\n{content}")
            context = "\n\n".join(context_parts)
        else:
            context = ""

        sources = [doc.metadata.get("source_file", "未知") for doc in documents]

        logger.info(f"[RAG] 检索到 {len(documents)} 个文档, context长度: {len(context)}")
        return documents, sources, context

    def _get_llm(self):
        """根据 config 中的 LLM_MODE 获取非流式 LLM 实例"""
        if LLM_MODE == "deepseek":
            print("[LLM] 使用在线 DeepSeek API")
            return ChatOpenAI(
                model=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_NUM_PREDICT
            )
        else:
            print("[LLM] 使用本地 Ollama")
            return Ollama(
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                num_predict=LLM_NUM_PREDICT
            )

    def _get_streaming_llm(self):
        """根据 config 中的 LLM_MODE 获取流式 LLM 实例（DeepSeek 支持 streaming=True）"""
        if LLM_MODE == "deepseek":
            print("[LLM] 流式-使用在线 DeepSeek API")
            return ChatOpenAI(
                model=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_NUM_PREDICT,
                streaming=True   # 开启流式
            )
        else:
            # Ollama 的流式是通过 astream 实现的，不需要单独设置参数
            print("[LLM] 流式-使用本地 Ollama")
            return Ollama(
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                num_predict=LLM_NUM_PREDICT
            )

    def chat(self, prompt: str) -> str:
        """
        非流式调用 LLM，返回完整回答
        ⚠️ 常改动：错误处理中的返回消息可根据需求修改
        """
        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)

            # 兼容不同 LLM 的返回格式
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)

            answer = answer.strip()
            return answer if answer else "抱歉，我没有找到相关信息。"
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            if "API key" in str(e) or "authentication" in str(e).lower():
                return "DeepSeek API 密钥无效，请检查 config.py 中的 DEEPSEEK_API_KEY 配置"
            return f"模型调用失败: {e}"

    async def chat_stream(self, prompt: str):
        """
        流式调用 LLM，异步生成器逐步返回 token
        ⚠️ 注意事项：Ollama 和 DeepSeek 的 astream 返回格式略有差异，已做兼容处理
        """
        try:
            llm = self._get_streaming_llm()

            if LLM_MODE == "deepseek":
                # DeepSeek (OpenAI 兼容) 的流式返回每个 chunk 有 content 属性
                async for chunk in llm.astream(prompt):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
                    elif isinstance(chunk, str):
                        yield chunk
            else:
                # Ollama 的流式返回直接是字符串
                async for chunk in llm.astream(prompt):
                    if isinstance(chunk, str):
                        yield chunk
                    elif hasattr(chunk, 'content'):
                        yield chunk.content

        except Exception as e:
            logger.error(f"LLM 流式调用失败: {e}")
            yield f"模型调用失败: {e}"

    def _clean_source_name(self, name: str) -> str:
        """
        清理来源文件名：去除日期后缀、扩展名和下划线
        ⚠️ 常改动：可调整正则规则以匹配不同的文件名格式
        """
        name = re.sub(r'_\d{8}', '', name)   # 去除 _20260101 格式
        name = re.sub(r'\.docx$|\.pdf$', '', name)
        name = name.replace('_', '')
        return name if len(name) > 2 else "民法典"

    def format_sources_response(self, docs: List[Document]) -> List[Dict]:
        """
        格式化来源信息，用于前端展示（最多取前 3 个文档）
        返回格式：[{"file": "《文件名》", "preview": "内容预览"}]
        """
        sources = []
        for doc in docs[:3]:
            source_file = doc.metadata.get("source_file", "知识库文档")
            # 提取纯文件名
            if '/' in source_file:
                source_file = source_file.split('/')[-1]
            if '\\' in source_file:
                source_file = source_file.split('\\')[-1]
            source_file = self._clean_source_name(source_file)

            content_preview = doc.page_content[:200]
            if len(doc.page_content) > 200:
                content_preview += "..."
            sources.append({
                "file": f"《{source_file}》",
                "preview": content_preview
            })
        return sources


def create_rag_engine(vector_store, embedding_func, use_reranker: bool = True) -> RAGEngine:
    """工厂方法：创建 RAG 引擎实例"""
    return RAGEngine(vector_store, embedding_func, use_reranker)