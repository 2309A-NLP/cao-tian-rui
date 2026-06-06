"""
向量存储模块
基于 Milvus 向量数据库 + BGE-M3 嵌入模型（使用 transformers 直接加载）
支持 GPU 自动检测 + 嵌入进度显示 + 先连后删保护
支持 BM25 全文检索 + 向量+BM25 混合检索 (RRF 融合)
"""
import os
import json
import time
import sys
import re
import numpy as np
from pathlib import Path
from typing import Optional

from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility,
)

from logger import get_logger, log_exception

logger = get_logger("vector")


class BGEM3Embedding:
    """BGE-M3 嵌入模型（兼容旧引用，委托给 embedding_provider）"""

    def __init__(self, model_path: str, device: str = "auto",
                 max_length: int = 8192):
        from embedding_provider import BGEM3Embedding as _RealBGE
        self._impl = _RealBGE(model_path, device=device, max_length=max_length)

    @property
    def actual_device(self) -> str:
        return self._impl.actual_device

    @property
    def dimension(self) -> int:
        return self._impl.dimension

    def load(self) -> bool:
        return self._impl.load()

    def encode(self, texts: list[str], batch_size: int = 32,
               show_progress: bool = True) -> np.ndarray:
        self._impl.batch_size = batch_size
        return self._impl.encode(texts, show_progress=show_progress)

    def encode_query(self, text: str) -> np.ndarray:
        return self._impl.encode_query(text)

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        self._impl.batch_size = batch_size
        return self._impl.encode_documents(texts)


class VectorStore:
    """
    向量存储
    - 嵌入模型: 通过 EmbeddingFactory 支持多种模型 (BGE-M3, M3E 等)
    - 向量库: Milvus
    """

    def __init__(
        self,
        model_path: str = "",
        milvus_host: str = "127.0.0.1",
        milvus_port: int = 19530,
        collection_name: str = "doc_chunks",
        device: str = "auto",
        batch_size: int = 32,
        model_type: str = "bge-m3",  # bge-m3 / m3e 等
    ):
        self.model_path = model_path
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.device = device
        self.batch_size = batch_size
        self.model_type = model_type
        self.dimension = 1024  # 默认，由模型加载后更新

        self.model = None  # BaseEmbeddingProvider 实例
        self.collection: Optional[Collection] = None
        self.connected = False
        self.chunks_metadata = []
        # BM25 全文检索
        self.bm25 = None           # BM25Okapi 实例
        self.bm25_texts = []       # 原始文本列表（对应 BM25 索引顺序）
        self.bm25_chunk_ids = []   # 每个 BM25 文档对应的 chunk_id
        self.bm25_ready = False    # BM25 索引是否已构建
        self._has_field_type = False  # 集合是否有 field_type 字段（兼容旧集合）

    # ── 嵌入模型 ──

    def load_model(self) -> bool:
        """加载嵌入模型（通过 EmbeddingFactory 支持多种模型）"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"嵌入模型目录不存在: {self.model_path}")
                return False
            from embedding_provider import EmbeddingFactory
            self.model = EmbeddingFactory.create(
                model_type=self.model_type,
                model_path=self.model_path,
                device=self.device,
                batch_size=self.batch_size,
            )
            if not self.model:
                logger.error(f"不支持的嵌入模型类型: {self.model_type}")
                return False
            result = self.model.load()
            if result:
                self.dimension = self.model.dimension
                logger.info(f"嵌入模型加载成功: {self.model_type}, dim={self.dimension}")
            return result
        except Exception as e:
            log_exception(logger, "嵌入模型加载失败", e)
            return False

    def embed_query(self, text: str) -> list[float]:
        """对查询文本进行嵌入"""
        if not self.model:
            raise RuntimeError("嵌入模型未加载")
        emb = self.model.encode_query(text)
        return emb.tolist() if isinstance(emb, np.ndarray) else emb

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量对文档进行嵌入"""
        if not self.model:
            raise RuntimeError("嵌入模型未加载")
        return self.model.encode_documents(texts, batch_size=self.batch_size)

    # ── Milvus 连接 ──

    def connect_milvus(self) -> bool:
        """连接 Milvus 并初始化集合"""
        try:
            logger.info(f"连接 Milvus: {self.milvus_host}:{self.milvus_port}")
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port,
                timeout=10,
            )

            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                # 检查 field_type 字段是否存在（兼容旧集合）
                schema_fields = [f.name for f in self.collection.schema.fields]
                self._has_field_type = "field_type" in schema_fields
                logger.info(f"Milvus 集合已存在: {self.collection_name}, field_type={'是' if self._has_field_type else '否'}")
                self.collection.load()
            else:
                self._create_collection()

            self.connected = True
            logger.info(f"Milvus 连接成功 (集合: {self.collection_name})")
            return True
        except Exception as e:
            log_exception(logger, "Milvus 连接失败", e)
            self.connected = False
            return False

    def _create_collection(self):
        """创建 Milvus 集合（含 field_type 字段）"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="field_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        schema = CollectionSchema(fields, description="RAG 文档分块向量存储")
        self.collection = Collection(self.collection_name, schema)
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self.collection.create_index("vector", index_params)
        self._has_field_type = True
        logger.info(f"Milvus 集合已创建: {self.collection_name}")

    # ── 数据操作 ──

    def document_exists(self, filename: str) -> bool:
        """检查知识库中是否已存在同名文件"""
        if not self.connected or not self.collection:
            return False
        try:
            self.collection.load()
            # 在 Milvus 的 text 字段中搜索文件名标记
            # 约定：索引时在 text 的 chunk 中加入 "【文件名】xxx.pdf" 标记
            expr = f'text like "%【文件名】{filename}%"'
            results = self.collection.query(expr=expr, output_fields=["text"], limit=1)
            return len(results) > 0
        except Exception:
            return False

    def list_documents(self) -> list[str]:
        """列出知识库中已索引的所有文件名"""
        if not self.connected or not self.collection:
            return []
        try:
            self.collection.load()
            # 查询所有包含【文件名】标记的 chunk
            expr = 'text like "%【文件名】%"'
            results = self.collection.query(expr=expr, output_fields=["text"], limit=1000)
            seen = set()
            for r in results:
                import re
                m = re.search(r'【文件名】(.+?)(?:\n|$)', r["text"])
                if m:
                    seen.add(m.group(1).strip())
            return sorted(seen)
        except Exception as e:
            logger.debug(f"列出文档失败: {e}")
            return []

    def is_document_indexed(self, filename: str) -> bool:
        """检查文档是否已经被索引"""
        return filename in self.list_documents()

    def index_documents(self, chunks: list[dict], filename: str = "") -> bool:
        """
        索引文档分块
        chunks: [{"text": str, "page": int, "chunk_id": int, "field_type": str}, ...]
        """
        if not self.model:
            if not self.load_model():
                return False
        if not self.connected:
            if not self.connect_milvus():
                return False

        try:
            texts = [c["text"] for c in chunks]
            # 给每个chunk文本加上公司/文件名前缀，确保检索时能识别来源
            if filename:
                # 从文件名提取公司名标识
                company_tag = ""
                if "兴图新科" in filename or "招股说明书1" in filename or "招股意向书" in filename:
                    company_tag = "【公司】武汉兴图新科电子股份有限公司\n"
                elif "力源信息" in filename or "招股说明书2" in filename:
                    company_tag = "【公司】武汉力源信息技术股份有限公司\n"
                else:
                    company_tag = f"【文件名】{filename}\n"
                
                # 每个chunk都加前缀（不只是第一个）
                for i in range(len(texts)):
                    texts[i] = f"{company_tag}{texts[i]}"
            logger.info(f"开始嵌入 {len(texts)} 个文档块...")
            embs = self.embed_documents(texts)

            # field_type 兼容：旧 chunks 可能没有 field_type 字段
            entities = [
                [c["chunk_id"] for c in chunks],
                [c["page"] for c in chunks],
                [c.get("field_type", "body") for c in chunks],
                [c["text"] for c in chunks],
                embs,
            ]

            self.collection.insert(entities)
            self.collection.flush()
            self.chunks_metadata.extend(chunks)

            # 索引完成后自动构建 BM25
            try:
                import jieba  # noqa: F401
                from rank_bm25 import BM25Okapi  # noqa: F401
                # 用所有已索引的 chunks 重建 BM25
                all_chunks = self.chunks_metadata
                if all_chunks:
                    self.build_bm25_index(all_chunks)
            except ImportError:
                logger.debug("BM25 依赖未安装，跳过 BM25 索引构建")

            count = self.collection.num_entities
            logger.info(f"文档索引完成: {len(chunks)} 个块, 集合总数={count}")
            return True
        except Exception as e:
            log_exception(logger, "文档索引失败", e)
            return False

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        相似度搜索（支持字段加权）
        返回: [{"text": str, "page": int, "chunk_id": int, "score": float, "field_type": str}, ...]
        """
        if not self.connected or not self.model:
            return []

        try:
            query_emb = self.embed_query(query)
            self.collection.load()
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            # 尝试获取 field_type（旧集合可能没有该字段）
            if self._has_field_type:
                output_fields = ["chunk_id", "page", "text", "field_type"]
            else:
                output_fields = ["chunk_id", "page", "text"]
            results = self.collection.search(
                data=[query_emb],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields,
            )

            # 字段类型加权系数
            field_boost = {"title": 1.3, "abstract": 1.15, "table": 1.1, "body": 1.0}

            hits = []
            for hit in results[0]:
                ft = hit.entity.get("field_type", "body") if "field_type" in output_fields else "body"
                base_score = hit.score
                boost = field_boost.get(ft, 1.0)
                hits.append({
                    "text": hit.entity.get("text"),
                    "page": hit.entity.get("page"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "score": round(base_score * boost, 4),
                    "field_type": ft,
                })

            logger.debug(f"向量搜索完成: query='{query[:50]}...', hits={len(hits)}")
            return hits
        except Exception as e:
            log_exception(logger, "向量搜索失败", e)
            return []

    # ── BM25 全文检索 ────────────────────────────────────

    def build_bm25_index(self, chunks: list[dict]):
        """从 chunks 构建 BM25 全文索引
        
        将每个 chunk 的文本分词后构建 BM25 倒排索引。
        安装依赖: pip install rank_bm25 jieba
        """
        if not chunks:
            logger.warning("BM25 索引构建失败: 无文本数据")
            return False
        try:
            import jieba
            from rank_bm25 import BM25Okapi

            texts = [c["text"] for c in chunks]
            # jieba 分词
            logger.info(f"构建 BM25 索引: {len(texts)} 个文档...")
            tokenized = []
            for i, text in enumerate(texts):
                tokens = list(jieba.cut(text))
                tokenized.append(tokens)
                if (i + 1) % 500 == 0:
                    logger.debug(f"  BM25 分词进度: {i+1}/{len(texts)}")

            self.bm25 = BM25Okapi(tokenized)
            self.bm25_texts = texts
            self.bm25_chunk_ids = [c["chunk_id"] for c in chunks]
            self.bm25_ready = True
            logger.info(f"BM25 索引构建完成: {len(texts)} 个文档")
            return True
        except ImportError as e:
            logger.warning(f"BM25 依赖未安装: {e}，请运行 pip install rank_bm25 jieba")
            return False
        except Exception as e:
            log_exception(logger, "BM25 索引构建失败", e)
            return False

    def search_bm25(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 全文检索（支持布尔查询/短语匹配/模糊匹配）
        
        查询语法:
          - 引号 "xxx" 精确短语匹配
          - AND / OR 布尔运算（默认空格=AND）
          - NOT -xxx 排除包含该词的文档
          - 普通关键词: BM25 相关性打分
        
        返回: [{"text": str, "page": int, "chunk_id": int, "score": float, "source": "bm25"}, ...]
        """
        if not self.bm25_ready or not self.bm25:
            return []
        try:
            import jieba

            # ── 1. 解析查询语法 ──
            phrases, must_have, must_not, should_have = self._parse_boolean_query(query)

            # ── 2. BM25 基础打分 ──
            # 对 should_have 中的关键词做 BM25 打分
            query_tokens = list(jieba.cut(query))
            # 过滤掉布尔运算符
            stop_ops = {'AND', 'OR', 'NOT', 'and', 'or', 'not'}
            query_tokens = [t for t in query_tokens if t not in stop_ops]
            scores = self.bm25.get_scores(query_tokens)

            # ── 3. 短语匹配加分 ──
            phrase_boost = [0.0] * len(scores)
            if phrases:
                for i, text in enumerate(self.bm25_texts):
                    for phrase in phrases:
                        if phrase in text:
                            phrase_boost[i] += 5.0  # 短语命中大幅加分

            # ── 4. 布尔过滤 ──
            final_scores = [0.0] * len(scores)
            for i in range(len(scores)):
                text = self.bm25_texts[i]

                # NOT 过滤: 包含排除词的文档直接跳过
                if must_not:
                    skip = False
                    for neg in must_not:
                        if neg in text:
                            skip = True
                            break
                    if skip:
                        continue

                # AND 过滤: 必须包含所有 must_have 词
                if must_have:
                    all_present = True
                    for mh in must_have:
                        if mh not in text:
                            all_present = False
                            break
                    if not all_present:
                        continue

                # 基础 BM25 分数 + 短语加分
                final_scores[i] = scores[i] + phrase_boost[i]

            # ── 5. 获取 top_k ──
            top_indices = sorted(
                range(len(final_scores)), key=lambda i: final_scores[i], reverse=True
            )[:top_k]

            hits = []
            for idx in top_indices:
                if final_scores[idx] <= 0:
                    continue
                chunk_id = self.bm25_chunk_ids[idx]
                page = 0
                for cm in self.chunks_metadata:
                    if cm.get("chunk_id") == chunk_id:
                        page = cm.get("page", 0)
                        break
                hits.append({
                    "text": self.bm25_texts[idx],
                    "page": page,
                    "chunk_id": chunk_id,
                    "score": round(float(final_scores[idx]), 4),
                    "source": "bm25",
                })
            return hits
        except Exception as e:
            log_exception(logger, "BM25 搜索失败", e)
            return []

    @staticmethod
    def _parse_boolean_query(query: str) -> tuple:
        """解析布尔查询语法
        
        Returns:
            (phrases, must_have, must_not, should_have)
            - phrases: 精确短语列表 ["xxx", ...]
            - must_have: 必须包含的词 (AND)
            - must_not: 必须不包含的词 (NOT)
            - should_have: 至少包含一个的词 (OR)
        """
        import re

        phrases = []
        must_not = []
        remaining = query

        # 提取引号短语 "xxx"
        for m in re.finditer(r'"([^"]+)"', query):
            phrases.append(m.group(1))
        remaining = re.sub(r'"[^"]+"', '', remaining)

        # 提取 NOT 词 (-xxx 或 NOT xxx)
        for m in re.finditer(r'(?:^|\s)-(\S+)', remaining):
            must_not.append(m.group(1))
        remaining = re.sub(r'(?:^|\s)-\S+', '', remaining)
        # NOT 关键字
        for m in re.finditer(r'\bNOT\s+(\S+)', remaining, re.IGNORECASE):
            must_not.append(m.group(1))
        remaining = re.sub(r'\bNOT\s+\S+', '', remaining, flags=re.IGNORECASE)

        # 剩余词: 检查是否有 OR
        remaining = remaining.strip()
        if ' OR ' in remaining.upper():
            parts = re.split(r'\s+OR\s+', remaining, flags=re.IGNORECASE)
            should_have = [p.strip() for p in parts if p.strip()]
            must_have = []
        else:
            # 默认 AND: 所有词都必须出现
            tokens = remaining.split()
            must_have = [t for t in tokens if len(t) >= 1]
            should_have = []

        return phrases, must_have, must_not, should_have

    def hybrid_search(self, query: str, top_k: int = 12,
                      alpha: float = 0.4, rrf_k: int = 60) -> list[dict]:
        """向量 + BM25 混合检索 (RRF 融合)
        
        Args:
            query: 查询文本
            top_k: 最终返回条数
            alpha: BM25 权重（0-1），向量权重=1-alpha
            rrf_k: RRF 融合常数（越大越平滑）
        Returns:
            [{"text": str, "page": int, "chunk_id": int, 
              "score": float, "vector_score": float, "bm25_score": float}, ...]
        """
        # 1. 向量检索
        vector_results = self.search(query, top_k=top_k * 3)
        # 2. BM25 检索
        bm25_results = self.search_bm25(query, top_k=top_k * 3)

        if not vector_results and not bm25_results:
            return []
        if not bm25_results:
            return vector_results[:top_k]
        if not vector_results:
            return bm25_results[:top_k]

        # 3. RRF 融合
        rrf_scores = {}  # chunk_id -> score

        for rank, hit in enumerate(vector_results):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1 - alpha) / (rrf_k + rank + 1)

        for rank, hit in enumerate(bm25_results):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + alpha / (rrf_k + rank + 1)

        # 4. 按 RRF 得分排序
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])

        # 5. 组装结果
        chunk_map = {}
        for hit in vector_results:
            chunk_map[hit["chunk_id"]] = hit
        for hit in bm25_results:
            if hit["chunk_id"] not in chunk_map:
                chunk_map[hit["chunk_id"]] = hit

        merged = []
        for cid, rrf_score in ranked[:top_k]:
            if cid in chunk_map:
                item = dict(chunk_map[cid])
                orig_vector_score = item.get("score", 0)
                item["score"] = round(rrf_score, 4)
                item["vector_score"] = orig_vector_score
                item["bm25_score"] = next(
                    (h["score"] for h in bm25_results if h["chunk_id"] == cid), 0
                )
                merged.append(item)

        logger.debug(f"混合检索: query='{query[:50]}', alpha={alpha}, "
                     f"向量={len(vector_results)}, BM25={len(bm25_results)}, "
                     f"融合={len(merged)}条")
        return merged

    def count(self) -> int:
        """获取向量数量"""
        if not self.connected or not self.collection:
            return 0
        try:
            self.collection.load()
            return self.collection.num_entities
        except Exception as e:
            log_exception(logger, "获取向量数量失败", e)
            return 0

    # ── 先连接，再删除（防止 ConnectionNotExistException）──

    def drop_collection(self):
        """删除集合（先确保已连接再删）"""
        try:
            # 确保连接存在
            if not self.connected:
                logger.info("drop_collection: 尚未连接，先建立连接")
                connections.connect(
                    alias="default",
                    host=self.milvus_host,
                    port=self.milvus_port,
                    timeout=10,
                )
                self.connected = True

            if utility.has_collection(self.collection_name):
                self.collection = None
                utility.drop_collection(self.collection_name)
                logger.info(f"Milvus 集合已删除: {self.collection_name}")
            else:
                logger.info(f"集合不存在，无需删除: {self.collection_name}")
        except Exception as e:
            log_exception(logger, "删除 Milvus 集合失败", e)

    def disconnect(self):
        """断开 Milvus 连接"""
        if self.connected:
            try:
                connections.disconnect("default")
                self.connected = False
                logger.info("Milvus 连接已断开")
            except Exception as e:
                log_exception(logger, "断开 Milvus 连接失败", e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
