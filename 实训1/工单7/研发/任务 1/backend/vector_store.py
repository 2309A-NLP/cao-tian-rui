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
        batch_size: int = 16,
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
        self._bm25_meta_by_id = {}  # chunk_id -> metadata，避免 search_bm25 里 O(N) 线性扫描
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
        return self.model.encode_documents(texts)

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

            # 连接后从已有数据初始化 BM25
            self._init_bm25_from_milvus()
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

    def index_documents(self, chunks: list[dict], filename: str = None,
                       config=None) -> bool:
        """批量索引文档块（嵌入 + 存入 Milvus + 自动构建 BM25）

        防崩溃策略：
        1. 小 batch 嵌入（SAFE_BATCH_SIZE=8），每个 batch 独立 try/except
        2. 单个 batch 失败不影响已成功入库的数据
        3. 嵌入或插入失败的 chunk 自动跳过并记录，不会导致整个进程崩溃
        """
        if not self.model:
            if not self.load_model():
                return False
        if not self.connected:
            if not self.connect_milvus():
                return False

        SAFE_BATCH_SIZE = 8  # 小 batch 嵌入，防止单个超大 chunk 炸掉 GPU
        total = len(chunks)
        logger.info(
            f"索引开始: {total} 个块, "
            f"模型设备={self.model.actual_device}, "
            f"batch_size={self.batch_size}, "
            f"safe_batch={SAFE_BATCH_SIZE}"
        )

        # 给每个chunk文本加上公司/文件名前缀，确保检索时能识别来源
        texts = []
        company_tag = ""
        if filename:
            if "兴图新科" in filename or "招股说明书1" in filename or "招股意向书" in filename:
                company_tag = "【公司】武汉兴图新科电子股份有限公司\n"
            elif "力源信息" in filename or "招股说明书2" in filename:
                company_tag = "【公司】武汉力源信息技术股份有限公司\n"
            else:
                company_tag = f"【文件名】{filename}\n"
        for c in chunks:
            texts.append(f"{company_tag}{c['text']}")

        succeeded_count = 0
        failed_count = 0

        # 分批嵌入 + 写入，每批独立 try/except
        start_time = time.time()
        for batch_start in range(0, total, SAFE_BATCH_SIZE):
            batch_end = min(batch_start + SAFE_BATCH_SIZE, total)
            batch_chunks = chunks[batch_start:batch_end]
            batch_texts = texts[batch_start:batch_end]

            try:
                # 嵌入当前 batch
                batch_embs = self.embed_documents(batch_texts)
                logger.debug(
                    f"  batch [{batch_start+1}-{batch_end}/{total}] "
                    f"嵌入完成, {len(batch_chunks)} 个块"
                )

                # 入库
                entities = [
                    [c["chunk_id"] for c in batch_chunks],
                    [c["page"] for c in batch_chunks],
                    [c.get("field_type", "body") for c in batch_chunks],
                    [c["text"] for c in batch_chunks],
                    batch_embs,
                ]
                self.collection.insert(entities)
                self.collection.flush()
                self.chunks_metadata.extend(batch_chunks)

                succeeded_count += len(batch_chunks)
                logger.debug(
                    f"  batch [{batch_start+1}-{batch_end}/{total}] "
                    f"入库完成, {len(batch_chunks)} 条"
                )

            except Exception as e:
                # 单个 batch 失败：逐个重试，跳过有问题的 chunk
                failed_count += len(batch_chunks)
                logger.warning(
                    f"  batch [{batch_start+1}-{batch_end}/{total}] "
                    f"整体失败 ({e}), 尝试逐个嵌入..."
                )
                for j, (chunk, text) in enumerate(zip(batch_chunks, batch_texts)):
                    try:
                        emb = self.embed_documents([text])
                        entities = [
                            [chunk["chunk_id"]],
                            [chunk["page"]],
                            [chunk.get("field_type", "body")],
                            [chunk["text"]],
                            emb,
                        ]
                        self.collection.insert(entities)
                        self.collection.flush()
                        self.chunks_metadata.append(chunk)
                        succeeded_count += 1
                        failed_count -= 1  # 修正计数
                        logger.debug(
                            f"    第 {batch_start+1+j} 个 chunk 单独嵌入成功"
                        )
                    except Exception as e2:
                        logger.warning(
                            f"    第 {batch_start+1+j} 个 chunk (页{chunk.get('page','?')}) "
                            f"嵌入失败，已跳过: {e2}"
                        )

        elapsed = time.time() - start_time
        logger.info(
            f"索引完成: 成功={succeeded_count}, 失败={failed_count}, "
            f"耗时 {elapsed:.1f}s"
        )

        # 索引完成后自动构建 BM25
        if succeeded_count > 0:
            try:
                import jieba  # noqa: F401
                from rank_bm25 import BM25Okapi  # noqa: F401
                all_chunks = self.chunks_metadata
                if all_chunks:
                    self.build_bm25_index(all_chunks)
            except ImportError:
                logger.debug("BM25 依赖未安装，跳过 BM25 索引构建")

        count = self.collection.num_entities
        logger.info(f"Milvus 集合总数={count}")
        return succeeded_count > 0

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
            field_boost = {"title": 1.6, "abstract": 1.3, "table": 1.2, "body": 1.0}

            hits = []
            for hit in results[0]:
                # 新版 pymilvus Hit 不支持 get() 的默认值参数
                try:
                    ft = hit.entity.get("field_type")
                except (TypeError, AttributeError):
                    ft = None
                if ft is None:
                    ft = "body" if "field_type" in output_fields else "body"
                base_score = hit.score
                boost = field_boost.get(ft, 1.0)
                hits.append({
                    "text": hit.entity.get("text"),
                    "page": hit.entity.get("page"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "score": round(base_score * boost, 4),
                    "field_type": ft,
                    "source": "vector",
                })

            logger.debug(f"向量搜索完成: query='{query[:50]}...', hits={len(hits)}")
            return hits
        except Exception as e:
            log_exception(logger, "向量搜索失败", e)
            return []

    # ── BM25 全文检索 ────────────────────────────────────

    def _init_bm25_from_milvus(self):
        """从 Milvus 已有数据初始化 BM25 索引（启动时加载）"""
        if not self.connected or not self.collection:
            return
        try:
            count = self.collection.num_entities
            if count == 0:
                logger.info("Milvus 无数据，跳过 BM25 初始化")
                return
            logger.info(f"从 Milvus 加载 BM25 索引: {count} 条记录")
            results = self.collection.query(
                expr="id >= 0",
                output_fields=["chunk_id", "page", "text", "field_type"],
                limit=count + 100,
            )
            if not results:
                logger.info("Milvus 查询返回空，跳过 BM25 初始化")
                return
            # 存入 chunks_metadata（BM25 从这个列表构建）
            self.chunks_metadata = []
            for r in results:
                self.chunks_metadata.append({
                    "chunk_id": r.get("chunk_id", 0),
                    "page": r.get("page", 0),
                    "text": r.get("text", ""),
                    "field_type": r.get("field_type", "body"),
                })
            logger.info(f"加载 {len(self.chunks_metadata)} 条记录到 chunks_metadata")
            self.build_bm25_index(self.chunks_metadata)
        except Exception as e:
            logger.warning(f"从 Milvus 初始化 BM25 失败: {e}（不影响向量检索）")

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
            # 预建 chunk_id -> metadata 索引（保留首次出现，与原 break-on-first 行为一致），
            # 供 search_bm25 做 O(1) 页码/字段查找，替代每命中一次的全量线性扫描。
            self._bm25_meta_by_id = {}
            for c in chunks:
                cid = c["chunk_id"]
                if cid not in self._bm25_meta_by_id:
                    self._bm25_meta_by_id[cid] = c
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
                cm = self._bm25_meta_by_id.get(chunk_id)
                if cm:
                    page = cm.get("page", 0)
                    field_type = cm.get("field_type", "body") or "body"
                else:
                    page = 0
                    field_type = "body"

                # field_type boost for BM25 ranking
                type_boost = 0.0
                if field_type == "title":
                    type_boost = 1.2
                elif field_type == "abstract":
                    type_boost = 0.8
                elif field_type == "table":
                    type_boost = 0.5

                score_value = float(final_scores[idx]) + type_boost
                hits.append({
                    "text": self.bm25_texts[idx],
                    "page": page,
                    "chunk_id": chunk_id,
                    "score": round(score_value, 4),
                    "field_type": field_type,
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
            # 中文查询无空格分隔（单个长token）→ 不设 must_have 硬过滤
            # 因为 BM25 打分 jieba 分词后会做语义匹配，AND 过滤反而把
            # 含 "发行股数" 但不含 "总股本" 的 chunk 全滤掉了
            if len(tokens) == 1 and len(tokens[0]) >= 8:
                must_have = []
            else:
                must_have = [t for t in tokens if len(t) >= 1]
            should_have = []

        return phrases, must_have, must_not, should_have

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
