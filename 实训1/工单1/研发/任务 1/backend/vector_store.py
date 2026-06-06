"""
向量存储模块
基于 Milvus 向量数据库 + BGE-M3 嵌入模型（使用 transformers 直接加载）
支持 GPU 自动检测 + 嵌入进度显示 + 先连后删保护
"""
import os
import json
import time
import sys
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
    """
    BGE-M3 嵌入模型的轻量封装
    直接使用 transformers（不依赖 sentence-transformers）
    支持 GPU 自动检测
    """

    def __init__(self, model_path: str, device: str = "auto",
                 max_length: int = 8192):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.dimension = 1024  # BGE-M3 固定
        self.normalize = True
        self._actual_device = "cpu"

    @property
    def actual_device(self) -> str:
        """返回实际使用的设备"""
        return self._actual_device

    def load(self) -> bool:
        """加载模型，自动检测 GPU"""
        try:
            from transformers import AutoTokenizer, AutoModel

            logger.info(f"加载 BGE-M3: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.model.eval()

            # 自动检测最佳设备
            import torch
            if self.device == "auto":
                if torch.cuda.is_available():
                    self._actual_device = "cuda"
                    # 兼容 transformers 5.x meta device 加载策略
                    if str(next(self.model.parameters()).device) == "meta":
                        self.model.to_empty(device="cuda")
                    else:
                        self.model = self.model.to("cuda")
                    logger.info(f"检测到 CUDA (GPU: {torch.cuda.get_device_name(0)})，使用 GPU")
                else:
                    self._actual_device = "cpu"
                    logger.info("未检测到 CUDA，使用 CPU")
            elif self.device == "cuda" and torch.cuda.is_available():
                self._actual_device = "cuda"
                # 兼容 transformers 5.x meta device 加载策略
                if str(next(self.model.parameters()).device) == "meta":
                    self.model.to_empty(device="cuda")
                else:
                    self.model = self.model.to("cuda")
                logger.info(f"使用 CUDA (GPU: {torch.cuda.get_device_name(0)})")
            else:
                self._actual_device = "cpu"
                logger.info("使用 CPU")

            logger.info(f"BGE-M3 加载成功 (dim={self.dimension}, device={self._actual_device})")
            return True
        except Exception as e:
            log_exception(logger, "BGE-M3 加载失败", e)
            return False

    def _get_pooling(self, model_output, attention_mask):
        """获取 sentence embedding（mean pooling）"""
        import torch
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: list[str], batch_size: int = 32,
               show_progress: bool = True) -> np.ndarray:
        """批量编码，带进度显示"""
        import torch
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )

            with torch.no_grad():
                if self._actual_device == "cuda":
                    encoded_input = {k: v.to("cuda") for k, v in encoded_input.items()}

                model_output = self.model(**encoded_input)
                embeddings = self._get_pooling(model_output, encoded_input["attention_mask"])

                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

                # 进度显示（写 stderr 避免干扰 stdout 管道）
                if show_progress and total_batches > 1:
                    progress = min(i + batch_size, total)
                    pct = progress / total * 100
                    msg = f"\r  嵌入进度: {progress}/{total} ({pct:.0f}%)  batch {batch_num}/{total_batches}"
                    print(msg, end="", file=sys.stderr, flush=True)

        if show_progress and total > 0:
            print(f"\r  嵌入完成: {total}/{total} (100%)    ", file=sys.stderr, flush=True)

        return np.concatenate(all_embeddings, axis=0)

    def encode_query(self, text: str) -> np.ndarray:
        """编码单个查询文本（加 instruction 前缀）"""
        prefixed = f"为这个句子生成表示以用于检索相关文章：{text}"
        return self.encode([prefixed], batch_size=1, show_progress=False)[0]

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """编码文档（不加 instruction 前缀）"""
        embs = self.encode(texts, batch_size=batch_size, show_progress=True)
        return embs.tolist()


class VectorStore:
    """
    向量存储
    - 嵌入模型: BAAI/bge-m3 (1024维)
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
    ):
        self.model_path = model_path
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.device = device
        self.batch_size = batch_size
        self.dimension = 1024  # BGE-M3 固定维度

        self.model: Optional[BGEM3Embedding] = None
        self.collection: Optional[Collection] = None
        self.connected = False
        self.chunks_metadata = []

    # ── 嵌入模型 ──

    def load_model(self) -> bool:
        """加载 BGE-M3 嵌入模型"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"嵌入模型目录不存在: {self.model_path}")
                return False
            self.model = BGEM3Embedding(self.model_path, device=self.device)
            return self.model.load()
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
                logger.info(f"Milvus 集合已存在: {self.collection_name}")
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
        """创建 Milvus 集合"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="page", dtype=DataType.INT64),
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
        chunks: [{"text": str, "page": int, "chunk_id": int}, ...]
        """
        if not self.model:
            if not self.load_model():
                return False
        if not self.connected:
            if not self.connect_milvus():
                return False

        try:
            texts = [c["text"] for c in chunks]
            # 如果传入了文件名，在第一个 chunk 的文本中嵌入文件名标记
            if filename:
                texts[0] = f"【文件名】{filename}\n{texts[0]}"
            logger.info(f"开始嵌入 {len(texts)} 个文档块...")
            embs = self.embed_documents(texts)

            entities = [
                [c["chunk_id"] for c in chunks],
                [c["page"] for c in chunks],
                [c["text"] for c in chunks],
                embs,
            ]

            self.collection.insert(entities)
            self.collection.flush()
            self.chunks_metadata.extend(chunks)

            count = self.collection.num_entities
            logger.info(f"文档索引完成: {len(chunks)} 个块, 集合总数={count}")
            return True
        except Exception as e:
            log_exception(logger, "文档索引失败", e)
            return False

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        相似度搜索
        返回: [{"text": str, "page": int, "chunk_id": int, "score": float}, ...]
        """
        if not self.connected or not self.model:
            return []

        try:
            query_emb = self.embed_query(query)
            self.collection.load()
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_emb],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["chunk_id", "page", "text"],
            )

            hits = []
            for hit in results[0]:
                hits.append({
                    "text": hit.entity.get("text"),
                    "page": hit.entity.get("page"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "score": round(hit.score, 4),
                })

            logger.debug(f"向量搜索完成: query='{query[:50]}...', hits={len(hits)}")
            return hits
        except Exception as e:
            log_exception(logger, "向量搜索失败", e)
            return []

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
