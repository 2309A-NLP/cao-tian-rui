"""
embedding_provider.py — 嵌入模型抽象层

支持多种嵌入模型（BGE-M3, M3E 等），通过工厂模式创建。
所有嵌入模型统一提供 encode() / encode_query() 接口。
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import logging

from logger import get_logger, log_exception

logger = get_logger("embedding")


# ──────────────────────────────────────────────
#  抽象基类
# ──────────────────────────────────────────────

class BaseEmbeddingProvider(ABC):
    """嵌入模型抽象基类"""

    def __init__(self, model_path: str, device: str = "auto",
                 max_length: int = 8192, batch_size: int = 32):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._actual_device = "cpu"
        self._dimension = 768  # 默认，由子类覆盖

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def actual_device(self) -> str:
        return self._actual_device

    @abstractmethod
    def load(self) -> bool:
        """加载模型"""
        ...

    @abstractmethod
    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """批量编码文本"""
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """编码单个查询文本"""
        return self.encode([text], show_progress=False)[0]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """编码文档文本"""
        embs = self.encode(texts, show_progress=True)
        return embs.tolist()


# ──────────────────────────────────────────────
#  BGE-M3 Provider
# ──────────────────────────────────────────────

class BGEM3Embedding(BaseEmbeddingProvider):
    """
    BGE-M3 嵌入模型（BAAI/bge-m3，1024维）
    直接使用 transformers（不依赖 sentence-transformers）
    支持 GPU 自动检测
    """

    def __init__(self, model_path: str, device: str = "auto",
                 max_length: int = 8192, batch_size: int = 32):
        super().__init__(model_path, device, max_length, batch_size)
        self._dimension = 1024
        self.normalize = True
        self.tokenizer = None
        self.model = None

    def load(self) -> bool:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            logger.info(f"加载 BGE-M3: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.model.eval()

            if self.device == "auto":
                if torch.cuda.is_available():
                    self._actual_device = "cuda"
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
                if str(next(self.model.parameters()).device) == "meta":
                    self.model.to_empty(device="cuda")
                else:
                    self.model = self.model.to("cuda")
                logger.info(f"使用 CUDA (GPU: {torch.cuda.get_device_name(0)})")
            else:
                self._actual_device = "cpu"
                logger.info("使用 CPU")

            logger.info(f"BGE-M3 加载成功 (dim={self._dimension}, device={self._actual_device})")
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

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        import torch
        import sys
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size

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

                # 每个 batch 后清理 GPU 缓存，防止显存泄漏导致渐进式变慢
                if self._actual_device == "cuda":
                    torch.cuda.empty_cache()

                if show_progress and total_batches > 1:
                    progress = min(i + self.batch_size, total)
                    pct = progress / total * 100
                    msg = f"\r  嵌入进度: {progress}/{total} ({pct:.0f}%)  batch {batch_num}/{total_batches}"
                    print(msg, end="", file=sys.stderr, flush=True)

        if show_progress and total > 0:
            print(f"\r  嵌入完成: {total}/{total} (100%)    ", file=sys.stderr, flush=True)

        return np.concatenate(all_embeddings, axis=0)

    def encode_query(self, text: str) -> np.ndarray:
        """编码查询。

        注意：BGE-M3 是 instruction-free 模型（与 bge-large-zh-v1.5 不同），
        查询不需要、也不应加指令前缀。文档侧入库时也未加前缀，
        因此查询必须与文档保持同一编码方式，否则 query/doc 向量不对称会损害召回。
        """
        return self.encode([text], show_progress=False)[0]


# ──────────────────────────────────────────────
#  M3E Provider (备用，可选的嵌入模型)
# ──────────────────────────────────────────────

class M3EEmbedding(BaseEmbeddingProvider):
    """
    M3E 嵌入模型（moka-ai/m3e-base，768维）
    更轻量，适合快速实验
    """

    def __init__(self, model_path: str, device: str = "auto",
                 max_length: int = 8192, batch_size: int = 32):
        super().__init__(model_path, device, max_length, batch_size)
        self._dimension = 768
        self.normalize = True
        self.tokenizer = None
        self.model = None

    def load(self) -> bool:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            logger.info(f"加载 M3E: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.eval()

            if self.device in ("auto", "cuda") and torch.cuda.is_available():
                self._actual_device = "cuda"
                self.model = self.model.to("cuda")
                logger.info(f"M3E 使用 GPU (dim={self._dimension})")
            else:
                self._actual_device = "cpu"
                logger.info(f"M3E 使用 CPU (dim={self._dimension})")

            return True
        except Exception as e:
            log_exception(logger, "M3E 加载失败", e)
            return False

    def _mean_pooling(self, model_output, attention_mask):
        import torch
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        import torch
        import sys
        all_embeddings = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            with torch.no_grad():
                if self._actual_device == "cuda":
                    encoded_input = {k: v.to("cuda") for k, v in encoded_input.items()}
                model_output = self.model(**encoded_input)
                embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
                if show_progress and total > self.batch_size:
                    pct = min(i + self.batch_size, total) / total * 100
                    print(f"\r  嵌入进度: {pct:.0f}%", end="", file=sys.stderr, flush=True)
        if show_progress and total > 0:
            print(f"\r  嵌入完成: {total}/{total} (100%)    ", file=sys.stderr, flush=True)
        return np.concatenate(all_embeddings, axis=0)


# ──────────────────────────────────────────────
#  嵌入模型工厂
# ──────────────────────────────────────────────

class EmbeddingFactory:
    """嵌入模型工厂"""

    _registry = {
        "bge-m3": BGEM3Embedding,
        "m3e": M3EEmbedding,
    }

    @classmethod
    def create(cls, model_type: str, **kwargs) -> Optional[BaseEmbeddingProvider]:
        """
        创建嵌入模型实例

        Args:
            model_type: "bge-m3" | "m3e"
            kwargs: 传递给构造函数的参数（model_path, device 等）
        """
        provider_cls = cls._registry.get(model_type)
        if not provider_cls:
            logger.warning(f"未知的嵌入模型类型: {model_type}，可用: {list(cls._registry.keys())}")
            return None
        return provider_cls(**kwargs)

    @classmethod
    def register(cls, name: str, provider_cls):
        """注册自定义嵌入模型"""
        cls._registry[name] = provider_cls

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._registry.keys())


# ─── Langchain 适配器（供 RAGAS 评估使用） ────────────────────────

class BGEM3LangchainEmbeddings:
    """将 BGE-M3 封装为 Langchain Embeddings 接口

    用于 RAGAS 评估（answer_relevancy, context_precision 等需要 embedding 的指标）。
    复用已有的 BGEM3Embedding 实例，避免重复加载模型。
    """

    def __init__(self, model_path: str = "", device: str = "auto",
                 batch_size: int = 16, existing_model=None):
        if existing_model is not None:
            self._model = existing_model
        else:
            from config import AppConfig
            # 自动检测 config.json
            import os as _os
            _cfg_path = _os.path.join(_os.path.dirname(__file__), "config.json")
            if not _os.path.exists(_cfg_path):
                _cfg_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "config.json")
            cfg = AppConfig.load(_cfg_path) if _os.path.exists(_cfg_path) else AppConfig.load()
            path = model_path or cfg.embedding_model_path
            emb = EmbeddingFactory.create(
                model_type=cfg.embedding_model_type or "bge-m3",
                model_path=path,
                device=device,
                batch_size=batch_size,
            )
            if not emb or not emb.load():
                raise RuntimeError(f"BGE-M3 加载失败: {path}")
            self._model = emb

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Langchain 接口：批量嵌入文档"""
        embs = self._model.encode_documents(texts)
        return [e.tolist() if hasattr(e, 'tolist') else e for e in embs]

    def embed_query(self, text: str) -> list[float]:
        """Langchain 接口：嵌入单个查询"""
        return self._model.encode_query(text).tolist()

    @property
    def dimension(self) -> int:
        return self._model.dimension
