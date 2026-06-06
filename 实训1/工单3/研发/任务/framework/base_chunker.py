"""
base_chunker.py — 文档分块策略抽象层

核心思想：
  策略模式。每种分块方式（固定窗口、句号切分、段落切分、语义切分……）
  都是一个独立的策略类，可自由切换和组合。

复用方式：
  1. 继承 BaseChunker，实现 chunk() 方法
  2. 用 @ChunkerFactory.register() 注册
  3. 业务代码：chunker = ChunkerFactory.create("sentence", chunk_size=512)
     chunks = chunker.chunk(long_text)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class Chunk:
    """
    一个文档片段。
    
    属性:
        text: 片段文本
        index: 在原始文档中的序号（从 0 开始）
        metadata: 额外信息（页码、章节标题、文件名等）
        start_char: 在原始文本中的起始字符位置
        end_char: 在原始文本中的结束字符位置
    """
    text: str
    index: int
    metadata: dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0

    def __len__(self) -> int:
        return len(self.text)


# ──────────────────────────────────────────────
# 分块策略抽象基类
# ──────────────────────────────────────────────

class BaseChunker(ABC):
    """
    分块策略抽象基类。
    
    所有分块器只需要实现一个方法：
      chunk(text, metadata) -> list[Chunk]
    
    参数:
        chunk_size: 目标块大小（字符数 / token 数，由子类决定）
        overlap: 相邻块之间的重叠字符/token 数
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) 必须小于 chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        将文本切分成多个片段。
        
        参数:
            text: 原始文本（可能是整个文档或一页内容）
            metadata: 附加到每个 Chunk 的元数据（如 {"page": 3}）
        
        返回:
            Chunk 列表，按原文顺序排列
        """
        ...

    # ── 辅助方法 ──

    def chunk_document(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        chunk() 的别名，语义更清晰。
        
        对于 PDF 等多页文档，可以逐页调用此方法，
        每页传入不同的 metadata（如 {"page": 1, "source": "report.pdf"}）。
        """
        return self.chunk(text, metadata)

    def __call__(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """直接调用实例即可分块"""
        return self.chunk(text, metadata)


# ──────────────────────────────────────────────
# 内置示例策略
# ──────────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    """
    固定窗口大小分块（最简单的基线策略）。
    按 chunk_size 字符数硬切，不做语义边界检测。
    
    适用场景:
        - 快速原型
        - 纯英文文本
        - 不需要语义完整性的简单检索
    """

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        meta = metadata or {}
        chunks = []
        start = 0
        idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            # 如果不是最后一块，往后找最近的换行符
            if end < text_len:
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start + self.chunk_size // 2:
                    end = newline_pos + 1

            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                metadata={**meta},
                start_char=start,
                end_char=end,
            ))

            # 带重叠的下一块起点
            next_start = end - (self.overlap if idx > 0 else 0)
            if next_start <= start:
                next_start = end  # 防止死循环
            start = next_start
            idx += 1

        return chunks


class SentenceChunker(BaseChunker):
    """
    按句子边界分块（中文/英文均适用）。
    按句号、问号、感叹号、换行符切分后，
    再合并到接近 chunk_size 的大小。
    
    适用场景:
        - 中文文档
        - 需要语义完整块（每段是完整的几句话）
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        super().__init__(chunk_size, overlap)
        # 中文和英文的句子结束符
        self._sentence_end = "。！？.!?\n"

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        meta = metadata or {}
        # 第一步：按句子拆分
        sentences = self._split_sentences(text)

        # 第二步：合并句子到接近 chunk_size
        chunks = []
        current_sentences: list[str] = []
        current_len = 0
        char_offset = 0
        idx = 0

        def _flush():
            nonlocal current_sentences, current_len, idx, char_offset
            if not current_sentences:
                return
            chunk_text = "".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                metadata={**meta},
                start_char=char_offset - len(chunk_text),
                end_char=char_offset,
            ))
            idx += 1
            # overlap: 保留最后若干句子
            overlap_chars = 0
            keep_sentences: list[str] = []
            for s in reversed(current_sentences):
                if overlap_chars + len(s) >= self.overlap:
                    keep_sentences.insert(0, s)
                    break
                keep_sentences.insert(0, s)
                overlap_chars += len(s)
            current_sentences = keep_sentences
            current_len = sum(len(s) for s in keep_sentences)

        for sentence in sentences:
            s_len = len(sentence)
            # 如果单句长度已经超过 chunk_size，直接自成一块
            if s_len > self.chunk_size:
                _flush()
                chunks.append(Chunk(
                    text=sentence,
                    index=idx,
                    metadata={**meta},
                    start_char=char_offset,
                    end_char=char_offset + s_len,
                ))
                char_offset += s_len
                idx += 1
                current_sentences = []
                current_len = 0
                continue

            if current_len + s_len > self.chunk_size:
                _flush()
            current_sentences.append(sentence)
            current_len += s_len
            char_offset += s_len

        _flush()  # 最后一块
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """按句子结束符拆分文本"""
        sentences = []
        buf = []
        for ch in text:
            buf.append(ch)
            if ch in self._sentence_end:
                sentences.append("".join(buf))
                buf = []
        if buf:
            sentences.append("".join(buf))
        return [s.strip() for s in sentences if s.strip()]


# ──────────────────────────────────────────────
# 分块器工厂
# ──────────────────────────────────────────────

class ChunkerFactory:
    """
    分块器工厂。
    
    用法:
        @ChunkerFactory.register("fixed")
        class FixedSizeChunker(BaseChunker):
            ...
        
        chunker = ChunkerFactory.create("sentence", chunk_size=256)
    """

    _registry: dict[str, type[BaseChunker]] = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
    }

    @classmethod
    def register(cls, name: str):
        def wrapper(chunker_cls: type[BaseChunker]):
            cls._registry[name] = chunker_cls
            return chunker_cls
        return wrapper

    @classmethod
    def create(cls, name: str = "sentence", **kwargs) -> BaseChunker:
        if name not in cls._registry:
            raise ValueError(
                f"未知的分块策略: {name}。"
                f" 已注册: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_strategies(cls) -> list[str]:
        return list(cls._registry.keys())
