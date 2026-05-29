# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
文档处理追踪器 - 保存文档处理的中间产物

负责：
- 保存原始提取文本（raw）
- 保存清洗后文本（cleaned）
- 保存分块结果（chunks）
- 保存向量化信息（vectors）
- 保存文档摘要（summary）
- 汇总处理报告（processing_reports.json）

⚠️ 常改动的地方：
1. output_dir（输出根目录），默认 "data/processed_docs"
2. 子目录名称（raw_dir, cleaned_dir, chunks_dir, vectors_dir, summary_dir）
3. JSON 文件保存时的 indent（缩进）和 ensure_ascii 设置
4. processing_reports.json 的累积方式（默认追加记录）

⚠️ 注意事项：
1. 所有保存方法会自动创建不存在的目录
2. 每处理一个文档都会单独保存 JSON 文件，便于追溯
3. 处理报告（processing_reports.json）以追加方式记录每次处理的统计信息
4. 分块保存时要求传入 LangChain Document 列表
5. 文件名基于源文件的基础名（stem）生成，避免冲突
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document


class DocumentTracker:
    """文档处理追踪器：将文档处理的各个阶段（原始、清洗、分块）保存为 JSON 文件"""

    def __init__(self, output_dir: str = "data/processed_docs"):
        """
        初始化追踪器，创建输出目录及子目录
        ⚠️ 常改动：output_dir 参数可改为绝对路径或从 config 读取
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 子目录定义
        self.raw_dir = self.output_dir / "raw"
        self.cleaned_dir = self.output_dir / "cleaned"
        self.chunks_dir = self.output_dir / "chunks"
        self.vectors_dir = self.output_dir / "vectors"
        self.summary_dir = self.output_dir / "summary"

        # 自动创建所有子目录
        for d in [self.raw_dir, self.cleaned_dir, self.chunks_dir, self.vectors_dir, self.summary_dir]:
            d.mkdir(exist_ok=True)

    def save_raw_text(self, file_name: str, texts: List[str]):
        """保存原始提取的文本（未清洗）"""
        output_file = self.raw_dir / f"{Path(file_name).stem}_raw.json"
        data = {
            "source_file": file_name,
            "timestamp": datetime.now().isoformat(),
            "text_count": len(texts),
            "texts": texts
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[文档追踪] 原始文本已保存: {output_file}")

    def save_cleaned_text(self, file_name: str, texts: List[str], stats: Dict = None):
        """保存清洗后的文本（附带统计信息，如原始段落数、清洗后段落数）"""
        output_file = self.cleaned_dir / f"{Path(file_name).stem}_cleaned.json"
        data = {
            "source_file": file_name,
            "timestamp": datetime.now().isoformat(),
            "text_count": len(texts),
            "stats": stats,
            "texts": texts
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[文档追踪] 清洗后文本已保存: {output_file}")

    def save_chunks(self, file_name: str, chunks: List[Document], strategy: str = None):
        """
        保存分块结果
        ⚠️ 常改动：可增加 chunk 的更多元数据字段
        """
        output_file = self.chunks_dir / f"{Path(file_name).stem}_chunks.json"
        data = {
            "source_file": file_name,
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "index": i,
                    "content": chunk.page_content,
                    "length": len(chunk.page_content),
                    "metadata": chunk.metadata
                }
                for i, chunk in enumerate(chunks)
            ]
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[文档追踪] 分块结果已保存: {output_file}")
        return len(chunks)

    def save_vector_info(self, file_name: str, chunk_count: int, vector_ids: List[str] = None):
        """保存向量化信息（记录哪些块被向量化以及对应的 ID）"""
        output_file = self.vectors_dir / f"{Path(file_name).stem}_vectors.json"
        data = {
            "source_file": file_name,
            "timestamp": datetime.now().isoformat(),
            "chunk_count": chunk_count,
            "vector_ids": vector_ids or []
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[文档追踪] 向量信息已保存: {output_file}")

    def save_summary(self, file_name: str, summary: str, metadata: Dict = None):
        """保存文档摘要（可用于文档级 RAG 或快速预览）"""
        output_file = self.summary_dir / f"{Path(file_name).stem}_summary.json"
        data = {
            "source_file": file_name,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "metadata": metadata or {}
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[文档追踪] 摘要已保存: {output_file}")

    def save_processing_report(self, file_name: str, full_stats: Dict):
        """
        保存完整处理报告（追加到 processing_reports.json）
        ⚠️ 注意事项：多次处理同一个文件会追加多条记录，需自行去重或按时间筛选
        """
        output_file = self.output_dir / "processing_reports.json"

        # 读取现有报告（如果存在）
        existing = []
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)

        # 追加新记录
        existing.append({
            "file": file_name,
            "timestamp": datetime.now().isoformat(),
            "stats": full_stats
        })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[文档追踪] 处理报告已更新: {output_file}")


# 全局单例实例
document_tracker = DocumentTracker()