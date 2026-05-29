# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
高级 PDF 解析器 - 使用 PyMuPDF (fitz)

负责：
- PDF 深度解析（文本提取）
- 图片提取（占位实现）
- 表格提取（占位实现）
- OCR（占位实现）
- 将解析结果转换为 LangChain Document

⚠️ 常改动的地方：
1. 如需真正的 OCR 功能，需要安装并实现 OCRProcessor（如 Tesseract、PaddleOCR）
2. 表格提取器（PDFTableExtractor）当前为空实现，可集成 camelot 或 tabula-py
3. 图片提取功能当前仅返回位置元信息，如需保存图片文件可扩展
4. parse_pdf_deep 方法中可增加更多元数据（如标题、作者、字体信息）

⚠️ 注意事项：
1. 依赖 PyMuPDF (fitz)，需安装：pip install PyMuPDF
2. 如果 fitz 不可用，所有解析功能将退化（返回空结果）
3. 当前版本仅提取纯文本，不进行 OCR（光学字符识别），扫描版 PDF 无法提取文字
4. 解析失败会记录错误日志，不影响主流程
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 检查是否有 fitz（PyMuPDF）
try:
    import fitz

    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("[警告] PyMuPDF 未安装，高级PDF解析不可用")


class PDFAdvancedParser:
    """高级 PDF 解析器（基于 PyMuPDF）"""

    def __init__(self):
        self.has_fitz = HAS_FITZ

    def parse_pdf_deep(self, pdf_path: str) -> Dict[str, Any]:
        """
        深度解析 PDF，提取每页文本
        返回格式: {"pages": [{"page_num": int, "text": str, "raw_text": str}], "tables": [], "images": []}
        ⚠️ 常改动：可增加更多页面属性（如 page.get_text("dict") 获取结构化信息）
        """
        if not self.has_fitz:
            return {"pages": [], "tables": [], "images": []}

        result = {"pages": [], "tables": [], "images": []}

        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, 1):
                # 提取纯文本（默认格式）
                text = page.get_text()
                result["pages"].append({
                    "page_num": page_num,
                    "text": text,
                    "raw_text": text
                })
            doc.close()
            logger.info(f"[Fitz] 解析成功: {Path(pdf_path).name}, {len(result['pages'])}页")
        except Exception as e:
            logger.error(f"PDF解析失败: {e}")

        return result

    def extract_images(self, pdf_path: str, output_dir: str = None) -> List[Dict]:
        """
        提取 PDF 中的图片信息（仅返回位置，不保存图片文件）
        ⚠️ 常改动：如需实际保存图片，可在此实现 fitz 的图片提取并写入磁盘
        """
        if not self.has_fitz:
            return []

        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, 1):
                for img_idx, img in enumerate(page.get_images()):
                    # 当前仅记录位置，可扩展为实际提取图片数据
                    images.append({"page": page_num, "index": img_idx})
            doc.close()
        except Exception as e:
            logger.error(f"提取图片失败: {e}")

        return images


class PDFTableExtractor:
    """表格提取器（占位实现，实际可集成 camelot 或 tabula-py）"""
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """目前返回空列表，等待实现"""
        return []  # 简化版，暂不实现


class OCRProcessor:
    """OCR 处理器（占位实现，实际可集成 Tesseract 或 PaddleOCR）"""
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.ocr = None

    def init_ocr(self):
        """初始化 OCR 引擎（需实现）"""
        pass

    def recognize_image(self, image_path: str) -> str:
        """对图片进行 OCR 识别（需实现）"""
        return ""


class AdvancedDocumentParser:
    """
    高级文档解析器（统一入口）
    ⚠️ 常改动：enable_ocr 和 use_gpu 参数当前未生效，需实现 OCR 功能后才可使用
    """

    def __init__(self, enable_ocr: bool = False, use_gpu: bool = False):
        self.pdf_parser = PDFAdvancedParser()
        self.table_extractor = PDFTableExtractor()
        self.enable_ocr = enable_ocr

    def parse_pdf_deep(self, pdf_path: str, output_dir: str = None) -> Dict[str, Any]:
        """调用 PDFAdvancedParser 进行深度解析"""
        return self.pdf_parser.parse_pdf_deep(pdf_path)

    def convert_to_documents(self, parsed_result: Dict) -> List[Document]:
        """
        将解析结果（`parse_pdf_deep` 的返回值）转换为 LangChain Document 列表
        每个页面生成一个 Document
        """
        from langchain_core.documents import Document
        docs = []
        for page in parsed_result.get("pages", []):
            doc = Document(
                page_content=page["text"],
                metadata={"source_page": page["page_num"]}
            )
            docs.append(doc)
        return docs