"""
PDF 文档解析模块
基于 PyMuPDF，支持表格检测（find_tables）、图片过滤、结构化文本提取
"""
import os
import json
import csv
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from logger import get_logger, log_exception

logger = get_logger("pdf")


# ─── 常量 ────────────────────────────────────────────────────────

# 图片过滤：小于此字节数视为页眉页脚小图标/logo，跳过
IMAGE_MIN_SIZE = 5000

# 图片过滤：文本区域附近多少像素内的图片视为"正文插图"而非页眉
IMAGE_PAGE_HEADER_HEIGHT = 60   # 页眉区域高度（像素），此区域内的图片跳过
IMAGE_PAGE_FOOTER_MARGIN = 40   # 距底部多少像素内的图片视为页脚，跳过

# 分块参数默认值
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


class PDFProcessor:
    """PDF 文档处理器 - 支持表格检测、图片提取、文本分块"""

    def __init__(self, pdf_path: str, output_dir: str = "output"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.doc: Optional[fitz.Document] = None
        self.page_count = 0
        self.tables = []          # [(page_num, table_text, rows, cols), ...]
        self.images = []          # [(page_num, image_bytes, ext, xref), ...]
        self.raw_text = ""        # 全文纯文本
        self.page_texts = []      # 每页文本
        self._skipped_images = 0
        self.processing_stats = {
            "pages": 0,
            "tables": 0,
            "images": 0,
            "skipped_images": 0,
            "total_chars": 0,
            "time_seconds": 0,
        }

    def load(self) -> bool:
        """加载 PDF 文档"""
        try:
            if not os.path.exists(self.pdf_path):
                logger.error(f"PDF 文件不存在: {self.pdf_path}")
                return False
            self.doc = fitz.open(self.pdf_path)
            self.page_count = len(self.doc)
            logger.info(f"PDF 加载成功: {os.path.basename(self.pdf_path)} ({self.page_count} 页)")
            return True
        except Exception as e:
            log_exception(logger, f"PDF 加载失败: {self.pdf_path}", e)
            return False

    # ── 表格检测 ────────────────────────────────────────────────

    def extract_tables(self, page) -> list[str]:
        """
        提取页面中的表格
        使用 PyMuPDF 的 find_tables()：基于文本对齐检测，对无框线表格有效
        返回: 表格文本列表（每个元素是制表符分隔的表格内容）
        """
        tables_text = []
        try:
            tf = page.find_tables()
            for table_idx, table in enumerate(tf.tables):
                # table.extract() 返回 list[list[str]] — 行×列
                rows = table.extract()
                if not rows or len(rows) < 2:
                    continue

                # 过滤全空行/列
                clean_rows = []
                for row in rows:
                    non_empty = [cell.strip() if cell else "" for cell in row]
                    if any(non_empty):
                        clean_rows.append(non_empty)

                if len(clean_rows) < 2:
                    continue

                num_cols = max(len(r) for r in clean_rows)
                num_rows = len(clean_rows)

                # 转为制表符分隔的纯文本
                table_lines = []
                for row in clean_rows:
                    table_lines.append("\t".join(row))

                table_text = "\n".join(table_lines)

                if len(table_text) > 20:  # 至少有一定内容
                    tables_text.append(table_text)
                    self.tables.append((page.number + 1, table_text, num_rows, num_cols))
                    logger.debug(f"  发现表格 (页 {page.number+1}): {num_rows}行×{num_cols}列, {len(table_text)}字符")

        except Exception as e:
            log_exception(logger, f"表格检测失败 (页 {page.number+1})", e)

        return tables_text

    # ── 图片提取 ────────────────────────────────────────────────

    def _is_valid_image(self, page, image_bytes: bytes, xref: int, page_height: float) -> bool:
        """
        判断图片是否是有价值的正文插图（而非页眉页脚/logo/图标）

        过滤规则：
        1. 太小（< IMAGE_MIN_SIZE 字节）→ 页眉logo/小图标
        2. 通过 PyMuPDF 图片包围盒判断位置（如果 API 支持）
        3. 极端宽高比（> 6:1）→ 装饰线/分割线
        """
        # 规则1：太小就跳过
        if len(image_bytes) < IMAGE_MIN_SIZE:
            return False

        # 规则2：通过 xref 获取图片包围盒（如果 PyMuPDF 版本支持）
        try:
            # 遍历页面上所有图片引用
            for img_info in page.get_images(full=True):
                if img_info[0] == xref:
                    # 获取图片的包围矩形
                    rects = page.get_image_rects(img_info[0])
                    if rects:
                        rect = rects[0]  # 取第一个包围盒
                        # 如果图片在页眉区域 → 跳过
                        if rect.y0 < IMAGE_PAGE_HEADER_HEIGHT:
                            logger.debug(f"  跳过页眉图片 (页 {page.number+1}): {len(image_bytes)}b, y0={rect.y0:.0f}")
                            return False
                        # 如果图片在页脚区域 → 跳过
                        if rect.y1 > page_height - IMAGE_PAGE_FOOTER_MARGIN:
                            logger.debug(f"  跳过页脚图片 (页 {page.number+1}): {len(image_bytes)}b, y1={rect.y1:.0f}")
                            return False
                        # 极端宽高比 → 装饰线
                        w = rect.width
                        h = rect.height
                        if w > 0 and h > 0 and (w / h > 6 or h / w > 6):
                            logger.debug(f"  跳过装饰线 (页 {page.number+1}): {len(image_bytes)}b, {w:.0f}x{h:.0f}")
                            return False
                    break
        except Exception:
            pass  # 如果 get_image_rects 不支持，只靠大小过滤

        return True

    def extract_images(self, page) -> list[tuple[int, bytes, str, int]]:
        """
        提取页面中的正文字插图（过滤页眉页脚/logo/小图标）
        返回: [(page_num, image_bytes, ext, xref), ...]
        """
        page_images = []
        page_height = page.rect.height
        try:
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                base_image = self.doc.extract_image(xref)
                if not base_image:
                    continue
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                if self._is_valid_image(page, image_bytes, xref, page_height):
                    page_images.append((page.number + 1, image_bytes, ext, xref))
                    self.images.append((page.number + 1, image_bytes, ext, xref))
                    logger.debug(f"  提取图片 (页 {page.number+1}): {len(image_bytes)} bytes, .{ext}")
        except Exception as e:
            log_exception(logger, f"图片提取失败 (页 {page.number+1})", e)
        return page_images

    # ── 保存中间产物 ────────────────────────────────────────────

    def save_tables(self, output_dir: str):
        """保存表格为 CSV + 索引 JSON"""
        table_dir = Path(output_dir) / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)

        for idx, (page_num, table_text, rows, cols) in enumerate(self.tables):
            # 保存为 .txt（制表符分隔，方便查看）
            txt_path = table_dir / f"page_{page_num:04d}_table_{idx+1}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== 第 {page_num} 页 表格 {idx+1} ({rows}行×{cols}列) ===\n\n")
                f.write(table_text)

            # 也保存为 .csv（方便程序读取）
            csv_path = table_dir / f"page_{page_num:04d}_table_{idx+1}.csv"
            try:
                lines = table_text.split("\n")
                with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    for line in lines:
                        writer.writerow(line.split("\t"))
            except Exception as e:
                logger.warning(f"  保存 CSV 失败 (表格 {idx+1}): {e}")

        # 表格索引报告
        report = []
        for idx, (page_num, table_text, rows, cols) in enumerate(self.tables):
            report.append({
                "table_id": idx + 1,
                "page": page_num,
                "rows": rows,
                "cols": cols,
                "length": len(table_text),
                "preview": table_text[:150],
            })
        report_path = Path(output_dir) / "tables_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"表格已保存: {len(self.tables)} 个表格 ({len(report)} 报告)")

    def save_images(self, output_dir: str):
        """保存过滤后的图片 + 元数据"""
        img_dir = Path(output_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        for idx, (page_num, img_bytes, ext, xref) in enumerate(self.images):
            file_path = img_dir / f"page_{page_num:04d}_{ext.upper()}_{idx+1}.{ext}"
            with open(file_path, "wb") as f:
                f.write(img_bytes)

        # 图片元数据
        report = []
        for idx, (page_num, img_bytes, ext, xref) in enumerate(self.images):
            report.append({
                "image_id": idx + 1,
                "page": page_num,
                "size_bytes": len(img_bytes),
                "format": ext,
                "xref": xref,
            })
        report_path = Path(output_dir) / "images_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 打印过滤统计
        logger.info(f"图片已保存: {len(self.images)} 张正文插图")
        if self._skipped_images > 0:
            logger.info(f"已过滤 {self._skipped_images} 张页眉页脚/小图标")

    # ── 主处理 ──────────────────────────────────────────────────

    def process(self, output_dir: Optional[str] = None) -> bool:
        """
        完整处理 PDF
        返回: 是否成功
        """
        start_time = time.time()

        if output_dir is None:
            output_dir = self.output_dir

        if not self.load():
            return False

        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            self.page_texts = []
            self._skipped_images = 0
            total_chars = 0

            for page_num, page in enumerate(self.doc):
                # 提取文本
                page_text = page.get_text("text", sort=True)

                # 提取表格（find_tables）并将表格内容拼接到文本中
                tables = self.extract_tables(page)
                if tables:
                    table_block = "\n\n【表格】\n" + "\n---\n".join(tables)
                    page_text += table_block

                self.page_texts.append(page_text)
                total_chars += len(page_text)

                # 提取图片（已含过滤）
                images = self.extract_images(page)
                total_images_on_page = len(page.get_images(full=True))
                self._skipped_images += max(0, total_images_on_page - len(images))

                if (page_num + 1) % 50 == 0:
                    logger.info(f"  处理进度: {page_num + 1}/{self.page_count} 页")

            # 合并全文
            self.raw_text = "\n".join(self.page_texts)

            # 保存中间产物
            self.save_tables(output_dir)
            self.save_images(output_dir)

            # 保存处理报告
            elapsed = time.time() - start_time
            self.processing_stats = {
                "pages": self.page_count,
                "tables": len(self.tables),
                "images": len(self.images),
                "skipped_images": self._skipped_images,
                "total_chars": total_chars,
                "time_seconds": round(elapsed, 2),
                "pdf_file": os.path.basename(self.pdf_path),
            }

            report_path = Path(output_dir) / "processing_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self.processing_stats, f, ensure_ascii=False, indent=2)

            logger.info(
                f"PDF 处理完成: {self.page_count} 页, "
                f"{len(self.tables)} 表格, {len(self.images)} 图片 "
                f"(过滤 {self._skipped_images} 张), "
                f"{total_chars:,} 字符, 耗时 {elapsed:.2f}s"
            )

            return True

        except Exception as e:
            log_exception(logger, "PDF 处理过程中发生异常", e)
            return False

    # ── 文本分块 ────────────────────────────────────────────────

    def get_chunks(self, chunk_size: int = DEFAULT_CHUNK_SIZE,
                   chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict]:
        """
        将文本分块，包含元数据
        返回: [{"text": str, "page": int, "chunk_id": int}, ...]
        """
        chunks = []
        chunk_id = 0

        for page_num, page_text in enumerate(self.page_texts):
            if not page_text.strip():
                continue

            text_len = len(page_text)
            start = 0
            while start < text_len:
                end = min(start + chunk_size, text_len)

                # 尽量在句号处断开
                if end < text_len:
                    for sep in ["。", "！", "？", "\n", "；", ".", "!", "?"]:
                        pos = page_text.rfind(sep, start, end)
                        if pos > start + chunk_size // 2:
                            end = pos + 1
                            break

                chunk_text = page_text[start:end].strip()
                if chunk_text and len(chunk_text) > 20:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num + 1,
                        "chunk_id": chunk_id,
                    })
                    chunk_id += 1

                start = end - chunk_overlap if end < text_len else text_len

        # 保存分块到 output
        chunk_dir = Path(self.output_dir) / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_file = chunk_dir / "all_chunks.json"
        chunk_data = [{"chunk_id": c["chunk_id"], "page": c["page"],
                        "text": c["text"], "text_length": len(c["text"])} for c in chunks]
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        logger.info(f"文本分块完成: {len(chunks)} 个块")
        return chunks

    # ── 资源管理 ────────────────────────────────────────────────

    def close(self):
        if self.doc:
            self.doc.close()
            self.doc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
