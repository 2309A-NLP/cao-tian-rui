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

    @staticmethod
    def _clean_cell(cell: str) -> str:
        """清理表格单元格：去空白/换行/Tab，统一空格"""
        import re
        c = cell.strip() if cell else ""
        c = re.sub(r'\s+', '', c)  # 去除内部所有空白/换行
        return c

    @staticmethod
    def _norm_label(label: str) -> str:
        """标准化标签语义，增加同义词映射方便检索"""
        mapping = {
            '国防领域': '军用领域',
            '军用': '军用领域',
            '民用领域': '民用领域',
            '民用': '民用领域',
            '视频指挥控制类': '视频指挥控制类产品',
            '视频预警控制类': '视频预警控制类产品',
        }
        return mapping.get(label, label)

    def _table_to_natural_language(self, rows: list[list[str]], page_num: int) -> str:
        """
        将表格转为自然语言描述（v2）
        修复：换行符污染cell、多级表头解析、同义词映射、去重输出
        """
        if not rows or len(rows) < 2:
            return ""

        import re

        # Step 1: 清理所有 cell
        clean_rows = []
        for row in rows:
            cr = [self._clean_cell(c) for c in row]
            if any(cr):
                clean_rows.append(cr)
        rows = clean_rows
        if len(rows) < 2:
            return ""

        # Step 2: 识别表头行和数据列映射
        # 第一行通常是表头，可能包含年份
        header = rows[0]

        # 识别年份列：哪一列的 header 或数据中包含年份数字
        year_col_map = {}  # col_idx -> year
        for ridx, row in enumerate(rows):
            for ci, cell in enumerate(row):
                years = re.findall(r'(20\d{2})', cell)
                if years:
                    year_col_map[ci] = years[-1]

        # 没有年份列的话，尝试从表头单元格找
        if not year_col_map:
            for ci, cell in enumerate(header):
                years = re.findall(r'(20\d{2})', cell)
                if years:
                    year_col_map[ci] = years[-1]

        # Step 3: 识别多级表头层级
        # 对于 "军用领域-视频指挥控制类" 这种多级表头，第1列是大类，第2列是子类
        # 构建标签栈：按行跟踪每一列所属的标签
        col_labels_stack = {}  # col_idx -> 当前该列的前缀标签

        # 从第一行往后，记录哪些列有"标签值"而非数据
        # 遍历表头行（通常在表的前1-3行）
        label_row_count = min(3, len(rows))
        for ridx in range(label_row_count):
            row = rows[ridx]
            # 检查这一行是否是纯表头行（大部分cell无数据/年份）
            has_data = any(re.search(r'[\d]', cell) for cell in row[1:])
            if has_data:
                break  # 遇到了包含数据的第一行，停止表头解析
            for ci in range(len(row)):
                cell = row[ci]
                if cell and not re.search(r'20\d{2}', cell) and not re.search(r'^[\d\-.,%]+$', cell):
                    col_labels_stack[ci] = cell

        # Step 4: 判断函数
        def is_data(val: str) -> bool:
            if not val or val in ('-', '—', '【】', '', '－', '无'):
                return False
            return bool(re.search(r'[\d]', val))

        def is_pct_val(val: str) -> bool:
            return '%' in val

        # Step 5: 构建标签继承 map — 逐行跟踪左侧标签
        current_labels = {}  # col_idx -> 积累的标签路径
        # 从第一行开始扫描每行的第一列，构建标签树
        for ridx, row in enumerate(rows):
            # 第一列是行标签（项目名）
            label0 = row[0].strip() if row else ""
            # 第二列也可能是子标签（当第一列空时继承）
            label1 = row[1].strip() if len(row) > 1 else ""

            # 向左上追溯标签
            if label0 and not re.search(r'[\d]', label0) and label0 not in ('金额', '占比', '类型', '项目'):
                # 这是一个有效的行标签
                # 先找最近的非空左侧标签
                for ri in range(ridx, -1, -1):
                    prev_label = rows[ri][0].strip() if rows[ri] else ""
                    if prev_label and not re.search(r'[\d]', prev_label):
                        current_labels[0] = self._norm_label(prev_label)
                        break
                # 同时检查第二列是否为子标签
                if label1 and not re.search(r'[\d]', label1) and label1 not in ('金额', '占比', '类型', '项目'):
                    current_labels[1] = label1
            elif not label0 or re.search(r'[\d]', label0) or label0 in ('金额', '占比'):
                pass  # 数据行，但可能左侧标签还在 current_labels 中有效

        # Step 6: 组装句子（去除重复和冗余）
        seen_sentences = set()
        sentences = []

        for ridx in range(1, len(rows)):
            row = rows[ridx]
            if not row or not any(re.search(r'[\d]', c) for c in row[1:]):
                continue

            # 提取行标签：从第一列+第二列构建
            cell0 = row[0].strip() if row else ""
            cell1 = row[1].strip() if len(row) > 1 else ""

            # 向上追溯有效的行标签
            row_label = ""
            for ri in range(ridx, -1, -1):
                rl = rows[ri][0].strip() if rows[ri] else ""
                if rl and not re.search(r'^[\d\-.,%]+$', rl) and rl not in ('金额', '占比', '类型', '项目'):
                    row_label = self._norm_label(rl)
                    break

            # 子标签（第二列中的文本）
            sub_label = ""
            if cell1 and not re.search(r'^[\d\-.,%]+$', cell1) and cell1 not in ('金额', '占比', '类型'):
                sub_label = cell1

            # 判断是否为合计行
            is_summary = row_label in ('合计', '小计', '总计')

            # 组装完整标签
            if row_label and sub_label and sub_label not in ('金额', '占比'):
                full_label = f"{row_label}-{sub_label}"
            elif row_label:
                full_label = row_label
            elif sub_label and row_label not in ('金额', '占比'):
                full_label = sub_label
            else:
                continue

            # 列前缀标签（来自表头的列标签，如"军用领域"覆盖各列）
            col_prefix = col_labels_stack.get(0, "")

            # 遍历数据列
            data_columns_seen = {}  # col_idx -> already processed (avoid duplicate)

            for ci in range(1, len(row)):
                if ci in data_columns_seen:
                    continue
                val = row[ci].strip()
                if not val or not is_data(val):
                    continue
                if val in ('-', '—', '【】', '', '－', '无'):
                    continue

                year = year_col_map.get(ci, "")
                year_prefix = f"{year}年" if year else ""

                # 如果列有表头前缀标签（col_prefix），且full_label不包含它时才加
                if col_prefix and col_prefix not in full_label and col_prefix not in ('金额', '占比', '类型', '项目'):
                    display_label = f"{col_prefix}-{full_label}"
                else:
                    display_label = full_label

                is_pct = is_pct_val(val)

                if is_summary and not is_pct:
                    s = f"{year_prefix}{display_label}为{val}"
                elif is_summary and is_pct:
                    s = f"{year_prefix}占比为{val}"
                elif is_pct:
                    # 找同一行最近的金额列作为引用的值
                    prev_val = ""
                    for pc in range(ci - 1, 0, -1):
                        pv = row[pc].strip()
                        if pv and is_data(pv) and not is_pct_val(pv):
                            prev_val = pv
                            break
                    if prev_val:
                        s = f"{year_prefix}{display_label}为{prev_val}万元，占比{val}"
                    else:
                        s = f"{year_prefix}{display_label}占比{val}"
                else:
                    # 普通金额：判断是否含%或含"万元"
                    if '%' in val:
                        s = f"{year_prefix}{display_label}占比{val}"
                    else:
                        s = f"{year_prefix}{display_label}为{val}万元"

                # 去重
                if s not in seen_sentences:
                    seen_sentences.add(s)
                    sentences.append(s)

        # 如果句子太少或没有，就用 tab 原文作为后备
        if len(sentences) < 2:
            table_lines = []
            for row in rows:
                table_lines.append("\t".join(row))
            return "\n".join(table_lines)

        return "；".join(sentences)

    def extract_tables(self, page) -> list[str]:
        """
        提取页面中的表格
        使用 PyMuPDF 的 find_tables()：基于文本对齐检测，对无框线表格有效
        返回: 表格文本列表（每个元素先是自然语言描述，再附 tab 格式原文）
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

                # 转为自然语言描述
                nl_text = self._table_to_natural_language(clean_rows, page.number + 1)
                
                # 同时也保留 tab 格式原文（方便精确数值核对）
                table_lines = []
                for row in clean_rows:
                    table_lines.append("\t".join(row))
                tab_text = "\n".join(table_lines)

                # 合并：自然语言在前，原文在后
                combined = nl_text + "\n" + tab_text if nl_text else tab_text

                if len(combined) > 20:  # 至少有一定内容
                    tables_text.append(combined)
                    self.tables.append((page.number + 1, combined, num_rows, num_cols))
                    logger.debug(f"  发现表格 (页 {page.number+1}): {num_rows}行×{num_cols}列, {len(combined)}字符")

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

    def process(self, output_dir: Optional[str] = None,
                ocr_images: bool = False,
                tesseract_cmd: str = "tesseract") -> bool:
        """
        完整处理 PDF
        返回: 是否成功

        ocr_images: 是否对提取的图片进行 OCR 识别（优先用 RapidOCR，降级到 tesseract）
        tesseract_cmd: tesseract 命令路径（Windows 下可能需要完整路径）
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
            self._ocr_texts = []  # (page_num, ocr_text)
            total_chars = 0

            for page_num, page in enumerate(self.doc):
                # 提取文本
                page_text = page.get_text("text", sort=True)

                # 提取表格（find_tables）并将表格内容拼接到文本中
                tables = self.extract_tables(page)
                if tables:
                    table_block = "\n\n【表格】\n" + "\n---\n".join(tables)
                    page_text += table_block

                # 提取图片（已含过滤）
                images = self.extract_images(page)
                total_images_on_page = len(page.get_images(full=True))
                self._skipped_images += max(0, total_images_on_page - len(images))

                # OCR 提取的图片文字并追加到页文本
                if ocr_images and images:
                    import subprocess
                    ocr_texts = []
                    for _, img_bytes, ext, _ in images:
                        try:
                            # 将图片字节写入临时文件供 tesseract 识别
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                                tmp.write(img_bytes)
                                tmp_path = tmp.name
                            # 调用 tesseract（中文+英文）
                            result = subprocess.run(
                                [tesseract_cmd, tmp_path, "stdout", "-l", "chi_sim+eng", "--psm", "6"],
                                capture_output=True, text=True, timeout=30
                            )
                            os.unlink(tmp_path)
                            if result.returncode == 0:
                                text = result.stdout.strip()
                                if text and len(text) > 10:
                                    ocr_texts.append(text)
                                    logger.debug(f"  OCR 第{page_num+1}页图片: {len(text)}字符")
                            else:
                                logger.debug(f"  OCR 失败 (第{page_num+1}页): {result.stderr[:100]}")
                        except FileNotFoundError:
                            logger.warning(f"tesseract 未安装，跳过 OCR（第{page_num+1}页）")
                            break
                        except Exception as e:
                            logger.debug(f"  OCR 异常 (第{page_num+1}页): {e}")
                    
                    if ocr_texts:
                        ocr_block = "\n\n【图片文字】\n" + "\n---\n".join(ocr_texts)
                        page_text += ocr_block
                        self._ocr_texts.append((page_num + 1, ocr_texts))

                self.page_texts.append(page_text)
                total_chars += len(page_text)

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

    # ── 字段类型检测 ──────────────────────────────────────────

    @staticmethod
    def _detect_field_type(text: str, page_num: int, chunk_id: int) -> str:
        """检测文本块的字段类型
        
        Returns: "title" | "abstract" | "body"
        """
        import re
        text_stripped = text.strip()
        first_line = text_stripped.split('\n')[0].strip() if '\n' in text_stripped else text_stripped

        # 标题检测: 短文本 + 编号模式
        is_short = len(first_line) < 60
        title_patterns = [
            r'^第[一二三四五六七八九十\d]+[章节篇]',  # 第X章
            r'^[一二三四五六七八九十]+[、.．]',       # 一、
            r'^\d+[、.．]\s*\S',                      # 1. xxx
            r'^[（(]\s*[一二三四五六七八九十\d]+\s*[）)]', # （一）
            r'^【[^】]+】\s*$',                        # 【标题】
            r'^目\s*录$',                              # 目录
            r'^摘\s*要$',                              # 摘要
            r'^ABSTRACT',                              # ABSTRACT
        ]
        if is_short:
            for pat in title_patterns:
                if re.match(pat, first_line):
                    return "title"

        # 摘要/概述检测
        abstract_keywords = ['摘要', '概述', '本章主要', '本节主要', '主要内容',
                             '本报告', '本公司', '本次发行', '本招股说明书']
        if any(kw in text_stripped[:200] for kw in abstract_keywords):
            if len(text_stripped) < 500:
                return "abstract"

        return "body"

    # ── 文本分块 ────────────────────────────────────────────────

    def get_chunks(self, chunk_size: int = DEFAULT_CHUNK_SIZE,
                   chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict]:
        """
        将文本分块，包含元数据
        返回: [{"text": str, "page": int, "chunk_id": int, "field_type": str}, ...]
        """
        chunks = []
        chunk_id = 0
        import re

        for page_num, page_text in enumerate(self.page_texts):
            if not page_text.strip():
                continue

            # 按【表格】标记拆分，表格部分作为一个整体不切分
            parts = re.split(r'(\n【表格】\n)', page_text)

            i = 0
            while i < len(parts):
                section = parts[i]
                if not section.strip():
                    i += 1
                    continue
                if section == '\n【表格】\n':
                    # 表格标记
                    i += 1
                    table_content = parts[i] if i < len(parts) else ""
                    combined = "\n【表格】\n" + table_content
                    if len(combined.strip()) > 20:
                        chunks.append({
                            "text": combined.strip(),
                            "page": page_num + 1,
                            "chunk_id": chunk_id,
                            "field_type": "table",
                        })
                        chunk_id += 1
                    i += 1
                    continue

                # 普通文本段：正常分块
                section_len = len(section)
                start = 0
                while start < section_len:
                    end = min(start + chunk_size, section_len)

                    # 尽量在句号处断开
                    if end < section_len:
                        for sep in ["。", "！", "？", "\n", "；", ".", "!", "?"]:
                            pos = section.rfind(sep, start, end)
                            if pos > start + chunk_size // 2:
                                end = pos + 1
                                break

                    chunk_text = section[start:end].strip()
                    if chunk_text and len(chunk_text) > 20:
                        field_type = self._detect_field_type(chunk_text, page_num, chunk_id)
                        chunks.append({
                            "text": chunk_text,
                            "page": page_num + 1,
                            "chunk_id": chunk_id,
                            "field_type": field_type,
                        })
                        chunk_id += 1

                    start = end - chunk_overlap if end < section_len else section_len

                i += 1

        # 保存分块到 output（按文件名区分）
        chunk_dir = Path(self.output_dir) / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        _pdf_name = "unknown"
        if hasattr(self, 'pdf_path') and self.pdf_path:
            _pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]

        out_path = chunk_dir / f"chunks_{_pdf_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"分块已保存: {out_path}")

        # 增量追加到 all_chunks.json
        all_path = chunk_dir / "all_chunks.json"
        if all_path.exists():
            try:
                with open(all_path, encoding="utf-8") as f:
                    all_data = json.load(f)
            except Exception:
                all_data = []
        else:
            all_data = []
        all_data.extend(chunks)
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        logger.info(f"分块完成: {len(chunks)} 个块 (chunk_size={chunk_size}, overlap={chunk_overlap})")
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
