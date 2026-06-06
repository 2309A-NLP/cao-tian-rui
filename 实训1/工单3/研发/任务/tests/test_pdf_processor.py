"""
测试：pdf_processor.py PDF 解析模块（不依赖真实 PDF）
使用 fpdf 生成的测试 PDF 或 Mock PyMuPDF
"""
import os
import sys
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pdf_processor import PDFProcessor


class TestPDFProcessorInit(unittest.TestCase):
    """测试 PDFProcessor 初始化"""

    def test_init_defaults(self):
        """测试默认初始化"""
        p = PDFProcessor("test.pdf", "output_test")
        self.assertEqual(p.pdf_path, "test.pdf")
        self.assertEqual(p.output_dir, "output_test")
        self.assertEqual(p._skipped_images, 0)
        self.assertIn("pages", p.processing_stats)
        self.assertEqual(p.page_count, 0)
        self.assertIsNone(p.doc)

    def test_init_sets_skipped_images(self):
        """测试 _skipped_images 在 __init__ 中已初始化"""
        p = PDFProcessor("test.pdf")
        self.assertEqual(p._skipped_images, 0)
        p._skipped_images = 5
        self.assertEqual(p._skipped_images, 5)


class TestPDFProcessorLoad(unittest.TestCase):
    """测试 PDFProcessor.load()"""

    def test_load_nonexistent_file(self):
        """测试加载不存在的 PDF 返回 False"""
        p = PDFProcessor("/tmp/nonexistent_test_file.pdf")
        result = p.load()
        self.assertFalse(result)

    def test_load_existing_pdf(self):
        """测试加载真实 PDF（使用 fpdf 生成测试文件）"""
        from fpdf import FPDF
        test_pdf = os.path.join(tempfile.mkdtemp(), "test.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Courier", "", 12)
        pdf.cell(0, 10, "Test PDF Content", ln=True)
        pdf.output(test_pdf)

        p = PDFProcessor(test_pdf)
        result = p.load()
        self.assertTrue(result)
        self.assertIsNotNone(p.doc)
        self.assertEqual(p.page_count, 1)
        p.close()

        os.remove(test_pdf)
        os.rmdir(os.path.dirname(test_pdf))


class TestPDFProcessorExtractText(unittest.TestCase):
    """测试 PDFProcessor 的文本提取逻辑（使用纯英文 PDF）"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        from fpdf import FPDF
        self.pdf_path = os.path.join(self.tmp_dir, "test.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Courier", "", 12)
        pdf.cell(0, 10, "Wuhan XingTu XinKe Electronics Co., Ltd.", ln=True)
        pdf.cell(0, 10, "Registered capital is 53.25 million yuan", ln=True)
        pdf.add_page()
        pdf.cell(0, 10, "Second page content", ln=True)
        pdf.output(self.pdf_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_process_and_get_chunks(self):
        """测试完整处理并获取分块"""
        output_dir = os.path.join(self.tmp_dir, "output")
        p = PDFProcessor(self.pdf_path, output_dir)
        success = p.process(output_dir)
        self.assertTrue(success)
        self.assertGreater(p.page_count, 0)

        # 获取分块
        chunks = p.get_chunks(chunk_size=512, chunk_overlap=0)
        self.assertGreater(len(chunks), 0)

        # 检查分块元数据
        first = chunks[0]
        self.assertIn("text", first)
        self.assertIn("page", first)
        self.assertIn("chunk_id", first)
        self.assertIsInstance(first["page"], int)
        self.assertGreater(first["page"], 0)
        self.assertIsInstance(first["chunk_id"], int)

        # 检查输出目录
        self.assertTrue(os.path.exists(os.path.join(output_dir, "chunks", "all_chunks.json")))
        p.close()

    def test_processing_stats(self):
        """测试处理统计信息"""
        output_dir = os.path.join(self.tmp_dir, "output")
        p = PDFProcessor(self.pdf_path, output_dir)
        p.process(output_dir)
        stats = p.processing_stats
        self.assertIn("pages", stats)
        self.assertIn("tables", stats)
        self.assertIn("images", stats)
        self.assertIn("total_chars", stats)
        self.assertIn("time_seconds", stats)
        self.assertGreater(stats["total_chars"], 0)
        p.close()

    def test_context_manager(self):
        """测试上下文管理器"""
        output_dir = os.path.join(self.tmp_dir, "output")
        with PDFProcessor(self.pdf_path, output_dir) as p:
            success = p.process(output_dir)
            self.assertTrue(success)
            self.assertIsNotNone(p.doc)
        # 退出上下文后 doc 应关闭
        self.assertIsNone(p.doc)

    def test_chunk_respects_size(self):
        """测试分块大小限制"""
        from fpdf import FPDF
        chunk_pdf = os.path.join(self.tmp_dir, "chunk_test.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Courier", "", 12)
        long_text = "This is a test PDF document for chunking verification. " * 30
        pdf.multi_cell(0, 7, long_text)
        pdf.add_page()
        pdf.multi_cell(0, 7, "Second page content for cross-page chunking test. " * 10)
        pdf.output(chunk_pdf)

        output_dir = os.path.join(self.tmp_dir, "output")
        p = PDFProcessor(chunk_pdf, output_dir)
        p.process(output_dir)
        # chunk_size=30 能在产生多个块的同时保证 >20 字符过滤条件
        chunks = p.get_chunks(chunk_size=30, chunk_overlap=0)
        self.assertGreater(len(chunks), 5)
        for c in chunks:
            self.assertLessEqual(len(c["text"]), 30 + 15)
        p.close()


class TestPDFProcessorImageFilter(unittest.TestCase):
    """测试图片过滤逻辑（使用 Mock）"""

    def setUp(self):
        self.p = PDFProcessor("dummy.pdf")
        self.mock_page = MagicMock()
        self.mock_page.number = 5
        self.mock_page.rect.height = 800

    def test_small_image_filtered(self):
        """测试小图片（<5KB）被过滤"""
        result = self.p._is_valid_image(
            self.mock_page, b"x" * 100, xref=1, page_height=800
        )
        self.assertFalse(result)

    def test_large_image_passes_size_check(self):
        """测试大图片通过大小检查"""
        result = self.p._is_valid_image(
            self.mock_page, b"x" * 10000, xref=999, page_height=800
        )
        self.assertTrue(result)

    def test_header_image_filtered(self):
        """测试页眉位置的图片被过滤"""
        self.mock_page.get_images.return_value = [(42,)]
        self.mock_page.get_image_rects.return_value = [
            MagicMock(y0=10, y1=50, width=100, height=30)
        ]
        result = self.p._is_valid_image(
            self.mock_page, b"x" * 10000, xref=42, page_height=800
        )
        self.assertFalse(result)

    def test_footer_image_filtered(self):
        """测试页脚位置的图片被过滤"""
        self.mock_page.get_images.return_value = [(55,)]
        self.mock_page.get_image_rects.return_value = [
            MagicMock(y0=770, y1=800, width=100, height=30)
        ]
        result = self.p._is_valid_image(
            self.mock_page, b"x" * 10000, xref=55, page_height=800
        )
        self.assertFalse(result)

    def test_wide_image_filtered(self):
        """测试极端宽高比的图片被过滤"""
        self.mock_page.get_images.return_value = [(77,)]
        self.mock_page.get_image_rects.return_value = [
            MagicMock(y0=100, y1=110, width=200, height=10)
        ]
        result = self.p._is_valid_image(
            self.mock_page, b"x" * 10000, xref=77, page_height=800
        )
        self.assertFalse(result)


class TestPDFProcessorTableExtraction(unittest.TestCase):
    """测试表格提取逻辑"""

    def setUp(self):
        self.p = PDFProcessor("dummy.pdf")
        self.mock_page = MagicMock()
        self.mock_page.number = 3

    def test_extract_tables_empty(self):
        """测试无表格页面"""
        self.mock_page.find_tables.return_value.tables = []
        tables = self.p.extract_tables(self.mock_page)
        self.assertEqual(tables, [])

    def test_extract_tables_single(self):
        """测试单个表格提取"""
        mock_table = MagicMock()
        mock_table.extract.return_value = [
            ["姓名", "年龄", "城市"],
            ["张三", "28", "北京"],
            ["李四", "32", "上海"],
        ]
        self.mock_page.find_tables.return_value.tables = [mock_table]
        tables = self.p.extract_tables(self.mock_page)
        self.assertEqual(len(tables), 1)
        self.assertIn("姓名", tables[0])
        # 验证保存到 self.p.tables
        self.assertEqual(len(self.p.tables), 1)
        page_num, text, rows, cols = self.p.tables[0]
        self.assertEqual(page_num, 4)

    def test_extract_tables_filter_empty_rows(self):
        """测试过滤全空行"""
        mock_table = MagicMock()
        mock_table.extract.return_value = [
            ["公司名称", "成立时间", "注册资本"],
            ["", "", ""],
            ["武汉兴图新科", "2010年", "5,325万元"],
        ]
        self.mock_page.find_tables.return_value.tables = [mock_table]
        tables = self.p.extract_tables(self.mock_page)
        self.assertEqual(len(tables), 1)
        lines = tables[0].split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn("武汉兴图新科", tables[0])

    def test_reject_single_row_as_table(self):
        """测试只有单行的不视为表格"""
        mock_table = MagicMock()
        mock_table.extract.return_value = [["只有一行"]]
        self.mock_page.find_tables.return_value.tables = [mock_table]
        tables = self.p.extract_tables(self.mock_page)
        self.assertEqual(tables, [])

    def test_reject_short_table(self):
        """测试长度 < 20 字符的表格被过滤"""
        mock_table = MagicMock()
        mock_table.extract.return_value = [["a"], ["b"]]
        self.mock_page.find_tables.return_value.tables = [mock_table]
        tables = self.p.extract_tables(self.mock_page)
        self.assertEqual(tables, [])


class TestPDFProcessorSave(unittest.TestCase):
    """测试保存中间产物"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.p = PDFProcessor("dummy.pdf")
        self.p.tables = [
            (1, "名称\t价格\n商品A\t100", 2, 2),
            (2, "项目\t数值\n总资产\t500万", 2, 2),
        ]
        self.p.images = [(1, b"fake_image_bytes", "png", 100)]
        self.p._skipped_images = 3

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_tables_creates_files(self):
        """测试保存表格创建文件"""
        self.p.save_tables(self.tmp_dir)
        table_dir = os.path.join(self.tmp_dir, "tables")
        self.assertTrue(os.path.exists(table_dir))
        files = os.listdir(table_dir)
        self.assertGreater(len(files), 0)
        txt_files = [f for f in files if f.endswith(".txt")]
        csv_files = [f for f in files if f.endswith(".csv")]
        self.assertGreater(len(txt_files), 0)
        self.assertGreater(len(csv_files), 0)

    def test_save_tables_report(self):
        """测试保存表格报告 JSON"""
        self.p.save_tables(self.tmp_dir)
        report_path = os.path.join(self.tmp_dir, "tables_report.json")
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(len(report), 2)
        self.assertIn("table_id", report[0])
        self.assertIn("page", report[0])

    def test_save_images_creates_files(self):
        """测试保存图片创建文件"""
        self.p.save_images(self.tmp_dir)
        img_dir = os.path.join(self.tmp_dir, "images")
        self.assertTrue(os.path.exists(img_dir))
        files = os.listdir(img_dir)
        self.assertGreater(len(files), 0)

    def test_save_images_report(self):
        """测试保存图片报告 JSON"""
        self.p.save_images(self.tmp_dir)
        report_path = os.path.join(self.tmp_dir, "images_report.json")
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(len(report), 1)


if __name__ == "__main__":
    unittest.main()
