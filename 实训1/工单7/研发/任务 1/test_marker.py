"""
测试 marker-pdf 对招股书的解析效果
在 Windows gpu_env 里运行: python test_marker.py
"""
import sys
import os
import time

# 测试前几页看效果（改 full_process=True 跑全量）
TEST_PAGES = 10  # 先测10页

pdf_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "招股说明书1-无水印.pdf")
output_dir = os.path.join(os.path.dirname(__file__), "output", "marker_test")
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(pdf_path):
    print(f"PDF 不存在: {pdf_path}")
    sys.exit(1)

print(f"PDF: {pdf_path}")
print(f"输出: {output_dir}")
print(f"测试前 {TEST_PAGES} 页...\n")

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

start = time.time()

# 配置：只处理前 N 页测试
config_parser = ConfigParser({
    "output_dir": output_dir,
    "page_range": f"0-{TEST_PAGES-1}",  # 前10页
    "force_ocr": True,          # 强制 OCR（即使有文本层也做）
    "output_format": "markdown",
})

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)
rendered = converter(pdf_path)

# 用 text_from_rendered 提取文本
text, _, images = text_from_rendered(rendered)

# 保存 markdown
md_path = os.path.join(output_dir, "test_output.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(text)

elapsed = time.time() - start
print(f"完成！耗时 {elapsed:.1f}s")
print(f"输出文件: {md_path}")
print(f"Markdown 长度: {len(text)} 字符")

# 打印前2000字符预览
print("\n" + "="*60)
print("预览（前2000字符）:")
print("="*60)
print(text[:2000])
