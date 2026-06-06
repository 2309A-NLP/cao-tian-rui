"""
测试工具模块
提供测试用的小型 PDF 生成、Mock LLM 等功能
"""
import os
import sys
import json

# 确保项目根路径在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def create_test_pdf(output_path: str) -> str:
    """
    创建一个包含中文文本和表格的测试 PDF
    返回 PDF 文件路径
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 字体设置
    font_path = os.path.join(os.path.dirname(__file__), "..", "fonts")
    # 尝试找到中文字体
    chinese_font = _find_chinese_font()
    if chinese_font:
        pdf.add_font("zh", "", chinese_font, uni=True)
        pdf.set_font("zh", "", 14)
    else:
        pdf.set_font("Courier", "", 12)

    # 第1页：标题和正文
    pdf.cell(0, 10, "测试文档 - 武汉兴图新科电子股份有限公司", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("zh", "", 11) if chinese_font else pdf.set_font("Courier", "", 10)
    text_lines = [
        "武汉兴图新科电子股份有限公司是一家专注于视频指挥和视频监控领域的",
        "高新技术企业。公司成立于2010年，注册资本为5,325万元。",
        "",
        "报告期内，公司来自军用领域的收入占比分别为82.10%、97.31%和94.84%。",
        "公司主营业务收入按客户类型分类如下：",
        "",
        "军用领域收入占比逐年上升，从82.10%增长至94.84%，",
        "显示出公司在军用市场的竞争优势。",
        "",
        "本次发行所募集的资金将用于以下项目：",
        "1. 下一代视频指挥系统研发项目",
        "2. 智能视频监控系统升级项目",
        "3. 补充流动资金",
        "",
    ]
    for line in text_lines:
        pdf.cell(0, 7, line, ln=True)

    # 第2页：表格
    pdf.add_page()
    pdf.set_font("zh", "", 14) if chinese_font else pdf.set_font("Courier", "", 12)
    pdf.cell(0, 10, "主营业务收入构成表", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("zh", "", 10) if chinese_font else pdf.set_font("Courier", "", 9)

    # 绘制表格
    col_widths = [60, 30, 30, 30]
    headers = ["客户类型", "2023年", "2022年", "2021年"]
    data = [
        ["军用领域", "11,021.81", "9,414.16", "7,802.48"],
        ["民用领域", "1,834.56", "255.36", "80.03"],
        ["合计", "12,856.37", "9,669.52", "7,882.51"],
    ]

    # 表头
    pdf.set_fill_color(200, 220, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C", fill=True)
    pdf.ln()

    # 数据行
    for row in data:
        for i, cell in enumerate(row):
            align = "C" if i > 0 else "L"
            pdf.cell(col_widths[i], 8, cell, border=1, align=align)
        pdf.ln()

    # 第3页：更多文本
    pdf.add_page()
    pdf.set_font("zh", "", 11) if chinese_font else pdf.set_font("Courier", "", 10)
    more_text = [
        "公司主要从事视频指挥和视频监控系统的研发、生产和销售。",
        "产品广泛应用于军队、公安、应急管理等领域。",
        "",
        "截至2023年末，公司拥有已授权专利56项，其中发明专利23项。",
        "公司在视频处理和指挥调度方面拥有多项核心技术。",
        "",
        "本次发行的保荐机构为华泰联合证券有限责任公司。",
    ]
    for line in more_text:
        pdf.cell(0, 7, line, ln=True)

    pdf.output(output_path)
    return output_path


def create_test_config(tmp_dir: str) -> dict:
    """创建测试用配置"""
    config = {
        "project_root": tmp_dir,
        "log_dir": os.path.join(tmp_dir, "logs"),
        "output_dir": os.path.join(tmp_dir, "output"),
        "knowledge_base_dir": os.path.join(tmp_dir, "knowledge_base"),
        "chunk_size": 256,
        "chunk_overlap": 32,
        "top_k": 5,
        "similarity_threshold": 0.0,
        "llm_provider": "mock",
        "llm_api_key": "",
        "llm_model": "mock",
        "embedding_model_path": "",
        "embedding_device": "cpu",
    }
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config


def _find_chinese_font() -> str:
    """查找系统中可用的中文字体"""
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/yahei.ttf",
        "/mnt/c/Windows/Fonts/msyhbd.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None
