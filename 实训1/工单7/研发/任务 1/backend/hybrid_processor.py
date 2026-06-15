"""
Hybrid PDF Processor v5 - Qwen3-VL

页面分类智能处理:
- pure_text: pymupdf直接提取（高效，无需API）
- table/image/mixed: Qwen3-VL-Plus 多模态提取（能区分占比 vs 增长率）
- 保存 VL 中间产物（PNG + Qwen3-VL markdown）
- 记录章节标题元数据
- 支持 300 页 VL API 配额
- 输出 Milvus 兼容的 Chunk 结构
"""
import os
import json
import time
import base64
import re
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field, asdict

import fitz

# ─── 常量 ────────────────────────────────────────────────────────

DEFAULT_API_BASE = "https://app-u613z0mda075e806.aistudio-app.com"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
BATCH_SIZE = 10
MAX_PAGES_PER_FILE = 300  # API限制：单文件最多300页VL调用（覆盖招股书全量页）

# 页面分类阈值
TEXT_DENSITY_THRESHOLD = 0.08  # 招股书正文页密度约 0.16~0.31，封面约0.16，图片页约 0.09
TABLE_ROWS_MIN = 3
IMAGE_MIN_SIZE = 5000

# 标题检测正则
# 招股书常见章节标题格式
SECTION_PATTERNS = [
    r'^第[一二三四五六七八九十]+节\s+(.+)$',          # 第一节 概览
    r'^第[一二三四五六七八九十]+章\s+(.+)$',           # 第一章 总则
    r'^\d+[、\.\s]\s*(.+)$',                           # 1、xxx / 1. xxx
    r'^[（(][一二三四五六七八九十][)）]\s*(.+)$',      # （一）xxx
    r'^[（(]\d+[)）]\s*(.+)$',                          # （1）xxx
]


# ─── 数据结构 ─────────────────────────────────────────────────────

@dataclass
class Chunk:
    """统一的数据块结构，兼容 Milvus"""
    page: int
    content_type: str  # "pure_text" | "table" | "image" | "mixed"
    content: str
    raw_html: Optional[str] = None
    image_caption: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


# ─── 目录解析器 ──────────────────────────────────────────────────

class TOCParser:
    """目录解析器：从 PDF 目录页解析出标题-页码映射表

    支持招股书/年报的常见目录格式：
      - 第一节 概览 ................................................................ 24
      - 一、本次发行的基本情况 ................................................... 24
      - 1、xxx ...................................................................... 25

    Returns:
        [{"level": 0|1|2, "title": str, "page": int}, ...]
        level=0: 一级章节 (第一节/第一章)
        level=1: 二级标题 (一、/1.)
        level=2: 三级标题
    """

    # 招股书常见页码：目录页在第6-9页（1-1-6 到 1-1-9）
    TOC_PAGE_RANGE = range(5, 11)  # 0-indexed: page 5-10

    def parse(self, doc) -> list[dict]:
        """从 PDF 文档中提取目录"""
        toc_entries = []

        for page_num in self.TOC_PAGE_RANGE:
            if page_num >= len(doc):
                break
            page = doc[page_num]
            text = page.get_text("text")
            entries = self._parse_toc_text(text)
            toc_entries.extend(entries)

        # 去重 + 按页码排序
        seen = set()
        unique = []
        for e in toc_entries:
            key = (e["title"], e["page"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def _parse_toc_text(self, text: str) -> list[dict]:
        """解析单页目录文本"""
        entries = []
        # 按行处理
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 跳过页眉
            if self._is_header(line):
                continue

            # 匹配 "标题 .... 页码" 格式
            # 用 .... 或 ... 或连续空格做分隔
            m = re.match(r'^(.+?)[\.\s]{3,}(\d+)$', line)
            if m:
                title = m.group(1).strip()
                page = int(m.group(2))

                # 去掉页码中的虚点
                title = re.sub(r'\s*\.\s*$', '', title).strip()

                # 判断层级
                level = self._detect_level(title)
                if title and page > 0:
                    entries.append({
                        "level": level,
                        "title": title,
                        "page": page,
                    })
        return entries

    @staticmethod
    def _is_header(text: str) -> bool:
        """检查是否为页眉"""
        headers = [
            "目录", "目  录",
            "武汉兴图新科电子", "招股说明书",
            "1-1-",
        ]
        for h in headers:
            if h in text:
                return True
        return False

    @staticmethod
    def _detect_level(title: str) -> int:
        """根据标题格式判断层级"""
        # 一级: 第X节 / 第X章
        if re.match(r'^第[一二三四五六七八九十]+[节章]', title):
            return 0
        # 二级: 一、/ 1、/ 1. / （一）
        if re.match(r'^[一二三四五六七八九十]+[、，,.]', title):
            return 1
        if re.match(r'^\d+[、,.]', title):
            return 1
        if re.match(r'^[（(][一二三四五六七八九十][）)]', title):
            return 1
        if re.match(r'^[（(]\d+[）)]', title):
            return 1
        # 三级
        return 2

    def build_page_section_map(self, doc) -> dict:
        """构建 page_num -> (section_title, subsection_title) 映射

        三段式策略：
        1. 前置内容（page 0-8）：用字体检测直接识别"重大事项提示"等非目录章节
        2. TOC 解析（从目录页）：精准解析章节-页码映射
        3. 向前填充：所有页继承最近的章节归属

        Returns:
            {page_num: (section_title, subsection_title)}
        """
        toc = self.parse(doc)
        if not toc:
            return {}

        toc.sort(key=lambda x: x["page"])

        page_map = {}
        current_section = ""
        current_subsection = ""
        total_pages = len(doc)

        # ── 第1步：TOC 解析 ──
        for entry in toc:
            level = entry["level"]
            title = entry["title"]
            toc_page = entry["page"]  # 排版页码（1-indexed）

            # 注意：已验证页眉 1-1-X = 文档 0-index page X
            # 所以 doc_page = toc_page
            doc_page = toc_page

            if level == 0:
                current_section = title
                current_subsection = ""
            elif level == 1:
                current_subsection = title

            page_map[doc_page] = (current_section, current_subsection)

        # ── 第2步：前置内容检测（封面后的前几页）──
        # 检测 page 1~8 中是否有标题行（16.1 SimHei 或最大字号+黑体）
        # 这些不在 TOC 中但属于正式内容：本次发行概况、重大事项提示等
        for p in range(1, min(15, total_pages)):
            if p not in page_map or not page_map[p][0]:
                page = doc[p]
                title_found = self._find_front_matter_title(page)
                if title_found:
                    # 标记为一级章节（level 0）
                    current_section = title_found
                    current_subsection = ""
                    page_map[p] = (current_section, current_subsection)

        # ── 第3步：向前填充 ──
        last_section = ""
        last_subsection = ""
        for p in range(total_pages):
            if p in page_map:
                last_section, last_subsection = page_map[p]
            else:
                page_map[p] = (last_section, last_subsection)

        return page_map

    def _find_front_matter_title(self, page) -> str:
        """检测前置页中是否有一级章节标题（16.1 SimHei 或等效）

        前置内容如"重大事项提示"、"本次发行概况"等虽然不是"第X节"格式，
        但字体规格和"第X节"一致（16.1 SimHei），是正式的一级标题。
        """
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                if not line["spans"]:
                    continue
                span = line["spans"][0]
                text = span["text"].strip()
                if not text:
                    continue
                # 过滤页眉页脚
                if self._is_header(text):
                    continue
                if re.match(r'^\d+$', text):  # 纯数字页码
                    continue
                # 检测一级标题特征
                fontsize = round(span["size"], 1)
                is_heiti = "Hei" in span["font"] or "黑" in span["font"]

                # 只匹配明确的前置章节标题
                front_titles = [
                    "本次发行概况", "重大事项提示",
                ]
                if fontsize >= 14.0 and is_heiti:
                    for ft in front_titles:
                        if text.startswith(ft):
                            return text[:60]

        return ""


# ─── 标题提取器 ───────────────────────────────────────────────────

class SectionTitleExtractor:
    """从 PDF 字体/字号分析提取章节标题

    招股书排版规范：
      - SimHei 16.1 → 一级标题（第X节）
      - SimHei 13.9 → 二级标题（一、/（一）/1.）
      - SimSun 10.6 → 正文
    自适应检测：找页面中最大字号的非页眉文本行作为标题
    """

    def __init__(self):
        self._titles_cache: Dict[int, tuple] = {}
        self._toc_map: Dict[int, tuple] = {}

    def build_from_toc(self, doc):
        """从文档目录构建页码-标题映射"""
        from hybrid_processor import TOCParser
        parser = TOCParser()
        self._toc_map = parser.build_page_section_map(doc)
        if self._toc_map:
            print(f"📖 已解析目录：{len(self._toc_map)} 页的章节归属")
            sections = set()
            for p, (s, ss) in sorted(self._toc_map.items()):
                if s:
                    sections.add(s)
            if sections:
                print(f"📑 发现 {len(sections)} 个一级章节")
                for s in sorted(sections, key=self._section_sort_key):
                    print(f"   {s}")

    def build_toc_map(self, doc):
        """外部调用，构建 TOC 映射并返回"""
        parser = TOCParser()
        self._toc_map = parser.build_page_section_map(doc)
        return self._toc_map

    @staticmethod
    def _section_sort_key(title: str) -> int:
        m = re.search(r'第([一二三四五六七八九十]+)[节章]', title)
        if m:
            cn_nums = "零一二三四五六七八九十"
            num_str = m.group(1)
            total = 0
            for c in num_str:
                total = total * 10 + cn_nums.index(c)
            return total
        return 99

    def extract_titles(self, page: fitz.Page, page_num: int, doc=None) -> tuple:
        """提取当前页的章节标题信息

        策略：
        1. 优先使用 TOC 目录映射（精确）
        2. 无 TOC → 用字体大小检测（pymupdf font spans）
        3. 兜底从文本正则匹配

        Returns:
            (section_title: str, subsection_title: str)
        """
        # 优先 TOC 映射
        if page_num in self._toc_map:
            return self._toc_map[page_num]

        # 检查缓存
        if page_num in self._titles_cache:
            return self._titles_cache[page_num]

        # 用字体大小检测
        section, subsection = self._detect_by_fontsize(page)
        if not section or not subsection:
            # 兜底：文本正则
            text = page.get_text("text")
            titles = self._find_titles(text)
            if not section:
                section = titles.get("section", "")
            if not subsection:
                subsection = titles.get("subsection", "")

        # 如果还没找到章节标题，从前面页面推断
        if not section and doc and page_num > 0:
            for prev in range(max(0, page_num - 10), page_num):
                prev_page = doc[prev]
                prev_titles = self._detect_by_fontsize(prev_page)
                if prev_titles[0]:
                    section = prev_titles[0]
                    break
                # 再试文本
                if not prev_titles[0]:
                    prev_text = prev_page.get_text("text")
                    prev_found = self._find_titles(prev_text)
                    if prev_found.get("section"):
                        section = prev_found["section"]
                        break

        result = (section, subsection)
        self._titles_cache[page_num] = result
        return result

    def _detect_by_fontsize(self, page: fitz.Page) -> tuple:
        """通过字体大小检测标题

        策略：
        - 找每一行的最大字号
        - 排除页眉（SimSun 9.1，短文本）
        - 排除页码（"1-1-xxx"）
        - 最大字号 + 黑体 = 一级标题
        - 次大字号 + 黑体 = 二级标题

        Returns:
            (section: str, subsection: str)
        """
        blocks = page.get_text("dict")["blocks"]
        candidates = []  # [(fontsize, is_bold, text), ...]

        for block in blocks:
            if block["type"] != 0:  # 非文本块
                continue
            for line in block["lines"]:
                if not line["spans"]:
                    continue
                span = line["spans"][0]
                text = span["text"].strip()
                if not text:
                    continue
                # 过滤页眉
                if self._is_header_footer(text):
                    continue
                # 过滤页码行
                if re.match(r'^\d+-\d+-\d+$', text):
                    continue
                fontsize = round(span["size"], 1)
                is_heiti = "Hei" in span["font"] or "黑" in span["font"]
                candidates.append((fontsize, is_heiti, text))

        if not candidates:
            return ("", "")

        # 按字号从大到小排序
        candidates.sort(key=lambda x: -x[0])

        section = ""
        subsection = ""

        # 找最大字号的非正文行
        base_fontsize = 10.6  # 正文基线
        for fsize, is_heiti, text in candidates:
            if fsize >= 15.0 and is_heiti:  # 一级标题
                section = text[:80]
                break
            elif fsize >= 14.0 and is_heiti:
                section = text[:80]
                break
            elif fsize >= 13.0 and is_heiti:  # 可能是二级
                if not section:
                    section = text[:80]
                else:
                    subsection = text[:80]
                break

        # 找二级标题（次大字号的 SimHei）
        if not subsection:
            for fsize, is_heiti, text in candidates:
                if text == section:
                    continue
                if fsize >= 12.0 and is_heiti:
                    # 判断是否像二级标题（以数字或括号开头）
                    if re.match(r'^[一二三四五六七八九十\d（(]', text):
                        subsection = text[:80]
                        break

        return (section, subsection)

    def _find_titles(self, text: str) -> dict:
        """从页面文本中查找标题（正则兜底）"""
        lines = text.split("\n")
        section_title = ""
        subsection_title = ""

        SECTION_KEYWORDS = [
            "第一节", "第二节", "第三节", "第四节", "第五节",
            "第六节", "第七节", "第八节", "第九节", "第十节",
            "第一章", "第二章", "第三章", "第四章", "第五章",
            "第六章", "第七章", "第八章", "第九章", "第十章",
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            for kw in SECTION_KEYWORDS:
                if line.startswith(kw):
                    section_title = line[:80]
                    break
            if section_title:
                break

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if self._is_header_footer(line):
                continue
            if 10 <= len(line) <= 80:
                if re.match(r'^[\u4e00-\u9fff\d（(]', line):
                    if "，" not in line and "。" not in line and "、" not in line[:40]:
                        subsection_title = line[:80]
                        break

        return {"section": section_title, "subsection": subsection_title}

    @staticmethod
    def _is_header_footer(text: str) -> bool:
        """检查是否为页眉页脚"""
        skip_patterns = [
            r"招股说明书",
            r"武汉兴图新科电子",
            r"中泰证券",
        ]
        for pat in skip_patterns:
            if re.match(pat, text):
                return True
        # 检查是否以"1-1-"开头的页码行
        if re.match(r'^\d+-\d+-\d+$', text):
            return True
        return False


# ─── API 调用器 ─────────────────────────────────────────────────────

class Qwen3VLClient:
    """Qwen3-VL-Plus 多模态 API 客户端（基于阿里云 DashScope）
    
    用于替代 PaddleOCR-VL，对非纯文字页（表格/图表/混合页）进行多模态内容提取。
    通过 DashScope 兼容 OpenAI SDK 格式调用。
    """
    
    def __init__(self, api_key: str = "", 
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model: str = "qwen3-vl-plus"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
    
    def parse_page(self, image_path: str, page_num: int, 
                   content_type: str = "table") -> dict:
        """使用 Qwen3-VL-Plus 解析非纯文字页面
        
        Args:
            image_path: 页面图片路径 (PNG)
            page_num: 页码（仅用于日志）
            content_type: "table" | "image" | "mixed"
        Returns:
            {"markdown": str, "success": bool, "error": str}
        """
        if not self.api_key:
            return {"markdown": "", "success": False, "error": "API Key 未配置"}
        
        prompt_templates = {
            "table": """将这张图片中的所有内容整理为结构化 Markdown。

要求：
1. 文字段落：保持原文输出
2. 表格：每个表格独立输出，列名写明含义（如"金额(亿元)"、"占比"、"增长率"）
3. 如果有多个图表/子图，按**物理位置**拆分（左侧/右侧/上方/下方），每个子图单独一个 `## 图表N（位置+类型）` 区块，各自带标题和列标注
4. 提取所有注释、来源说明、脚注
5. 原始文字描述和图表数据分开输出，不要混在一起

输出格式：
- 文字段落用 `## 正文` 开头
- 每个子图用 `## 图表N（左侧占比图/右侧增长率图/上方柱状图...）` 开头，独立表格
- 表格列名必须明确标注数据含义""",

            "image": """将这张图片中的所有内容整理为结构化 Markdown。

要求：
1. 提取所有文字内容
2. 如果包含图表（柱状图、折线图、饼图、流程图、组织结构图等），按子图拆分
3. 每个子图用 `## 图表N（类型+位置）` 开头，描述图表的标题、坐标轴、数据趋势
4. 层级关系用缩进表示
5. 保留注释和说明文字

输出格式：
- 文字段落保持原文
- 每个图表独立区块""",

            "mixed": """将这张图片中的所有内容整理为结构化 Markdown。

要求：
1. 文字段落：保持原文输出，用 `## 正文` 开头
2. 表格/图表：按物理位置拆分（左侧/右侧/上方/下方），每个子图独立一个 `## 图表N（位置+类型）` 区块
3. 表格列名写清楚含义（"占比"、"增长率"、"金额"等），不要笼统写成"数量/百分比"
4. 来源说明、脚注单独列出
5. 文字描述和图表数据分开，不要混在一起

输出格式：
- `## 正文` → 纯文字段落
- `## 图表1（左侧占比图）` → 子图1的数据
- `## 图表2（右侧增长率图）` → 子图2的数据
- `## 来源` → 来源说明"""
        }
        
        prompt = prompt_templates.get(content_type, prompt_templates["table"])
        return self._call_api(image_path, prompt, page_num)
    
    def _call_api(self, image_path: str, prompt: str, page_num: int) -> dict:
        """调用 DashScope API"""
        import base64
        import httpx
        
        try:
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            
            img_base64 = f"data:image/png;base64,{img_data}"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_base64}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            }
            
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
            
            content = data["choices"][0]["message"]["content"]
            
            if content and content.strip():
                return {"markdown": content.strip(), "success": True, "error": ""}
            else:
                return {"markdown": "", "success": False, "error": "模型返回空内容"}
                
        except httpx.TimeoutException:
            return {"markdown": "", "success": False, "error": "API 请求超时"}
        except httpx.HTTPStatusError as e:
            return {"markdown": "", "success": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
        except Exception as e:
            return {"markdown": "", "success": False, "error": str(e)}


# ─── 页面分类器 ─────────────────────────────────────────────────────

class PDFPageClassifier:
    """轻量级页面分类器"""

    def __init__(self):
        self.text_density_threshold = TEXT_DENSITY_THRESHOLD
        self.table_rows_min = TABLE_ROWS_MIN

    def classify_page(self, page: fitz.Page) -> str:
        """
        返回：pure_text | table | image | mixed

        策略：
        1. 检测真实图片（排除 1x1 像素的遮罩/占位图）
        2. 计算文字密度
        3. 文字密集 + 无图 → pure_text
           文字密集 + 有图 → mixed
           文字稀疏 + 有图 → image
           文字稀疏 + 无图 → mixed（兜底）
        """
        # Level 1: 检测真实图片（过虑 1x1 占位图和极小遮罩）
        images = self._get_real_images(page)
        has_real_images = len(images) > 0

        # Level 2: 文字密度
        text_density = self._calculate_text_density(page)

        if text_density > self.text_density_threshold:
            # 文字密集 → 纯文本或混合
            if has_real_images:
                return "mixed"
            return "pure_text"
        else:
            # 文字稀疏 → 图片或图表
            if has_real_images:
                return "image"
            return "mixed"  # 兜底

    def _get_real_images(self, page: fitz.Page) -> list:
        """获取真实图片（排除 1x1 遮罩和极小占位图）

        一些扫描版 PDF 把每个字符/背景做成 1x1 的 mask 图片，
        需要过滤掉这些假图片。
        """
        all_images = page.get_image_info(xrefs=True)
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        real_images = []
        for img in all_images:
            bbox = img.get("bbox", (0, 0, 0, 0))
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            img_area = w * h
            # 过滤极小图片（小于页面面积 1% 的通常是遮罩/字符图）
            if img_area < page_area * 0.005:
                continue
            # 过滤 1x1 占位图
            if img["width"] <= 2 and img["height"] <= 2:
                continue
            real_images.append(img)

        return real_images

    def _calculate_text_density(self, page: fitz.Page) -> float:
        """计算文字占页面的比例"""
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        if page_area == 0:
            return 0.0

        text_blocks = page.get_text("blocks")
        text_area = 0.0

        for block in text_blocks:
            # blocks 格式: (x0, y0, x1, y1, text, block_no, block_type)
            text = block[4]
            if isinstance(text, str) and text.strip():
                block_rect = fitz.Rect(block[:4])
                text_area += block_rect.width * block_rect.height

        return text_area / page_area


# ─── 主处理器 ─────────────────────────────────────────────────────

class HybridPDFProcessor:
    """混合策略 PDF 处理器"""

    def __init__(self, pdf_path: str, output_dir: str = "output",
                 qwen3_api_key: Optional[str] = None,
                 qwen3_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 qwen3_model: str = "qwen3-vl-plus"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.api_client: Optional[Qwen3VLClient] = None
        if qwen3_api_key:
            self.api_client = Qwen3VLClient(
                api_key=qwen3_api_key,
                base_url=qwen3_base_url,
                model=qwen3_model,
            )

        self.pdf = None
        self.classifier = PDFPageClassifier()
        self.title_extractor = SectionTitleExtractor()
        self.stats = {
            "total": 0,
            "pure_text": 0,
            "table": 0,
            "image": 0,
            "mixed": 0,
            "errors": 0,
            "total_time": 0,
            "vl_api_calls": 0,
            "vl_api_errors": 0,
            "vl_pure_text_skipped": 0,  # 纯文本页无需调 API
            "vl_quota_exhausted": 0,    # 超 VL 配额回退到 pymupdf 的页数
        }
        self._vl_calls_in_file = 0  # 当前文件的 VL API 调用计数器
        self.timings = []
        self._vl_archive_dir = ""
        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            from logger import get_logger
            self._logger = get_logger("hybrid")
        return self._logger
        self._vl_archive_dir = ""  # 在 process 时设置

    def load(self) -> bool:
        """加载 PDF"""
        try:
            self.pdf = fitz.open(self.pdf_path)
            # 自动解析目录，建立标题-页码映射
            self.title_extractor.build_from_toc(self.pdf)
            return True
        except Exception as e:
            print(f"❌ PDF 加载失败：{e}")
            return False

    def process(self, pages: Optional[List[int]] = None,
                resume: bool = False) -> List[Chunk]:
        """
        处理 PDF

        Args:
            pages: 指定页码列表（None 表示全部）
            resume: 是否从 checkpoint 恢复（跳过已完成的页）
        Returns:
            Chunk 列表
        """
        if not self.pdf:
            if not self.load():
                return []

        start_time = time.time()
        chunks: List[Chunk] = []

        # 不再截断页数——VL API 限制仅对非 pure_text 页单独控制
        # pure_text 页不受 VL 配额影响，全部用 pymupdf 处理
        if pages is None:
            total_pages = len(self.pdf)
            print(f"📄 文件共 {total_pages} 页，VL API 限制每文件 {MAX_PAGES_PER_FILE} 页")
            if total_pages > MAX_PAGES_PER_FILE:
                print(f"   ⚠️ 超出部分（{total_pages - MAX_PAGES_PER_FILE} 页）的表格/图片/混合页将回退到 pymupdf 纯文本提取")
            pages = list(range(total_pages))
        page_list = pages
        total = len(page_list)
        pdf_basename = Path(self.pdf_path).stem
        self._vl_archive_dir = os.path.join(self.output_dir, "hybrid_vl_output", pdf_basename)
        os.makedirs(self._vl_archive_dir, exist_ok=True)

        page_list = pages
        total = len(page_list)

        # ── Checkpoint 恢复 ──────────────────────────────────────
        checkpoint_path = os.path.join(self._vl_archive_dir, "_checkpoint.json")
        done_set: set[int] = set()
        if resume:
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, "r") as f:
                        cp = json.load(f)
                    done_set = set(cp.get("done_pages", []))
                    # 从已保存的 VL 结果中恢复 chunks
                    restored_chunks = self._load_checkpoint_chunks(checkpoint_path)
                    if restored_chunks is not None:
                        chunks = restored_chunks
                    print(f"🔄 检测到 checkpoint：{len(done_set)} 页已完成")
                    print(f"📄 将从第 {sorted(done_set)[-1]+2 if done_set else 1} 页继续")
                except Exception as e:
                    print(f"⚠️ 读取 checkpoint 失败: {e}，将从头开始")
                    done_set = set()
            else:
                print("ℹ️ 未找到 checkpoint，从头开始处理")

        print(f"📄 开始处理：{os.path.basename(self.pdf_path)}")
        print(f"📊 处理页数：{total} (API限制: {MAX_PAGES_PER_FILE}页)")
        print(f"💾 VL 中间产物保存目录：{self._vl_archive_dir}")
        if resume:
            print(f"🔄 恢复模式：跳过 {len(done_set)} 页")

        for i, page_num in enumerate(page_list):
            # 跳过已完成的页
            if page_num in done_set:
                continue

            page = self.pdf[page_num]
            page_start = time.time()

            try:
                # 1. 分类
                page_type = self.classifier.classify_page(page)
                self.stats[page_type] += 1

                # 2. 路由处理
                chunk = self._route_process(page, page_num, page_type)
                chunks.append(chunk)

                # 3. 计时
                page_time = time.time() - page_start
                self.timings.append(page_time)

                # 4. 日志
                title_info = ""
                if chunk.metadata.get("section_title"):
                    title_info = f" 章节:{chunk.metadata['section_title'][:30]}"
                if chunk.metadata.get("subsection_title"):
                    title_info += f" > {chunk.metadata['subsection_title'][:20]}"

                print(f"[{page_num+1:3d}/{total}] "
                      f"Type={page_type:10s} "
                      f"Time={page_time:.2f}s"
                      f"{title_info}")

                self.stats["total"] += 1

                # 5. 每 10 页写入 checkpoint
                if (page_num + 1) % 10 == 0:
                    self._save_checkpoint(checkpoint_path, done_set | {p for p in page_list[:page_list.index(page_num)+1]}, chunks)

            except Exception as e:
                self.stats["errors"] += 1
                print(f"❌ Page {page_num+1} 错误：{e}")
                continue

        self.stats["total_time"] = time.time() - start_time
        self._print_summary()
        self._save_vl_archive_manifest()

        # 清理 checkpoint（处理完就删掉，代表本次任务彻底完成）
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("✅ 处理完成，已清理 checkpoint")

        return chunks

    def _checkpoint_path(self) -> str:
        """返回 checkpoint 文件路径"""
        pdf_basename = Path(self.pdf_path).stem
        return os.path.join(self.output_dir, "hybrid_vl_output", pdf_basename, "_checkpoint.json")

    def _save_checkpoint(self, path: str, done_pages: set[int], chunks: list):
        """保存 checkpoint，记录已完成的页号和已生成的 chunks"""
        try:
            chunk_data = [c.to_dict() for c in chunks]
            cp = {
                "done_pages": sorted(done_pages),
                "chunks": chunk_data,
                "stats": self.stats,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cp, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存 checkpoint 失败: {e}")

    def _load_checkpoint_chunks(self, path: str) -> Optional[List[Chunk]]:
        """从 checkpoint 恢复 chunks"""
        try:
            with open(path, "r") as f:
                cp = json.load(f)
            if "chunks" in cp and cp["chunks"]:
                # 恢复统计
                cp_stats = cp.get("stats", {})
                for k, v in cp_stats.items():
                    self.stats[k] = v
                return [Chunk.from_dict(c) for c in cp["chunks"]]
        except Exception:
            pass
        return None

    def _route_process(self, page: fitz.Page, page_num: int, page_type: str) -> Chunk:
        """路由到对应处理器"""

        if page_type == "pure_text":
            # 纯文字页面：使用pymupdf提取（高效，无需API）
            return self._process_text_pymupdf(page, page_num)

        # 非纯文本页：检查 VL API 配额
        if self._vl_calls_in_file >= MAX_PAGES_PER_FILE:
            self.stats["vl_quota_exhausted"] += 1
            print(f"  ⚠️ VL 配额已用满 ({MAX_PAGES_PER_FILE}页)，"
                  f"第 {page_num+1} 页回退到 pymupdf (type={page_type})")
            return self._fallback_text(page, page_num, page_type)

        if page_type == "table":
            # 表格页面：使用PaddleOCR-VL（更准确）
            self._vl_calls_in_file += 1
            return self._process_table_paddleocr(page, page_num)
        elif page_type == "image":
            # 图表页面：使用PaddleOCR-VL（更准确）
            self._vl_calls_in_file += 1
            return self._process_image_paddleocr(page, page_num)
        else:  # mixed
            # 混合页面：使用PaddleOCR-VL 统一解析
            self._vl_calls_in_file += 1
            return self._process_mixed_hybrid(page, page_num)

    def _process_text_pymupdf(self, page: fitz.Page, page_num: int) -> Chunk:
        """纯文本处理：使用pymupdf提取（高效，无需API），自动清洗页眉页脚"""
        content = page.get_text("text")
        # 清洗页眉页脚：前两行通常是公司名+页眉和页码
        cleaned = self._clean_header_footer(content)

        # 提取标题信息
        section_title, subsection_title = self.title_extractor.extract_titles(
            page, page_num, self.pdf
        )

        return Chunk(
            page=page_num,
            content_type="pure_text",
            content=cleaned,
            metadata={
                "source": "pymupdf",
                "method": "direct_extract",
                "section_title": section_title,
                "subsection_title": subsection_title,
            }
        )

    def _clean_header_footer(self, text: str) -> str:
        """清洗页眉页脚噪声

        招股书每页前两行固定为：
          L0: 公司名称 + 招股说明书（申报稿）
          L1: 1-1-XX (页码) 或纯数字页码
        """
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # 过滤公司名+招股书头
            if re.match(r'^武汉.{0,20}股份有限公司\s+招股', stripped):
                continue
            # 过滤页码行 1-1-XX 或纯数字
            if re.match(r'^\d+-\d+-\d+$', stripped):
                continue
            if re.match(r'^\d+$', stripped) and len(stripped) < 5:
                continue
            # 过滤空行（多行空白保留一个）
            if not stripped:
                cleaned_lines.append("")
            else:
                cleaned_lines.append(stripped)

        # 去除首尾空行
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()

        # 合并连续空行为一个空行
        result = []
        prev_empty = False
        for line in cleaned_lines:
            if not line:
                if not prev_empty:
                    result.append("")
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False

        return "\n".join(result)

    def _save_page_image(self, page: fitz.Page, page_num: int) -> str:
        """将页面保存为临时图片，返回路径"""
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_dir = os.path.join(self._vl_archive_dir, "page_images")
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, f"page_{page_num:04d}.png")
        pix.save(img_path)
        return img_path

    def _load_vl_cache(self, page_num: int, api_method: str) -> Optional[dict]:
        """检查是否已有 VL 缓存结果，有则返回，无则返回 None"""
        result_dir = os.path.join(self._vl_archive_dir, "vl_results")
        result_path = os.path.join(result_dir, f"page_{page_num:04d}_{api_method}.json")
        if not os.path.exists(result_path):
            return None
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("success") and data.get("markdown", "").strip():
                return data
        except Exception:
            pass
        return None

    def _save_vl_result(self, page_num: int, vl_result: dict,
                        api_method: str) -> str:
        """保存 VL API 返回的原始结果到文件，返回保存路径"""
        result_dir = os.path.join(self._vl_archive_dir, "vl_results")
        os.makedirs(result_dir, exist_ok=True)

        result_path = os.path.join(result_dir, f"page_{page_num:04d}_{api_method}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(vl_result, f, ensure_ascii=False, indent=2)

        # 也保存 markdown 原文方便阅读
        if vl_result.get("markdown"):
            md_path = os.path.join(result_dir, f"page_{page_num:04d}_{api_method}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(vl_result["markdown"])

        return result_path

    def _extract_table_name(self, page: fitz.Page, page_num: int, markdown: str = "") -> str:
        """尝试提取表格名称"""
        # 策略1：从 markdown 结果中找表头
        if markdown:
            lines = markdown.strip().split("\n")
            for line in lines[:5]:
                line = line.strip()
                if re.match(r'^[#*|\s]*$', line):
                    continue
                if "表" in line or "Table" in line:
                    return line.strip()[:80]

        # 策略2：从 PDF 文本中找"表 X" 或 "表格 X"
        text = page.get_text("text")
        m = re.search(r'表[格]?[\s]*\d+[-\d]*[：:。\s]*(.+?)(?:\n|$)', text)
        if m:
            label = m.group(0).strip()[:80]
            return label

        return ""

    def _route_process(self, page: fitz.Page, page_num: int, page_type: str) -> Chunk:
        """路由到对应处理器"""

        if page_type == "pure_text":
            # 纯文字页面：使用pymupdf提取（高效，无需API）
            return self._process_text_pymupdf(page, page_num)

        # 非纯文本页：检查 VL API 配额
        if self._vl_calls_in_file >= MAX_PAGES_PER_FILE:
            self.stats["vl_quota_exhausted"] += 1
            print(f"  ⚠️ VL 配额已用满 ({MAX_PAGES_PER_FILE}页)，"
                  f"第 {page_num+1} 页回退到 pymupdf (type={page_type})")
            return self._fallback_text(page, page_num, page_type)

        # 非文字页统一使用 Qwen3-VL-Plus
        self._vl_calls_in_file += 1
        return self._process_non_text_qwen3(page, page_num, page_type)

    def _process_non_text_qwen3(self, page: fitz.Page, page_num: int,
                                page_type: str) -> Chunk:
        """非纯文字页面：使用 Qwen3-VL-Plus 多模态提取
        
        替代旧版 _process_table_paddleocr / _process_image_paddleocr / _process_mixed_hybrid
        三个方法。Qwen3-VL-Plus 能更好地区分多图表页面中的占比 vs 增长率等复杂布局。
        """
        if not self.api_client:
            return self._fallback_text(page, page_num, page_type)

        try:
            # 检查是否已有 VL 缓存（避免重复调用 API）
            cached_result = self._load_vl_cache(page_num, "qwen3_vl")
            if cached_result is not None:
                section_title, subsection_title = self.title_extractor.extract_titles(
                    page, page_num, self.pdf
                )
                table_name = ""
                if page_type == "table":
                    table_name = self._extract_table_name(page, page_num, cached_result["markdown"])
                return Chunk(
                    page=page_num,
                    content_type=page_type,
                    content=cached_result["markdown"],
                    metadata={
                        "source": "qwen3_vl",
                        "method": "qwen3-vl-plus(cached)",
                        "section_title": section_title,
                        "subsection_title": subsection_title,
                        "table_name": table_name,
                    }
                )

            img_path = self._save_page_image(page, page_num)
            section_title, subsection_title = self.title_extractor.extract_titles(
                page, page_num, self.pdf
            )

            # 调用 Qwen3-VL-Plus
            result = self.api_client.parse_page(
                img_path, page_num, content_type=page_type
            )
            save_path = self._save_vl_result(page_num, result, "qwen3_vl")

            if result["success"] and result["markdown"].strip():
                # 对于表格页，尝试提取表名
                table_name = ""
                if page_type == "table":
                    table_name = self._extract_table_name(page, page_num, result["markdown"])

                return Chunk(
                    page=page_num,
                    content_type=page_type,
                    content=result["markdown"],
                    metadata={
                        "source": "qwen3_vl",
                        "method": "qwen3-vl-plus",
                        "section_title": section_title,
                        "subsection_title": subsection_title,
                        "table_name": table_name,
                        "vl_archive": save_path,
                        "page_image": img_path,
                    }
                )

            # Qwen3-VL 返回空结果，回退到 pymupdf
            print(f"  ⚠️ Qwen3-VL 第{page_num+1}页返回空，回退到 pymupdf")
            return self._fallback_text(page, page_num, page_type)

        except Exception as e:
            print(f"⚠️ Qwen3-VL 第{page_num+1}页处理失败 ({page_type})，回退到 pymupdf: {e}")
            return self._fallback_text(page, page_num, page_type)


    def _fallback_text(self, page: fitz.Page, page_num: int,
                       content_type: str = "mixed") -> Chunk:
        """回退到 pymupdf 文本提取"""
        section_title, subsection_title = self.title_extractor.extract_titles(
            page, page_num, self.pdf
        )
        content = page.get_text("text")
        return Chunk(
            page=page_num,
            content_type=content_type,
            content=content,
            metadata={
                "source": "pymupdf_fallback",
                "section_title": section_title,
                "subsection_title": subsection_title,
            }
        )

    def _save_vl_archive_manifest(self):
        """保存 VL 中间产物的清单文件"""
        manifest = {
            "pdf_file": self.pdf_path,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": self.stats,
            "archive_dir": self._vl_archive_dir,
        }
        manifest_path = os.path.join(self._vl_archive_dir, "_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _print_summary(self):
        """打印统计"""
        print("\n" + "="*50)
        print("📊 处理统计")
        print("="*50)
        for key, value in self.stats.items():
            print(f"  {key:15s}: {value}")

        if self.timings:
            avg_time = sum(self.timings) / len(self.timings)
            print(f"  平均耗时        : {avg_time:.2f}s")
        if self._vl_archive_dir:
            print(f"  VL中间产物    : {self._vl_archive_dir}")
        print("="*50)

    @staticmethod
    def _strip_html_tags(text: str) -> str:
        """清洗文本中的 HTML 标签，保留纯文本内容

        处理 <div>, <img>, <table>, <tr>, <td>, <br>, <a> 等常见标签。
        - <table> 整块转换为纯文本表格（竖线分隔）
        - <img> 直接移除
        - 其他标签保留内部文本
        """
        # 1. 先把 <table>...</table> 整块提取并转换为纯文本
        def _replace_table(m):
            return "\n" + HybridPDFProcessor._html_table_to_text(m.group(0)) + "\n"

        text = re.sub(r'<table[^>]*>.*?</table>', _replace_table, text, flags=re.DOTALL)

        # 2. 移除 <img ...>
        text = re.sub(r'<img[^>]*>', '', text)
        # 3. 移除 <a ...> 和 </a>
        text = re.sub(r'<a\s+[^>]*>', '', text)
        text = re.sub(r'</a>', '', text)
        # 4. 移除剩余的 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 5. 解码 HTML 实体
        text = text.replace('&nbsp;', ' ').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&amp;', '&').replace('&quot;', '"')
        # 6. 清理多余空白
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _html_table_to_text(html: str) -> str:
        """将 HTML 表格转换为纯文本表格格式

        从 <table> 标签中提取每行每列的值，
        输出为竖线分隔的表格，方便检索。
        """
        if not html:
            return ""

        rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)
        if not rows:
            return ""

        table_lines = []
        for row in rows:
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)
            if cells:
                cleaned = []
                for cell in cells:
                    cell_text = re.sub(r'<[^>]+>', '', cell)
                    cell_text = cell_text.strip()
                    cell_text = re.sub(r'\s+', ' ', cell_text)
                    # 空单元格保留占位
                    cleaned.append(cell_text if cell_text else "-")
                table_lines.append(" | ".join(cleaned))

        return "\n".join(table_lines) if table_lines else ""

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 8000) -> str:
        """截断过长的文本，防止 GPU OOM

        Args:
            text: 原始文本
            max_chars: 最大字符数（中文场景下约等于 token 数的 1.5 倍）
        Returns:
            截断后的文本，末尾添加截断标记
        """
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        return truncated + "\n\n[...以下内容已截断，原始文本过长]"

    @staticmethod
    def merge_by_section(index_chunks: list[dict],
                         max_chars: int = 1200,
                         overlap: int = 150) -> list[dict]:
        """按章节合并相邻页，再智能分块

        流程：
        1. 同一个 section_title + subsection_title 的相邻页合并为一个文本块
        2. 合并后的文本超过 max_chars 时，在句号处智能切分（带 overlap）
        3. 表格/图片类型保持独立不合并

        Args:
            index_chunks: to_index_format() 的输出（逐页的）
            max_chars: 合并后单块最大字符数
            overlap: 切分时相邻块重叠字符数
        Returns:
            按章节合并并分块后的 chunks，chunk_id 重新编号
        """
        SEPARATORS = ["\n\n", "。", "！", "？", "\n", "；", ".", "!", "?"]

        # ── 第1步：按 (section_title, subsection_title, page_连续) 归并 ──
        merged = []  # [(text, page_min, page_max, field_type, metadata_dict), ...]
        current_text = ""
        current_pages = []
        current_meta = {}

        def flush_current():
            nonlocal current_text, current_pages, current_meta
            if current_text.strip():
                merged.append((
                    current_text.strip(),
                    min(current_pages) if current_pages else 0,
                    max(current_pages) if current_pages else 0,
                    current_pages,
                    dict(current_meta),
                ))
            current_text = ""
            current_pages = []
            current_meta = {}

        for chunk in index_chunks:
            section = chunk.get("section_title", "")
            subsection = chunk.get("subsection_title", "")
            field_type = chunk.get("field_type", "body")
            page = chunk["page"]
            text = chunk["text"]

            # 表格/图片：独立一个块，不合并
            if field_type in ("table", "image"):
                flush_current()
                merged.append((text, page, page, [page], {
                    "field_type": field_type,
                    "section_title": section,
                    "subsection_title": subsection,
                    "table_name": chunk.get("table_name", ""),
                    "source": chunk.get("source", ""),
                    "method": chunk.get("method", ""),
                }))
                continue

            # 同一标题连续：合并
            same_section = (
                current_meta.get("section_title") == section
                and current_meta.get("subsection_title") == subsection
            )
            if same_section:
                current_text += "\n" + text
                current_pages.append(page)
            else:
                flush_current()
                current_text = text
                current_pages = [page]
                current_meta = {
                    "field_type": "body",
                    "section_title": section,
                    "subsection_title": subsection,
                    "source": chunk.get("source", ""),
                    "method": chunk.get("method", ""),
                }
        flush_current()

        # ── 第2步：超长块智能切分 ──
        result = []
        new_id = 0

        for text, p_min, p_max, pages, meta in merged:
            field_type = meta.get("field_type", "body")

            # 表格/图片 或 短文本（800字以下）：保持原样
            if field_type in ("table", "image") or len(text) <= max_chars:
                result.append({
                    "text": text,
                    "page": p_min,  # 使用起始页
                    "chunk_id": new_id,
                    "field_type": field_type,
                    "section_title": meta.get("section_title", ""),
                    "subsection_title": meta.get("subsection_title", ""),
                    "table_name": meta.get("table_name", ""),
                    "source": meta.get("source", ""),
                    "method": meta.get("method", ""),
                })
                new_id += 1
                continue

            # 长文本：在句号处切分
            start = 0
            text_len = len(text)
            while start < text_len:
                end = min(start + max_chars, text_len)

                if end < text_len:
                    best_pos = start
                    for sep in SEPARATORS:
                        pos = text.rfind(sep, start, end)
                        if pos > start + max_chars // 2:
                            best_pos = pos + len(sep)
                            break
                    if best_pos > start:
                        end = best_pos

                chunk_text = text[start:end].strip()
                if chunk_text and len(chunk_text) > 20:
                    result.append({
                        "text": chunk_text,
                        "page": p_min,
                        "chunk_id": new_id,
                        "field_type": field_type,
                        "section_title": meta.get("section_title", ""),
                        "subsection_title": meta.get("subsection_title", ""),
                    })
                    new_id += 1

                step = max(max_chars - overlap, 1)
                new_start = start + step
                if new_start <= start:
                    break
                start = new_start

        return result

    def to_index_format(self, chunks, filename=None):
        """将 Chunk 列表转换为 vector_store.index_documents() 接受的 dict 格式

        page 转为 1-indexed 以与标准处理器统一
        元数据字段：section_title, subsection_title, table_name, source, method

        Returns:
            [{"text": str, "page": int, "chunk_id": int, "field_type": str,
              "section_title": str, "subsection_title": str, "table_name": str,
              "source": str, "method": str}, ...]
        """
        result = []
        for i, chunk in enumerate(chunks):
            type_map = {
                "pure_text": "body",
                "table": "table",
                "image": "image",
                "mixed": "body",
            }
            field_type = type_map.get(chunk.content_type, "body")

            # 构建带元数据的文本（文件名优先，用于公司路由过滤）
            text_parts = []
            if filename:
                text_parts.insert(0, f"【文件】{filename}")
            if chunk.metadata.get("section_title"):
                text_parts.append(f"【章节】{chunk.metadata['section_title']}")
            if chunk.metadata.get("subsection_title"):
                text_parts.append(f"【子标题】{chunk.metadata['subsection_title']}")
            if chunk.metadata.get("table_name"):
                text_parts.append(f"【表名】{chunk.metadata['table_name']}")
            if chunk.metadata.get("source"):
                text_parts.append(f"【来源】{chunk.metadata['source']}")

            text = chunk.content

            # 清洗 HTML 标签：VL 返回的 markdown 可能包含 <table> <div> 等
            text = self._strip_html_tags(text)

            if chunk.raw_html:
                # 把 raw_html 的 HTML 表格转为纯文本格式
                table_text = self._html_table_to_text(chunk.raw_html)
                if table_text:
                    text = text + "\n\n[表格]\n" + table_text
            if chunk.image_caption:
                text = text + "\n\n[图片描述]\n" + chunk.image_caption

            # 有元数据前缀的加入文本
            if text_parts:
                text = "\n".join(text_parts) + "\n" + text

            # 截断超长文本，防止 GPU OOM
            text = self._truncate_text(text, max_chars=8000)

            page_1based = chunk.page + 1

            record = {
                "text": text.strip(),
                "page": page_1based,
                "chunk_id": i,
                "field_type": field_type,
                "section_title": chunk.metadata.get("section_title", ""),
                "subsection_title": chunk.metadata.get("subsection_title", ""),
                "table_name": chunk.metadata.get("table_name", ""),
                "source": chunk.metadata.get("source", ""),
                "method": chunk.metadata.get("method", ""),
            }
            result.append(record)

        return result

    def save_output(self, chunks: List[Chunk], output_file: str = None):
        """保存输出"""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not output_file:
            base_name = Path(self.pdf_path).stem
            output_file = f"{base_name}_hybrid_chunks.json"

        output_path = output_path / output_file

        data = [chunk.to_dict() for chunk in chunks]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 已保存到：{output_path}")

        return output_path


# ─── 独立运行入口 ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="混合策略 PDF 处理器")
    parser.add_argument("--pdf", required=True, help="PDF 路径")
    parser.add_argument("--api-key", help="Qwen3-VL API Key")
    parser.add_argument("--pages", type=str, help="指定页码范围，如 0-50 或 0,10,20")
    parser.add_argument("--output-dir", default="output", help="输出目录")

    args = parser.parse_args()

    # 解析页码
    pages = None
    if args.pages:
        if "-" in args.pages:
            start, end = map(int, args.pages.split("-"))
            pages = range(start, end + 1)
        else:
            pages = [int(x) for x in args.pages.split(",")]

    processor = HybridPDFProcessor(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        qwen3_api_key=args.api_key,
    )

    chunks = processor.process(pages=pages)
    processor.save_output(chunks)
    processor.to_index_format(chunks)
