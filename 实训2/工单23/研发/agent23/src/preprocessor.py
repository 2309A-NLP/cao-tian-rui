"""
问题预处理器：语言检测 + 格式约束提取 + 初始搜索词生成。

特点：
- 纯规则处理，不调用 LLM，运行极快（微秒级）
- 结果封装为 QuestionMeta 数据类，供 react_loop 和 answer_generator 使用
- 支持中文/英文题目，以及混合语言题目

处理步骤：
  1. _detect_lang       : 通过中文字符占比判断语言
  2. _extract_format_hint: 正则匹配格式约束（如"用阿拉伯数字"）
  3. _make_search_query : 去除疑问词和格式说明，提取搜索词
"""
import re              # 标准库：正则表达式
from dataclasses import dataclass  # 标准库：数据类装饰器


@dataclass
class QuestionMeta:
    """
    问题元数据：预处理结果的结构化表示。

    Attributes:
        raw:          原始问题字符串（完整，未修改）
        lang:         检测到的语言，"zh"（中文）或 "en"（英文）
        format_hint:  提取到的格式约束字符串，如"阿拉伯数字"；无约束时为 ""
        search_query: 生成的初始搜索词（去除疑问词和格式说明后的精简版）
    """
    raw: str
    lang: str           # "zh" | "en"
    format_hint: str    # 格式约束，例如"用阿拉伯数字" / "format like: X"，无则为 ""
    search_query: str   # 初始搜索词（供 tool_search 使用）


# 中文字符范围正则：
# [一-鿿] 覆盖 CJK 统一汉字基本区（U+4E00~U+9FFF）
# [㐀-䶿] 覆盖 CJK 扩展 A 区（U+3400~U+4DBF）
_ZH_PATTERN = re.compile(r"[一-鿿㐀-䶿]")

# 格式约束关键词匹配规则（pattern, 语言标记）
# 使用正则捕获组提取约束内容，第一个捕获组为约束值
_FORMAT_PATTERNS = [
    # 中文格式要求：
    (r"要求格式形如[：:]\s*(.+?)(?:[。，,]|$)", "zh"),          # 要求格式形如：XX年XX月
    (r"格式[如如：:](.+?)(?:[。，,]|$)", "zh"),                  # 格式如：XX
    (r"请(以|用|按照?)([一-鿿\w]+格式|数字|阿拉伯数字|中文数字)回答", "zh"),  # 请用阿拉伯数字回答
    (r"(仅|只)回答([一-鿿\w]+)", "zh"),                          # 仅回答数字
    # 英文格式要求：
    (r"format (?:like|as|such as)[：:\s]+(.+?)(?:[.,]|$)", "en"),  # format like: Alibaba Group
    (r"answer (?:in|with) (?:the format|format)[：:\s]+(.+?)(?:[.,]|$)", "en"),  # answer in format: XX
    (r"in the form[：:\s]+(.+?)(?:[.,]|$)", "en"),                # in the form: XX
    (r"respond with (?:only )?(.+?)(?:[.,]|$)", "en"),            # respond with only the number
]

# 中文停用词集合：生成搜索词时去掉这些词，保留实体词和关键词
_STOP_WORDS_ZH = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "这个",
    "哪个", "什么", "请问", "请", "回答", "问题",
}


def preprocess(question: str) -> QuestionMeta:
    """
    对原始问题做预处理，生成 QuestionMeta。

    :param question: 原始问题字符串（用户输入的原文）
    :return:         QuestionMeta 数据类实例
    """
    lang = _detect_lang(question)                   # 第1步：检测语言
    format_hint = _extract_format_hint(question)    # 第2步：提取格式约束
    search_query = _make_search_query(question, lang)  # 第3步：生成搜索词

    return QuestionMeta(
        raw=question,
        lang=lang,
        format_hint=format_hint,
        search_query=search_query,
    )


def _detect_lang(text: str) -> str:
    """
    通过中文字符占比判断语言。
    若中文字符数量占总字符数 > 15%，判定为中文题；否则为英文题。

    :param text: 输入文本
    :return:     "zh"（中文）或 "en"（英文）
    """
    zh_count = len(_ZH_PATTERN.findall(text))  # 统计中文字符数量
    ratio = zh_count / max(len(text), 1)        # 计算占比（max 避免空字符串除零）
    return "zh" if ratio > 0.15 else "en"       # 15% 阈值区分中英文


def _extract_format_hint(question: str) -> str:
    """
    用正则匹配题目中的格式约束说明，提取约束内容字符串。
    找不到格式约束时返回空字符串。

    :param question: 原始问题
    :return:         格式约束字符串，如 "阿拉伯数字"；无则返回 ""
    """
    for pattern, _ in _FORMAT_PATTERNS:  # 遍历所有格式约束匹配规则
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            # 有捕获组时取第一个捕获组（约束值），否则取整个匹配
            hint = m.group(1) if m.lastindex else m.group(0)
            return hint.strip()  # 去首尾空格后返回
    return ""  # 所有规则都未匹配，无格式约束


def _make_search_query(question: str, lang: str) -> str:
    """
    从原始问题生成初始搜索词。

    处理步骤：
    1. 去掉格式说明部分（不是搜索目标）
    2. 去掉常见疑问词（不影响搜索意义）
    3. 中文题去停用词
    4. 截断到 80 字以内（搜索词不宜过长）

    :param question: 原始问题
    :param lang:     语言代码 "zh" 或 "en"
    :return:         精简后的搜索词字符串（最长 80 字）
    """
    q = question  # 以原始问题为起点做逐步清洗

    # ── 去除中文格式说明句 ──────────────────────────────────────────────────────
    q = re.sub(r"要求格式形如.+?(?:[。，,]|$)", "", q)   # 去"要求格式形如..."
    q = re.sub(r"请(用|以|按照?)([一-鿿\w]+)回答", "", q)  # 去"请用X回答"
    q = re.sub(r"（[^）]+）", "", q)                      # 去括号说明（全角括号）

    # ── 去除英文格式说明句 ──────────────────────────────────────────────────────
    q = re.sub(r"format (?:like|as|such as)[：:\s]+.+?(?:[.,]|$)", "", q, flags=re.IGNORECASE)
    q = re.sub(r"in the form[：:\s]+.+?(?:[.,]|$)", "", q, flags=re.IGNORECASE)
    q = re.sub(r"answer (?:only )?(?:in|with)[：:\s]+.+?(?:[.,]|$)", "", q, flags=re.IGNORECASE)

    # ── 去掉结尾的疑问词 ──────────────────────────────────────────────────────
    q = re.sub(r"[，。？?！!]+$", "", q)                          # 去末尾标点
    q = re.sub(r"(是什么|叫什么|的名字|的名称|的英文名)\s*$", "", q)  # 去中文疑问词
    q = re.sub(r"what (?:is|was|are|were) (?:the )?(?:name of )?", "", q, flags=re.IGNORECASE)

    # ── 中文题：逐字过滤停用词 ──────────────────────────────────────────────────
    if lang == "zh":
        words = list(q)  # 中文无法按空格分词，逐字处理
        q = "".join(w for w in words if w not in _STOP_WORDS_ZH)

    # 去首尾空格并截断到 80 字（搜索词过长会降低搜索精度）
    q = q.strip()[:80]

    # 如果处理后为空（问题全是停用词），回退到使用原始问题前80字
    return q if q else question[:80]
