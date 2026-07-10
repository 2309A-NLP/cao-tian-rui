"""
答案自检器（M4）：answer_generator 之后的最后一道门。
只做规则检查，不再调 LLM，目标是拦截"答案对但格式不干净"的情况。

处理场景：
1. 答案夹带解释句 → 截取第一个逗号/句号前的名词短语
2. 答案超长（>80字）→ 截取第一句
3. 答案为空或占位词 → 标记为低质量

设计原则：纯规则、无 LLM 调用、速度极快，作为最后的格式卫兵。
"""
import re           # 标准库：正则表达式，用于模式匹配和文本截断
import logging      # 标准库：日志记录，用于输出调试/变更信息
from dataclasses import dataclass  # 标准库：数据类装饰器，自动生成 __init__/__repr__ 等方法

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# 触发截断的谓语结构正则（说明答案里带了解释性句子，需要截断）
_PREDICATE_PATTERNS = [
    r"[，,]\s*(?:是|为|曾|曾经|担任|被|于|在)",   # 中文谓语：XXX，是/为/曾担任
    r"\s+(?:is|was|are|were|served|became)\s+",    # 英文谓语：XXX is/was/served
    r"[，,]\s+(?:a|an|the)\s+",                    # 英文冠词短语：XXX, a professor
]

# 低质量占位词集合（不区分大小写，统一转小写比较）
_LOW_QUALITY = {
    "unknown",
    "unable to determine",
    "无法确定",
    "不知道",
    "",              # 空字符串
    "目前未知",
    "暂时未知",
    "尚不清楚",
    "不清楚",
    "无法回答",
    "i don't know",
    "i cannot determine",
    "cannot be determined",
}


@dataclass
class ValidateResult:
    """
    自检结果数据类。

    Attributes:
        answer:  处理后的最终答案（可能与输入不同）
        changed: 是否对答案做了修改
        issues:  发现的问题列表，用于调试溯源（如 "truncated_long"）
    """
    answer: str          # 处理后的答案
    changed: bool        # 是否做了修改
    issues: list         # 发现的问题列表


def validate(answer: str, format_hint: str = "", lang: str = "en") -> ValidateResult:
    """
    对 answer_generator 输出的答案做规则自检和格式清洗。

    处理顺序：
    1. 空值/占位词检查 → 直接返回（标记 low_quality，不做截断）
    2. 有格式约束时跳过截断（格式已在 answer_generator 中处理）
    3. 超长截断（>80字取第一句）
    4. 谓语解释截断（取名词短语部分）
    5. 去首尾空格和引号

    :param answer:      answer_generator 输出的答案字符串
    :param format_hint: 题目格式约束（有则跳过截断，格式已在上游处理）
    :param lang:        问题语言 "zh"（中文）或 "en"（英文）
    :return:            ValidateResult 数据类实例
    """
    original = answer  # 保存原始答案，用于最后判断是否发生变化
    issues = []        # 记录本次发现的问题列表

    # 1. 空值/占位词检查：如果答案是已知低质量词，直接标记返回，不做截断
    if answer.strip().lower() in _LOW_QUALITY:
        logger.debug("Validator: low quality answer %r", answer)
        # changed=False 表示没有修改（保持原样让上层决定如何处理）
        return ValidateResult(answer=answer, changed=False, issues=["low_quality"])

    # 2. 如果题目有格式约束，跳过截断逻辑（格式已由 answer_generator 专门处理过）
    if format_hint:
        # 仅做首尾去空格，不截断
        return ValidateResult(answer=answer.strip(), changed=False, issues=[])

    # 3. 截断：答案超过 80 字时，取第一句作为精简答案
    if len(answer) > 80:
        first_sentence = _first_sentence(answer, lang)  # 按语言规则取第一句
        if first_sentence and len(first_sentence) < len(answer):
            # 记录截断信息（截断前→截断后的字符数）
            issues.append(f"truncated_long: {len(answer)}→{len(first_sentence)}")
            answer = first_sentence  # 用第一句替换原始答案

    # 4. 截断：答案含谓语解释结构（说明答案带了解释），取第一个逗号/句号前的名词短语
    if not format_hint:  # 无格式约束才做截断（有格式约束已在上面返回）
        cleaned = _strip_explanation(answer, lang)
        if cleaned != answer and len(cleaned) >= 1:  # 截断后至少有1个字符才采用
            issues.append("stripped_explanation")
            answer = cleaned  # 采用截断后的名词短语

    # 5. 最终清洗：去首尾空格和各种引号（英文单/双引号、中文引号）
    answer = answer.strip().strip('"\'""''')

    changed = answer != original  # 对比原始答案，判断是否做了修改
    if changed:
        logger.info("Validator changed answer: %r → %r (issues=%s)",
                    original[:60], answer[:60], issues)

    return ValidateResult(answer=answer, changed=changed, issues=issues)


# ── 内部函数 ──────────────────────────────────────────────────────────────────

def _first_sentence(text: str, lang: str) -> str:
    """
    取文本的第一句话。
    中文按句号/问号/感叹号分割，英文按句号+空格分割。

    :param text: 输入文本
    :param lang: 语言代码（"zh" 或 "en"），当前实现两者逻辑相同
    :return:     第一句话（不含句末标点）
    """
    # 匹配中文句子结束符（句号、感叹号、问号的全角/半角形式）
    m = re.search(r"[。！？!?]", text)
    if m:
        return text[:m.start()].strip()  # 取标点前的部分

    # 匹配英文句子结束（句号后跟空格，避免误截小数点）
    m = re.search(r"\.\s", text)
    if m:
        return text[:m.start()].strip()

    return text  # 找不到句子分隔符时返回原文


def _strip_explanation(text: str, lang: str) -> str:
    """
    检测到谓语结构时，截断到第一个谓语前的名词短语。

    示例：
      "詹姆斯·洛克哈特，是UCLA教授" → "詹姆斯·洛克哈特"
      "John Smith, a professor at UCLA" → "John Smith"
      "Albert Einstein, who was born in..." → "Albert Einstein"

    :param text: 输入答案文本
    :param lang: 语言代码（当前未区分，对中英文统一处理）
    :return:     截断后的名词短语；若未检测到谓语则返回原文
    """
    for pattern in _PREDICATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            candidate = text[:m.start()].strip()  # 取谓语之前的名词部分
            # 候选不能太短（至少2个字符，否则可能是误截）
            if len(candidate) >= 2:
                return candidate
    return text  # 未检测到谓语结构，返回原文
