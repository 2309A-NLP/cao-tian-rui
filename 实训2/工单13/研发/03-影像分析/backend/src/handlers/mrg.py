"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
MRG Handler：医疗影像报告生成（Medical Report Generation）。

MRG 任务的核心逻辑：
1. 将医疗影像和可选的临床背景发送给 VLM
2. 要求 VLM 以 JSON 格式输出结构化报告（四个字段）
3. 解析 JSON（支持三种格式：裸 JSON、Markdown 代码块、文本中嵌入 JSON）
4. 最多重试 3 次，连续失败则降级为纯文本报告

熔断降级机制：
- 3 次 JSON 解析全部失败时，将 VLM 原始输出放入 findings 字段
- impression 字段注明"结构化解析失败"，提示人工复核
"""

# json：Python 内置模块，用于 JSON 序列化和反序列化
import json

# re：Python 内置正则表达式模块，用于从文本中提取 JSON 代码块
import re

# Optional：类型提示，表示可以为 None
from typing import Optional

# 从父包导入所需模块（使用相对导入）
from ..config import DISCLAIMER      # 医疗免责声明文本
from ..logger import get_logger      # 日志工厂函数
from ..models import ReportSchema    # 结构化报告数据模型
from ..vlm_client import get_vlm_client  # VLM 客户端工厂函数

# 获取本模块专用的日志记录器
logger = get_logger("wt13.mrg")

# ── MRG 任务的系统提示词 ──
# 明确告知 VLM 角色（放射科医师）、输出格式（严格 JSON）和内容要求
MRG_SYSTEM_PROMPT = (
    "你是一位经验丰富的放射科医师。请根据用户提供的医疗影像，撰写一份结构化影像诊断报告。\n"
    "严格按 JSON 格式输出，不要输出任何 JSON 外的文字（包括 markdown 代码块）。\n"
    "JSON Schema：\n"
    "{\n"
    '  "chief_complaint": "主诉/检查目的（string，简短）",\n'
    '  "findings": "影像所见（string，客观描述可见的解剖结构、异常信号、大小、位置、边界等）",\n'
    '  "impression": "印象/结论（string，基于所见的诊断意见，若不确定用可能/待排除等措辞）",\n'
    '  "recommendation": "建议（string，后续检查/临床处置/复查间隔）"\n'
    "}\n"
    "要求：\n"
    "1) 用规范医学术语，中文回答；\n"
    "2) 若图像质量不足以撰写报告，findings 中说明并将其余字段留 \"图像质量不足，建议重新采集\"；\n"
    "3) 不诊断、不做出确定性诊疗决定，措辞用\"考虑\"、\"提示\"、\"建议进一步检查\"；\n"
    "4) 输出必须是合法 JSON，可被 json.loads 直接解析。"
)

# 最大 JSON 解析重试次数（连续失败超过此次数则降级）
_MAX_PARSE_RETRIES = 3


class MRGHandler:
    """
    医疗影像报告生成处理器。

    封装了 MRG 任务的完整流程：VLM 调用 → JSON 解析 → 结构化报告构建。
    支持带临床背景的上下文输入（extra_context）。
    """

    def __init__(self):
        """初始化处理器，获取 VLM 客户端单例。"""
        self.vlm = get_vlm_client()  # 获取全局 VLM 客户端（单例，避免重复创建）

    def run(self, image_b64: str, extra_context: str = "", image_format: str = "jpeg") -> tuple[ReportSchema, str]:
        """
        执行 MRG 任务：生成结构化医疗影像诊断报告。

        参数：
            image_b64 (str)：图片的 base64 编码字符串
            extra_context (str)：可选的临床背景信息（如患者年龄、症状），默认为空
            image_format (str)：图片格式（"jpeg" 或 "png"），默认 "jpeg"

        返回值：
            tuple[ReportSchema, str]：
            - ReportSchema：包含四个字段的结构化报告对象
            - str：格式化为 Markdown 的可读报告文本
        """
        # 基础提示词
        prompt = "请为这张医疗影像撰写结构化诊断报告。"
        if extra_context:
            # 若有临床背景，追加到提示词中
            prompt += f"\n\n临床背景/患者信息：{extra_context}"

        last_raw = ""  # 记录最后一次 VLM 的原始输出（用于降级时的兜底）

        # 重试循环：最多尝试 _MAX_PARSE_RETRIES 次
        for attempt in range(1, _MAX_PARSE_RETRIES + 1):
            # 调用 VLM 生成报告文本
            raw = self.vlm.chat_vision(
                image_b64=image_b64,
                prompt=prompt,
                system=MRG_SYSTEM_PROMPT,
                max_tokens=1500,          # 报告字段多，给予较大的 token 预算
                temperature=0.2,          # 较低温度：报告需要准确一致，减少随机性
                image_format=image_format,
            )
            last_raw = raw  # 保存原始输出

            # 尝试将 VLM 输出解析为 JSON 字典
            parsed = _try_parse_json(raw)
            if parsed is not None:
                # 解析成功：构建结构化报告对象
                report = ReportSchema(
                    chief_complaint=parsed.get("chief_complaint", ""),
                    findings=parsed.get("findings", ""),
                    impression=parsed.get("impression", ""),
                    recommendation=parsed.get("recommendation", ""),
                )
                # 将结构化报告格式化为 Markdown 文本
                text = _format_report_text(report)
                return report, text  # 成功返回

            # 解析失败：记录警告日志，下次重试时在 prompt 中强调格式要求
            logger.warning(
                "MRG JSON 解析失败",
                extra={"payload": {"attempt": attempt, "raw_snippet": raw[:200]}},
            )
            # 在 prompt 前追加格式强调，引导模型下次输出合法 JSON
            prompt = "请严格返回合法 JSON，不要添加任何解释或 markdown 代码块。" + prompt

        # ── 3 次解析全部失败：触发熔断降级 ──
        logger.error("MRG 熔断降级：JSON 解析连续失败", extra={"payload": {"raw_snippet": last_raw[:300]}})

        # 降级策略：将原始输出放入 findings，标注解析失败状态
        report = ReportSchema(
            chief_complaint="",                                     # 主诉置空
            findings=last_raw.strip(),                              # 原始输出作为"所见"
            impression="（结构化解析失败，以上为模型原始输出）",       # 明确标注降级
            recommendation="建议人工复核报告",                       # 提示人工介入
        )
        return report, _format_report_text(report)


# 预编译正则：用于匹配 Markdown 代码块中的 JSON
# ```json ... ``` 或 ``` ... ``` 格式
# re.DOTALL：使 . 也匹配换行符（JSON 可能有多行）
# re.IGNORECASE：大小写不敏感（```JSON 和 ```json 都能匹配）
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _try_parse_json(text: str) -> Optional[dict]:
    """
    尝试从文本中解析 JSON 字典，支持三种格式：

    1. 裸 JSON：整个文本就是合法的 JSON 字符串
    2. Markdown 代码块：JSON 被 ```json...``` 包裹
    3. 文本中嵌入：JSON 对象嵌在其他文字中（通过查找第一个 { 和最后一个 } 提取）

    参数：
        text (str)：待解析的文本

    返回值：
        Optional[dict]：成功解析返回字典，否则返回 None
    """
    # 输入为空直接返回 None
    if not text:
        return None

    # ── 策略1：尝试直接解析整个文本为 JSON ──
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):  # 确保是字典（而非数组等其他 JSON 类型）
            return obj
    except json.JSONDecodeError:
        pass  # 解析失败，继续下一种策略

    # ── 策略2：尝试从 Markdown 代码块中提取 JSON ──
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))  # m.group(1) 是第一个括号捕获的内容（JSON 部分）
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass  # 提取的内容不是合法 JSON，继续

    # ── 策略3：寻找文本中第一个 { 和最后一个 }，尝试提取嵌入的 JSON ──
    l = text.find("{")     # 第一个 { 的位置（-1 表示不存在）
    r = text.rfind("}")    # 最后一个 } 的位置（-1 表示不存在）
    if l >= 0 and r > l:   # 找到了一对 {}
        try:
            obj = json.loads(text[l: r + 1])  # 提取 { 到 } 之间的文本
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass  # 提取的内容不是合法 JSON

    # 三种策略都失败，返回 None
    return None


def _format_report_text(r: ReportSchema) -> str:
    """
    将结构化报告对象格式化为 Markdown 可读文本。

    参数：
        r (ReportSchema)：结构化报告对象

    返回值：
        str：Markdown 格式的报告文本，末尾附有免责声明
    """
    lines = ["## 医疗影像诊断报告"]  # 报告标题

    # 各字段非空时才追加（避免显示空白节）
    if r.chief_complaint:
        lines.append(f"\n**主诉/检查目的**\n{r.chief_complaint}")

    if r.findings:
        lines.append(f"\n**影像所见**\n{r.findings}")

    if r.impression:
        lines.append(f"\n**印象**\n{r.impression}")

    if r.recommendation:
        lines.append(f"\n**建议**\n{r.recommendation}")

    # 末尾追加免责声明（⚠️ 符号起视觉警示作用）
    lines.append(f"\n---\n⚠️ {DISCLAIMER}")

    # 将所有行用换行符连接为单个字符串
    return "\n".join(lines)
