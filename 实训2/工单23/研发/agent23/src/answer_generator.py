"""
答案生成器：把 ReAct 的原始输出归一化成精确答案。
步骤：提取 → 去噪 → 语言对齐 → 格式对齐。

处理流程：
  1. _basic_clean   : 规则清洗，去前缀/引号/markdown 符号
  2. _is_low_quality: 判断答案是否需要补救
  3. _extract_from_trace: 从 ReAct 轨迹重新提取（补救路径）
  4. _llm_normalize : 调 LLM 做语言/格式对齐（轻量调用）
  5. _final_clean   : 最后去首尾空格和多余前缀
"""
import re           # 标准库：正则表达式，用于文本匹配与替换
import logging      # 标准库：日志记录，用于输出调试/错误信息
import llm_client   # 本项目模块：封装了对阿里云百炼 Qwen 的 API 调用
from prompts import build_normalize_messages  # 本项目模块：构建答案归一化所需的消息列表
from preprocessor import QuestionMeta         # 本项目模块：问题元数据数据类（语言、格式约束等）
from react_loop import ReactResult            # 本项目模块：ReAct 循环的结果数据类

# 获取当前模块的日志记录器，日志会带上模块名方便排查
logger = logging.getLogger(__name__)


def generate_answer(react_result: ReactResult, meta: QuestionMeta) -> str:
    """
    将 ReAct 循环的原始输出归一化为精确答案。

    :param react_result: ReAct 循环产出，包含 final_answer、trace 等字段
    :param meta:         问题元数据，包含语言、格式约束等信息
    :return:             归一化后的最终答案字符串
    """
    raw = react_result.final_answer  # 取出 ReAct 循环给出的原始答案字符串

    # Step 1: 基础清洗（纯规则处理，不调用 LLM，速度快）
    answer = _basic_clean(raw)

    # Step 2: 如果答案明显质量太差（空/占位词/超长），从 trace 记录里重新提取
    if _is_low_quality(answer) and react_result.trace:
        # trace 非空说明有搜索历史，让 LLM 从中重新抽取答案
        answer = _extract_from_trace(react_result, meta)

    # Step 3: 语言/格式对齐（用一次轻量 LLM 调用修正语言或格式不匹配的答案）
    if _needs_normalization(answer, meta):
        answer = _llm_normalize(answer, meta)

    # Step 4: 最终再做一次清洗，去掉 LLM 可能加上的"Answer:"前缀
    answer = _final_clean(answer)

    # 记录最终答案，[:80] 防止日志行过长
    logger.info("Answer: %r (exit=%s, rounds=%d)",
                answer[:80], react_result.exit_reason, react_result.rounds_used)
    return answer


# ── 内部函数 ──────────────────────────────────────────────────────────────────

def _basic_clean(text: str) -> str:
    """
    基础规则清洗：去除常见噪音前缀、引号、markdown 符号。

    :param text: 原始答案文本
    :return:     清洗后的文本
    """
    # 去掉 "Final Answer:" 前缀（LLM 有时会把格式标记带入答案）
    text = re.sub(r"^Final Answer[：:]\s*", "", text, flags=re.IGNORECASE).strip()

    # 去掉首尾的各种引号（英文单/双引号、中文引号）
    text = re.sub(r'^[\"\'""'']|[\"\'""'']$', "", text).strip()

    # 去掉 markdown 加粗/斜体标记（如 **内容** 或 *内容*），只保留内容
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)

    # 将多余换行合并为单个空格，使答案更整洁
    text = re.sub(r"\n+", " ", text).strip()
    return text


def _is_low_quality(answer: str) -> bool:
    """
    判断答案质量是否太差，需要走补救路径。

    判断条件：
    - 答案为空或长度小于 2（几乎无内容）
    - 包含 "Unable to determine" 等降级兜底语
    - 超过 500 字（答案不应如此冗长，说明 LLM 没有正确提取）

    :param answer: 待检查的答案字符串
    :return:       True 表示质量差，需要补救
    """
    if not answer or len(answer) < 2:  # 空答案或极短答案
        return True
    if "Unable to determine" in answer:  # 常见降级兜底语
        return True
    if len(answer) > 500:  # 答案不应该超过500字，超长说明提取失败
        return True
    return False  # 质量可接受


def _extract_from_trace(react_result: ReactResult, meta: QuestionMeta) -> str:
    """
    从完整的 ReAct 轨迹中，让 LLM 重新提取精确答案。
    在 ReAct 循环本身没有给出好答案时作为补救手段。

    :param react_result: 包含完整 trace（思考+动作+观察）的 ReactResult
    :param meta:         问题元数据，用于确定语言
    :return:             LLM 从 trace 中提取的答案；失败时返回原始 final_answer
    """
    trace_text = ""  # 将 trace 步骤拼成可读文本
    for step in react_result.trace:
        # 每步由 Thought / Action / Observation 三部分组成
        trace_text += f"Thought: {step.thought}\n"
        trace_text += f"Action: {step.action}(\"{step.action_input}\")\n"
        trace_text += f"Observation: {step.observation}\n\n"

    # 构建 system + user 消息，指示 LLM 从 trace 中提取答案
    messages = [
        {
            "role": "system",
            "content": (
                "Based on the research trace below, extract the precise answer to the question. "
                "Output ONLY the answer, nothing else. "
                # 根据问题语言要求 LLM 用对应语言回答
                f"Answer in {'Chinese' if meta.lang == 'zh' else 'English'}."
            )
        },
        {
            "role": "user",
            "content": f"Question: {meta.raw}\n\nResearch trace:\n{trace_text}\n\nAnswer:"
        }
    ]
    try:
        # 调用 LLM，限制 max_tokens=256，只需提取短答案
        return llm_client.chat(messages, max_tokens=256).strip()
    except Exception as e:
        # LLM 调用失败时记录警告并回退到原始答案
        logger.warning("Trace extraction failed: %s", e)
        return react_result.final_answer  # 降级返回原始答案


def _needs_normalization(answer: str, meta: QuestionMeta) -> bool:
    """
    判断答案是否需要调用 LLM 做格式/语言归一化。

    触发条件：
    - 题目有格式约束（format_hint 非空）
    - 中文题但答案中英文字符占比过高
    - 英文题但答案含过多中文字符

    :param answer: 当前答案
    :param meta:   问题元数据
    :return:       True 表示需要归一化
    """
    # 有格式要求时必须归一化（如"用阿拉伯数字"、"format like: X"）
    if meta.format_hint:
        return True

    # 中文题但答案英文占比过高（超过10个英文字符且中文占比不足30%）
    if meta.lang == "zh":
        zh_count = len(re.findall(r"[一-鿿]", answer))   # 统计中文字符数
        en_count = len(re.findall(r"[a-zA-Z]", answer))  # 统计英文字符数
        if en_count > 10 and zh_count < en_count * 0.3:  # 英文太多，中文太少
            return True
    # 英文题但答案含中文字符（可能语言混杂）
    elif meta.lang == "en":
        zh_count = len(re.findall(r"[一-鿿]", answer))   # 统计中文字符数
        if zh_count > 5:  # 超过5个中文字符就认为需要修正
            return True

    return False  # 不需要归一化


def _llm_normalize(answer: str, meta: QuestionMeta) -> str:
    """
    调用 LLM 对答案做一次格式/语言修正。

    :param answer: 需要修正的答案
    :param meta:   问题元数据（含原始问题和格式约束）
    :return:       修正后的答案；LLM 失败时返回原答案
    """
    # build_normalize_messages 构建归一化专用的 system+user 消息
    messages = build_normalize_messages(answer, meta.raw, meta.format_hint)
    try:
        # max_tokens=256 足够返回精简答案
        return llm_client.chat(messages, max_tokens=256).strip()
    except Exception as e:
        # 归一化失败不影响主流程，记录警告后返回原答案
        logger.warning("LLM normalization failed: %s", e)
        return answer  # 降级返回未修正的答案


def _final_clean(text: str) -> str:
    """
    最后一次清洗：去首尾空格，去 LLM 可能加上的"Answer:"类前缀。

    :param text: 待清洗文本
    :return:     清洗后的文本
    """
    text = text.strip()  # 去首尾空白字符

    # 去掉 "Answer: " / "答案: " / "回答: " 等前缀（有些 LLM 会自动加上）
    text = re.sub(r"^(?:Answer|答案|回答)[：:]\s*", "", text, flags=re.IGNORECASE)

    text = text.strip()  # 再次去首尾空格（去前缀后可能有残留空格）
    return text
