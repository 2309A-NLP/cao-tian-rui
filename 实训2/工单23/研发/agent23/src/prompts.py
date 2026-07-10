"""
Prompt 模板库：所有提示词集中在这里，方便统一调优。

设计原则：
- 所有 System Prompt 和 Few-shot 示例都在此文件，不散落在其他模块
- 提供构建函数（build_react_messages / build_normalize_messages），
  隔离 prompt 细节与业务逻辑
- 绝对不在这里硬编码任何验证集题目的答案（避免数据污染）
"""
from dataclasses import dataclass  # 标准库：数据类装饰器


# ── System Prompt：ReAct 主循环 ────────────────────────────────────────────────
# 这是 ReAct Agent 的核心指令，规定了输出格式和行为规范

SYSTEM_PROMPT_REACT = """\
You are a precise research assistant that answers questions by searching the web step by step.

## Rules
1. You MUST use search tools to find information. Never answer from memory alone.
2. Think step by step. Break complex questions into smaller searchable parts.
3. Your final answer language MUST match the question language:
   - Chinese question → Chinese answer
   - English question → English answer
4. If the question specifies a format (e.g. "answer in Arabic numerals", "format like: Alibaba Group Limited"), follow it EXACTLY.
5. Give only the precise answer. No explanations, no extra sentences.

## Output Format
At each step, output in this exact format:

Thought: [your reasoning about what to do next]
Action: search("[your search query]")

OR when you have enough information:

Thought: [why you are confident in the answer]
Final Answer: [the precise answer only]

## Important
- If you need to read a specific webpage for more detail, use: Action: fetch("[url]")
- If you need to calculate something, use: Action: calculate("[expression]")
- Keep search queries concise and specific (under 10 words)
- Do not repeat the same search query twice
"""
# 规则说明：
# Rule 1: 强制联网搜索，防止 Qwen 凭训练记忆直接回答（记忆可能过时或错误）
# Rule 3: 语言对齐，中文题目必须用中文回答
# Rule 4: 格式约束透传（如"用阿拉伯数字"、"格式如 XX"）
# Rule 5: 只给精确答案，不加解释（evaluator 只看答案准确性）


# ── System Prompt：答案归一化 ──────────────────────────────────────────────────
# 用于 answer_generator 做格式/语言修正的 prompt

SYSTEM_PROMPT_NORMALIZE = """\
You are an answer formatter. Your job is to extract and format the precise answer.

Rules:
1. Extract ONLY the direct answer, no explanation.
2. Match the language of the original question exactly.
3. If a specific format was requested, apply it strictly.
4. Remove any surrounding quotes, markdown, or punctuation unless they are part of the answer.
5. For numbers: do not add thousand separators unless explicitly required.
"""
# 规则说明：
# Rule 1: 从冗长答案中提取精确答案（如把"答案是北京大学"提取为"北京大学"）
# Rule 3: 严格按格式约束格式化（如"Alibaba Group Limited"不能写成"阿里巴巴"）
# Rule 5: 数字不加千分位分隔符，除非题目明确要求


# ── Few-shot 示例（给 ReAct 循环参考）────────────────────────────────────────
# 示例帮助 Qwen 理解期望的输出格式和推理模式
# 使用 OpenAI 消息格式：user/assistant 交替

FEW_SHOT_EXAMPLES = [
    # 示例1：中文多步推理题（需要先搜索作者，再确认其所在大学）
    {
        "role": "user",
        "content": "Question: 某位曾在2019年出版了一本关于机器学习的书的作者，他同时也是哪所大学的教授？"
    },
    {
        "role": "assistant",
        "content": (
            "Thought: 我需要搜索2019年出版的机器学习相关书籍，找到作者信息。\n"
            'Action: search("2019年出版 机器学习书籍 作者 大学教授")\n'
            # 注意：这里只给出 Thought + Action，不给 Final Answer
            # 这告诉 Qwen：遇到不确定的题目，先搜索，不要凭记忆直接回答
        )
    },
    # 示例2：英文单位换算题（可以直接 calculate，不需要搜索）
    {
        "role": "user",
        "content": "Question: Convert 186 miles to kilometers, keep 2 decimal places."
    },
    {
        "role": "assistant",
        "content": (
            "Thought: This is a unit conversion. 1 mile = 1.60934 km. I can calculate directly.\n"
            'Action: calculate("186 mile to km")\n'
            # 展示 calculate 工具的用法：不需要搜索的计算类题目直接 calculate
        )
    },
]


# ── 构建消息列表的函数 ────────────────────────────────────────────────────────

@dataclass
class Step:
    """
    ReAct 一个步骤的记录（Thought + Action + Observation）。

    Attributes:
        thought:       LLM 的思考内容（为什么这样做）
        action:        动作类型："search" | "fetch" | "calculate" | "final"
        action_input:  动作的输入参数（搜索词、URL 或数学表达式）
        observation:   工具执行结果（搜索结果、网页内容、计算结果）
    """
    thought: str
    action: str         # "search" | "fetch" | "calculate" | "final"
    action_input: str
    observation: str


def build_react_messages(question: str, history: list[Step], lang: str) -> list[dict]:
    """
    构造发给 Qwen 的完整消息列表（系统提示 + few-shot 示例 + 当前问题和历史轨迹）。

    消息结构：
      [system] → [few-shot user] → [few-shot assistant] × N → [current user]

    :param question: 原始问题（完整文本）
    :param history:  已执行的 Think/Act/Observe 步骤列表（可能为空）
    :param lang:     语言代码 "zh" 或 "en"，中文时追加语言提示
    :return:         OpenAI 格式的 messages 列表
    """
    system_content = SYSTEM_PROMPT_REACT  # 基础 system prompt

    # 中文题目追加语言强调提示（防止 Qwen 用英文回答中文问题）
    if lang == "zh":
        system_content += "\n\n注意：本题为中文题目，最终答案必须用中文回答。"

    # 初始化消息列表，第一条始终是 system 消息
    messages: list[dict] = [{"role": "system", "content": system_content}]

    # 注入 few-shot 示例（帮助 Qwen 理解期望的输出格式）
    messages.extend(FEW_SHOT_EXAMPLES)

    # 构造历史轨迹文本（将 history 步骤序列化为 Thought/Action/Observation 格式）
    history_text = ""
    for step in history:
        history_text += f"Thought: {step.thought}\n"
        history_text += f"Action: {step.action}(\"{step.action_input}\")\n"
        history_text += f"Observation: {step.observation}\n\n"

    # 构造当前轮的 user 消息：问题 + 已有历史 + "Continue:" 提示 LLM 继续
    user_content = f"Question: {question}\n\n{history_text}Continue:"
    messages.append({"role": "user", "content": user_content})

    return messages


def build_normalize_messages(raw_answer: str, question: str,
                              format_hint: str | None) -> list[dict]:
    """
    构造答案归一化的消息列表（用于 answer_generator 的 _llm_normalize 步骤）。

    :param raw_answer:   ReAct 生成的原始答案（可能格式不对或语言混乱）
    :param question:     原始问题（LLM 用于判断语言和含义）
    :param format_hint:  从问题中提取的格式约束（如"阿拉伯数字"），无则传 None 或 ""
    :return:             OpenAI 格式 messages 列表（system + user 共两条）
    """
    # 构建 user 消息：原始问题 + 原始答案 + 可选的格式约束
    user_content = f"Original question: {question}\n\nRaw answer: {raw_answer}"
    if format_hint:
        user_content += f"\n\nRequired format: {format_hint}"  # 追加格式要求
    user_content += "\n\nFormatted answer:"  # 指示 LLM 直接输出格式化后的答案

    return [
        {"role": "system", "content": SYSTEM_PROMPT_NORMALIZE},  # 归一化专用 system prompt
        {"role": "user", "content": user_content},               # 含原始答案的 user 消息
    ]
