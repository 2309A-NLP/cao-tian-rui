"""
ReAct 循环控制器：核心调度逻辑。

ReAct（Reasoning + Acting）模式：
  Think（推理）→ Act（执行工具）→ Observe（观察结果）→ 循环

退出条件（按优先级）：
  1. LLM 输出 "Final Answer: ..." → 正常退出（exit_reason="final_answer"）
  2. 达到 MAX_ROUNDS 轮 → 强制退出（exit_reason="max_rounds"）
  3. 整体超时 → 强制退出（exit_reason="timeout"）
  4. LLM 不可用 → 强制退出（exit_reason="llm_error"）
  5. 连续2轮观察完全相同 → 强制退出（exit_reason="no_new_info"）
  6. 没有解析到有效 Action → 强制退出（exit_reason="no_action"）

强制退出时：从历史 trace 中取最后一条非空 observation 作为降级答案。
"""
import re          # 标准库：正则表达式，用于解析 LLM 输出的 Thought/Action/Final Answer
import time        # 标准库：时间函数，用于超时检测
import logging     # 标准库：日志记录
from dataclasses import dataclass, field  # 标准库：数据类工具

import llm_client    # 本项目：LLM 调用客户端
import tool_search   # 本项目：搜索工具（IQS/Bing/SerpAPI 三级降级）
import tool_fetch    # 本项目：网页抓取工具
import tool_calc     # 本项目：数学/单位换算工具
from prompts import Step, build_react_messages  # 本项目：Step 数据类 + 消息构建函数
from preprocessor import QuestionMeta           # 本项目：问题元数据数据类
from config import Config                        # 本项目：全局配置

logger = logging.getLogger(__name__)  # 当前模块日志记录器


@dataclass
class ReactResult:
    """
    ReAct 循环的最终结果。

    Attributes:
        final_answer:  未归一化的原始答案字符串（由 answer_generator 进一步处理）
        trace:         完整的推理轨迹列表（每步含 Thought/Action/Observation）
        rounds_used:   实际使用的搜索轮数
        evidence_urls: 本次推理过程中访问的 URL 列表（来源证据）
        exit_reason:   退出原因标记，用于统计分析
    """
    final_answer: str               # 未归一化的原始答案
    trace: list[Step] = field(default_factory=list)    # 推理轨迹
    rounds_used: int = 0
    evidence_urls: list[str] = field(default_factory=list)  # 证据 URL 列表
    exit_reason: str = ""  # "final_answer" | "max_rounds" | "no_new_info" | "timeout" | "llm_error" | "no_action"


def run_react(meta: QuestionMeta, req_id: str = "", set_status=None) -> ReactResult:
    """
    执行 ReAct 循环，驱动 LLM 通过多轮工具调用解答问题。

    :param meta:       预处理结果（含原始问题、语言、格式约束、初始搜索词）
    :param req_id:     请求 ID，用于前端状态轮询（可选，空字符串时跳过状态更新）
    :param set_status: 状态更新回调函数 set_status(req_id, **kwargs)（可选）
    :return:           ReactResult 包含答案、轨迹、轮数等信息
    """
    def _update(action: str, query: str = ""):
        """内部辅助：更新前端展示的当前处理状态。"""
        if set_status and req_id:
            set_status(req_id, round=round_num, action=action, query=query, done=False)

    history: list[Step] = []    # 推理轨迹（Think/Act/Observe 历史）
    evidence_urls: list[str] = []  # 本次访问的 URL 列表
    last_observation = ""       # 上一轮的 observation，用于检测重复
    same_obs_count = 0          # 连续相同 observation 的计数
    start_time = time.time()    # 记录开始时间，用于总体超时检测
    round_num = 0               # 当前轮数（用于状态更新）

    for round_num in range(1, Config.MAX_ROUNDS + 1):  # 从第1轮到 MAX_ROUNDS 轮

        # ── 总体超时检查（每轮开始时检查）──────────────────────────────────────
        elapsed = time.time() - start_time
        if elapsed > Config.TOTAL_TIMEOUT_S:
            logger.warning("Total timeout after %.1fs at round %d", elapsed, round_num)
            return _force_exit(history, evidence_urls, round_num, "timeout")

        # ── Think：调用 LLM 进行推理 ────────────────────────────────────────────
        # 构建包含历史轨迹的消息列表
        messages = build_react_messages(meta.raw, history, meta.lang)
        try:
            # max_tokens=300：Thought + Action 不需要太多 token
            llm_output = llm_client.chat(messages, max_tokens=300)
        except llm_client.LLMUnavailableError as e:
            # LLM 彻底不可用（主备模型都失败），强制退出
            logger.error("LLM unavailable at round %d: %s", round_num, e)
            return _force_exit(history, evidence_urls, round_num, "llm_error")

        # 调试日志：记录 LLM 本轮的完整输出
        logger.debug("Round %d LLM output:\n%s", round_num, llm_output)

        # ── 检查是否得到最终答案 ────────────────────────────────────────────────
        final = _parse_final_answer(llm_output)  # 尝试提取 "Final Answer: ..." 内容
        if final is not None:
            # 特殊情况：第1轮且没有任何搜索历史就给出最终答案
            # 强制先搜索一轮，防止 Qwen 凭训练记忆直接回答（答案可能过时）
            if round_num == 1 and not history:
                logger.warning("Round 1 final answer without search, forcing search: %r", final[:60])
                thought = _parse_thought(llm_output)  # 提取本轮 Thought 内容

                # 生成兜底搜索词：优先用 preprocessor 生成的搜索词，否则截取原始问题
                forced_query = meta.search_query or meta.raw[:60]
                _update("search", forced_query)  # 通知前端：正在强制搜索

                # 执行强制搜索
                observation = _execute_action("search", forced_query, evidence_urls)
                history.append(Step(
                    thought=thought,
                    action="search",
                    action_input=forced_query,
                    observation=observation,
                ))
                continue  # 带着搜索结果进入下一轮，让 Qwen 基于证据重新决策

            # 正常情况：已有搜索历史，直接返回最终答案
            return ReactResult(
                final_answer=final,
                trace=history,
                rounds_used=round_num,
                evidence_urls=evidence_urls,
                exit_reason="final_answer",
            )

        # ── Act：解析 LLM 输出的动作 ────────────────────────────────────────────
        action, action_input = _parse_action(llm_output)  # 提取 Action 类型和输入
        thought = _parse_thought(llm_output)               # 提取 Thought 内容

        if not action:
            # 未解析到有效 Action（LLM 输出格式错误），强制退出
            logger.warning("No valid action at round %d, output: %r", round_num, llm_output[:200])
            return _force_exit(history, evidence_urls, round_num, "no_action")

        # 通知前端：正在执行哪个工具（search/fetch/calculate）、使用什么参数
        _update(action, action_input)

        # ── Observe：执行工具，获取观察结果 ────────────────────────────────────
        observation = _execute_action(action, action_input, evidence_urls)

        # 连续两轮观察完全相同，说明没有获取到新信息，提前退出（避免死循环）
        if observation == last_observation and observation:
            same_obs_count += 1
            if same_obs_count >= 2:  # 连续2次相同才退出（给1次容错机会）
                return _force_exit(history, evidence_urls, round_num, "no_new_info")
        else:
            same_obs_count = 0  # 有新信息时重置计数

        last_observation = observation  # 更新上一轮 observation 记录

        # 将本轮 Think/Act/Observe 记录到历史轨迹
        history.append(Step(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        ))

    # 达到最大轮数（MAX_ROUNDS）仍未得到最终答案，强制退出
    return _force_exit(history, evidence_urls, Config.MAX_ROUNDS, "max_rounds")


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────

def _parse_final_answer(text: str) -> str | None:
    """
    从 LLM 输出中提取 "Final Answer: ..." 后的内容。
    未找到时返回 None（表示本轮还未得出答案）。

    :param text: LLM 完整输出文本
    :return:     最终答案字符串，或 None
    """
    # re.IGNORECASE：兼容 "final answer:"/"Final Answer:" 等大小写变体
    m = re.search(r"Final Answer[：:]\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()  # 取捕获组内容，去首尾空格
    return None  # 未找到 Final Answer 标记


def _parse_thought(text: str) -> str:
    """
    从 LLM 输出中提取 "Thought: ..." 内容。
    未找到时返回原文前200字作为兜底。

    :param text: LLM 完整输出文本
    :return:     Thought 内容字符串
    """
    # re.DOTALL：使 . 匹配换行符（Thought 内容可能多行）
    m = re.search(r"Thought[：:]\s*(.+?)(?=\nAction|$)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else text[:200]  # 未找到时取前200字作为兜底


def _parse_action(text: str) -> tuple[str, str]:
    """
    解析 "Action: ..." 行，提取动作类型和输入参数。

    支持的格式：
      Action: search("query string")
      Action: fetch("https://example.com")
      Action: calculate("3.14 * 2 ** 2")

    :param text: LLM 完整输出文本
    :return:     (action_type, action_input) 元组；未找到时返回 ("", "")
    """
    m = re.search(
        r'Action[：:]\s*(search|fetch|calculate)\s*\(\s*["\']?(.*?)["\']?\s*\)',
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        return m.group(1).lower(), m.group(2).strip()  # action 类型转小写，input 去空格
    return "", ""  # 未找到有效 Action


def _execute_action(action: str, action_input: str,
                    evidence_urls: list[str]) -> str:
    """
    根据动作类型调用对应工具，返回 Observation 字符串。
    所有工具调用的异常都在此捕获，保证 ReAct 循环不中断。

    :param action:        动作类型："search" | "fetch" | "calculate"
    :param action_input:  动作的输入参数
    :param evidence_urls: 可变列表，记录本次访问的 URL（副作用修改）
    :return:              Observation 字符串（工具执行结果的文本表示）
    """
    try:
        if action == "search":
            # 调用搜索工具，返回最多 SEARCH_TOP_K 条结果
            result = tool_search.web_search(action_input)
            if not result.results:
                return "No results found."  # 无结果时返回空结果提示

            lines = []
            for i, r in enumerate(result.results, 1):  # enumerate 从1开始计数
                # 格式化每条搜索结果：序号、标题、摘要、URL
                lines.append(f"[{i}] {r['title']}\n{r['snippet']}\nURL: {r['url']}")
                if r["url"]:
                    evidence_urls.append(r["url"])  # 记录来源 URL
            return "\n\n".join(lines)  # 多条结果用空行分隔

        elif action == "fetch":
            # 调用网页抓取工具，获取网页正文内容
            evidence_urls.append(action_input)  # 记录访问的 URL
            result = tool_fetch.web_fetch(action_input)
            if not result.success:
                return f"Failed to fetch: {action_input}"  # 抓取失败时返回失败提示
            return f"Title: {result.title}\n\n{result.content}"  # 返回标题 + 正文

        elif action == "calculate":
            # 调用计算器工具，处理数学表达式或单位换算
            result = tool_calc.calculate(action_input)
            if not result.success:
                return f"Calculation failed: {action_input}"  # 计算失败提示
            unit = f" {result.unit}" if result.unit else ""  # 单位（可能为空）
            return f"Result: {result.value}{unit}"  # 返回计算结果（含单位）

        else:
            # 未知动作类型（理论上不会发生，_parse_action 只解析已知类型）
            return f"Unknown action: {action}"

    except Exception as e:
        # 工具执行异常统一捕获，记录错误日志后返回错误描述
        # 不抛出异常，让 ReAct 循环继续（LLM 会根据错误信息决定下一步）
        logger.error("Action execution error [%s(%r)]: %s", action, action_input, e)
        return f"Error executing {action}: {e}"


def _force_exit(history: list[Step], evidence_urls: list[str],
                rounds: int, reason: str) -> ReactResult:
    """
    强制退出时构造降级结果。
    尽量从历史 trace 中提取最后一条有意义的 observation 作为兜底答案，
    宁可给一个不完整的答案，也不返回空字符串。

    :param history:       历史轨迹列表
    :param evidence_urls: 已访问的 URL 列表
    :param rounds:        已使用的轮数
    :param reason:        退出原因标记
    :return:              ReactResult（含降级答案）
    """
    fallback = ""
    # 从最后一步开始向前遍历，找第一条有意义的 observation 作为兜底
    for step in reversed(history):
        if step.observation and step.observation != "No results found.":
            fallback = step.observation[:300]  # 截断到300字，避免答案过长
            break

    logger.warning("Force exit: reason=%s, rounds=%d, fallback=%r",
                   reason, rounds, fallback[:80])

    return ReactResult(
        final_answer=fallback or "Unable to determine the answer.",  # 无历史时返回默认兜底语
        trace=history,
        rounds_used=rounds,
        evidence_urls=evidence_urls,
        exit_reason=reason,
    )
