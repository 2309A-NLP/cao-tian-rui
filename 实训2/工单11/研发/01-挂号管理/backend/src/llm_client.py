"""
工单11 医疗挂号Agent - LLM 客户端
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-挂号管理任务

兜底策略（按序降级）：
  1. 百炼 qwen3.7-plus（主路径，最快最新）
  2. 硅基流动 Qwen2.5-32B（百炼不可用时）
  3. 硅基流动 Qwen2.5-14B（32B也失败时）
  4. 规则正则（本地，无 LLM）—— 服务完全不可用时最低保证
"""
import json    # 标准库：JSON 序列化（此处主要供 rule_based_intent 内使用）
import time    # 标准库：time.sleep()（重试延迟）和 time.perf_counter()（计时）
from typing import Any, Optional   # 标准库：类型注解

# openai：OpenAI 官方 Python SDK，兼容 OpenAI 协议的服务商均可使用
import openai

# 从配置模块读取 LLM 相关配置
from src.config import (
    BAILIAN_API_KEY, BAILIAN_BASE_URL, BAILIAN_MODEL,
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL,
    VALID_DEPARTMENTS, VALID_TITLES,
)
from src import logger   # 结构化日志模块
from src.utils import fuzzy_match_dept, fuzzy_match_title


# ─────────────────────────────────────────────
# 客户端初始化（两个端点，各自独立客户端）
# ─────────────────────────────────────────────

# 百炼客户端（主模型）
_bailian_client = openai.OpenAI(
    api_key=BAILIAN_API_KEY,
    base_url=BAILIAN_BASE_URL,
    timeout=30.0,
)

# 硅基流动客户端（兜底）
_siliconflow_client = openai.OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
    timeout=30.0,
)

# 三级模型链：(客户端, 模型ID, 标签)
_MODEL_CHAIN = [
    (_bailian_client,     BAILIAN_MODEL,                 "百炼/qwen3.7-plus"),
    (_siliconflow_client, SILICONFLOW_MODEL,              "硅基/32B"),
    (_siliconflow_client, "Qwen/Qwen2.5-14B-Instruct",   "硅基/14B"),
]

_MAX_RETRIES = 3                     # 单个模型最多重试次数
_RETRY_DELAYS = [1.0, 2.0, 4.0]     # 指数退避延迟（秒）


def _call_llm(
    client: openai.OpenAI,
    messages: list,
    tools: Optional[list] = None,
    model: str = "",
    trace_id: str = "-",
) -> Any:
    """
    底层 LLM 调用函数，带指数退避重试（最多 3 次）。

    参数：
      client    : openai.OpenAI 客户端实例（百炼或硅基流动）
      messages  : 对话历史列表
      tools     : Function Calling 工具定义列表，None 表示不启用工具
      model     : 模型 ID
      trace_id  : 链路追踪 ID

    重试策略：
      网络超时/连接错误 → 重试；429/503 → 等待后重试；其他 4xx → 立即终止
    """
    t0 = time.perf_counter() * 1000
    last_exc = None

    for attempt in range(_MAX_RETRIES):
        try:
            kwargs = {"model": model, "messages": messages, "temperature": 0}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            # Qwen3 系列默认开启 CoT 思考模式，挂号场景不需要，关掉节省时间
            if "qwen3" in model.lower():
                kwargs["extra_body"] = {"enable_thinking": False}

            resp = client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() * 1000 - t0
            logger.log_llm_call(
                phase=f"attempt{attempt+1}",
                messages=messages,
                response=resp.choices[0].message.content or "(tool_call)",
                elapsed_ms=elapsed,
                trace_id=trace_id,
            )
            return resp

        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            last_exc = exc
            logger.log_warning(f"LLM 网络异常 attempt={attempt+1}: {exc}", trace_id=trace_id)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAYS[attempt])

        except openai.APIStatusError as exc:
            last_exc = exc
            logger.log_error(
                f"LLM API 错误 attempt={attempt+1}: status={exc.status_code}",
                trace_id=trace_id, exc=exc,
            )
            if exc.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAYS[attempt])
            else:
                break

    raise last_exc


def call_llm_with_fallback(
    messages: list,
    tools: Optional[list] = None,
    trace_id: str = "-",
) -> Any:
    """
    三级模型 fallback：百炼qwen3.7-plus → 硅基32B → 硅基14B → 抛异常（由agent做规则兜底）。

    异常处理层级（从内到外）：
      1. _call_llm 内部：单模型 3 次重试（网络层）
      2. call_llm_with_fallback：逐级切换模型
      3. agent.run_agent（上层）：全部失败 → 规则正则兜底
    """
    last_exc = None
    for idx, (client, model_id, label) in enumerate(_MODEL_CHAIN):
        try:
            if idx > 0:
                logger.log_warning(
                    f"切换到兜底模型 [{label}]: 上一级失败: {last_exc}",
                    trace_id=trace_id,
                )
            return _call_llm(client, messages, tools, model=model_id, trace_id=trace_id)
        except Exception as exc:
            last_exc = exc
            logger.log_error(
                f"模型 [{label}] 全部重试失败: {exc}",
                trace_id=trace_id, exc=exc,
            )

    # 三级均失败，向上抛出，由 agent.run_agent 的 except 块捕获做规则兜底
    logger.log_error("三级 LLM 均失败，返回异常给 agent 做规则兜底", trace_id=trace_id, exc=last_exc)
    raise last_exc


# ─────────────────────────────────────────────
# 规则兜底（LLM 完全不可用时）
# ─────────────────────────────────────────────

def rule_based_intent(query: str) -> dict:
    """
    用正则表达式做最基础的意图识别 + 槽位抽取，作为 LLM 不可用时的最后保障。

    精度有限（不如 LLM），但能保证系统在没有网络或 API 配额的情况下不崩溃，
    至少能回答"查一下内科有没有号"这类简单问题。

    参数：
      query : 用户自然语言输入

    返回：
      {"intent": str, "slots": dict, "fallback_mode": True}
      intent 可能值：cancel / doctor_schedule / book / query
      slots 包含：dept, title, alias, doctor_name, time_text
    """
    import re   # 标准库：正则表达式（局部导入，避免模块顶层过多依赖）
    q = query.lower()   # 转小写便于正则匹配（中英文混合场景）

    # 从文本中模糊匹配科室名（如"内科"/"消化内科"）
    dept = fuzzy_match_dept(query)
    # 从文本中匹配号源类型（如"专家号"/"普通"）
    title = fuzzy_match_title(query)

    # 意图识别：按优先级顺序检查关键词
    if re.search(r"取消|退号|退掉", q):
        intent = "cancel"            # 取消挂号意图
    elif re.search(r"排班|坐诊|出诊", q):
        intent = "doctor_schedule"   # 查询医生排班意图
    elif re.search(r"挂.*号|预约|约.*号", q):
        intent = "book"              # 挂号意图
    else:
        intent = "query"             # 默认：查询号源意图

    # 家属别名提取（简单正则，优先匹配常见称呼）
    alias_m = re.search(r"(大宝|二宝|小明|老爸|我自己|我)", query)
    alias = alias_m.group(1) if alias_m else "我"   # 找不到别名默认"我"（本人）

    # 医生姓名提取（"张建国医生"→"张建国"，匹配 2-4 个汉字后接职称）
    doctor_m = re.search(r"([一-龥]{2,4})(医生|大夫|医师|主任)", query)
    doctor_name = doctor_m.group(1) if doctor_m else None   # 未提及医生则为 None

    return {
        "intent": intent,
        "slots": {
            "dept": dept,
            "title": title,
            "alias": alias,
            "doctor_name": doctor_name,
            "time_text": query,   # 把全文传给时间解析器，由 parse_time 进一步处理
        },
        "fallback_mode": True,   # 标记为规则兜底模式，上层可据此降低置信度
    }
