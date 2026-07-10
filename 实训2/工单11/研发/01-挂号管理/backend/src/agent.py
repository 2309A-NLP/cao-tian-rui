"""
工单11 医疗挂号Agent — 核心调度逻辑
======================================
文件职责（本文件只做这几件事）：
  1. _try_parse_tool_call_from_text()
       当 LLM 把工具调用写进普通文本而非 tool_calls 字段时，
       用正则从文本里抢救出工具名和参数。

  2. _dispatch_tool()
       工具分发器：根据 LLM 返回的工具名，调用 tools_read / tools_write
       对应函数，把结果序列化成 JSON 字符串返回给 LLM。

  3. _pre_validate()
       规则层前置校验：在调 LLM 前先拦截明显错误
       （不存在的用户、无效日期、不存在的科室），节省 token。

  4. run_agent()
       对外唯一接口，驱动 LLM 多轮工具循环，
       带挂号守卫（防止 LLM 找到号源却忘了挂），
       最终返回自然语言回复。

关联文件：
  - agent_prompts.py  → system prompt 与 Function Calling Schema
  - tools_read.py     → 4个查询类工具
  - tools_write.py    → 2个写入类工具（挂号/取消）
  - llm_client.py     → LLM 调用封装（多模型 fallback）
"""
import json   # 标准库：JSON 序列化和反序列化（工具参数/结果的格式）
import re     # 标准库：正则表达式（文本中提取工具调用 JSON、意图检测）
import time   # 标准库：perf_counter 高精度计时（记录总耗时）
from datetime import date, timedelta   # 标准库：日期操作（默认日期范围）
from typing import Optional            # 标准库：Optional 类型注解

from src import logger                                    # 结构化日志
from src.agent_prompts import build_system_prompt, TOOLS_SCHEMA   # System prompt 和工具定义
from src.config import VALID_DEPARTMENTS, MAX_FUTURE_DAYS          # 业务常量
from src.database import get_connection                             # 数据库连接
from src.llm_client import call_llm_with_fallback, rule_based_intent  # LLM 调用和规则兜底
import src.tools_read as tools_read    # 4 个只读工具
import src.tools_write as tools_write  # 2 个写入工具
from src.utils import parse_time, validate_date_range, fuzzy_match_dept  # 辅助函数


# ──────────────────────────────────────────────────────────────────────────────
# 思考过程（Trace）辅助
# ──────────────────────────────────────────────────────────────────────────────

# 工具名 → 前端展示的中文标签映射表
_TOOL_LABELS: dict[str, str] = {
    "get_family_member":   "查询家属",
    "query_schedule":      "查询号源",
    "book_appointment":    "执行挂号",
    "cancel_appointment":  "取消挂号",
    "get_user_history":    "查历史记录",
    "get_doctor_schedule": "查医生排班",
}


def _make_trace_step(step: int, name: str, args: dict, result_json: str) -> dict:
    """
    把一次工具调用转为前端可渲染的 trace 步骤字典。

    对大列表字段（schedules / history）只保留条数摘要（如"[共5条]"），
    避免把完整数据都放进 trace 导致响应体过大。

    参数：
      step        : 步骤序号（从 1 开始）
      name        : 工具名
      args        : 工具调用参数（dict）
      result_json : 工具返回结果的 JSON 字符串

    返回：前端展示所需的步骤字典
    """
    try:
        out = json.loads(result_json)   # 解析工具结果 JSON
    except Exception:
        out = {}   # JSON 解析失败时使用空字典（容错）
    summary = dict(out)   # 浅拷贝，避免修改原始结果
    # 对 schedules/history 大列表只保留条数，不展示全部内容
    for key in ("schedules", "history"):
        if isinstance(summary.get(key), list):
            summary[key] = f"[共{len(out[key])}条]"
    return {
        "step":   step,
        "tool":   name,
        "label":  _TOOL_LABELS.get(name, name),   # 有中文标签则用中文，否则用工具名
        "input":  args,
        "output": summary,
        "ok":     bool(out.get("ok", False)),      # 工具是否成功
    }


# ──────────────────────────────────────────────────────────────────────────────
# 工具调用文本解析器（LLM 降级兜底）
# ──────────────────────────────────────────────────────────────────────────────

def _try_parse_tool_call_from_text(content: str) -> Optional[tuple[str, dict]]:
    """
    从 LLM 纯文本回复中尝试提取工具调用信息（格式容错处理）。

    背景：
      标准行为是 LLM 通过 message.tool_calls 字段返回工具调用，
      但少数模型（或限流降级时）会把 JSON 写在普通文本里，例如：
        {"name": "query_schedule", "arguments": {"dept": "内科"}}
      本函数负责识别并提取这类"嵌入文本的工具调用"。

    依次尝试三种格式：
      1. Markdown 代码块：  ```json { ... } ```
      2. 嵌入文本的 JSON：  {"name":..., "arguments":...}（含 name 和 arguments 键）
      3. 去掉注释前缀后直接解析（//注释 或 #注释 开头的 JSON）

    参数：
      content : LLM 返回的文本内容（可能包含工具调用 JSON）

    返回：
      (工具名, 参数字典) 如果成功提取
      None              如果无法识别或输入为空
    """
    if not content:
        return None   # 空内容直接返回 None

    candidates = []   # 候选 JSON 字符串列表，逐一尝试解析

    # ── 格式1：提取 ```json...``` 代码块 ──
    # re.search 在字符串中搜索第一个匹配
    # [\s\S]*? 非贪婪匹配任意字符（含换行），用于提取代码块内容
    code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    if code_block:
        candidates.append(code_block.group(1).strip())   # group(1) 取括号内的捕获组

    # ── 格式2：搜索含 "name" 和 "arguments" 键的 JSON 对象 ──
    # 匹配 {...  "name"... "arguments": {...} ...} 这样的嵌套 JSON
    for m in re.finditer(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\{[\s\S]*?\}\s*\}', content):
        candidates.append(m.group(0))

    # ── 格式3：去掉 // 或 # 注释前缀后尝试解析 ──
    # 有些模型会在 JSON 前加一行注释说明
    clean = re.sub(r'^(?://[^\n]*\n|#[^\n]*\n|\s)+', '', content)
    if clean.startswith("{"):
        candidates.append(clean)

    # 逐一尝试解析候选 JSON，找到第一个合法的工具调用格式就返回
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)   # 尝试 JSON 反序列化
            if (
                isinstance(parsed, dict)           # 必须是字典
                and "name" in parsed               # 有 name 字段（工具名）
                and "arguments" in parsed          # 有 arguments 字段（参数）
                and isinstance(parsed["arguments"], dict)  # arguments 必须是字典
            ):
                return parsed["name"], parsed["arguments"]   # 成功，返回工具名+参数
        except (json.JSONDecodeError, ValueError):
            continue   # 解析失败，尝试下一个候选
    return None   # 所有候选均失败，返回 None


# ──────────────────────────────────────────────────────────────────────────────
# 工具分发器
# ──────────────────────────────────────────────────────────────────────────────

def _dispatch_tool(name: str, args: dict, user_id: int, trace_id: str) -> str:
    """
    根据 LLM 指定的工具名，调用对应的 tools_read / tools_write 函数，
    返回序列化为 JSON 字符串的工具结果。

    为什么返回字符串而非 dict？
      OpenAI Function Calling 协议规定：tool 消息的 content 字段必须是字符串，
      不能是对象。LLM 会读取这个字符串了解工具执行结果。

    参数：
      name     : 工具名（如 "query_schedule"）
      args     : 工具参数字典（LLM 填写的参数）
      user_id  : 当前操作用户 ID（写操作工具需要）
      trace_id : 日志追踪 ID

    返回：工具执行结果的 JSON 字符串（如 '{"ok": true, "schedules": [...]}'）
    """
    today = date.today()   # 默认日期，用于补全缺省的 date_start/date_end

    # ── 根据工具名分发到对应函数 ──
    if name == "get_family_member":
        # 查家属信息，alias 参数由 LLM 从用户消息中提取
        result = tools_read.get_family_member(user_id, args["alias"], trace_id=trace_id)

    elif name == "query_schedule":
        # 查询号源，date_start/date_end 有默认值（未传则用今天和最大日期）
        result = tools_read.query_schedule(
            dept=args["dept"],
            date_start=args.get("date_start", today.isoformat()),
            date_end=args.get("date_end", (today + timedelta(days=MAX_FUTURE_DAYS)).isoformat()),
            title=args.get("title"),
            time_pref=args.get("time_pref"),
            doctor_name=args.get("doctor_name"),
            trace_id=trace_id,
        )

    elif name == "book_appointment":
        # 挂号，patient_id 和 sch_id 都是必填（LLM 必须先调查询工具获取这两个 ID）
        result = tools_write.book_appointment(
            user_id=user_id,
            patient_id=args["patient_id"],
            sch_id=args["sch_id"],#sch_id	日程ID / 排班编号	日程或排班的唯一标识符
            trace_id=trace_id,
        )

    elif name == "cancel_appointment":
        # 取消挂号：优先用 reg_id 精确定位，没有则靠 alias 解析 patient_id 再模糊查找
        cond: dict = {}
        if args.get("reg_id"):
            cond["reg_id"] = args["reg_id"]   # 精确定位路径
        else:
            # 模糊定位路径：先用昵称查 patient_id，再组合其他条件
            alias = args.get("patient_alias") or "我"   # 未指定昵称则默认"我"
            member = tools_read.get_family_member(user_id, alias, trace_id=trace_id)
            if member["ok"]:
                cond["patient_id"] = member["patient_id"]
            # 把其他辅助条件（科室/类型/日期）也加入查找条件
            for k in ("dept", "title", "date"):
                if k in args:
                    cond[k] = args[k]
        result = tools_write.cancel_appointment(user_id=user_id, cond=cond, trace_id=trace_id)

    elif name == "get_user_history":
        # 查历史挂号，dept 可选
        result = tools_read.get_user_history(
            user_id=user_id, dept=args.get("dept"), trace_id=trace_id
        )

    elif name == "get_doctor_schedule":
        # 查医生完整排班
        result = tools_read.get_doctor_schedule(
            doctor_name=args["doctor_name"],
            date_start=args.get("date_start", today.isoformat()),
            date_end=args.get("date_end", (today + timedelta(days=MAX_FUTURE_DAYS)).isoformat()),
            trace_id=trace_id,
        )
    else:
        # 未知工具名（LLM 幻觉或新工具未注册）
        result = {"ok": False, "error": f"未知工具: {name}"}

    # ensure_ascii=False：允许中文直接编码而不转义为 \uXXXX
    # cls=logger._DateEncoder：处理结果中可能存在的 date/datetime 对象
    return json.dumps(result, ensure_ascii=False, cls=logger._DateEncoder)


# ──────────────────────────────────────────────────────────────────────────────
# 规则层前置校验
# ──────────────────────────────────────────────────────────────────────────────

def _pre_validate(user_id: int, query: str, trace_id: str) -> Optional[str]:
    """
    规则层快速拦截，在调用 LLM 之前排除明显错误输入，节省 token 费用。

    检查顺序：
      1. 用户是否存在（防幽灵 user_id，如未登录直接传了 user_id=0）
      2. 时间表达是否合法（能解析成有效日期且在允许范围内）
      3. 科室是否在白名单中（避免 LLM 查询不存在的科室）

    参数：
      user_id  : 用户 ID
      query    : 用户自然语言输入
      trace_id : 日志追踪 ID

    返回：
      None : 全部检查通过，可以继续走 LLM 流程
      str  : 校验失败原因，直接返回给用户，不走 LLM
    """
    # ── 检查1：用户必须存在于数据库 ──
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT ua_id FROM user_account WHERE ua_id=%s", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return f"用户不存在（ID={user_id}），请确认登录状态。"

    # ── 检查2：时间表达合法性 ──
    # 取消/排班查询可以引用过去日期（"取消上周的号"），跳过范围检查
    is_cancel = bool(re.search(r"取消|退号|退掉", query))
    is_schedule_query = bool(re.search(r"排班|坐诊|出诊", query))
    # "下周/这周"不带具体星期几时，放给 LLM 处理（parse_time 会解析到周首/周末，LLM 再细化）
    is_week_only = bool(re.search(r"(下周|上周|这周|本周)(?![一二三四五六日天])", query))

    # 只有包含明确时间词时才做校验（避免对"帮我挂内科的号"这类不含时间词的查询做无效校验）
    time_keywords = re.search(
        r"今天|明天|后天|大后天|上周|下周|这周|本周|[上下]午|\d+点|最近|最早", query
    )
    if time_keywords and not is_cancel and not is_schedule_query and not is_week_only:
        target_date, _ = parse_time(query, trace_id=trace_id)
        if target_date is None:
            return '抱歉，我无法理解您说的时间，请换个方式描述（如"明天下午2点"）。'
        err = validate_date_range(target_date, trace_id=trace_id)
        if err:
            return err   # 日期超出范围（过去或太远的未来）

    # ── 检查3：科室白名单（仅当用户明确提到"XX科"时才检查）──
    # 只有文本中有"科"字时才做科室匹配，避免对不含科室的查询（如"明天有号吗"）做误报
    dept = fuzzy_match_dept(query)
    if "科" in query and dept is None:
        return f"抱歉，未找到您提到的科室，可用科室为：{', '.join(VALID_DEPARTMENTS)}。"

    return None   # 所有检查通过


# ──────────────────────────────────────────────────────────────────────────────
# 对外主入口
# ──────────────────────────────────────────────────────────────────────────────

def run_agent(user_id: int, query: str, trace_id: Optional[str] = None) -> dict:
    """
    处理用户自然语言请求，返回包含回复文本和工具调用 trace 的字典。

    返回格式：{"reply": str, "trace": list[dict]}

    整体流程：
      Step1. 规则前置校验（不走 LLM，快速拦截无效输入）
      Step2. 多轮 LLM 工具调用循环（最多 8 轮）：
               每轮：LLM 返回 → 有 tool_calls 则执行工具 → 结果追加进历史 → 继续下一轮
      Step3. 挂号守卫：有"约/挂/预约"意图但 LLM 跳过 book_appointment 时，
               自动强制补调一次（防止 LLM"查到号源但忘记挂号"）
      Step4. 所有 LLM 调用均失败时，降级为规则兜底（简易查询模式）

    参数：
      user_id  : 登录用户 ID（来自 JWT / session）
      query    : 用户原始自然语言输入
      trace_id : 链路追踪 ID，None 时自动生成
    """
    if trace_id is None:
        trace_id = logger.new_trace()   # 自动生成 12 位随机 hex trace_id

    t0 = time.perf_counter() * 1000   # 记录函数开始时间（毫秒）
    logger.log_info(
        f"agent.start: user_id={user_id}, query={query!r}",
        trace_id=trace_id, user_id=user_id, query=query,
    )

    # ── Step 1：规则前置校验（快速失败，不浪费 LLM token）──
    pre_err = _pre_validate(user_id, query, trace_id)
    if pre_err:
        logger.log_info(f"agent.pre_validate rejected: {pre_err}", trace_id=trace_id)
        return {"reply": pre_err, "trace": []}

    # ── 初始化 LLM 对话历史 ──
    # system prompt 作为第一条消息（包含今天日期、规则等）
    # user 消息紧随其后
    messages = [
        {"role": "system", "content": build_system_prompt()},   # 系统提示词（含当天日期）
        {"role": "user",   "content": query},                   # 用户输入
    ]

    trace: list[dict] = []   # 工具调用步骤列表（用于前端展示思考过程）
    step_num = 0              # 工具调用计数器

    today = date.today()
    # 挂号意图检测正则（用于守卫逻辑）
    _BOOK_INTENT_RE = re.compile(r"约|挂|预约")

    # ── Step 2：LLM 多轮工具调用循环 ──
    try:
        max_rounds = 8              # 防死循环上限（正常流程最多 4 步，留安全余量）
        tools_called: set[str] = set()   # 追踪本次会话已调用的工具名（守卫检测用）
        book_guard_fired = False          # 挂号守卫最多触发一次，防止无限循环

        for round_idx in range(max_rounds):
            # 调用 LLM（主模型失败自动切备用模型）
            resp = call_llm_with_fallback(messages, tools=TOOLS_SCHEMA, trace_id=trace_id)
            msg = resp.choices[0].message   # 取第一个候选回复（choices[0]）

            # ── 分支A：LLM 返回纯文字（没有 tool_calls 字段）──
            if not msg.tool_calls:
                content = (msg.content or "").strip()   # 提取文本内容，去除首尾空白

                # 尝试从文本里抢救工具调用（有些模型不遵循 Function Calling 协议）
                parsed_fake = _try_parse_tool_call_from_text(content)
                if parsed_fake:
                    tc_name, tc_args = parsed_fake
                    fake_id = f"fake_{round_idx}"   # 伪造一个 tool_call_id
                    logger.log_warning(
                        f"LLM put tool call in text, dispatching: {tc_name}",
                        trace_id=trace_id,
                    )
                    # 把伪造的工具调用插入消息历史（让 LLM 以为自己正常调了工具）
                    messages.append({
                        "role": "assistant", "content": None,
                        "tool_calls": [{
                            "id": fake_id, "type": "function",
                            "function": {
                                "name": tc_name,
                                "arguments": json.dumps(tc_args, ensure_ascii=False),
                            },
                        }],
                    })
                    step_num += 1
                    _result_str = _dispatch_tool(tc_name, tc_args, user_id, trace_id)
                    trace.append(_make_trace_step(step_num, tc_name, tc_args, _result_str))
                    # 把工具结果追加到消息历史
                    messages.append({
                        "role": "tool", "tool_call_id": fake_id,
                        "content": _result_str,
                    })
                    tools_called.add(tc_name)
                    continue   # 继续下一轮，让 LLM 基于工具结果生成回复

                # ── 挂号守卫（Book Guard）逻辑 ──
                # 触发条件：
                #   1. 用户有挂号意图（"约/挂/预约"）
                #   2. LLM 没有调 book_appointment
                #   3. LLM 已调过 query_schedule（查到了号源），或完全没调任何工具
                #   4. 守卫未触发过（最多触发一次）
                _has_book_intent = bool(_BOOK_INTENT_RE.search(query))
                _schedule_called = "query_schedule" in tools_called
                _book_called     = "book_appointment" in tools_called
                _no_tools        = len(tools_called) == 0

                if (
                    not book_guard_fired
                    and _has_book_intent
                    and not _book_called
                    and (_schedule_called or _no_tools)
                ):
                    book_guard_fired = True   # 标记守卫已触发，避免无限循环
                    logger.log_warning(
                        "book_guard: LLM skipped book_appointment, forcing it",
                        trace_id=trace_id,
                    )
                    # 从历史消息中提取 patient_id 和号源列表（用于强制挂号）
                    _patient_id, _all_schedules = None, []
                    for _m in messages:
                        if _m.get("role") == "tool":
                            try:
                                _tr = json.loads(_m["content"])
                                if "patient_id" in _tr and _patient_id is None:
                                    _patient_id = _tr["patient_id"]   # 取家属查询结果里的 patient_id
                                if _tr.get("ok") and _tr.get("schedules"):
                                    _all_schedules = _tr["schedules"]  # 取号源列表
                            except Exception as _eg:
                                logger.log_warning(
                                    f"book_guard parse error: {_eg}", trace_id=trace_id
                                )

                    # 只有恰好一个号源时才自动代用户选（多个号源时用户需要自己选，守卫不代选）
                    _sch_id = (
                        _all_schedules[0].get("sch_id")
                        if len(_all_schedules) == 1
                        else None
                    )

                    if _patient_id and _sch_id:
                        # 强制调用 book_appointment
                        logger.log_info(
                            f"book_guard: forcing book(patient={_patient_id}, sch={_sch_id})",
                            trace_id=trace_id,
                        )
                        fake_id = f"guard_book_{round_idx}"
                        book_args = {"patient_id": _patient_id, "sch_id": _sch_id}
                        # 插入伪造的 book_appointment 工具调用
                        messages.append({
                            "role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": fake_id, "type": "function",
                                "function": {
                                    "name": "book_appointment",
                                    "arguments": json.dumps(book_args, ensure_ascii=False),
                                },
                            }],
                        })
                        step_num += 1
                        _guard_result_str = _dispatch_tool(
                            "book_appointment", book_args, user_id, trace_id
                        )
                        trace.append(_make_trace_step(
                            step_num, "book_appointment", book_args, _guard_result_str
                        ))
                        messages.append({
                            "role": "tool", "tool_call_id": fake_id,
                            "content": _guard_result_str,
                        })
                        tools_called.add("book_appointment")
                        continue  # 继续下一轮，让 LLM 生成含 reg_id 的最终回复
                    else:
                        # 无法提取所需 ID（号源多于一条，用户需要选择）→ 文字提醒 LLM
                        logger.log_warning(
                            "book_guard: cannot extract IDs, injecting reminder",
                            trace_id=trace_id,
                        )
                        if content:
                            messages.append({"role": "assistant", "content": content})
                        # 注入提醒消息，让 LLM 知道它漏掉了挂号步骤
                        messages.append({
                            "role": "user",
                            "content": (
                                "你刚才查到了号源，但还没有实际挂号。"
                                "用户说的是【约/挂/预约】，请立即调用 book_appointment 完成挂号。"
                            ),
                        })
                        continue

                # ── 正常结束：LLM 给出了最终文字回复 ──
                reply = content or "抱歉，我暂时无法处理您的请求。"
                elapsed = time.perf_counter() * 1000 - t0
                logger.log_info(
                    f"agent.done: elapsed={elapsed:.0f}ms",
                    trace_id=trace_id, reply=reply, elapsed_ms=elapsed,
                )
                return {"reply": reply, "trace": trace}

            # ── 分支B：LLM 返回 tool_calls（需要执行工具）──
            # ChatCompletionMessage 对象不能直接 JSON 化，需要手动转成 dict
            messages.append({
                "role": "assistant",
                "content": msg.content,   # 可能为 None（LLM 直接调工具不附带文字）
                "tool_calls": [
                    {
                        "id": tc.id, "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,   # 已是 JSON 字符串
                        },
                    }
                    for tc in msg.tool_calls   # 一轮 LLM 可能并发调多个工具
                ],
            })

            # 逐一执行本轮所有工具调用
            _book_result_this_round = None   # 记录本轮是否有成功的挂号结果
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)   # 解析工具参数 JSON
                except json.JSONDecodeError:
                    args = {}   # 参数 JSON 格式错误时使用空字典（容错）
                logger.log_info(
                    f"agent.tool_dispatch: {name}({args})",
                    trace_id=trace_id, round=round_idx, tool=name, args=args,
                )
                tool_result_str = _dispatch_tool(name, args, user_id, trace_id)
                step_num += 1
                trace.append(_make_trace_step(step_num, name, args, tool_result_str))
                # 把工具结果追加到消息历史（role="tool"，与 tool_call_id 对应）
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": tool_result_str,
                })
                tools_called.add(name)

                # ── 短路优化：book_appointment 成功时直接用模板生成回复 ──
                # 跳过最后一轮"让 LLM 润色回复"的 LLM 调用，减少一次 API 耗时
                if name == "book_appointment":
                    try:
                        _br = json.loads(tool_result_str)
                        if _br.get("ok"):
                            _book_result_this_round = _br   # 保存挂号成功结果
                    except Exception:
                        pass

            if _book_result_this_round:
                # 从用户查询中提取家属称呼（用于回复"已为大宝预约成功"）
                _alias_m = re.search(
                    r"(大宝|二宝|小宝|老爸|老妈|妈妈|爸爸|爷爷|奶奶|宝宝|我自己|我)", query
                )
                _alias = _alias_m.group(1) if _alias_m else "您"   # 未找到称呼则用"您"
                _br = _book_result_this_round
                # 用模板字符串直接生成回复（不再调 LLM 润色）
                reply = (
                    f"已为{_alias}预约成功！"
                    f"挂号编号：{_br['reg_id']}，"
                    f"{_br['dept']} {_br['sch_type']}，"
                    f"医生：{_br['doctor_name']}，"
                    f"时间：{_br['date']} {_br['time_slot']}，"
                    f"费用：{_br['fee']} 元。请按时就诊。"
                )
                elapsed = time.perf_counter() * 1000 - t0
                logger.log_info(
                    f"agent.fast_path: book_ok, elapsed={elapsed:.0f}ms",
                    trace_id=trace_id, reply=reply, elapsed_ms=elapsed,
                )
                return {"reply": reply, "trace": trace}
            # 本轮没有挂号成功结果 → 继续下一轮，携带工具结果让 LLM 决定下一步

        # 超出 max_rounds 仍未结束（极端情况）
        return {"reply": "抱歉，处理您的请求时遇到了问题，请稍后再试或联系人工客服。", "trace": trace}

    except Exception as exc:
        # ── LLM 全部失败 → 规则兜底 ──
        # 用简单正则提取意图和科室，查一条号源告知用户（最低保证服务可用）
        logger.log_error("LLM 全部失败，启用规则兜底", trace_id=trace_id, exc=exc)
        fb   = rule_based_intent(query)   # 规则意图识别
        dept = fb["slots"].get("dept")    # 提取科室槽位

        if fb["intent"] == "query" and dept:
            # 仅"查询"意图且有科室时才能提供有意义的规则兜底
            r = tools_read.query_schedule(
                dept=dept,
                date_start=today.isoformat(),
                date_end=(today + timedelta(days=7)).isoformat(),   # 默认查最近一周
                title=fb["slots"].get("title"),
                trace_id=trace_id,
            )
            if r["ok"] and r["schedules"]:
                s = r["schedules"][0]   # 取第一条号源
                return {
                    "reply": (
                        f"[简易模式] {dept} 最近可用号：{s['doctor_name']} "
                        f"{s['date']} {s['time_slot']} {s['title']}号，剩余 {s['available']} 个。"
                    ),
                    "trace": trace,
                }

        # 完全无法服务时给出友好提示
        return {"reply": "[简易模式] 当前智能服务暂时不可用，请直接前往医院挂号窗口或拨打热线。", "trace": trace}
