"""
工单11 医疗挂号Agent - 工具函数辅助层
包含时间解析、日期校验、科室/号源类型模糊匹配等纯计算函数（无副作用）。
"""
import re   # 标准库：正则表达式
from datetime import date, datetime, timedelta   # 标准库：日期/时间操作
from typing import Optional, Tuple               # 标准库：类型注解

# dateparser：第三方自然语言日期解析库
# 能将"明天下午3点"、"next Monday"等自然语言转换为 datetime 对象
# 支持中文、英文等多种语言
import dateparser

from src.config import MAX_FUTURE_DAYS, VALID_DEPARTMENTS, VALID_TITLES
from src import logger   # 结构化日志


# ─────────────────────────────────────────────
# 时间解析
# ─────────────────────────────────────────────

# dateparser 解析设置
_DATEPARSER_SETTINGS = {
    "PREFER_DAY_OF_MONTH": "first",    # 当只说月份时，默认取每月第一天
    "PREFER_DATES_FROM": "future",     # 优先解析为未来日期（避免"明天"被解析为过去）
    "DATE_ORDER": "YMD",               # 日期格式顺序：年-月-日（中文习惯）
    "RETURN_AS_TIMEZONE_AWARE": False, # 返回不带时区信息的 naive datetime
    "RELATIVE_BASE": None,             # 相对时间的基准（每次调用时动态设置为当前时刻）
}

# 中文时间关键词规范化映射表（把中文转成 dateparser 更好识别的格式）
# 每条格式：(正则模式, 替换字符串或函数)
_CN_NORM = [
    (r"后天", "the day after tomorrow"),                          # 后天 → 英文
    (r"大后天", "in 3 days"),                                     # 大后天 → 3天后
    (r"下个?(周|星期)([一二三四五六日天])", r"next week \2"),     # 下周X → next week X
    (r"这个?(周|星期)([一二三四五六日天])", r"this week \2"),     # 这周X → this week X
    (r"上个?(周|星期)([一二三四五六日天])", r"last week \2"),     # 上周X → last week X
    # "上午3点" → "AM 3:00"，"下午3点" → "PM 3:00"
    (r"[上下]午(\d{1,2})点", lambda m: f"{'AM' if '上' in m.group(0) else 'PM'} {m.group(1)}:00"),
]

# 中文星期几 → Python weekday 数字（0=周一 ... 6=周日）
_WEEKDAY_CN = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}


def _cn_weekday_delta(keyword: str, is_last: bool = False) -> Optional[date]:
    """
    将"下周三"/"上周三"中的星期几汉字转为具体日期。

    参数：
      keyword : 星期几的汉字（如"三"/"五"/"日"）
      is_last : True 表示"上周"，False 表示"下周"

    返回：计算出的具体 date，无法识别时返回 None

    算法：
      1. 把汉字星期几转成 weekday 数字（0-6）
      2. 计算今天距目标星期几的天数差
      3. is_last=False（下周）：确保 diff > 0（下周的某天一定在今天之后）
         is_last=True（上周）：确保 diff < 0（上周的某天一定在今天之前）
    """
    wd = _WEEKDAY_CN.get(keyword)   # 汉字星期 → 数字
    if wd is None:
        return None   # 不认识的汉字（如"零"），返回 None
    today_wd = date.today().weekday()   # 今天是周几（0-6）
    diff = wd - today_wd               # 目标星期几 - 今天星期几 = 天数差

    if is_last:
        # 上周：需要 diff 在 -7 到 -1 之间（上周的天一定是负数天差）
        diff -= 7
        if diff >= 0:
            diff -= 7
    else:
        # 下周：需要 diff 在 1 到 7 之间（下周的天一定是正数天差）
        diff += 7
        if diff <= 0:
            diff += 7
    return date.today() + timedelta(days=diff)


def parse_time(text: str, trace_id: str = "-") -> Tuple[Optional[date], Optional[str]]:
    """
    将用户自然语言时间表达解析为 (目标日期, 时间偏好)。

    参数：
      text     : 用户输入的文本（如"明天下午2点"/"下周三上午"）
      trace_id : 日志追踪 ID

    返回：
      (date, time_pref)
        date      : 解析出的日期，无法解析时为 None
        time_pref : "HH:MM" 格式的时间偏好（如"14:00"），未提及时为 None
      (None, None) 表示完全无法解析

    策略（按顺序，第一个匹配成功即返回）：
      1. 正则匹配常见模式（今天/明天/后天/大后天/上下周），速度 <1ms
      2. dateparser 库解析（支持中英文自然语言），耗时约 5ms
      3. 均失败 → 返回 (None, None)，上层会把原文直接交给 LLM 处理
    """
    now = datetime.now()   # 当前时刻（用于 dateparser 的 RELATIVE_BASE）
    today = now.date()     # 今天的日期
    time_pref = None       # 时间偏好，默认为 None（未指定具体时间）

    # ── 先提取时间偏好（如"下午2点" → "14:00"）──
    time_match = re.search(r"([上下]午)\s*(\d{1,2})\s*点", text)
    if time_match:
        period = time_match.group(1)    # "上午" 或 "下午"
        hour = int(time_match.group(2)) # 小时数字（如 2、3、10）
        if period == "下午" and hour < 12:
            hour += 12   # 下午 2 点 → 14 点（12小时制转24小时制）
        time_pref = f"{hour:02d}:00"   # 格式化为 "HH:00"

    # ── 快速规则匹配（今天/明天/后天/大后天）──
    if "今天" in text or "今日" in text:
        return today, time_pref   # 今天
    if "明天" in text or "明日" in text:
        return today + timedelta(days=1), time_pref   # 明天 = 今天+1
    if "后天" in text:
        return today + timedelta(days=2), time_pref   # 后天 = 今天+2
    if "大后天" in text:
        return today + timedelta(days=3), time_pref   # 大后天 = 今天+3

    # ── 上周/下周 + 星期几（或仅上下周不带具体星期）──
    for kw, is_last in [("上周", True), ("上个星期", True), ("上个礼拜", True),
                        ("下周", False), ("下个星期", False), ("下个礼拜", False)]:
        if kw in text:
            # 在关键词后5个字符内查找星期几汉字（如"下周三"中的"三"）
            m = re.search(r"[一二三四五六日天]", text[text.index(kw):text.index(kw) + 5])
            if m:
                # 带具体星期几：计算精确日期
                d = _cn_weekday_delta(m.group(0), is_last=is_last)
                if d:
                    logger.log_debug(f"parse_time 规则匹配: {text!r} → {d}",
                                     trace_id=trace_id)
                    return d, time_pref
            else:
                # 不带具体星期几：取下周一或上周一作为代表
                diff = (7 - today.weekday()) if not is_last else -(today.weekday() + 7)
                d = today + timedelta(days=diff)
                logger.log_debug(f"parse_time 周范围起点: {text!r} → {d}",
                                 trace_id=trace_id)
                return d, time_pref

    # ── "最近"/"最早" → 今天（查最早可用号源）──
    if re.search(r"最近|最早|earliest", text):
        return today, time_pref

    # ── "这周"/"本周" + 星期几 ──
    this_week_m = re.search(r"(这周|本周|这个星期)[^\d一二三四五六日天]*([一二三四五六日天])", text)
    if this_week_m:
        wd = _WEEKDAY_CN.get(this_week_m.group(2))   # 汉字星期 → 数字
        if wd is not None:
            diff = wd - today.weekday()
            if diff < 0:
                diff += 7   # 已过的星期几 → 顺延到下周（本周视为未来）
            return today + timedelta(days=diff), time_pref

    # ── dateparser 兜底：处理更复杂的时间表达 ──
    try:
        settings = dict(_DATEPARSER_SETTINGS, RELATIVE_BASE=now)  # 注入当前时刻作为相对时间基准
        parsed = dateparser.parse(text, languages=["zh"], settings=settings)
        # 过滤掉超过 1 年的解析结果（防止 dateparser 误解析成遥远的未来/过去）
        if parsed and abs((parsed.date() - today).days) <= 365:
            logger.log_debug(f"parse_time dateparser: {text!r} → {parsed.date()}",
                             trace_id=trace_id)
            # 若 dateparser 解析出了具体时间（非 0 点），且用户未明确指定时间偏好
            if not time_pref and parsed.hour != 0:
                time_pref = f"{parsed.hour:02d}:{parsed.minute:02d}"
            return parsed.date(), time_pref
    except Exception:
        pass   # dateparser 内部异常：忽略，继续走兜底逻辑

    # 所有解析策略均失败
    logger.log_warning(f"parse_time 无法解析: {text!r}", trace_id=trace_id)
    return None, None


def validate_date_range(target: date, trace_id: str = "-") -> Optional[str]:
    """
    校验目标日期是否在可挂号范围 [今天, 今天+MAX_FUTURE_DAYS] 内。

    参数：
      target   : 要校验的目标日期

    返回：
      None     : 日期合法，通过校验
      str      : 错误原因描述（直接返回给用户）

    边界情况：
      target < today          → "不可挂过去日期的号"
      target > today+14天     → "超出挂号范围"
    """
    today = date.today()
    if target < today:
        return "不可挂过去日期的号"   # 过去的日期无法挂号
    if (target - today).days > MAX_FUTURE_DAYS:
        return f"超出挂号范围（最多可预约 {MAX_FUTURE_DAYS} 天内的号）"
    return None   # 合法


# ─────────────────────────────────────────────
# 槽位校验辅助（用于规则层快速校验，异常拦截不走 LLM）
# ─────────────────────────────────────────────

def fuzzy_match_dept(text: str) -> Optional[str]:
    """
    从用户输入文本中模糊匹配科室名（白名单精确子串匹配）。

    参数：
      text : 用户输入的文本

    返回：
      白名单中的科室名（如"内科"），未匹配到则返回 None

    实现方式：
      遍历 VALID_DEPARTMENTS 列表，检查科室名是否是 text 的子串。
      例如 text="帮我挂内科的号" 能匹配到 "内科"。
    """
    for dept in VALID_DEPARTMENTS:
        if dept in text:   # 科室名作为子串存在于用户输入中
            return dept
    return None   # 文本中未提及任何已知科室


def fuzzy_match_title(text: str) -> Optional[str]:
    """
    从用户输入文本中匹配号源类型（专家号/普通号）。

    返回 "专家" / "普通" / None（未提及则返回 None，表示不限类型）。
    """
    if "专家" in text:
        return "专家"
    if "普通" in text or "普" in text:   # "普"也能匹配（如"普通号"/"普号"）
        return "普通"
    return None


def build_date_range(target: date, span_days: int = 1) -> Tuple[str, str]:
    """
    根据目标日期和跨度天数，返回 (start_iso, end_iso) 日期范围字符串。

    参数：
      target    : 起始日期
      span_days : 日期范围跨度（默认 1 天，即只查当天）

    返回：
      (start_iso, end_iso)，格式 "YYYY-MM-DD"，用于 query_schedule 的入参

    例如：build_date_range(date(2026,7,10), 3) → ("2026-07-10", "2026-07-12")
    """
    return target.isoformat(), (target + timedelta(days=span_days - 1)).isoformat()
