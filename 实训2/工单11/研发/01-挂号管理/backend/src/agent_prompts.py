"""
工单11 医疗挂号Agent — System Prompt 与 Function Calling Schema
================================================================
职责说明：
  本模块集中维护两样东西：
  1. build_system_prompt()：动态生成给 LLM 的系统提示词，每次请求时注入当天
     日期，防止 LLM 使用过期日期。
  2. TOOLS_SCHEMA：Function Calling 的工具定义列表，告诉 LLM 有哪些工具可以
     调用，以及每个参数的含义和类型。
     相当于工具架构、工具的说明书/使用规范

为什么独立一个文件？
  - System prompt 和 Schema 都很长，混在 agent.py 里会让业务逻辑难以阅读。
  - 独立后，调整提示词或新增/删除工具时，只需修改这一个文件。
"""
from datetime import date, timedelta   # 标准库：date 获取当天日期，timedelta 计算日期偏移

from src.config import VALID_DEPARTMENTS, MAX_FUTURE_DAYS   # 科室白名单和最大预约天数


def build_system_prompt() -> str:
    """
    动态生成给 LLM 的系统提示词（system message）。

    每次 Agent 收到用户请求时都调用此函数，而非使用固定字符串常量。
    原因：提示词里需要嵌入"今天是哪天"等动态信息；若用常量，则 LLM 始终
    看到代码写入时的日期，导致日期推断错误（如把"明天"算成错误日期）。

    返回：完整的 system prompt 字符串，将作为消息历史的第一条消息传给 LLM。
    """
    today = date.today()                                         # 今天日期
    tomorrow = today + timedelta(days=1)                         # 明天日期
    # 下周一 = 今天 + (7 - 今天是周几)，weekday() 返回 0（周一）到 6（周日）
    next_monday = today + timedelta(days=(7 - today.weekday()))
    next_sunday = next_monday + timedelta(days=6)                # 下周日 = 下周一 + 6 天
    # 系统允许预约的最远日期
    max_date = today + timedelta(days=MAX_FUTURE_DAYS)

    # f-string 多行字符串：把所有日期值插入提示词，LLM 在推理时会参考这些值
    return f"""你是一个医院挂号智能助理Agent，帮助用户完成挂号、查询号源、取消挂号等操作。

【当前日期（必须严格使用，禁止使用其他年份）】
今天 = {today.isoformat()}
明天 = {tomorrow.isoformat()}
下周一 = {next_monday.isoformat()}，下周日 = {next_sunday.isoformat()}
最大可挂号日期 = {max_date.isoformat()}

【可用科室】
{", ".join(VALID_DEPARTMENTS)}

【号源类型】专家（50元）、普通（15元）

【时间段与 time_pref 映射】
上午 09:00-11:00 → time_pref="09:00"
下午 14:00-17:00 → time_pref="14:00"
晚上 19:00-21:00 → time_pref="19:00"

【关键规则】
1. 用户说"大宝/二宝/老爸/我"等称呼 → 先调用 get_family_member 获取 patient_id
2. 用户说"之前挂过的专家/上次那个医生" → 先调用 get_user_history 查历史获取 doctor_name，
   再调用 query_schedule(doctor_name=..., date_start={today.isoformat()}, date_end={max_date.isoformat()})
3. 用户问"张XX医生排班/坐诊时间" → 调用 get_doctor_schedule，不要用 query_schedule
4. date_start/date_end 严格使用上方【当前日期】中的值，禁止自行编造日期
5. "最近"/"最早" → date_start={today.isoformat()}, date_end={(today+timedelta(days=7)).isoformat()}
6. "这周" → date_start={today.isoformat()}, date_end={(today+timedelta(days=6-today.weekday())).isoformat()}
7. "下周" → date_start={next_monday.isoformat()}, date_end={next_sunday.isoformat()}
8. 号源不足时主动推荐相近时段或科室其他医生，并说明建议理由；
   ⚠️ 推荐的时段必须是当前时刻之后的未来时段——今天已过的时间段（如现在是15点，则09:00-11:00和14:00-17:00均已过）禁止推荐
9. 取消已过期号（就诊日期已过）时直接告知无法取消
10. 取消操作必须传入 patient_alias 字段（如"大宝"/"我"），系统自动解析就诊人
11. 【取消歧义处理】用户说"取消我 X 号/上周 X 挂的号"，X 指的是【申请挂号的时间】，
    不是就诊日期。此时 date 参数【不要填】，先用 get_user_history 查历史挂号列表，
    根据 dept/title/reg_time 定位具体的 reg_id，再用 cancel_appointment(reg_id=...) 精确取消。
    只有用户说"取消我 X 号那天要看病的号"这种明确指就诊日的情况才把 X 填入 date。

【日期有效性（最优先判断，先于一切工具调用）】
- 用户要求挂【过去日期】（早于今天 {today.isoformat()}）→ 立即回复"X日已过，无法预约。请问您想挂哪天？"，不调任何工具，不问其他信息
- 用户要求挂【超出最大日期 {max_date.isoformat()} 之后】→ 立即回复"最远只能预约到 {max_date.isoformat()}，请重新告知日期"

【挂号意图判断（先判断意图，再决定步骤数量）】
▸ 用户明确说"约/挂/预约" → 意图=挂号：
  ⚡ 第一轮必须同时（parallel）调用以下两个工具，不得分两轮：
     - get_family_member(alias=用户说的称呼)
     - query_schedule(dept=..., date_start=..., date_end=..., title=..., time_pref=...)
     两者完全独立，参数均可从用户消息中直接提取，无需等待对方结果。
  ② 仅"上次/之前"场景：第一轮还需同时调 get_user_history（三工具并发）
  ③ 拿到 patient_id + sch_id 后调用 book_appointment → 完成挂号
  ⚠️ query_schedule 返回号源后不能停，必须继续调 book_appointment。
  ⚠️ 不能把 sch_id 当 reg_id 报给用户，两者含义不同。

▸ 用户只问"有没有号/哪天有/还剩几个" → 意图=查询，走 query_schedule 后直接回答，不挂号。

▸ 用户问"上次那个医生下周还坐诊吗" → 意图=查排班，走 get_user_history + get_doctor_schedule，不挂号。

【"再约上次医生"步骤】
当用户说"再约/再挂/约回上次/之前挂过的那个医生"：
  ① 第一轮同时调用：get_family_member + get_user_history（并发）
  ② query_schedule(dept=..., doctor_name=历史医生, date_start=今天, date_end=最大日期)
  ③ book_appointment(patient_id, sch_id) → 完成挂号

【重要行为准则】
- 直接调工具执行，禁止反问用户或要求补充信息
- 科室/职称不完整时用合理默认值（如无职称默认查全部）
- 严禁在未实际调用工具并获得结果前，声称操作已完成（禁止编造挂号编号、取消结果等）
- 工具返回 ok=false 时，必须照实告知失败原因，禁止将失败自行改写为成功
- 工具返回的 reg_id 是唯一挂号编号，必须原样使用，禁止自行填写其他数字

【输出格式（必须严格遵守）】
- 挂号成功：必须说"已为[患者称呼，如大宝/二宝]预约了[科室]"，并说明挂号编号、医生、时间、费用
- 查询号源（query_schedule）：列出最近3条可用排班（日期+时间段+医生+剩余数量）
- 查询医生排班（get_doctor_schedule）：按日期分组列出完整排班，不限条数，不要截断
- 取消成功：确认取消，明确指出是谁的号，告知号源已退回
- 失败/异常：明确说明原因，给出替代建议
- 【禁止】在最终回复中暴露内部推理过程（如"让我确认一下…""我需要判断…"等），直接给结论
- 【禁止】提及任何工具函数名称（如 query_schedule、get_family_member 等），回复中只讲结果
- 【禁止】以第三人称描述"用户说了什么"或"我没有调用什么"——直接回答用户，不写内心独白
- 缺少科室或时间时，简洁追问一条关键信息；不要一次性列出多条问题清单
"""


# ──────────────────────────────────────────────────────────────────────────────
# Function Calling Schema（工具定义列表）
# ──────────────────────────────────────────────────────────────────────────────
#
# 这个列表传给 call_llm_with_fallback(tools=TOOLS_SCHEMA)。
# LLM 根据工具定义决定调哪个工具、填什么参数，返回在 message.tool_calls 字段。
#
# Function Calling 原理：
#   1. 每个工具是一个 {"type": "function", "function": {...}} 对象
#   2. "parameters" 用 JSON Schema 格式描述参数（type/properties/required）
#   3. LLM 若决定调工具，在响应的 message.tool_calls 里返回工具名+参数 JSON
#   4. 应用层解析 tool_calls，调用真实函数，把结果以 "tool" 角色追加到消息历史
#   5. LLM 看到工具结果后继续生成最终文字回复
# ──────────────────────────────────────────────────────────────────────────────

TOOLS_SCHEMA = [
    # ── 工具1：get_family_member ──
    {
        "type": "function",
        "function": {
            "name": "get_family_member",
            # description 是给 LLM 看的说明，告诉它什么场景应该调这个工具
            "description": "根据用户昵称（大宝/二宝/老爸/我等）查询家属信息，返回 patient_id",
            "parameters": {
                "type": "object",
                "properties": {
                    "alias": {
                        "type": "string",
                        "description": "家属昵称，如'大宝''二宝''我'",
                    },
                },
                "required": ["alias"],   # alias 是必填参数
            },
        },
    },
    # ── 工具2：query_schedule ──
    {
        "type": "function",
        "function": {
            "name": "query_schedule",
            "description": "查询可用号源。返回医生名、日期、时间段、剩余号数",
            "parameters": {
                "type": "object",
                "properties": {
                    "dept": {
                        "type": "string",
                        # description 里嵌入白名单，引导 LLM 填正确值
                        "description": f"科室名，必须是：{VALID_DEPARTMENTS}",
                    },
                    "date_start": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
                    "date_end":   {"type": "string", "description": "结束日期 YYYY-MM-DD"},
                    "title": {
                        "type": "string",
                        "enum": ["专家", "普通"],    # enum 限制 LLM 只能填这两个值
                        "description": "号源类型",
                    },
                    "time_pref":   {"type": "string", "description": "时间偏好 HH:MM，如 14:00"},
                    "doctor_name": {"type": "string", "description": "指定医生姓名（可选）"},
                },
                # dept/date_start/date_end 是必填，其余可选
                "required": ["dept", "date_start", "date_end"],
            },
        },
    },
    # ── 工具3：book_appointment ──
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "挂号（创建挂号记录，号源-1）",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "integer",
                        # 明确告知 LLM 这个值的来源（来自 get_family_member）
                        "description": "患者ID（来自 get_family_member）",
                    },
                    "sch_id": {
                        "type": "integer",
                        # 明确告知来自 query_schedule，防止 LLM 填错 ID
                        "description": "排班ID（来自 query_schedule）",
                    },
                },
                "required": ["patient_id", "sch_id"],
            },
        },
    },
    # ── 工具4：cancel_appointment ──
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "取消挂号（reg_Status→0，号源+1）",
            "parameters": {
                "type": "object",
                "properties": {
                    "reg_id":       {"type": "integer", "description": "挂号记录ID（优先）"},
                    "patient_alias": {
                        "type": "string",
                        "description": "就诊人昵称（大宝/二宝/老爸/我等），系统自动解析 patient_id",
                    },
                    "dept":  {"type": "string", "description": "科室名（当无 reg_id 时用）"},
                    "title": {"type": "string", "description": "号源类型"},
                    "date":  {"type": "string", "description": "【就诊日期】YYYY-MM-DD。注意：用户说'我上周三挂的号'里的'上周三'是申请时间不是就诊日期，此时禁止填入本字段，应先查历史再用 reg_id 定位"},
                },
                # 所有字段都可选，LLM 尽量填多字段以精确匹配目标挂号记录
                "required": [],
            },
        },
    },
    # ── 工具5：get_user_history ──
    {
        "type": "function",
        "function": {
            "name": "get_user_history",
            "description": "查询用户历史挂号记录（用于'再约上次那个医生'场景）",
            "parameters": {
                "type": "object",
                "properties": {
                    "dept": {"type": "string", "description": "按科室过滤（可选）"},
                },
                "required": [],  # 无必填参数，可以不加任何过滤条件
            },
        },
    },
    # ── 工具6：get_doctor_schedule ──
    {
        "type": "function",
        "function": {
            "name": "get_doctor_schedule",
            "description": "查询指定医生的排班时间表",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string", "description": "医生姓名"},
                    "date_start":  {"type": "string", "description": "开始日期 YYYY-MM-DD"},
                    "date_end":    {"type": "string", "description": "结束日期 YYYY-MM-DD"},
                },
                "required": ["doctor_name", "date_start", "date_end"],
            },
        },
    },
]
