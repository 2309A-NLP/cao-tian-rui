"""
Agent 引擎 — Function Calling 循环核心
工单编号：人工智能 NLP-Agent 数字人项目-日程提醒智能体任务

流程：user_msg → LLM(带tools) → tool_calls? → 执行工具 → 结果回填 → 再调LLM → 回复
"""
import json
import uuid
import time
from datetime import datetime, timedelta
from backend.llm_provider import LLMProvider
from backend.tools import TOOLS_SCHEMA, ToolExecutor
from backend.database import DatabaseManager
from backend.logger import get_logger

logger = get_logger("engine")


def _now() -> datetime:
    return datetime.now()


_TODAY = _now().strftime("%Y-%m-%d")
_TOMORROW = (_now() + timedelta(days=1)).strftime("%Y-%m-%d")
_TOMORROW_WEEKDAY = ["一", "二", "三", "四", "五", "六", "日"][(_now() + timedelta(days=1)).weekday()]
_WEEKDAY_CN = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}


def _build_system_prompt() -> str:
    """构建带实时时间的 System Prompt"""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    weekday = _WEEKDAY_CN[now.weekday()]
    tomorrow_weekday = _WEEKDAY_CN[(now + timedelta(days=1)).weekday()]

    return f"""你是"小暖"，一位温柔、细心的日程提醒助手。你的职责是帮助主人管理日程，做到增删改查准确无误。

## ⏰ 当前时间信息
- 现在是 **{now.strftime('%Y年%m月%d日 %H:%M')}**（星期{weekday}）
- 今天的日期是 **{today}**
- 明天的日期是 **{tomorrow}**（周{tomorrow_weekday}）

## 🔴 致命规则（违反任何一条都不可接受）

### 规则1：强制使用工具（100%数据库调用）
**任何涉及日程数据的操作（增加/查询/修改/删除），都必须调用工具操作数据库 schedules 表。**
**在任何情况下，你都要调用数据库。**
绝对不允许凭猜测或记忆直接回答日程相关的问题。用户问日程，必须先查数据库再回答。

### 规则2：序号 ≠ 数据库ID（极其重要）
查询结果中的格式是 `#序号 [DB_ID=X]`。
- **#1、#2、#3...** 是显示序号（按时间排序的第1/2/3个）
- **DB_ID=X** 是数据库中的真实ID
- **当用户说"日程1""删日程1""第1个"等，'1'指的是显示序号 #1**
- 你必须：去聊天记录里找到上一次查询结果中 #1 对应的 DB_ID，然后用 DB_ID 调用工具。
- **不要直接把用户说的"1"当成 DB_ID 传参！**
- 如果你不确定某个序号对应哪个 DB_ID，就先调用 query_schedules 重新查询。

### 规则3：不完整信息必须追问
如果用户输入缺少必填字段（title/时间），**不要**强行调用 add_schedule。
列出缺少的信息，逐项追问用户。
```
示例：
用户："添加一个日程"
你："好的主人～请问：
1️⃣ 要提醒的是什么事呢？
2️⃣ 什么时间呢？（比如：下午3点、明天上午8点）"
```

### 规则4：时间解析
你必须把用户的口语时间转为标准24小时格式：
| 用户说的 | 转换成 |
|---------|--------|
| "下午5点""傍晚5点""5pm" | 17:00 |
| "上午8点""早上8点""8am" | 08:00 |
| "中午12点""正午" | 12:00 |
| "晚上8点""8pm" | 20:00 |
| "下午3点半" | 15:30 |
| "半夜12点" | 00:00 |
| "明天"的日程 | 日期={tomorrow} |
| "后天"的日程 | 日期={day_after} |
| 没说日期 | 默认今天 {today} |

**特殊格式：竖线分隔输入**
用户有时会用竖线格式输入，如 `15:15|0000001|提醒我买咖啡`：
- 第1段 `15:15` → schedule_time，直接使用
- 第2段 `0000001` → 用户编号，忽略，不存储
- 第3段 `提醒我买咖啡` → 事项内容，去掉"提醒我/提醒您"后得到 title="买咖啡"
解析后正常调用 add_schedule。

### 规则5：删除必须先确认
流程：搜索 → 展示（带序号和DB_ID）→ 等待用户说"确认/可以/是的" → 执行删除。
**绝对禁止**在用户未明确确认的情况下直接删除。
当用户说"删除日程1""取消日程2"等，'1'和'2'是显示序号，你需要从最近一次查询结果中找到 #1/#2 对应的 DB_ID。

### 规则6：循环日程识别
根据用户表达自动判断重复类型：
- "每天""天天" → daily
- "每周X""每个星期X" → weekly, rule={{"weekdays":[X]}}
- "工作日" → weekly, rule={{"weekdays":[1,2,3,4,5]}}
- "周末" → weekly, rule={{"weekdays":[6,7]}}
- "每月Y号/日" → monthly, rule={{"day":Y}}
- 无循环说明 → none

### 规则7：不要重复调用
同一条添加/删除/修改请求，只调用一次对应工具。工具返回成功后不要再重复。

### 规则9：展示给用户时隐藏 DB_ID
工具返回的格式是 `#序号 [DB_ID=X] 时间 事项`，**展示给用户时只显示序号和内容，不要把 [DB_ID=X] 写进回复**。
DB_ID 是你内部使用的，用户不需要知道。
正确示例：`1. **09:00 站会** (循环日程)`
错误示例：`1. **#1 [DB_ID=28] 09:00 站会**`

### 规则8：已过时间也可以添加
用户说的时间即使已经过了当前时间，也要正常添加。例如用户说"15:15提醒我买咖啡"而当前是15:30，仍然要调用 add_schedule 正常添加。数据库保存所有日程，是否过期由系统判断。

## 📋 日程操作指引

### 添加日程
用户："添加日程：下午5点开会"
→ 解析：title="开会", schedule_date={today}, schedule_time="17:00"
→ 调用 add_schedule
→ 回复："好的主人～已为您添加日程：今天 17:00「开会」⏰"

用户："明天上午8点提醒我买咖啡"
→ title="买咖啡", date={tomorrow}, time="08:00"
→ 调用 add_schedule
→ 回复："已记录～明天 08:00「买咖啡」☕"

用户："每个工作日早上9点站会"
→ title="站会", date={today}, time="09:00", repeat_type="weekly", repeat_rule='{{"weekdays":[1,2,3,4,5]}}'
→ 调用 add_schedule

### 查询日程
用户："我今天的日程有哪些？"
→ 调用 query_schedules(schedule_date="{today}")
→ 把结果按时间顺序列出来，格式：`#1 [DB_ID=4] 17:00 开会`

用户："看看我的日程"
→ 调用 query_schedules() 查询全部
→ 按日期+时间列出

### 修改日程
用户："把日程3改到下午4点"
→ 调用 update_schedule(schedule_id=3, schedule_time="16:00")
→ 展示修改前后对比

### 删除日程
用户："删除日程1"
→ 用户说的"1"是显示序号！去聊天记录找我最近一次查询结果
→ 看到 #1 [DB_ID=4] 下午5点开会
→ 所以调 delete_schedule(confirmed=false, schedule_id=4, keyword="开会")
→ 展示查到的日程内容，问"确认删除这条吗？"
→ 用户说"确认"后 → delete_schedule(confirmed=true, schedule_id=4)

## 💬 回复风格
- 温柔亲切，用"主人"称呼用户
- 用 emoji 让回复更生动（📅⏰✅🗑️☕🔁）
- 列出日程时按时间排序，格式清晰
- 操作成功时回显完整信息（日期+时间+事项）
- 到时间提醒的话术（任意选择）：
  "温馨提醒：（事项）的时间到啦，主人！✨"
  "主人！是时候（事项）了喔~ 🌸"
  "亲爱的主人，现在是（事项）的时候啦！💕"
  "嘿，主人，该（事项）了哦~ ⏰"

## 🚀 开场
首次对话时简短打招呼，告诉主人可以帮你管理日程。"""


class Session:
    """会话：持有完整消息链"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"sch_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.messages: list[dict] = [{"role": "system", "content": _build_system_prompt()}]
        self.opening_sent = False

    def add(self, msg_dict: dict = None, **msg):
        if msg_dict is not None:
            self.messages.append(msg_dict)
        else:
            self.messages.append(msg)

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})


OPENING_MESSAGE = (
    "🌸 主人你好呀～我是你的专属日程助手「小暖」！\n\n"
    "我可以帮你：\n"
    "📝 **添加日程**：比如「添加日程：下午5点开会」\n"
    "📋 **查看日程**：比如「我今天有哪些日程？」\n"
    "✏️ **修改日程**：比如「把日程3改到下午4点」\n"
    "🗑️ **删除日程**：比如「删除日程1」\n"
    "🔁 **循环日程**：比如「每个工作日早上9点站会」\n\n"
    "你尽管用最日常的话跟我说，我会帮你安排得明明白白的～"
)


class AgentEngine:
    """日程提醒 Agent 引擎"""

    def __init__(self, llm: LLMProvider, db: DatabaseManager, max_rounds: int = 5):
        self.llm = llm
        self.executor = ToolExecutor(db)
        self.max_rounds = max_rounds
        self.sessions: dict[str, Session] = {}

    def get_session(self, session_id: str = None) -> Session:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        sess = Session(session_id)
        self.sessions[sess.session_id] = sess
        return sess

    def chat(self, user_message: str, session_id: str = None) -> dict:
        """同步对话：返回完整回复"""
        session = self.get_session(session_id)
        is_first_msg = len(session.messages) <= 1
        session.add_user(user_message)
        tool_calls_made = []
        called_tools_this_turn = set()

        for rnd in range(self.max_rounds):
            logger.info(f"Agent 第{rnd+1}轮, 消息数={len(session.messages)}")

            try:
                resp = self.llm.chat(session.messages, tools=TOOLS_SCHEMA)
            except Exception as e:
                err = f"抱歉主人，服务暂时不可用：{e}"
                session.add_assistant(err)
                return {"reply": err, "session_id": session.session_id, "tool_calls_made": tool_calls_made}

            finish = resp.get("finish_reason", "stop")

            if finish == "tool_calls":
                tcs = resp.get("tool_calls", [])
                if not tcs:
                    continue

                session.add({"role": "assistant", "content": resp.get("content") or "",
                             "tool_calls": tcs})

                for tc in tcs:
                    fn = tc.get("function", {})
                    name = str(fn.get("name", ""))
                    args = self._parse_args(str(fn.get("arguments", "{}")))

                    # 防重复调用
                    dedup_key = f"{name}|{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                    if dedup_key in called_tools_this_turn:
                        logger.warning(f"跳过重复调用: {dedup_key}")
                        result_text = f"刚才已执行过 {name}，跳过重复调用。"
                        session.add({"role": "tool", "tool_call_id": tc.get("id", ""),
                                     "content": result_text})
                        continue

                    called_tools_this_turn.add(dedup_key)
                    result_text = self.executor.execute(name, args)
                    session.add({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": result_text})
                    tool_calls_made.append({"name": name, "arguments": args})
                continue

            # finish == "stop" → 返回文本
            content = resp.get("content", "")
            if not content:
                content = "已处理。"

            if is_first_msg and not session.opening_sent:
                content = OPENING_MESSAGE + "\n\n" + content
            session.opening_sent = True
            session.add_assistant(content)
            return {"reply": content, "session_id": session.session_id, "tool_calls_made": tool_calls_made}

        err = "主人，处理步骤较多，请简化需求后重试哦～"
        session.add_assistant(err)
        return {"reply": err, "session_id": session.session_id, "tool_calls_made": tool_calls_made}

    def chat_stream(self, user_message: str, session_id: str = None):
        """
        流式对话生成器（真实 LLM token 流）。Yield 事件:
          {"type": "token", "content": "字"}
          {"type": "tool_call", "name": "...", "arguments": {...}}
          {"type": "tool_result", "name": "...", "result": "..."}
          {"type": "done", "session_id": "..."}
        """
        session = self.get_session(session_id)
        is_first_msg = len(session.messages) <= 1
        session.add_user(user_message)
        called_tools_this_turn = set()

        # 首次消息先推送开场白（token 逐字流出）
        if is_first_msg and not session.opening_sent:
            for ch in OPENING_MESSAGE:
                yield {"type": "token", "content": ch}
            yield {"type": "token", "content": "\n\n"}
            session.opening_sent = True

        for rnd in range(self.max_rounds):
            logger.info(f"Agent stream 第{rnd+1}轮")

            content_buf = []
            tool_calls_list = []
            finish_reason = "stop"

            try:
                for event in self.llm.chat_stream(session.messages, tools=TOOLS_SCHEMA):
                    etype = event.get("type")
                    if etype == "token":
                        token = event["content"]
                        content_buf.append(token)
                        yield {"type": "token", "content": token}
                    elif etype == "tool_calls":
                        tool_calls_list = event.get("tool_calls", [])
                    elif etype == "done":
                        finish_reason = event.get("finish_reason", "stop")
            except Exception as e:
                yield {"type": "token", "content": f"抱歉主人，服务不可用：{e}"}
                yield {"type": "done", "session_id": session.session_id}
                return

            if finish_reason == "tool_calls" and tool_calls_list:
                tcs = tool_calls_list
                session.add({"role": "assistant",
                             "content": "".join(content_buf) or "",
                             "tool_calls": tcs})

                for tc in tcs:
                    fn = tc.get("function", {})
                    name = str(fn.get("name", ""))
                    args = self._parse_args(str(fn.get("arguments", "{}")))
                    yield {"type": "tool_call", "name": name, "arguments": args}

                    dedup_key = f"{name}|{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                    if dedup_key in called_tools_this_turn:
                        result_text = f"刚才已执行过 {name}，跳过重复调用。"
                    else:
                        called_tools_this_turn.add(dedup_key)
                        result_text = self.executor.execute(name, args)

                    session.add({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": result_text})
                    yield {"type": "tool_result", "name": name, "result": result_text[:300]}
                continue

            # 文本回复 — token 已实时推出去了，只需记会话历史
            content = "".join(content_buf) or "已处理。"
            session.add_assistant(content)
            yield {"type": "done", "session_id": session.session_id}
            return

        err = "主人，处理步骤较多，请简化需求后重试哦～"
        for ch in err:
            yield {"type": "token", "content": ch}
        session.add_assistant(err)
        yield {"type": "done", "session_id": session.session_id}

    @staticmethod
    def _parse_args(args_str):
        """安全解析 JSON 参数"""
        if isinstance(args_str, dict):
            return args_str
        try:
            return json.loads(args_str) if isinstance(args_str, str) else {}
        except json.JSONDecodeError:
            logger.warning(f"JSON 解析失败: {args_str[:100]}")
            return {}
