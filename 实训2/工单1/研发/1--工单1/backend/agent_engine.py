"""
Agent 引擎 — Function Calling 循环核心
工单编号：人工智能 NLP-Agent 数字人项目-记账本任务

流程：user_msg → LLM(带tools) → tool_calls? → 执行工具 → 结果回填 → 再调LLM → 回复
"""
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Generator
from backend.llm_provider import LLMProvider
from backend.tools import TOOLS_SCHEMA, ToolExecutor
from backend.database import DatabaseManager
from backend.logger import get_logger

logger = get_logger("engine")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _yesterday() -> str:
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def _this_month_range() -> tuple:
    now = datetime.now()
    return now.replace(day=1).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


first_day, today_str = _this_month_range()

SYSTEM_PROMPT = f"""你是"小家专属记账本"助手。你的唯一职责是帮助家庭管理账目。

## 日期信息
今天是 {today_str}。"昨天"是 {_yesterday()}。本月范围：{first_day} 至 {today_str}。
用户说"今天"→{today_str}，"昨天"→{_yesterday()}，"这个月"→{first_day}至{today_str}。

## 致命规则
1. **强制使用工具**：任何涉及账目数据的操作，都必须调用工具操作数据库 money_notes 表。绝对不允许凭猜测或记忆直接回答。
2. **不完整信息必须追问**：如果用户输入缺少必填字段（member/amount/type/category/item/record_date），不要强行调用 add_record。列出缺少哪些信息，逐项追问用户。例如用户说"昨天花了200块"——缺了成员(member)和物品(item)，你应该问："请问：1)是谁花的？爸爸、妈妈还是女儿？2)买了什么东西呢？"
3. **add_record 只调用一次**：对于同一条记账请求，只调用一次 add_record。工具执行后返回"添加成功"即表示已完成，不要再重复调用。
5. **删除必须先搜索再确认**：
   - 用户说"删除X"→ 立刻调用 delete_record(confirmed=false, keyword="X")。重要：即使不知道用户说的"我"是谁，也先不传member参数搜索全部成员，不要先追问"您是谁"。
   - 示例："删除我的报销记录"→ delete_record(confirmed=false, keyword="报销")，搜索全部成员。搜索结果出来后告知用户找到几条、分别是谁的。
   - 把搜索到的记录展示给用户。
   - 明确问"确认删除吗？"
   - 用户确认后→ 调用 delete_record(confirmed=true) 执行。
   - 示例："删除我的报销记录"而你不知道"我"是谁 → 直接调 delete_record(confirmed=false, keyword="报销") 搜索，不要先问"您是谁"。

## 指代消解（"我"是谁？）
当用户说"我"时，按以下优先级判断：
- 上文中用户已经说过"我是XX"或明确透露过自己的身份 → 使用该身份
- 上文中用户替某个成员记过账（如"我买三体花了50元"之后又说"我昨天花了200"）→ 默认是同一个成员，但可以温柔确认
- 完全无法判断 → 追问"请问您是哪位家庭成员？爸爸、妈妈还是女儿？"

## 成员
爸爸、妈妈、女儿。

## 分类识别（category）
- 买书/教材/文具/培训 → "学习"
- 买衣/鞋子/包包/化妆品 → "购物"
- 吃饭/外卖/买菜/水果/零食 → "餐饮"
- 旅游/机票/酒店/景点 → "旅游"
- 工资/奖金/兼职收入 → "工资收入"
- 报销/退款 → "报销收入"
- 看病/买药/体检 → "医疗"
- 公交/地铁/加油/停车 → "交通"
- 房租/水电/物业 → "住房"
- 其他 → "其他"

## 收支判断
"花了""买了""消费""付了""交费"等 → 支出
"收到""收入""进账""工资""报销""赚了"等 → 收入

## 开场白
程序会自动处理开场白，你无需操心。

## 回复风格
- 友好亲切
- 记账成功回显：日期 + 成员 + 分类 + 物品 + 金额
- 查账结果清晰列出"""


class Session:
    """会话：持有完整消息链"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"ses_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.opening_sent = False  # 是否已发送开场白

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
    "您好，欢迎使用咱们小家专属记账本！"
    "请按照\"x年x月x日，谁做什么事收入/支出多少钱\"的格式来输入。"
    "请告诉我你的账目需求吧！"
)


class AgentEngine:
    """记账 Agent 引擎"""

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
        is_first_msg = len(session.messages) <= 1  # 首条消息（只有 system prompt）
        session.add_user(user_message)
        tool_calls_made = []
        called_tools_this_turn = set()  # 防重复调用

        for rnd in range(self.max_rounds):
            logger.info(f"Agent 第{rnd+1}轮, 消息数={len(session.messages)}")

            try:
                resp = self.llm.chat(session.messages, tools=TOOLS_SCHEMA)
            except Exception as e:
                err = f"抱歉，服务暂时不可用：{e}"
                session.add_assistant(err)
                return {"reply": err, "session_id": session.session_id, "tool_calls_made": tool_calls_made}

            finish = resp.get("finish_reason", "stop")

            if finish == "tool_calls":
                tcs = resp.get("tool_calls", [])
                if not tcs:
                    continue
                session.add({"role": "assistant", "content": resp.get("content") or None,
                             "tool_calls": tcs})
                for tc in tcs:
                    fn = tc.get("function", {})  # type: dict
                    name = str(fn.get("name", ""))
                    args = self._parse_args(str(fn.get("arguments", "{}")))

                    # 防重复：add_record 同参数只执行一次
                    dedup_key = f"{name}|{args.get('member','')}|{args.get('item','')}|{args.get('amount','')}|{args.get('record_date','')}"
                    if name == "add_record" and dedup_key in called_tools_this_turn:
                        logger.warning(f"跳过重复的 add_record: {dedup_key}")
                        result_text = f"该记录已存在，无需重复添加：{args.get('member')} {args.get('item')} {args.get('amount')}元"
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
            # 首条消息且未发过开场白：拼接开场白
            if is_first_msg and not session.opening_sent:
                content = OPENING_MESSAGE + "\n\n" + content
            session.opening_sent = True
            session.add_assistant(content)
            return {"reply": content, "session_id": session.session_id, "tool_calls_made": tool_calls_made}

        err = "处理步骤较多，请简化需求后重试。"
        session.add_assistant(err)
        return {"reply": err, "session_id": session.session_id, "tool_calls_made": tool_calls_made}

    def chat_stream(self, user_message: str, session_id: str = None):
        """
        流式对话生成器。Yield 事件:
          {"type": "token", "content": "字"}
          {"type": "tool_call", "name": "add_record", "arguments": {...}}
          {"type": "tool_result", "name": "add_record", "result": "..."}
          {"type": "done", "session_id": "..."}
        """
        session = self.get_session(session_id)
        is_first_msg = len(session.messages) <= 1  # 首条消息
        session.add_user(user_message)
        called_tools_this_turn = set()  # 防重复调用

        for rnd in range(self.max_rounds):
            logger.info(f"Agent stream 第{rnd+1}轮")

            try:
                resp = self.llm.chat(session.messages, tools=TOOLS_SCHEMA)
            except Exception as e:
                yield {"type": "token", "content": f"抱歉，服务不可用：{e}"}
                yield {"type": "done", "session_id": session.session_id}
                return

            finish = resp.get("finish_reason", "stop")

            if finish == "tool_calls":
                tcs = resp.get("tool_calls", [])
                if not tcs:
                    continue

                session.add({"role": "assistant", "content": resp.get("content") or None,
                             "tool_calls": tcs})

                for tc in tcs:
                    fn = tc.get("function", {})  # type: dict
                    name = str(fn.get("name", ""))
                    args = self._parse_args(str(fn.get("arguments", "{}")))
                    yield {"type": "tool_call", "name": name, "arguments": args}

                    # 防重复
                    dedup_key = f"{name}|{args.get('member','')}|{args.get('item','')}|{args.get('amount','')}"
                    if name == "add_record" and dedup_key in called_tools_this_turn:
                        result_text = f"该记录已存在，无需重复添加"
                    else:
                        called_tools_this_turn.add(dedup_key)
                        result_text = self.executor.execute(name, args)

                    session.add({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": result_text})
                    yield {"type": "tool_result", "name": name,
                           "result": result_text[:300]}
                continue

            # 有文本回复
            content = resp.get("content", "") or "已处理。"
            # 首条消息且未发过开场白：先推开场白
            if is_first_msg and not session.opening_sent:
                for ch in OPENING_MESSAGE:
                    yield {"type": "token", "content": ch}
                yield {"type": "token", "content": "\n\n"}
                content = OPENING_MESSAGE + "\n\n" + content
            for ch in content:
                yield {"type": "token", "content": ch}
            session.opening_sent = True
            session.add_assistant(content)
            yield {"type": "done", "session_id": session.session_id}
            return

        err = "处理步骤较多，请简化需求后重试。"
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
