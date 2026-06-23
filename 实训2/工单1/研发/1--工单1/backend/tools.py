"""
工具定义与执行器 — Function Calling Schema + DB 操作
工单编号：人工智能 NLP-Agent 数字人项目-记账本任务
"""
import json
from datetime import date
from backend.database import DatabaseManager
from backend.logger import get_logger

logger = get_logger("tools")

# ═══════════════════════════════════════════════
#  工具 Schema 定义（OpenAI Function Calling 格式）
# ═══════════════════════════════════════════════

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "add_record",
            "description": (
                "添加一条记账记录到数据库。"
                "重要规则："
                "1. 如果用户输入中缺少必填字段（member/amount/type/category/item/record_date），不要调用此工具，而是直接询问用户补充。"
                "2. 金额 amount 统一填正数，通过 type 区分'收入'还是'支出'。"
                "3. 日期 record_date 格式为 YYYY-MM-DD。'今天'使用当前日期，'昨天'使用昨天日期。"
                "4. category 是分类，如：购物、餐饮、学习、旅游、工资收入、报销收入、医疗、交通、住房、娱乐、其他。"
                "5. '花钱了''买了''消费''付了'等→type='支出'；'进账''收入''收到''工资''报销'等→type='收入'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "member": {
                        "type": "string",
                        "enum": ["爸爸", "妈妈", "女儿"],
                        "description": "家庭成员，从用户输入中识别（'我'需要根据上下文判断）"
                    },
                    "amount": {
                        "type": "number",
                        "description": "金额，正数"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["收入", "支出"],
                        "description": "收入还是支出"
                    },
                    "category": {
                        "type": "string",
                        "description": "消费/收入分类，根据物品性质推断。如买书→学习，买衣服鞋子→购物，吃饭外卖→餐饮，旅游→旅游，工资→工资收入，报销→报销收入"
                    },
                    "item": {
                        "type": "string",
                        "description": "具体物品或事项名称"
                    },
                    "record_date": {
                        "type": "string",
                        "description": "记账日期，格式 YYYY-MM-DD"
                    },
                    "note": {
                        "type": "string",
                        "description": "额外备注，用户未提供时留空"
                    }
                },
                "required": ["member", "amount", "type", "category", "item", "record_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_records",
            "description": (
                "查询/查看记账记录。仅当用户提问目的是'查看''查询''花了多少''买了什么''明细''汇总''统计'时使用。"
                "注意：如果用户说'删除'——请使用 delete_record 工具，不要用本工具。"
                "所有参数都可选，不传就是查全部。返回匹配的记录列表和汇总信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "member": {
                        "type": "string",
                        "enum": ["爸爸", "妈妈", "女儿"],
                        "description": "按成员筛选"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "开始日期 YYYY-MM-DD。如用户说'这个月'，则用当月第一天"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "结束日期 YYYY-MM-DD。如用户说'这个月'，则用当月最后一天"
                    },
                    "category": {
                        "type": "string",
                        "description": "按分类筛选。注意：'买书'→'学习'，'买衣服/买鞋'→'购物'"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["收入", "支出"],
                        "description": "按收支类型筛选"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "按物品名称模糊搜索"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_record",
            "description": (
                "删除记账记录。当用户说'删除''去掉''移除''取消''不要了'等表示要删除的关键词时，"
                "必须调用此工具（不要用 query_records 代替，因为本工具 confirmed=false 时就是搜索功能）。"
                "关键流程："
                "1. 先调用本工具 confirmed=false, keyword='用户要删的东西' ——执行搜索。"
                "2. 把搜索到的记录展示给用户，等待确认。"
                "3. 用户明确确认后（说'确认''可以''是的''好的'等），调用本工具 confirmed=true 执行删除。"
                "绝对不允许在用户未明确确认的情况下执行删除。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "要删除记录的关键词，用于匹配 item 字段"
                    },
                    "member": {
                        "type": "string",
                        "enum": ["爸爸", "妈妈", "女儿"],
                        "description": "限制在哪个成员的记录中查找"
                    },
                    "record_id": {
                        "type": "integer",
                        "description": "要删除的具体记录ID（从 query_records 结果中获取）"
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "用户是否已确认删除。首次查询时传 false，用户确认后传 true"
                    }
                },
                "required": ["confirmed"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": (
                "获取指定时间范围内的收支汇总统计。"
                "当用户问'这个月花了多少钱''这个月的开销''家里支出情况'等需要总体数据的问题时，调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "member": {
                        "type": "string",
                        "enum": ["爸爸", "妈妈", "女儿"],
                        "description": "按成员筛选，查全家则不填"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "开始日期 YYYY-MM-DD，必填"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "结束日期 YYYY-MM-DD，必填"
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["member", "category", "type"],
                        "description": "汇总维度，不填则按收支类型汇总"
                    }
                },
                "required": ["start_date", "end_date"]
            }
        }
    }
]


class ToolExecutor:
    """工具执行器：接收 LLM 的 tool_call 请求，操作数据库，返回结果"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def execute(self, tool_name: str, arguments: dict) -> str:
        """
        执行单个工具调用，返回结果文本（会被回填到 LLM messages 中）。

        Args:
            tool_name: 工具名称
            arguments: LLM 提取的参数 JSON

        Returns:
            执行结果文本（给 LLM 阅读）
        """
        logger.info(f"执行工具: {tool_name}, 参数: {json.dumps(arguments, ensure_ascii=False)}")
        try:
            if tool_name == "add_record":
                return self._add_record(arguments)
            elif tool_name == "query_records":
                return self._query_records(arguments)
            elif tool_name == "delete_record":
                return self._delete_record(arguments)
            elif tool_name == "get_summary":
                return self._get_summary(arguments)
            else:
                return f"错误：未知工具 '{tool_name}'"
        except Exception as e:
            logger.error(f"工具执行异常: {tool_name}, {e}")
            return f"错误：执行 {tool_name} 时发生异常: {str(e)}"

    def _add_record(self, args: dict) -> str:
        """添加记账记录"""
        member = args.get("member", "")
        amount = args.get("amount", 0)
        record_type = args.get("type", "")
        category = args.get("category", "")
        item = args.get("item", "")
        record_date = args.get("record_date", "")
        note = args.get("note", "")

        # 字段校验
        missing = []
        if not member:
            missing.append("成员(member)")
        if not amount:
            missing.append("金额(amount)")
        if not record_type:
            missing.append("收支类型(type)")
        if not category:
            missing.append("分类(category)")
        if not item:
            missing.append("物品(item)")
        if not record_date:
            missing.append("日期(record_date)")

        if missing:
            return f"未能添加记录，缺少必填字段：{', '.join(missing)}。请向用户追问这些信息。"

        # 金额取正
        amount = abs(float(amount))

        result = self.db.add_record(member, amount, record_type, category, item, record_date, note)
        if result["success"]:
            return f"添加成功。{result['message']}"
        else:
            return f"添加失败：{result['message']}"

    def _query_records(self, args: dict) -> str:
        """查询记账记录"""
        member = args.get("member")
        start_date = args.get("start_date")
        end_date = args.get("end_date")
        category = args.get("category")
        record_type = args.get("type")
        keyword = args.get("keyword")

        result = self.db.query_records(
            member=member, start_date=start_date, end_date=end_date,
            category=category, record_type=record_type, keyword=keyword
        )

        if not result["success"]:
            return f"查询失败：{result.get('message', '未知错误')}"

        if result["total_count"] == 0:
            return "查询结果：无匹配记录。请告诉用户没有找到相关的账目记录。"

        # 构造结构化返回
        lines = [f"查询到 {result['total_count']} 条记录："]
        if result.get("summary_text"):
            lines.append(f"汇总：{result['summary_text']}")

        for i, r in enumerate(result["records"], 1):
            sign = "-" if r.get("type") == "支出" else "+"
            lines.append(
                f"  [{r.get('id')}] {r.get('record_date','')} | {r.get('member','')} | "
                f"{r.get('category','')} | {r.get('item','')} | {sign}¥{r.get('amount',0):.2f}"
            )

        return "\n".join(lines)

    def _delete_record(self, args: dict) -> str:
        """删除记账记录（分两步：先查后删）"""
        confirmed = args.get("confirmed", False)
        keyword = args.get("keyword")

        if not confirmed:
            # 第一步：搜索匹配，展示给用户确认
            if not keyword:
                return "请提供要删除记录的关键词（如物品名称），以便搜索匹配的记录。"

            member = args.get("member")
            result = self.db.delete_by_keyword(keyword=keyword, member=member)
            if not result["success"]:
                return f"未找到匹配'{keyword}'的记录。请告诉用户没有找到相关记录。"

            # 返回匹配的记录，等待用户确认
            lines = [f"找到以下 {result['count']} 条匹配'{keyword}'的记录："]
            for r in result["matched"]:
                lines.append(
                    f"  [{r.get('id')}] {r.get('record_date')} | {r.get('member')} | "
                    f"{r.get('item')} | ¥{r.get('amount')}"
                )
            lines.append("请向用户展示以上记录，询问'确认删除这几条记录吗？'。等待用户明确确认后再操作。")
            return "\n".join(lines)

        # 第二步：用户已确认，执行删除
        record_id = args.get("record_id")
        if record_id:
            result = self.db.delete_by_id(record_id)
        elif keyword:
            # 按关键词删除
            member = args.get("member")
            search = self.db.query_records(keyword=keyword, member=member)
            if search["total_count"] == 0:
                return f"没有找到匹配'{keyword}'的记录可删除。"

            deleted = []
            for r in search["records"]:
                dr = self.db.delete_by_id(r["id"])
                if dr["success"]:
                    deleted.append(dr["message"])
            if deleted:
                return "删除完成。\n" + "\n".join(deleted)
            return "删除操作未执行，没有记录被删除。"
        else:
            return "需要提供 record_id 或 keyword 才能删除。请先搜索要删除的记录。"

        return result["message"]

    def _get_summary(self, args: dict) -> str:
        """汇总统计"""
        start_date = args.get("start_date")
        end_date = args.get("end_date")
        member = args.get("member")
        group_by = args.get("group_by")

        if not start_date or not end_date:
            return "汇总需要指定日期范围(start_date, end_date)。"

        result = self.db.get_summary(start_date, end_date, member=member, group_by=group_by)

        if not result["success"]:
            return f"汇总查询失败：{result.get('message', '')}"

        rows = result["summary_rows"]
        if not rows:
            return f"{start_date}至{end_date}期间无记录。"

        lines = [f"{start_date} 至 {end_date} 汇总："]
        for r in rows:
            if group_by == "member":
                lines.append(f"  {r.get('member')}: {r.get('type')} ¥{r.get('total', 0):.2f}")
            elif group_by == "category":
                lines.append(f"  {r.get('category')}: {r.get('type')} ¥{r.get('total', 0):.2f} ({r.get('cnt')}笔)")
            else:
                lines.append(f"  {r.get('type')}: ¥{r.get('total', 0):.2f} ({r.get('cnt')}笔)")

        return "\n".join(lines)
