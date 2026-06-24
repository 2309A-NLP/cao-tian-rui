"""
工具定义与执行器 — Function Calling Schema + 日程数据库操作
工单编号：人工智能 NLP-Agent 数字人项目-日程提醒智能体任务

4 个工具：add_schedule / query_schedules / update_schedule / delete_schedule
"""
import json
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
            "name": "add_schedule",
            "description": (
                "添加一条日程到数据库 schedules 表。\n"
                "## 致命规则\n"
                "1. 如果用户输入缺少必填字段（title 或 schedule_time），绝对不要调用此工具。列出缺少的信息，逐项追问用户。\n"
                "2. 日期 schedule_date 格式 YYYY-MM-DD。未明确说明时默认是今天。\n"
                "3. 时间 schedule_time 格式 HH:MM（24小时制），如 08:00 / 17:00 / 14:30。\n"
                "4. **时间解析映射**（用户口语到标准时间）：\n"
                "   - '下午5点''傍晚5点' -> 17:00\n"
                "   - '上午8点''早上8点' -> 08:00\n"
                "   - '中午12点' -> 12:00\n"
                "   - '晚上8点' -> 20:00\n"
                "   - '下午3点半' -> 15:30\n"
                "   - '半夜12点''午夜12点' -> 00:00\n"
                "   - '明早''明天早上' -> 第二天 08:00(需确认具体时间)\n"
                "5. **循环日程识别**（repeat_type + repeat_rule）：\n"
                "   - '每天''天天''每日' -> repeat_type='daily', repeat_rule=''\n"
                "   - '每周X''每个星期X' -> repeat_type='weekly', repeat_rule='{\"weekdays\":[X]}' (周一=1...周日=7)\n"
                "   - '工作日''每个工作日' -> repeat_type='weekly', repeat_rule='{\"weekdays\":[1,2,3,4,5]}'\n"
                "   - '周末' -> repeat_type='weekly', repeat_rule='{\"weekdays\":[6,7]}'\n"
                "   - '每月Y号''每月Y日' -> repeat_type='monthly', repeat_rule='{\"day\":Y}'\n"
                "   - '每年X月Y日' -> repeat_type='yearly', repeat_rule='{\"month\":X,\"day\":Y}'\n"
                "   - 无循环说明 -> repeat_type='none', repeat_rule=''\n"
                "6. remind_before：用户说'提前X分钟提醒'时才填，默认0（准时提醒）。\n"
                "7. title 是日程事项内容，如'开会''买咖啡''健身'，不要包含时间信息。\n"
                "   **title 必须去掉'提醒我''提醒您''帮我''记得''别忘了'等前缀**，只保留事项本身。\n"
                "   示例：'提醒我开会'→ title='开会'，'帮我记得吃药'→ title='吃药'，'别忘了买咖啡'→ title='买咖啡'\n"
                "8. add_schedule 只调用一次，不要对同一条日程重复添加。\n"
                "9. 无论日程时间是否已过，都正常添加。数据库存储所有日程，提醒系统会自动处理。用户说的时间即使已过也要添加。\n"
                "10. 重复检测：工具会自动检查同天同名同时间是否已存在。若检测到重复，工具返回提示，你需要询问用户是否确认添加；用户确认后再次调用并传 force=true。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "日程事项内容，如'开会''买咖啡''健身'，不要包含时间"
                    },
                    "schedule_date": {
                        "type": "string",
                        "description": "日程日期，格式 YYYY-MM-DD。未明确说明时用今天的日期"
                    },
                    "schedule_time": {
                        "type": "string",
                        "description": "日程时间，24小时制格式 HH:MM。如 08:00、17:00、14:30。需要将用户口语转为标准格式"
                    },
                    "repeat_type": {
                        "type": "string",
                        "enum": ["none", "daily", "weekly", "monthly", "yearly"],
                        "description": "重复类型，默认 none"
                    },
                    "repeat_rule": {
                        "type": "string",
                        "description": "重复规则JSON字符串，如'{\"weekdays\":[1,3,5]}'。无循环时留空"
                    },
                    "remind_before": {
                        "type": "integer",
                        "description": "提前多少分钟提醒，默认0表示准时提醒"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "用户已确认重复添加时传 true，跳过重复检查。默认不传（false）"
                    }
                },
                "required": ["title", "schedule_date", "schedule_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_schedules",
            "description": (
                "查询日程记录。当用户问'今天有什么日程''我的日程有哪些''查看日程''看看日程'时使用。\n"
                "所有参数可选，不传就是查全部。返回匹配的日程列表。\n"
                "注意：用户说'删除''取消''去掉'时，使用 delete_schedule 工具，不要用本工具。\n"
                "返回格式中每一条都带 #序号 [DB_ID=X] 格式，例如 '#2 [DB_ID=5] 17:00 开会'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "schedule_date": {
                        "type": "string",
                        "description": "按日期查询，格式 YYYY-MM-DD。'今天'用当天日期"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "按标题关键词模糊搜索"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "completed", "cancelled"],
                        "description": "按状态筛选。不填默认查全部状态"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_schedule",
            "description": (
                "修改/更新已有的日程记录。\n"
                "当用户说'修改''改成''改一下''更新''调整''推迟''提前'等表示要修改的关键词时，调用此工具。\n"
                "## 重要流程\n"
                "1. 如果用户只说了要改什么（如'把开会改到下午3点'），但没有说日程ID，"
                "则需要先调用 query_schedules 查出该日程的 ID，展示给用户确认后再调用本工具。\n"
                "2. 如果用户明确说了日程ID（如'修改日程3的时间为10:00'），'3'是显示序号，"
                "需要从最近查询结果中找到 #3 对应的 DB_ID 再传入。\n"
                "3. schedule_id 是必填项（传DB_ID）。\n"
                "4. 只需传要修改的字段，未传的字段保持不变。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "schedule_id": {
                        "type": "integer",
                        "description": "要修改的日程的数据库真实ID（DB_ID）。用户说的'日程X'中的X是序号不是DB_ID！"
                    },
                    "title": {
                        "type": "string",
                        "description": "新的事项内容"
                    },
                    "schedule_date": {
                        "type": "string",
                        "description": "新的日期，格式 YYYY-MM-DD"
                    },
                    "schedule_time": {
                        "type": "string",
                        "description": "新的时间，24小时制 HH:MM"
                    },
                    "repeat_type": {
                        "type": "string",
                        "enum": ["none", "daily", "weekly", "monthly", "yearly"],
                        "description": "新的重复类型"
                    },
                    "repeat_rule": {
                        "type": "string",
                        "description": "新的重复规则JSON"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "completed", "cancelled"],
                        "description": "修改状态"
                    },
                    "remind_before": {
                        "type": "integer",
                        "description": "新的提前提醒分钟数"
                    }
                },
                "required": ["schedule_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_schedule",
            "description": (
                "删除日程记录。当用户说'删除''去掉''取消''移除'等关键词时，必须调用此工具。\n"
                "## 关键：序号 vs 数据库ID\n"
                "查询结果格式：'#2 [DB_ID=5] 17:00 开会'。\n"
                "- #2 是显示序号（按时间排的第2个）\n"
                "- DB_ID=5 是数据库中的真实ID\n"
                "**用户说'日程1''删除日程1''取消日程1'中的'1'是显示序号#1！**\n"
                "不要直接把1当schedule_id传。正确做法：\n"
                "1. 从最近一次查询结果中找到 #1 对应的 DB_ID\n"
                "2. 把 DB_ID 作为 schedule_id 传入\n"
                "3. 如果你不记得#1对应哪个DB_ID，先调 query_schedules 查一次\n"
                "\n"
                "操作流程：\n"
                "1. 传 schedule_id(DB_ID) 或 keyword 来定位要删除的日程。\n"
                "2. confirmed=false 时，工具返回该日程详情，你展示给用户确认。\n"
                "3. 用户确认后，再发起一次 tool_call，传 confirmed=true + 相同的 schedule_id。\n"
                "4. 绝对不允许在用户未明确确认的情况下执行删除。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "schedule_id": {
                        "type": "integer",
                        "description": "数据库中的真实ID（DB_ID）。注意：用户说的'日程1'中的'1'是显示序号，不是DB_ID！必须从查询结果中找到#1对应的DB_ID传入"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "关键词搜索要删除的日程（不知道ID时用）"
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "用户是否已确认删除。首次查询时传 false，用户确认后传 true"
                    }
                },
                "required": ["confirmed"]
            }
        }
    }
]


class ToolExecutor:
    """工具执行器：接收 LLM 的 tool_call 请求，操作数据库，返回结果"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def execute(self, tool_name: str, arguments: dict) -> str:
        """执行单个工具调用，返回结果文本（回填给 LLM）"""
        logger.info(f"执行工具: {tool_name}, 参数: {json.dumps(arguments, ensure_ascii=False)}")
        try:
            if tool_name == "add_schedule":
                return self._add_schedule(arguments)
            elif tool_name == "query_schedules":
                return self._query_schedules(arguments)
            elif tool_name == "update_schedule":
                return self._update_schedule(arguments)
            elif tool_name == "delete_schedule":
                return self._delete_schedule(arguments)
            else:
                return f"错误：未知工具 '{tool_name}'"
        except Exception as e:
            logger.error(f"工具执行异常: {tool_name}, {e}")
            return f"错误：执行 {tool_name} 时发生异常: {str(e)}"

    # ── 添加日程 ──

    def _add_schedule(self, args: dict) -> str:
        title = args.get("title", "").strip()
        schedule_date = args.get("schedule_date", "").strip()
        schedule_time = args.get("schedule_time", "").strip()
        repeat_type = args.get("repeat_type", "none")
        repeat_rule = args.get("repeat_rule", "")
        remind_before = args.get("remind_before", 0)
        force = args.get("force", False)

        # 字段完整性校验
        missing = []
        if not title:
            missing.append("事项内容(title)")
        if not schedule_date:
            missing.append("日期(schedule_date)")
        if not schedule_time:
            missing.append("时间(schedule_time)")

        if missing:
            return f"### 无法添加，缺少必填信息：{', '.join(missing)}\n请向用户追问以上缺失信息，不要强行调用。"

        # 校验时间格式
        import re
        if not re.match(r'^\d{2}:\d{2}(:\d{2})?$', schedule_time):
            return f"### 时间格式错误：'{schedule_time}' 不是有效的 HH:MM 格式。请转换为24小时制后重试。"

        # 重复检查
        if not force:
            existing = self.db.query_schedules(schedule_date=schedule_date, status="active")
            records = existing.get("records", [])
            time_prefix = schedule_time[:5]  # HH:MM

            # 同名日程（无论时间）
            same_title = [r for r in records if r.get("title") == title]
            # 同时间段日程（无论名称）
            same_time = [r for r in records if str(r.get("schedule_time", ""))[:5] == time_prefix]

            if same_title:
                r = same_title[0]
                return (
                    f"### 检测到同名日程\n"
                    f"已存在：{str(r['schedule_time'])[:5]} 「{r['title']}」\n"
                    f"请询问用户：'主人，您今天已有「{title}」的日程了，是否要重复添加一条？'\n"
                    f"用户确认后，再次调用 add_schedule 并传 force=true。"
                )
            if same_time:
                r = same_time[0]
                return (
                    f"### 检测到时间冲突\n"
                    f"该时间段已有：{time_prefix} 「{r['title']}」\n"
                    f"请询问用户：'主人，{time_prefix} 已有「{r['title']}」，是否还要在同一时间再加一条「{title}」？'\n"
                    f"用户确认后，再次调用 add_schedule 并传 force=true。"
                )

        result = self.db.add_schedule(
            title=title,
            schedule_date=schedule_date,
            schedule_time=schedule_time,
            repeat_type=repeat_type,
            repeat_rule=repeat_rule,
            remind_before=remind_before
        )

        if result["success"]:
            loop_text = ""
            if repeat_type != "none":
                loop_map = {"daily": "每天", "weekly": "每周", "monthly": "每月", "yearly": "每年"}
                loop_text = f"，重复：{loop_map.get(repeat_type, repeat_type)}"
            return f"✅ {result['message']}{loop_text}\nschedule_id={result['schedule_id']}"
        else:
            return f"❌ 添加失败：{result['message']}"

    # ── 查询日程 ──

    def _query_schedules(self, args: dict) -> str:
        schedule_date = args.get("schedule_date")
        keyword = args.get("keyword")
        status = args.get("status")

        result = self.db.query_schedules(
            schedule_date=schedule_date,
            keyword=keyword,
            status=status
        )

        if not result["success"]:
            return f"查询失败：{result.get('message', '未知错误')}"

        if result["total_count"] == 0:
            date_hint = f" {schedule_date}" if schedule_date else ""
            kw_hint = f" 关键词'{keyword}'" if keyword else ""
            return f"查询结果：{date_hint}{kw_hint} 没有找到匹配的日程。"

        # 构造结构化返回 — 带序号(#N)和DB_ID
        lines = [f"## 已为您查询到 {result['total_count']} 条日程："]
        for idx, r in enumerate(result["records"], 1):
            sid = r.get("id")
            sdate = r.get("schedule_date", "")
            stime = r.get("schedule_time", "")[:5]  # HH:MM
            title = r.get("title", "")
            st_status = r.get("status", "active")
            repeat = r.get("repeat_type", "none")

            status_icon = {"active": " ", "completed": "[完成]", "cancelled": "[已取消]"}.get(st_status, "")
            loop_icon = " [循环]" if repeat != "none" else ""

            lines.append(
                f"  #{idx} [DB_ID={sid}] {stime} {title}{loop_icon}"
            )

        return "\n".join(lines)

    # ── 修改日程 ──

    def _update_schedule(self, args: dict) -> str:
        schedule_id = args.get("schedule_id")
        if not schedule_id:
            return "### 缺少 schedule_id，无法确定要修改哪条日程。请先查询日程然后让用户指定要修改的日程。"

        update_fields = {}
        for field in ["title", "schedule_date", "schedule_time", "repeat_type",
                       "repeat_rule", "status", "remind_before"]:
            if field in args and args[field] is not None and args[field] != "":
                update_fields[field] = args[field]

        if not update_fields:
            return "### 没有提供要修改的字段，请明确要修改的内容。"

        result = self.db.update_schedule(schedule_id, **update_fields)
        if result["success"]:
            old_rec = result.get("old", {})
            old_line = f"{old_rec.get('schedule_date','')} {str(old_rec.get('schedule_time',''))[:5]} {old_rec.get('title','')}"
            new_rec = result.get("updated", {})
            new_line = f"{new_rec.get('schedule_date','')} {str(new_rec.get('schedule_time',''))[:5]} {new_rec.get('title','')}"
            return f"✅ 已更新日程 id={schedule_id}\n  修改前：{old_line}\n  修改后：{new_line}"
        else:
            return f"❌ 更新失败：{result['message']}"

    # ── 删除日程 ──

    def _delete_schedule(self, args: dict) -> str:
        confirmed = args.get("confirmed", False)
        schedule_id = args.get("schedule_id")
        keyword = args.get("keyword")

        if not confirmed:
            # 第一步：查找要删除的日程并展示给用户确认
            if schedule_id:
                result = self.db.query_schedules(schedule_id=schedule_id)
                if result["total_count"] > 0:
                    # 找到了 exact DB_ID match
                    r = result["records"][0]
                    return (
                        f"## 确认删除？\n"
                        f"  [DB_ID={r['id']}] {r['schedule_date']} {str(r['schedule_time'])[:5]} | {r['title']}\n"
                        f"向用户展示确认后，调用本工具 confirmed=true 执行删除。"
                    )
                else:
                    # 未找到 DB_ID，当作显示序号 auto-map
                    logger.info(f"开始 auto-map: schedule_id={schedule_id} 当作序号处理")
                    all_records = self.db.query_schedules(status="active")
                    if all_records["total_count"] > 0 and 1 <= schedule_id <= all_records["total_count"]:
                        mapped = all_records["records"][schedule_id - 1]
                        # 直接删除（序号已确定，无需再确认）
                        del_result = self.db.delete_schedule(mapped["id"])
                        if del_result["success"]:
                            return (
                                f"已自动将序号#{schedule_id}映射到DB_ID={mapped['id']}并执行删除。\n"
                                f"已删除：{mapped['schedule_date']} {str(mapped['schedule_time'])[:5]} | {mapped['title']}\n"
                                f"告诉用户已删除完成，无需再确认。"
                            )
                        return f"删除失败：{del_result['message']}"
                    return f"未找到 DB_ID={schedule_id} 的日程，也无法映射为序号。"

            if keyword:
                result = self.db.query_schedules(keyword=keyword)
                if result["total_count"] == 0:
                    return f"未找到包含'{keyword}'的日程。"

                lines = [f"找到以下 {result['total_count']} 条匹配'{keyword}'的日程："]
                for idx, r in enumerate(result["records"], 1):
                    lines.append(f"  #{idx} [DB_ID={r['id']}] {str(r['schedule_time'])[:5]} {r['title']}")
                lines.append("\n展示给用户，让用户指定序号。等用户给出序号后，调用 delete_schedule 传 schedule_id=序号 执行删除。")
                return "\n".join(lines)

            return "需要 schedule_id 或 keyword 参数才能搜索。"

        # 第二步：用户已确认，执行删除
        if not schedule_id:
            return "需要提供 schedule_id 才能执行删除。"

        # 自动序号映射（同第一步一样）
        check = self.db.query_schedules(schedule_id=schedule_id)
        if check["total_count"] == 0:
            all_records = self.db.query_schedules(status="active")
            if all_records["total_count"] > 0 and 1 <= schedule_id <= all_records["total_count"]:
                schedule_id = all_records["records"][schedule_id - 1]["id"]

        result = self.db.delete_schedule(schedule_id)
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ 删除失败：{result['message']}"
