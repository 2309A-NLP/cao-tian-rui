"""
后台提醒线程 — 定时扫描到期日程，写入提醒队列
工单编号：人工智能 NLP-Agent 数字人项目-日程提醒智能体任务

设计思路：
- 后台线程每30秒检查一次数据库中到期且未提醒的日程
- 检查到到期日程后，标记为已提醒，把提醒消息加入内存队列
- API 端点 /api/reminders/pending 供前端轮询，拉取新提醒
- 这样提醒机制和 Agent 本身解耦，后续可以替换为推送/短信/邮件
"""
import threading
import time
from collections import deque
from datetime import datetime
from backend.logger import get_logger

logger = get_logger("reminder")

# 提醒消息队列：[(schedule_id, title, schedule_time, reminded_at), ...]
# 用 deque 避免 O(n) pop(0)
reminder_queue: deque = deque()
MAX_QUEUE_SIZE = 100

# 最近提醒的快照（用于前端展示）
last_reminders: list = []


def add_reminder(schedule_id: int, title: str, schedule_time: str):
    """向队列添加一条提醒"""
    now = datetime.now()
    msg = {
        "schedule_id": schedule_id,
        "title": title,
        "schedule_time": schedule_time,
        "reminded_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        # 随机选一个提醒话术
        "message": _pick_reminder_message(title),
    }
    reminder_queue.append(msg)
    last_reminders.append(msg)

    # 限制队列大小
    while len(reminder_queue) > MAX_QUEUE_SIZE:
        reminder_queue.popleft()
    if len(last_reminders) > 50:
        last_reminders.pop(0)

    logger.info(f"提醒入队: {title} @ {schedule_time}")


def _pick_reminder_message(title: str) -> str:
    """随机选一条温馨提醒话术"""
    import random
    templates = [
        f"温馨提醒：「{title}」的时间到啦，主人！✨",
        f"主人！是时候「{title}」了喔~ 🌸",
        f"亲爱的主人，现在是「{title}」的时候啦！💕",
        f"嘿，主人，该「{title}」了哦~ ⏰",
        f"叮咚～「{title}」的时间到咯，别忘了呀！🎀",
    ]
    return random.choice(templates)


def pop_pending_reminders() -> list:
    """弹出所有待推送的提醒（前端轮询用）"""
    result = []
    while reminder_queue:
        result.append(reminder_queue.popleft())
    return result


class ReminderThread(threading.Thread):
    """
    后台提醒线程
    每30秒扫描一次数据库，找到到期未提醒的日程，加入提醒队列
    """

    def __init__(self, db_getter, interval: int = 30):
        """
        db_getter: 无参函数，返回 DatabaseManager 实例（延迟获取，避免循环导入）
        interval: 扫描间隔（秒）
        """
        super().__init__(daemon=True, name="reminder-thread")
        self.db_getter = db_getter
        self.interval = interval
        self._stop_event = threading.Event()

    def run(self):
        logger.info(f"提醒线程启动，扫描间隔={self.interval}s")
        while not self._stop_event.is_set():
            try:
                db = self.db_getter()
                if db:
                    due = db.get_due_schedules()
                    for rec in due:
                        # 标记已提醒
                        db.mark_reminded(rec["id"])
                        # 加入提醒队列
                        add_reminder(
                            schedule_id=rec["id"],
                            title=rec["title"],
                            schedule_time=str(rec.get("schedule_time", ""))[:5],
                        )
                    if due:
                        logger.info(f"本轮发现 {len(due)} 条到期日程")
            except Exception as e:
                logger.error(f"提醒线程扫描异常: {e}")

            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        logger.info("提醒线程已停止")
