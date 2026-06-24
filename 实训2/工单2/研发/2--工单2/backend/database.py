"""
数据库管理模块 — schedules 表的 CRUD
工单编号：人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
"""
import pymysql
from pymysql.cursors import DictCursor
from typing import Optional
from datetime import date, datetime, time as dt_time, timedelta
import json

from backend.logger import get_logger

logger = get_logger("db")


class DatabaseManager:
    """日程数据库管理器：建表 + 日程增删改查 + 提醒查询"""

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn: Optional[pymysql.Connection] = None

    def connect(self) -> bool:
        """连接数据库，自动建库建表"""
        try:
            self.conn = pymysql.connect(
                host=self.host, port=self.port,
                user=self.user, password=self.password,
                charset="utf8mb4",
                cursorclass=DictCursor,
                connect_timeout=5,
            )
            with self.conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                    f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            self.conn.select_db(self.database)
            self._init_tables()
            logger.info(f"数据库连接成功: {self.host}:{self.port}/{self.database}")
            return True
        except pymysql.Error as e:
            logger.error(f"数据库连接失败: {e}")
            self.conn = None
            return False

    def _ensure(self) -> bool:
        """确保连接有效，断了就重连"""
        if self.conn is None or not self.conn.open:
            logger.warning("DB 断开，重连中...")
            return self.connect()
        try:
            self.conn.ping(reconnect=True)
        except pymysql.Error:
            return self.connect()
        return True

    def _init_tables(self):
        """创建 schedules 日程表"""
        sql = """
        CREATE TABLE IF NOT EXISTS `schedules` (
            `id`            INT AUTO_INCREMENT PRIMARY KEY COMMENT '日程ID',
            `title`         VARCHAR(256) NOT NULL            COMMENT '日程事项内容',
            `schedule_date` DATE         NOT NULL            COMMENT '日程日期',
            `schedule_time` TIME         NOT NULL            COMMENT '日程时间',
            `repeat_type`   VARCHAR(16)  DEFAULT 'none'      COMMENT '重复类型: none/daily/weekly/monthly/yearly',
            `repeat_rule`   VARCHAR(512) DEFAULT ''          COMMENT '重复规则JSON，如{"weekdays":[1,3,5]}',
            `status`        VARCHAR(16)  DEFAULT 'active'    COMMENT '状态: active/completed/cancelled/pending_remind',
            `remind_before` INT          DEFAULT 0           COMMENT '提前提醒分钟数',
            `last_reminded` DATETIME     DEFAULT NULL        COMMENT '上次提醒时间',
            `created_at`    DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            `updated_at`    DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            INDEX `idx_date`        (`schedule_date`),
            INDEX `idx_status`      (`status`),
            INDEX `idx_date_status` (`schedule_date`, `status`),
            INDEX `idx_datetime`    (`schedule_date`, `schedule_time`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='日程提醒表';
        """
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.conn.commit()
        logger.info("schedules 表就绪")

    # ═══════════════════════════════════════════
    #  序列化辅助
    # ═══════════════════════════════════════════

    @staticmethod
    def _serialize(record: dict) -> dict:
        """把 datetime/date/time/Decimal 转为字符串，方便 JSON"""
        for key in list(record.keys()):
            val = record[key]
            if isinstance(val, datetime):
                record[key] = val.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(val, date):
                record[key] = val.strftime("%Y-%m-%d")
            elif isinstance(val, timedelta):
                # MySQL TIME 字段 → PyMySQL 返回 timedelta
                total_secs = int(val.total_seconds())
                record[key] = f"{total_secs // 3600:02d}:{(total_secs % 3600) // 60:02d}:{total_secs % 60:02d}"
            elif isinstance(val, dt_time):
                record[key] = val.strftime("%H:%M:%S")
            elif hasattr(val, 'to_eng_string'):  # Decimal
                record[key] = float(val)
        return record

    # ═══════════════════════════════════════════
    #  1. 添加日程
    # ═══════════════════════════════════════════

    def add_schedule(self, title: str, schedule_date: str, schedule_time: str,
                     repeat_type: str = "none", repeat_rule: str = "",
                     remind_before: int = 0) -> dict:
        """
        添加一条日程
        schedule_date: YYYY-MM-DD
        schedule_time: HH:MM 或 HH:MM:SS
        """
        if not self._ensure():
            return {"success": False, "message": "数据库连接失败"}

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO schedules (title, schedule_date, schedule_time,
                       repeat_type, repeat_rule, remind_before)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (title, schedule_date, schedule_time, repeat_type, repeat_rule, remind_before)
                )
                self.conn.commit()
                sid = cur.lastrowid
                logger.info(f"日程添加成功: id={sid}, {schedule_date} {schedule_time}, {title}")
                return {
                    "success": True,
                    "message": f"已添加日程：{schedule_date} {schedule_time}，「{title}」",
                    "schedule_id": sid
                }
        except pymysql.Error as e:
            logger.error(f"添加日程失败: {e}")
            return {"success": False, "message": f"添加失败: {e}"}

    # ═══════════════════════════════════════════
    #  2. 查询日程
    # ═══════════════════════════════════════════

    def query_schedules(self, schedule_date: str = None, keyword: str = None,
                        status: str = None, schedule_id: int = None) -> dict:
        """
        查询日程
        - schedule_date: 按日期查 (YYYY-MM-DD)，不传则查全部
        - keyword: 按标题模糊搜索
        - status: 按状态筛选
        - schedule_id: 按ID精确查(改删前先查)
        """
        if not self._ensure():
            return {"success": False, "records": [], "total_count": 0, "message": "数据库连接失败"}

        conditions = []
        params = []

        if schedule_date:
            conditions.append("schedule_date = %s")
            params.append(schedule_date)
        if keyword:
            conditions.append("title LIKE %s")
            params.append(f"%{keyword}%")
        if status:
            conditions.append("status = %s")
            params.append(status)
        if schedule_id:
            conditions.append("id = %s")
            params.append(schedule_id)

        where = " AND ".join(conditions) if conditions else "1=1"

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM schedules WHERE {where} ORDER BY schedule_date ASC, schedule_time ASC",
                    params
                )
                records = cur.fetchall()
                for r in records:
                    self._serialize(r)

                logger.info(f"查询日程: {len(records)} 条")
                return {
                    "success": True,
                    "records": records,
                    "total_count": len(records),
                }
        except pymysql.Error as e:
            logger.error(f"查询日程失败: {e}")
            return {"success": False, "records": [], "total_count": 0, "message": str(e)}

    # ═══════════════════════════════════════════
    #  3. 修改日程
    # ═══════════════════════════════════════════

    def update_schedule(self, schedule_id: int, **kwargs) -> dict:
        """
        按 ID 更新日程字段
        可更新字段: title, schedule_date, schedule_time, repeat_type, repeat_rule, status, remind_before
        """
        if not self._ensure():
            return {"success": False, "message": "数据库连接失败"}

        allowed = {"title", "schedule_date", "schedule_time",
                   "repeat_type", "repeat_rule", "status", "remind_before"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return {"success": False, "message": "没有可更新的字段"}

        # 先查旧值
        old = self.query_schedules(schedule_id=schedule_id)
        if old["total_count"] == 0:
            return {"success": False, "message": f"未找到日程 id={schedule_id}"}
        old_rec = old["records"][0]

        try:
            set_clause = ", ".join(f"`{k}` = %s" for k in updates)
            values = list(updates.values()) + [schedule_id]

            with self.conn.cursor() as cur:
                cur.execute(
                    f"UPDATE schedules SET {set_clause} WHERE id = %s",
                    values
                )
                self.conn.commit()

            new = self.query_schedules(schedule_id=schedule_id)
            logger.info(f"日程更新成功: id={schedule_id}, {updates}")
            return {
                "success": True,
                "message": f"已更新日程 id={schedule_id}",
                "old": old_rec,
                "updated": new["records"][0] if new["records"] else {},
            }
        except pymysql.Error as e:
            logger.error(f"更新日程失败: {e}")
            return {"success": False, "message": f"更新失败: {e}"}

    # ═══════════════════════════════════════════
    #  4. 删除日程
    # ═══════════════════════════════════════════

    def delete_schedule(self, schedule_id: int) -> dict:
        """按 ID 删除日程"""
        if not self._ensure():
            return {"success": False, "message": "数据库连接失败"}

        try:
            # 先查，方便回显
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM schedules WHERE id = %s", (schedule_id,))
                rec = cur.fetchone()

            if not rec:
                return {"success": False, "message": f"未找到日程 id={schedule_id}"}

            self._serialize(rec)

            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM schedules WHERE id = %s", (schedule_id,))
                self.conn.commit()

            logger.info(f"删除成功: id={schedule_id}, {rec.get('schedule_date')} {rec.get('schedule_time')} {rec.get('title')}")
            return {
                "success": True,
                "message": f"已删除日程 {schedule_id}，内容：{rec['schedule_time'][:5]} {rec['title']}",
                "deleted": rec
            }
        except pymysql.Error as e:
            logger.error(f"删除失败: {e}")
            return {"success": False, "message": f"删除失败: {e}"}

    # ═══════════════════════════════════════════
    #  5. 提醒专用：获取到期未提醒的日程
    # ═══════════════════════════════════════════

    def get_due_schedules(self) -> list:
        """
        获取当前时间已到、且今天还没提醒过的 active 日程。
        支持循环日程（daily/weekly/monthly/yearly）。
        """
        if not self._ensure():
            return []

        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        now_time_str = now.strftime("%H:%M")
        weekday_num = now.isoweekday()   # 1=周一 … 7=周日
        month_day = now.day
        month_num = now.month

        try:
            with self.conn.cursor() as cur:
                # 一次取出所有 active、时间已到、今天未提醒的候选行
                cur.execute("""
                    SELECT * FROM schedules
                    WHERE status = 'active'
                      AND schedule_time <= %s
                      AND (last_reminded IS NULL OR DATE(last_reminded) < %s)
                    ORDER BY schedule_time ASC
                """, (now_time_str, today_str))
                candidates = cur.fetchall()

            results = []
            for r in candidates:
                self._serialize(r)
                rtype = r.get("repeat_type", "none")
                sdate = r.get("schedule_date", "")

                if rtype == "none":
                    if sdate == today_str:
                        results.append(r)

                elif rtype == "daily":
                    if sdate <= today_str:
                        results.append(r)

                elif rtype == "weekly":
                    if sdate <= today_str:
                        rule = {}
                        try:
                            rule = json.loads(r.get("repeat_rule") or "{}")
                        except (json.JSONDecodeError, TypeError):
                            pass
                        if weekday_num in rule.get("weekdays", []):
                            results.append(r)

                elif rtype == "monthly":
                    if sdate <= today_str:
                        rule = {}
                        try:
                            rule = json.loads(r.get("repeat_rule") or "{}")
                        except (json.JSONDecodeError, TypeError):
                            pass
                        if rule.get("day") == month_day:
                            results.append(r)

                elif rtype == "yearly":
                    if sdate <= today_str:
                        rule = {}
                        try:
                            rule = json.loads(r.get("repeat_rule") or "{}")
                        except (json.JSONDecodeError, TypeError):
                            pass
                        if rule.get("month") == month_num and rule.get("day") == month_day:
                            results.append(r)

            # 结束隐式事务，确保下次扫描读到最新数据（PyMySQL 默认非 autocommit）
            self.conn.commit()
            return results
        except pymysql.Error as e:
            logger.error(f"获取到期日程失败: {e}")
            return []

    def mark_reminded(self, schedule_id: int) -> bool:
        """标记日程已提醒（更新 last_reminded）"""
        if not self._ensure():
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE schedules SET last_reminded = NOW() WHERE id = %s",
                    (schedule_id,)
                )
                self.conn.commit()
            return True
        except pymysql.Error as e:
            logger.error(f"标记提醒失败: {e}")
            return False

    def close(self):
        if self.conn and self.conn.open:
            self.conn.close()
