"""
数据库管理模块 — MySQL 连接 + money_notes 表
工单编号：人工智能 NLP-Agent 数字人项目-记账本任务
"""
import pymysql
from pymysql.cursors import DictCursor
from typing import Optional
from datetime import date, datetime
from decimal import Decimal
from backend.logger import get_logger

logger = get_logger("db")


class DatabaseManager:
    """数据库管理器：建表 + 记账 CRUD"""

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
        """确保连接有效"""
        if self.conn is None or not self.conn.open:
            logger.warning("DB 断开，重连中...")
            return self.connect()
        try:
            self.conn.ping(reconnect=True)
        except pymysql.Error:
            return self.connect()
        return True

    def _init_tables(self):
        """创建 money_notes 表"""
        sql = """
        CREATE TABLE IF NOT EXISTS `money_notes` (
            `id`          INT AUTO_INCREMENT PRIMARY KEY COMMENT '记录ID',
            `member`      VARCHAR(16) NOT NULL COMMENT '成员：爸爸/妈妈/女儿',
            `amount`      DECIMAL(10,2) NOT NULL COMMENT '金额',
            `type`        VARCHAR(8) NOT NULL COMMENT '收入/支出',
            `category`    VARCHAR(32) NOT NULL DEFAULT '' COMMENT '分类',
            `item`        VARCHAR(256) NOT NULL DEFAULT '' COMMENT '物品/事项',
            `record_date` DATE NOT NULL COMMENT '记账日期',
            `note`        VARCHAR(512) DEFAULT '' COMMENT '备注',
            `created_at`  DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            `updated_at`  DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '修改时间',
            INDEX `idx_member` (`member`),
            INDEX `idx_date` (`record_date`),
            INDEX `idx_type` (`type`),
            INDEX `idx_category` (`category`),
            INDEX `idx_member_date` (`member`, `record_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='家庭记账本';
        """
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.conn.commit()
        logger.info("money_notes 表就绪")

    # ═══════════════════════════════════════════
    #  CRUD
    # ═══════════════════════════════════════════

    def add_record(self, member: str, amount: float, record_type: str,
                   category: str, item: str, record_date: str,
                   note: str = "") -> dict:
        """添加一条记账记录"""
        if not self._ensure():
            return {"success": False, "message": "数据库连接失败"}
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO money_notes (member, amount, type, category, item, record_date, note)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (member, amount, record_type, category, item, record_date, note)
                )
                self.conn.commit()
                rid = cur.lastrowid
                logger.info(f"记账成功: id={rid}, {member}, {record_type}, {item}, ¥{amount}")
                return {
                    "success": True,
                    "message": f"已记录：{record_date}，{member}，{category}，{item}，{'-' if record_type=='支出' else '+'}{amount}元",
                    "record_id": rid
                }
        except pymysql.Error as e:
            logger.error(f"记账失败: {e}")
            return {"success": False, "message": f"记账失败: {e}"}

    def query_records(self, member: str = None, start_date: str = None,
                      end_date: str = None, category: str = None,
                      record_type: str = None, keyword: str = None) -> dict:
        """查询记录，返回列表和汇总"""
        if not self._ensure():
            return {"success": False, "records": [], "total_count": 0, "message": "数据库连接失败"}

        conditions = []
        params = []

        if member:
            conditions.append("member = %s")
            params.append(member)
        if start_date:
            conditions.append("record_date >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("record_date <= %s")
            params.append(end_date)
        if category:
            conditions.append("category = %s")
            params.append(category)
        if record_type:
            conditions.append("type = %s")
            params.append(record_type)
        if keyword:
            conditions.append("item LIKE %s")
            params.append(f"%{keyword}%")

        where = " AND ".join(conditions) if conditions else "1=1"

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM money_notes WHERE {where} ORDER BY record_date DESC, id DESC",
                    params
                )
                records = cur.fetchall()

                # 汇总统计
                cur.execute(
                    f"SELECT type, SUM(amount) as total FROM money_notes WHERE {where} GROUP BY type",
                    params
                )
                summary = {row["type"]: float(row["total"]) for row in cur.fetchall()}

                # 把 Decimal 转 float 方便 JSON 序列化
                for r in records:
                    r["amount"] = float(r["amount"])
                    for dt_key in ("record_date", "created_at", "updated_at"):
                        if r.get(dt_key) and isinstance(r[dt_key], (date, datetime)):
                            r[dt_key] = str(r[dt_key])

                summary_text = ""
                if summary:
                    parts = []
                    for t, v in summary.items():
                        parts.append(f"{t}合计 ¥{v:.2f}")
                    summary_text = "，".join(parts)

                logger.info(f"查询账目: {len(records)} 条, {summary_text}")
                return {
                    "success": True,
                    "records": records,
                    "total_count": len(records),
                    "summary": summary,
                    "summary_text": summary_text,
                }
        except pymysql.Error as e:
            logger.error(f"查询账目失败: {e}")
            return {"success": False, "records": [], "total_count": 0, "message": str(e)}

    def delete_by_id(self, record_id: int) -> dict:
        """按 ID 删除记录"""
        if not self._ensure():
            return {"success": False, "message": "数据库连接失败"}
        try:
            # 先查一下，方便回显
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM money_notes WHERE id = %s", (record_id,))
                rec = cur.fetchone()

            if not rec:
                return {"success": False, "message": f"未找到记录 id={record_id}"}

            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM money_notes WHERE id = %s", (record_id,))
                self.conn.commit()

            logger.info(f"删除成功: id={record_id}, {rec.get('member')}, {rec.get('item')}, ¥{rec.get('amount')}")
            return {
                "success": True,
                "message": f"已删除记录：{rec['record_date']}，{rec['member']}，{rec['item']}，¥{rec['amount']}",
                "deleted": rec
            }
        except pymysql.Error as e:
            logger.error(f"删除失败: {e}")
            return {"success": False, "message": f"删除失败: {e}"}

    def delete_by_keyword(self, keyword: str, member: str = None) -> dict:
        """按关键词匹配删除（先搜后删）"""
        result = self.query_records(keyword=keyword, member=member)
        if not result["success"] or result["total_count"] == 0:
            return {"success": False, "message": f"没有找到包含'{keyword}'的记录", "matched": []}

        matched = result["records"]
        # 返回匹配结果，让 Agent 确认后再删
        return {
            "success": True,
            "matched": matched,
            "count": len(matched),
            "message": f"找到 {len(matched)} 条匹配'{keyword}'的记录，请确认是否删除"
        }

    def get_summary(self, start_date: str, end_date: str,
                    member: str = None, group_by: str = None) -> dict:
        """按维度汇总"""
        if not self._ensure():
            return {"success": False, "message": "数据库连接失败"}

        member_filter = "AND member = %s" if member else ""
        params = [start_date, end_date]
        if member:
            params.insert(0, member)

        results = {}
        try:
            with self.conn.cursor() as cur:
                if group_by == "member":
                    cur.execute(f"""
                        SELECT member, type, SUM(amount) as total
                        FROM money_notes
                        WHERE record_date >= %s AND record_date <= %s {member_filter}
                        GROUP BY member, type
                        ORDER BY member
                    """, params)
                elif group_by == "category":
                    cur.execute(f"""
                        SELECT category, type, SUM(amount) as total, COUNT(*) as cnt
                        FROM money_notes
                        WHERE record_date >= %s AND record_date <= %s {member_filter}
                        GROUP BY category, type
                        ORDER BY total DESC
                    """, params)
                else:
                    # 默认：总览
                    cur.execute(f"""
                        SELECT type, SUM(amount) as total, COUNT(*) as cnt
                        FROM money_notes
                        WHERE record_date >= %s AND record_date <= %s {member_filter}
                        GROUP BY type
                    """, params)

                rows = cur.fetchall()
                for r in rows:
                    r["total"] = float(r["total"]) if "total" in r else 0

            logger.info(f"汇总查询: {len(rows)} 组, group_by={group_by}")
            return {"success": True, "summary_rows": rows, "group_by": group_by}
        except pymysql.Error as e:
            logger.error(f"汇总失败: {e}")
            return {"success": False, "message": str(e)}

    def close(self):
        if self.conn and self.conn.open:
            self.conn.close()
