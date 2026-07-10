"""
工单11 医疗挂号Agent — 查询类工具函数（只读，不修改数据库）
==============================================================
本文件包含 4 个查询工具，全部只执行 SELECT，不做 INSERT/UPDATE：

  - get_family_member()   根据用户ID + 家属昵称（如"大宝"）查 patient_id
  - query_schedule()      查询可预约的号源（有剩余量的排班）
  - get_user_history()    查询用户的历史挂号记录
  - get_doctor_schedule() 查询某医生的完整排班（含已满号源，用于展示出诊时间）

设计原则：
  每个函数统一返回 {"ok": True/False, ...} 格式的 dict，
  LLM 根据 ok 字段判断工具调用是否成功，失败时 error 字段说明原因。
  所有函数末尾统一调用 logger.log_tool_call() 记录入参、出参、耗时。

辅助函数（私有，不对外暴露）：
  _now_ms()           获取当前时刻的毫秒时间戳（用于计算耗时）
  _match_time_slot()  把用户偏好时间（如"14:00"）映射到时间段字符串
  _validate_dept()    校验科室名是否在白名单中
  _validate_title()   校验号源类型是否合法
"""
import time   # 标准库：time.perf_counter()，高精度计时器
from datetime import date, datetime, timedelta   # 标准库：日期时间操作
from typing import Optional   # 标准库：Optional 类型注解

from src.config import (
    VALID_DEPARTMENTS,   # 科室白名单列表
    VALID_TITLES,        # 号源类型白名单（专家/普通）
    MAX_FUTURE_DAYS,     # 最大可预约天数
)
from src.database import get_connection   # 从连接池获取 MySQL 连接
from src import logger                    # 结构化日志


# ──────────────────────────────────────────────────────────────────────────────
# 私有辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _now_ms() -> float:
    """
    返回当前时刻的毫秒时间戳，用于计算工具函数的执行耗时。
    time.perf_counter() 是高精度单调时钟，适合测量短时间间隔。
    """
    return time.perf_counter() * 1000


def _match_time_slot(time_pref: Optional[str]) -> Optional[str]:
    """
    将用户时间偏好（HH:MM 格式）映射到数据库中存储的时间段字符串。

    数据库里排班时间段只有三种固定值：
      "09:00-11:00" / "14:00-17:00" / "19:00-21:00"

    映射规则（按小时分段）：
      0-11点（hour < 12）   → 上午段 "09:00-11:00"
      12-17点（12 <= h <18）→ 下午段 "14:00-17:00"
      18点及以后（h >= 18） → 晚上段 "19:00-21:00"

    参数：
      time_pref : "HH:MM" 格式字符串（如 "09:00"/"14:00"/"19:00"），None 表示不指定

    返回：
      对应的时间段字符串，None 表示不过滤时间段（查全天）
    """
    if not time_pref:
        return None   # 未指定时间偏好，不过滤时间段
    hour = int(time_pref.split(":")[0])   # 取冒号前的小时数字
    if hour < 12:
        return "09:00-11:00"    # 上午
    elif hour < 18:
        return "14:00-17:00"    # 下午
    else:
        return "19:00-21:00"    # 晚上


def _validate_dept(dept: str) -> Optional[str]:
    """
    校验科室名是否在白名单 VALID_DEPARTMENTS 中。

    返回：合法时返回原始科室名，非法时返回 None。
    用途：在执行 SQL 之前做快速校验，非法科室直接返回错误，不查数据库。
    """
    return dept if dept in VALID_DEPARTMENTS else None


def _validate_title(title: Optional[str]) -> Optional[str]:
    """
    校验号源类型（"专家"/"普通"）是否合法。

    参数：
      title : 号源类型字符串，None 表示不过滤（查全部类型）

    返回：合法返回原值，None 透传，非法字符串返回 None（相当于不过滤）。
    """
    if title is None:
        return None   # None 表示不过滤，透传
    return title if title in VALID_TITLES else None   # 非法类型视为不过滤


# ──────────────────────────────────────────────────────────────────────────────
# Tool 1：get_family_member — 查家属信息
# ──────────────────────────────────────────────────────────────────────────────

def get_family_member(user_id: int, alias: str, trace_id: str = "-") -> dict:
    """
    根据用户 ID 和家属昵称（alias）查询患者信息，返回 patient_id。

    数据库关系链：
      user_account（用户表，存登录用户）
        ↓ 1:N（一个用户可以绑定多个家属）
      family_relation（家属关联表，存 user_id + patient_id + alias 昵称）
        ↓
      patient（患者表，存患者真实姓名、性别等）

    使用场景：
      用户说"帮大宝挂内科" → LLM 调 get_family_member(alias="大宝")
      获取到 patient_id → 再调 book_appointment(patient_id=..., sch_id=...)

    参数：
      user_id  : 已登录用户的 ID（来自 JWT 或 session）
      alias    : 家属昵称（如"大宝"/"二宝"/"我"）
      trace_id : 日志追踪 ID

    返回：
      成功：{"ok": True, "patient_id": 123, "patient_name": "张小宝"}
      失败：{"ok": False, "error": "未找到家属: alias='大宝'..."}
    """
    t0 = _now_ms()   # 记录函数开始时间
    params = {"user_id": user_id, "alias": alias}   # 方便日志记录入参
    try:
        conn = get_connection()   # 从连接池取连接
        cur = conn.cursor()       # 获取 DictCursor 游标

        # ── 步骤1：先确认用户存在 ──
        # 防止用未登录/已注销账户的 user_id 越权查询他人家属
        cur.execute("SELECT ua_id FROM user_account WHERE ua_id = %s", (user_id,))
        if not cur.fetchone():   # fetchone() 返回 None 表示查不到
            result = {"ok": False, "error": f"用户不存在: user_id={user_id}"}
            logger.log_tool_call("get_family_member", params, result, _now_ms() - t0, trace_id)
            conn.close()   # 归还连接到连接池
            return result

        # ── 步骤2：通过昵称查 patient_id，同时 JOIN patient 表取真实姓名 ──
        cur.execute(
            """SELECT fr.patient_id, p.p_Name
               FROM family_relation fr
               JOIN patient p ON p.p_ID = fr.patient_id
               WHERE fr.user_id = %s AND fr.alias = %s""",
            (user_id, alias),
        )
        row = cur.fetchone()   # 每个昵称在同一用户下唯一，取第一条即可
        conn.close()

        if not row:
            # 用户存在但该昵称下没有绑定家属
            result = {
                "ok": False,
                "error": f"未找到家属: alias='{alias}'（该用户不存在此家属）",
            }
        else:
            # 找到了，返回 patient_id 和真实姓名
            result = {
                "ok": True,
                "patient_id": row["patient_id"],   # 患者 ID（后续挂号用）
                "patient_name": row["p_Name"],      # 患者真实姓名
            }

        logger.log_tool_call("get_family_member", params, result, _now_ms() - t0, trace_id)
        return result

    except Exception as exc:
        # 数据库异常（连接失败、SQL 错误等）
        logger.log_error("get_family_member 异常", trace_id, exc=exc, params=params)
        return {"ok": False, "error": "数据库操作失败，请稍后重试"}


# ──────────────────────────────────────────────────────────────────────────────
# Tool 2：query_schedule — 查可用号源
# ──────────────────────────────────────────────────────────────────────────────

def query_schedule(
    dept: str,
    date_start: str,
    date_end: str,
    title: Optional[str] = None,
    time_pref: Optional[str] = None,
    doctor_name: Optional[str] = None,
    trace_id: str = "-",
) -> dict:
    """
    查询指定科室在日期范围内有剩余号源（sch_available > 0）的排班列表。

    参数说明：
      dept        : 科室名称，必须在白名单中（如"内科"/"儿科"）
      date_start  : 查询开始日期，YYYY-MM-DD 格式
      date_end    : 查询结束日期，超出 MAX_FUTURE_DAYS 会被截断到最大允许日期
      title       : 号源类型过滤，"专家" 或 "普通"，None 表示不过滤（查全部）
      time_pref   : 时间偏好 HH:MM（如"14:00"），会映射为时间段字符串进行过滤
      doctor_name : 指定医生姓名（支持模糊匹配，如"张"能匹配"张伟"），None 不过滤

    后处理（过滤逻辑）：
      过滤掉今天已结束的时间段：
        例如现在是 15 点，则上午段（11点结束）和下午段（17点结束？不对，11点是上午段结束）
        实际：09:00-11:00 上午段 11 点结束，14:00-17:00 下午段 17 点结束

    返回格式：
      成功：{"ok": True, "schedules": [{"sch_id":..., "doctor_name":..., ...}, ...]}
      失败：{"ok": False, "error": "科室不存在: '...', 可用科室: [...]"}
    """
    t0 = _now_ms()
    params = dict(
        dept=dept, date_start=date_start, date_end=date_end,
        title=title, time_pref=time_pref, doctor_name=doctor_name,
    )

    # ── 前置校验：科室必须在白名单中 ──
    if not _validate_dept(dept):
        result = {"ok": False, "error": f"科室不存在: '{dept}'，可用科室: {VALID_DEPARTMENTS}"}
        logger.log_tool_call("query_schedule", params, result, _now_ms() - t0, trace_id)
        return result

    # 强制 date_end 不超过最大预约范围（防止 LLM 传入遥远未来的日期）
    max_end = (date.today() + timedelta(days=MAX_FUTURE_DAYS)).isoformat()
    if date_end > max_end:
        date_end = max_end   # 截断到最大允许日期

    validated_title = _validate_title(title)        # 校验并规范化号源类型
    time_slot = _match_time_slot(time_pref)          # 时间偏好 → 时间段字符串

    try:
        conn = get_connection()
        cur = conn.cursor()

        # ── 构建动态 SQL（只添加用户指定的过滤条件）──
        # 基础查询：从 schedule JOIN doctor JOIN department，只取有余量的行
        sql = """
            SELECT s.sch_id, s.d_id, d.d_Name AS doctor_name,
                   dep.dep_Name AS dept,
                   s.sch_type AS title,
                   s.sch_date AS date,
                   s.sch_time_slot AS time_slot,
                   s.sch_available AS available,
                   s.sch_total AS total
            FROM schedule s
            JOIN doctor d ON d.d_ID = s.d_id
            JOIN department dep ON dep.dep_ID = d.dep_ID
            WHERE dep.dep_Name = %s
              AND s.sch_date BETWEEN %s AND %s
              AND s.sch_available > 0
        """
        args = [dept, date_start, date_end]   # 基础参数：科室、日期范围

        # 可选过滤条件（用户不指定则不添加 WHERE 子句）
        if validated_title:
            sql += " AND s.sch_type = %s"      # 按号源类型过滤
            args.append(validated_title)
        if time_slot:
            sql += " AND s.sch_time_slot = %s"  # 按时间段过滤
            args.append(time_slot)
        if doctor_name:
            # LIKE %name%：模糊匹配，允许用户输入"张"匹配"张伟"
            sql += " AND d.d_Name LIKE %s"
            args.append(f"%{doctor_name}%")

        sql += " ORDER BY s.sch_date, s.sch_time_slot LIMIT 20"  # 按日期/时间升序，最多返回20条
        cur.execute(sql, args)
        rows = cur.fetchall()   # 获取所有匹配行
        conn.close()

        schedules = list(rows)   # 转为普通列表

        # ── 后过滤：去掉今天已结束的时间段 ──
        # 避免 LLM 向用户推荐过去已无法就诊的时段
        now = datetime.now()
        today_str = date.today().isoformat()
        # 各时间段的结束时间（小时），过了这个时间就算"已过"
        _slot_end_hour = {"09:00-11:00": 11, "14:00-17:00": 17, "19:00-21:00": 21}
        schedules = [
            s for s in schedules
            # 非今天的排班不过滤；今天的只保留还未结束的时间段
            if s["date"].isoformat() != today_str
            or now.hour < _slot_end_hour.get(s["time_slot"], 24)
        ]

        result = {"ok": True, "schedules": schedules}
        logger.log_tool_call("query_schedule", params, result, _now_ms() - t0, trace_id)
        return result

    except Exception as exc:
        logger.log_error("query_schedule 异常", trace_id, exc=exc, params=params)
        return {"ok": False, "error": "查询排班失败，请稍后重试"}


# ──────────────────────────────────────────────────────────────────────────────
# Tool 3：get_user_history — 查历史挂号记录
# ──────────────────────────────────────────────────────────────────────────────

def get_user_history(
    user_id: int,
    dept: Optional[str] = None,
    trace_id: str = "-",
) -> dict:
    """
    查询该用户（及其所有家属）的历史挂号记录，可按科室过滤。

    使用场景：
      用户说"再约上次那个内科医生" → LLM 调用此工具，找到上次内科挂号的
      doctor_name，再用 query_schedule(doctor_name=...) 查该医生的最新号源。

    实现逻辑（避免 N+1 查询）：
      1. 先从 family_relation 查出该用户名下所有 patient_id（一次查询）
      2. 用 IN (p1, p2, ...) 一次性查所有家属的挂号记录（一次查询）
      3. 按 reg_ID 倒序（最新的在前），最多返回 50 条

    参数：
      user_id  : 已登录用户的 ID
      dept     : 可选，按科室名过滤历史记录
      trace_id : 日志追踪 ID

    返回：
      成功：{"ok": True, "history": [{"reg_id":..., "doctor_name":..., ...}, ...]}
      失败：{"ok": False, "error": "..."}
    """
    t0 = _now_ms()
    params = {"user_id": user_id, "dept": dept}

    try:
        conn = get_connection()
        cur = conn.cursor()

        # ── 步骤1：确认用户存在 ──
        cur.execute("SELECT 1 FROM user_account WHERE ua_id = %s", (user_id,))
        if not cur.fetchone():
            conn.close()
            result = {"ok": False, "error": f"用户不存在: {user_id}"}
            logger.log_tool_call("get_user_history", params, result, _now_ms() - t0, trace_id)
            return result

        # ── 步骤2：查出该用户名下所有家属的 patient_id 列表 ──
        cur.execute(
            "SELECT patient_id FROM family_relation WHERE user_id = %s", (user_id,)
        )
        # 列表推导式提取每行的 patient_id（DictCursor 返回 dict）
        patient_ids = [r["patient_id"] for r in cur.fetchall()]

        if not patient_ids:
            # 用户未绑定任何家属（刚注册用户），返回空历史
            conn.close()
            result = {"ok": True, "history": []}
            logger.log_tool_call("get_user_history", params, result, _now_ms() - t0, trace_id)
            return result

        # ── 步骤3：用 IN 一次性查所有家属的挂号记录 ──
        # 动态生成占位符 "%s,%s,..." 与 patient_ids 数量一致
        placeholders = ",".join(["%s"] * len(patient_ids))
        sql = f"""
            SELECT r.reg_ID AS reg_id, r.d_ID AS d_id,
                   d.d_Name AS doctor_name,
                   dep.dep_Name AS dept,
                   s.sch_date AS date,
                   s.sch_time_slot AS time_slot,
                   s.sch_type AS title,
                   r.reg_Status AS reg_status,
                   r.reg_Time AS reg_time,
                   p.p_Name AS patient_name
            FROM register r
            JOIN doctor d ON d.d_ID = r.d_ID
            JOIN department dep ON dep.dep_ID = r.dep_ID
            JOIN schedule s ON s.sch_id = r.sch_id
            JOIN patient p ON p.p_ID = r.p_ID
            WHERE r.p_ID IN ({placeholders})
        """
        args = patient_ids[:]   # 复制列表，避免修改原数组
        if dept:
            sql += " AND dep.dep_Name = %s"   # 可选的科室过滤
            args.append(dept)
        sql += " ORDER BY r.reg_ID DESC LIMIT 50"  # 最新记录在前，最多 50 条

        cur.execute(sql, args)
        rows = cur.fetchall()
        conn.close()

        result = {"ok": True, "history": list(rows)}
        logger.log_tool_call("get_user_history", params, result, _now_ms() - t0, trace_id)
        return result

    except Exception as exc:
        logger.log_error("get_user_history 异常", trace_id, exc=exc, params=params)
        return {"ok": False, "error": "查询历史记录失败，请稍后重试"}


# ──────────────────────────────────────────────────────────────────────────────
# Tool 4：get_doctor_schedule — 查医生完整排班
# ──────────────────────────────────────────────────────────────────────────────

def get_doctor_schedule(
    doctor_name: str,
    date_start: str,
    date_end: str,
    trace_id: str = "-",
) -> dict:
    """
    查询指定医生在日期范围内的完整排班（包括已满号源），用于展示出诊时间表。

    与 query_schedule 的核心区别：
      query_schedule      → 只查 sch_available > 0 的号源（用户想挂号时使用）
      get_doctor_schedule → 查所有排班，不管是否有余量（用于查医生出诊时间时使用）

    使用场景：
      用户说"张建国医生下周还出诊吗？" →
      LLM 调用此工具，展示张建国下周所有出诊时间（包括已满的）。

    参数：
      doctor_name : 医生姓名（支持模糊匹配，如"张"匹配"张建国"）
      date_start  : 查询开始日期，YYYY-MM-DD 格式
      date_end    : 查询结束日期，YYYY-MM-DD 格式
      trace_id    : 日志追踪 ID

    返回：
      成功：{"ok": True, "doctor": "张建国", "schedules": [...]}
      失败：{"ok": False, "error": "未找到医生: '张XX'"}
    """
    t0 = _now_ms()
    params = {"doctor_name": doctor_name, "date_start": date_start, "date_end": date_end}

    try:
        conn = get_connection()
        cur = conn.cursor()

        # ── 步骤1：模糊查找医生 ──
        # 允许用户输入"张医生"→ LIKE "%张%" 匹配"张伟"/"张建国"等
        cur.execute(
            "SELECT d_ID, d_Name, d_Profession FROM doctor WHERE d_Name LIKE %s",
            (f"%{doctor_name}%",),   # f-string 构造 LIKE 模式
        )
        docs = cur.fetchall()
        if not docs:
            # 没有匹配的医生
            conn.close()
            result = {"ok": False, "error": f"未找到医生: '{doctor_name}'"}
            logger.log_tool_call(
                "get_doctor_schedule", params, result, _now_ms() - t0, trace_id
            )
            return result

        # 多个同名医生时取第一个（实际业务可以根据科室进一步精确匹配）
        d_id        = docs[0]["d_ID"]     # 医生数据库 ID
        actual_name = docs[0]["d_Name"]   # 医生实际姓名（用于回复用户）

        # ── 步骤2：查询该医生的完整排班（含已满号源）──
        cur.execute(
            """SELECT sch_date AS date, sch_time_slot AS time_slot,
                      sch_type AS title, sch_available AS available, sch_total AS total
               FROM schedule
               WHERE d_id = %s AND sch_date BETWEEN %s AND %s
               ORDER BY sch_date, sch_time_slot""",   # 按日期和时间段升序
            (d_id, date_start, date_end),
        )
        rows = cur.fetchall()
        conn.close()

        # ── 后过滤：同样去掉今天已结束的时间段 ──
        now = datetime.now()
        today_str = date.today().isoformat()
        _slot_end_hour = {"09:00-11:00": 11, "14:00-17:00": 17, "19:00-21:00": 21}
        rows = [
            s for s in rows
            if s["date"].isoformat() != today_str
            or now.hour < _slot_end_hour.get(s["time_slot"], 24)
        ]

        result = {"ok": True, "doctor": actual_name, "schedules": list(rows)}
        logger.log_tool_call(
            "get_doctor_schedule", params, result, _now_ms() - t0, trace_id
        )
        return result

    except Exception as exc:
        logger.log_error("get_doctor_schedule 异常", trace_id, exc=exc, params=params)
        return {"ok": False, "error": "查询医生排班失败，请稍后重试"}
