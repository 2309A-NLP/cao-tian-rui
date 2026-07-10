"""
工单11 医疗挂号Agent — 写入类工具函数（会修改数据库）
======================================================
本文件包含 2 个写入工具，均会修改数据库，使用 InnoDB 事务保证原子性：

  - book_appointment()    挂号：INSERT register + schedule.sch_available -= 1
  - cancel_appointment()  取消：UPDATE register + schedule.sch_available += 1

为什么要单独一个文件？
  写操作比读操作风险高（影响数据一致性），单独隔离后：
  - 容易做代码审查（只需重点检查这个文件）
  - 方便加事务测试（mock 事务上下文）

事务方式：
  使用 database.py 提供的 db_transaction() 上下文管理器，
  正常退出 with 块时自动 commit，抛出异常时自动 rollback。

并发安全：
  book_appointment 里用 SELECT ... FOR UPDATE 加行锁，
  防止多人同时抢同一个号源导致 sch_available 变成负数。
"""
import time   # 标准库：高精度计时器
from datetime import date, datetime, timedelta   # 标准库：日期时间操作
from typing import Optional   # 标准库：Optional 类型注解

from src.config import (
    VALID_DEPARTMENTS,  # 科室白名单
    MAX_FUTURE_DAYS,    # 最大可预约天数（14天）
    REGISTER_FEE,       # 号源费用映射 {"专家": 50, "普通": 15}
)
from src.database import db_transaction   # 事务上下文管理器
from src import logger                    # 结构化日志


def _now_ms() -> float:
    """返回当前毫秒时间戳，用于计算工具函数的执行耗时。"""
    return time.perf_counter() * 1000


# ──────────────────────────────────────────────────────────────────────────────
# Tool 5：book_appointment — 挂号
# ──────────────────────────────────────────────────────────────────────────────

def book_appointment(
    user_id: int,
    patient_id: int,
    sch_id: int,
    trace_id: str = "-",
) -> dict:
    """
    为患者在指定排班上挂号，原子性地完成以下步骤：
      1. 校验操作用户是否存在（防止越权调用）
      2. 校验患者是否存在
      3. 用 SELECT ... FOR UPDATE 锁定排班行（防并发争抢）
      4. 校验排班日期合法性（不能挂过去的号）
      5. 校验当天时间段是否已过（上午段已过则不能挂）
      6. 校验是否超出最大预约天数
      7. 校验号源余量是否 > 0
      8. 检查是否重复挂号（同患者同排班只能挂一次）
      9. UPDATE schedule.sch_available -= 1（条件式，防并发）
      10. INSERT register（生成挂号编号 reg_id）

    参数：
      user_id    : 操作人（已登录用户），用于安全验证
      patient_id : 就诊患者 ID（来自 get_family_member 工具）
      sch_id     : 排班 ID（来自 query_schedule 工具）
      trace_id   : 日志追踪 ID

    返回格式：
      成功：{
        "ok": True, "reg_id": 123, "doctor_name": "张伟",
        "dept": "内科", "sch_type": "普通", "date": "2026-07-10",
        "time_slot": "09:00-11:00", "fee": 15, "msg": "挂号成功！..."
      }
      失败：{"ok": False, "error": "...具体原因..."}
    """
    t0 = _now_ms()
    params = {"user_id": user_id, "patient_id": patient_id, "sch_id": sch_id}

    try:
        # db_transaction() 是上下文管理器：进入 with 块取连接，退出时自动 commit/rollback
        with db_transaction() as conn:
            cur = conn.cursor()

            # ── 校验1：操作用户存在（防幽灵 user_id）──
            cur.execute("SELECT 1 FROM user_account WHERE ua_id = %s", (user_id,))
            if not cur.fetchone():
                result = {"ok": False, "error": f"用户不存在: {user_id}"}
                logger.log_tool_call(
                    "book_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            # ── 校验2：患者存在 ──
            cur.execute("SELECT 1 FROM patient WHERE p_ID = %s", (patient_id,))
            if not cur.fetchone():
                result = {"ok": False, "error": f"患者不存在: {patient_id}"}
                logger.log_tool_call(
                    "book_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            # ── 校验3：排班存在并加行锁（SELECT ... FOR UPDATE）──
            # FOR UPDATE 原理：
            #   在事务内对指定行加"排他锁"，其他事务想修改该行必须等本事务提交/回滚后才能进行。
            #   这样保证了"检查 sch_available > 0"和"UPDATE sch_available -= 1"是原子的，
            #   多个并发请求不会同时通过检查然后都减 1 导致 available 变成负数。
            cur.execute(
                """SELECT s.sch_id, s.d_id, s.sch_date, s.sch_time_slot, s.sch_type,
                          s.sch_available, dep.dep_ID, dep.dep_Name, d.d_Name AS doctor_name
                   FROM schedule s
                   JOIN doctor d ON d.d_ID = s.d_id
                   JOIN department dep ON dep.dep_ID = d.dep_ID
                   WHERE s.sch_id = %s
                   FOR UPDATE""",   # 加行锁，直到事务结束
                (sch_id,),
            )
            sch = cur.fetchone()
            if not sch:
                result = {"ok": False, "error": f"排班不存在: sch_id={sch_id}"}
                logger.log_tool_call(
                    "book_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            sch_date = sch["sch_date"]   # MySQL DictCursor 返回 Python date 对象
            today    = date.today()

            # ── 校验4：不能挂过去日期的号 ──
            if sch_date < today:
                result = {"ok": False, "error": "不可挂过去日期的号"}
                logger.log_tool_call(
                    "book_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            # ── 校验5：今天的排班，判断时间段是否已过 ──
            # 例如：现在 12 点，上午段（09:00-11:00）11 点已结束，不能挂
            if sch_date == today:
                _slot_end_hour = {"09:00-11:00": 11, "14:00-17:00": 17, "19:00-21:00": 21}
                end_hour = _slot_end_hour.get(sch["sch_time_slot"], 24)
                if datetime.now().hour >= end_hour:
                    result = {
                        "ok": False,
                        "error": f"该时段（{sch['sch_time_slot']}）今日已过，请选择其他时段",
                    }
                    logger.log_tool_call(
                        "book_appointment", params, result, _now_ms() - t0, trace_id
                    )
                    return result

            # ── 校验6：不能超出最大预约天数（默认 14 天）──
            if (sch_date - today).days > MAX_FUTURE_DAYS:
                result = {
                    "ok": False,
                    "error": f"超出可挂号范围（最多 {MAX_FUTURE_DAYS} 天内）",
                }
                logger.log_tool_call(
                    "book_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            # ── 校验7：号源是否有余量 ──
            if sch["sch_available"] <= 0:
                result = {
                    "ok": False,
                    "error": f"号源已满（sch_id={sch_id}）",
                    "sch_type": sch["sch_type"],  # 告知是哪种类型号源满了
                    "dept": sch["dep_Name"],
                }
                logger.log_tool_call(
                    "book_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            # ── 校验8：重复挂号检查（同患者同排班且状态为有效）──
            cur.execute(
                "SELECT reg_ID FROM register WHERE p_ID = %s AND sch_id = %s AND reg_Status = 1",
                (patient_id, sch_id),
            )
            if cur.fetchone():
                result = {"ok": False, "error": "已挂过此号源，请勿重复挂号"}
                logger.log_tool_call(
                    "book_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            # ── 执行1：号源减 1（条件式 UPDATE，防并发抢光后再减）──
            # WHERE sch_available > 0 是二重保险：
            #   即使两个事务同时通过了上面的 available > 0 检查，
            #   只有一个能成功执行 UPDATE，另一个 rowcount=0 会在下面被拦截
            cur.execute(
                "UPDATE schedule SET sch_available = sch_available - 1 "
                "WHERE sch_id = %s AND sch_available > 0",
                (sch_id,),
            )
            if cur.rowcount == 0:
                # rowcount=0 说明这条 UPDATE 没有影响任何行（号源被并发抢光了）
                result = {"ok": False, "error": "号源已被抢完，请重新查询"}
                logger.log_tool_call(
                    "book_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            # ── 执行2：INSERT 挂号记录 ──
            fee = REGISTER_FEE.get(sch["sch_type"], 15)   # 按号源类型计算费用，默认 15 元
            reg_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 挂号时间
            cur.execute(
                """INSERT INTO register
                   (dep_ID, p_ID, w_ID, d_ID, sch_id, reg_Time, reg_Fee, reg_Order, reg_Status)
                   VALUES (%s, %s, NULL, %s, %s, %s, %s, 1, 1)""",
                # w_ID=NULL（诊室工位，此处不指定），reg_Status=1（有效），reg_Order=1（排号）
                (sch["dep_ID"], patient_id, sch["d_id"], sch_id, reg_time, fee),
            )
            reg_id = cur.lastrowid   # lastrowid 获取 INSERT 后自动生成的主键 ID（即挂号编号）

        # with db_transaction() 正常退出：自动 commit，事务提交成功
        result = {
            "ok":          True,
            "reg_id":      reg_id,                  # 挂号编号（唯一标识）
            "doctor_name": sch["doctor_name"],       # 医生姓名
            "dept":        sch["dep_Name"],          # 科室名
            "sch_type":    sch["sch_type"],          # 号源类型（专家/普通）
            "date":        str(sch_date),            # 就诊日期
            "time_slot":   sch["sch_time_slot"],     # 时间段
            "fee":         fee,                      # 费用（元）
            "msg": (
                f"挂号成功！挂号编号: {reg_id}，"
                f"{sch['dep_Name']} {sch['sch_type']}，"
                f"医生：{sch['doctor_name']}，"
                f"{sch_date} {sch['sch_time_slot']}，"
                f"费用: {fee} 元"
            ),
        }
        logger.log_tool_call("book_appointment", params, result, _now_ms() - t0, trace_id)
        return result

    except Exception as exc:
        # 数据库异常或事务回滚时的兜底
        logger.log_error("book_appointment 异常", trace_id, exc=exc, params=params)
        return {"ok": False, "error": "挂号操作失败，请稍后重试"}


# ──────────────────────────────────────────────────────────────────────────────
# Tool 6：cancel_appointment — 取消挂号
# ──────────────────────────────────────────────────────────────────────────────

def cancel_appointment(user_id: int, cond: dict, trace_id: str = "-") -> dict:
    """
    取消挂号，原子性地完成以下步骤：
      1. 根据 cond 查找目标挂号记录
      2. 校验记录状态（已取消则报错）
      3. 校验就诊日期（已过则不能取消）
      4. UPDATE register.reg_Status = 0（标记为取消）
      5. UPDATE schedule.sch_available += 1（退回号源）

    cond 参数格式（灵活匹配，优先级：reg_id > 组合条件）：
      {"reg_id": 123}                           — 精确指定挂号编号（最精确）
      {"patient_id": 456, "dept": "内科"}        — 按患者+科室找最新一条
      {"patient_id": 456, "date": "2026-07-10"} — 按患者+就诊日期找

    参数：
      user_id  : 操作人 ID（当前仅用于日志，实际业务可加权限校验）
      cond     : 查找条件字典（见上方说明）
      trace_id : 日志追踪 ID

    返回格式：
      成功：{"ok": True, "reg_id": 123, "msg": "挂号已取消（编号 123），号源已退回"}
      失败：{"ok": False, "error": "...具体原因..."}
    """
    t0 = _now_ms()
    params = {"user_id": user_id, "cond": cond}

    try:
        with db_transaction() as conn:
            cur = conn.cursor()

            # ── 查找目标挂号记录（两种查找方式）──
            if "reg_id" in cond:
                # 方式A：精确查找——直接用挂号 ID，最简单最可靠
                cur.execute(
                    """SELECT r.reg_ID, r.sch_id, r.reg_Status, s.sch_date, r.p_ID
                       FROM register r JOIN schedule s ON s.sch_id = r.sch_id
                       WHERE r.reg_ID = %s""",
                    (cond["reg_id"],),
                )
            else:
                # 方式B：模糊查找——按患者/科室/日期/类型等组合条件，取最新一条
                # 适用于用户说"取消我内科的号"但不知道具体 reg_id 的场景
                dept       = cond.get("dept")        # 科室名（可选）
                reg_date   = cond.get("date")        # 就诊日期（可选）
                title      = cond.get("title")       # 号源类型（可选）
                patient_id = cond.get("patient_id")  # 患者 ID（可选）

                sql = """
                    SELECT r.reg_ID, r.sch_id, r.reg_Status, s.sch_date, r.p_ID
                    FROM register r
                    JOIN schedule s ON s.sch_id = r.sch_id
                    JOIN doctor d ON d.d_ID = r.d_ID
                    JOIN department dep ON dep.dep_ID = r.dep_ID
                    WHERE r.reg_Status = 1
                """
                args = []
                # 动态追加 WHERE 条件（只有指定的条件才加入）
                if patient_id:
                    sql += " AND r.p_ID = %s"
                    args.append(patient_id)
                if dept:
                    sql += " AND dep.dep_Name = %s"
                    args.append(dept)
                if reg_date:
                    sql += " AND s.sch_date = %s"
                    args.append(reg_date)
                if title:
                    sql += " AND s.sch_type = %s"
                    args.append(title)
                # ORDER BY reg_ID DESC 取最新挂号，LIMIT 1 只取一条
                sql += " ORDER BY r.reg_ID DESC LIMIT 1"
                cur.execute(sql, args)

            row = cur.fetchone()   # 取查询结果
            if not row:
                # 找不到符合条件的挂号记录
                result = {"ok": False, "error": "未找到符合条件的挂号记录"}
                logger.log_tool_call(
                    "cancel_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            # ── 校验1：已取消的记录不能再次取消 ──
            if row["reg_Status"] == 0:
                result = {"ok": False, "error": "该挂号记录已取消"}
                logger.log_tool_call(
                    "cancel_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            # ── 校验2：就诊日期已过则不能取消（过期号无法退号）──
            sch_date = row["sch_date"]   # 就诊日期（Python date 对象）
            if sch_date < date.today():
                result = {"ok": False, "error": "不可取消已过期挂号（就诊日期已过）"}
                logger.log_tool_call(
                    "cancel_appointment", params, result, _now_ms() - t0, trace_id
                )
                return result

            reg_id = row["reg_ID"]   # 挂号编号
            sch_id = row["sch_id"]   # 对应的排班 ID

            # ── 执行取消（两步必须在同一事务内，保证原子性）──
            # 步骤A：标记挂号记录为取消状态（reg_Status=0）
            cur.execute("UPDATE register SET reg_Status = 0 WHERE reg_ID = %s", (reg_id,))
            # 步骤B：退回号源（sch_available 加 1）
            cur.execute(
                "UPDATE schedule SET sch_available = sch_available + 1 WHERE sch_id = %s",
                (sch_id,),
            )
            # with 块正常退出时自动 commit，两步操作原子提交

        result = {
            "ok":     True,
            "reg_id": reg_id,
            "msg":    f"挂号已取消（编号 {reg_id}），号源已退回",
        }
        logger.log_tool_call("cancel_appointment", params, result, _now_ms() - t0, trace_id)
        return result

    except Exception as exc:
        # 异常时 db_transaction 已自动 rollback，两个 UPDATE 均被撤销
        logger.log_error("cancel_appointment 异常", trace_id, exc=exc, params=params)
        return {"ok": False, "error": "取消挂号失败，请稍后重试"}
