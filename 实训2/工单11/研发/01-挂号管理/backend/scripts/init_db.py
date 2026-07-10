"""
工单11 医疗挂号Agent - 数据库初始化脚本
执行: python scripts/init_db.py
功能:
  1. 建表（执行 schema_mysql.sql）
  2. 插入种子数据（科室/医生/排班/用户/家属/历史挂号）
  排班日期范围: 2026-07-05 ~ 2026-07-19
"""
import os        # 标准库：读取操作系统环境变量（如 DB_HOST）
import sys       # 标准库：操作 Python 运行时，如修改模块搜索路径 sys.path
from datetime import date, timedelta   # 标准库：date 表示日期，timedelta 表示时间差
from pathlib import Path               # 标准库：面向对象的路径操作，比 os.path 更易用

# 确保 src 可导入
# __file__ 是当前脚本路径，.parent.parent 向上两级到项目根目录，插入到模块搜索路径首位
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pymysql：纯 Python 实现的 MySQL 客户端库，用来连接 MySQL 数据库执行 SQL
import pymysql
import pymysql.cursors   # cursors 子模块，提供 DictCursor（查询结果以字典形式返回）
# python-dotenv：读取 .env 文件里的 KEY=VALUE 配置，加载到 os.environ，方便本地开发
from dotenv import load_dotenv

# 从当前脚本向上两级找到 .env 文件并加载（数据库密码等不能硬编码到代码里）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 从环境变量（或 .env）读取数据库连接参数，构建连接配置字典
DB_CONFIG = dict(
    host     = os.getenv("DB_HOST", "127.0.0.1"),   # 数据库主机，默认本机
    port     = int(os.getenv("DB_PORT", "3306")),    # 端口，MySQL 默认 3306
    user     = os.getenv("DB_USER", "root"),          # 数据库用户名
    password = os.getenv("DB_PASSWORD", "root"),      # 数据库密码
    database = os.getenv("DB_NAME", "medical_agent"), # 数据库名
    charset  = "utf8mb4",                             # 字符集，支持完整 Unicode（含 emoji）
    cursorclass = pymysql.cursors.DictCursor,         # 让查询结果以 dict 返回，row["列名"] 更直观
)

# SCHEMA_FILE：与当前脚本同目录的 schema_mysql.sql 文件路径（存放建表 DDL）
SCHEMA_FILE = Path(__file__).parent / "schema_mysql.sql"


# ─────────── 数据定义 ───────────

# 科室种子数据：(dep_ID, dep_Name, dep_Address)
DEPARTMENTS = [
    (1, "内科",   "门诊楼2F 201"),
    (2, "外科",   "门诊楼3F 301"),
    (3, "儿科",   "门诊楼1F 101"),
    (4, "皮肤科", "门诊楼2F 205"),
    (5, "眼科",   "门诊楼4F 401"),
    (6, "牙科",   "门诊楼4F 405"),
    (7, "消化内科","门诊楼2F 208"),
    (8, "心内科", "门诊楼3F 308"),
]

# 医生种子数据：每条记录格式 (d_ID, d_Name, d_Sex, d_Profession, dep_ID)
# d_Profession：职称（主任医师/副主任医师/主治医师）
# dep_ID：所属科室 ID，与 DEPARTMENTS 的 dep_ID 对应
DOCTORS = [
    # 内科（dep_ID=1）
    (1,  "王建明", "男", "主任医师",   1),
    (2,  "刘晓华", "女", "副主任医师", 1),
    (3,  "陈磊",   "男", "主治医师",   1),
    # 外科（dep_ID=2）
    (4,  "张伟",   "男", "主任医师",   2),
    (5,  "李志强", "男", "副主任医师", 2),
    (6,  "赵阳",   "男", "主治医师",   2),
    # 儿科（dep_ID=3）
    (7,  "李华",   "男", "主任医师",   3),
    (8,  "王芳",   "女", "副主任医师", 3),
    (9,  "张明",   "男", "主治医师",   3),
    # 皮肤科（dep_ID=4）
    (10, "刘敏",   "女", "主任医师",   4),
    (11, "吴倩",   "女", "副主任医师", 4),
    (12, "周杰",   "男", "主治医师",   4),
    # 眼科（dep_ID=5）
    (13, "张建国", "男", "主任医师",   5),
    (14, "陈静",   "女", "副主任医师", 5),
    (15, "林强",   "男", "主治医师",   5),
    # 牙科（dep_ID=6）
    (16, "孙美丽", "女", "主任医师",   6),
    (17, "郭凯",   "男", "副主任医师", 6),
    (18, "黄磊",   "男", "主治医师",   6),
    # 消化内科（dep_ID=7）
    (19, "马云峰", "男", "主任医师",   7),
    (20, "杨帆",   "男", "副主任医师", 7),
    (21, "薛涛",   "男", "主治医师",   7),
    # 心内科（dep_ID=8）
    (22, "白雪",   "女", "主任医师",   8),
    (23, "秦浩",   "男", "副主任医师", 8),
    (24, "沈冰",   "女", "主治医师",   8),
]

# 每位医生的出诊规律字典，键=医生 ID，值=(出诊周几列表, 时间段列表, 号源类型)
# 规则：主任/副主任医师出专家号，主治医师出普通号（同一医生同一时段只有一种号）
# 周一=0 ... 周日=6（Python weekday() 的返回约定）
DOCTOR_PATTERN = {
    # 内科：王建明(主任)=专家, 刘晓华(副主任)=专家, 陈磊(主治)=普通
    1:  ([0,2,4],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    2:  ([1,3],      ["09:00-11:00","14:00-17:00"], ["专家"]),
    3:  ([0,1,2,3,4],["09:00-11:00"],               ["普通"]),
    # 外科：张伟(主任)=专家, 李志强(副主任)=专家, 赵阳(主治)=普通
    4:  ([1,3,5],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    5:  ([0,2,4],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    6:  ([0,1,2,3,4],["14:00-17:00"],               ["普通"]),
    # 儿科：李华(主任)=专家, 王芳(副主任)=专家, 张明(主治)=普通
    7:  ([0,2,4,5],  ["09:00-11:00","14:00-17:00"], ["专家"]),
    8:  ([1,3,5],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    9:  ([0,1,2,3,4],["09:00-11:00","14:00-17:00"], ["普通"]),
    # 皮肤科：刘敏(主任)=专家, 吴倩(副主任)=专家, 周杰(主治)=普通
    10: ([1,3,5],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    11: ([0,2,4],    ["09:00-11:00"],               ["专家"]),
    12: ([0,1,2,3,4],["14:00-17:00"],               ["普通"]),
    # 眼科：张建国(主任)=专家, 陈静(副主任)=专家, 林强(主治)=普通
    13: ([0,2,4],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    14: ([1,3,5],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    15: ([0,1,2,3,4],["09:00-11:00"],               ["普通"]),
    # 牙科：孙美丽(主任)=专家, 郭凯(副主任)=专家, 黄磊(主治)=普通
    16: ([1,3,5],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    17: ([0,2,4],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    18: ([0,1,2,3,4],["14:00-17:00"],               ["普通"]),
    # 消化内科：马云峰(主任)=专家, 杨帆(副主任)=专家, 薛涛(主治)=普通
    19: ([0,2,4],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    20: ([1,3,5],    ["09:00-11:00"],               ["专家"]),
    21: ([0,1,2,3,4],["14:00-17:00"],               ["普通"]),
    # 心内科：白雪(主任)=专家, 秦浩(副主任)=专家, 沈冰(主治)=普通
    22: ([1,3,5],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    23: ([0,2,4],    ["09:00-11:00","14:00-17:00"], ["专家"]),
    24: ([0,1,2,3,4],["09:00-11:00"],               ["普通"]),
}

# 号源配额：专家号每场8个，普通号每场15个
QUOTA = {"专家": 8, "普通": 15}

# 测试用户账户数据：(ua_id, ua_Name, ua_Phone, ua_Email)
USERS = [
    (1, "张三", "13800138001", "zhangsan@example.com"),
    (2, "李四", "13800138002", "lisi@example.com"),
]

# 患者数据：(p_ID, p_Name, p_Sex, p_Address, p_Birth, ps_ID)
# ps_ID：就诊状态 ID，关联 patientstatus 表
PATIENTS = [
    (1, "张三",   "男", "北京市朝阳区",  "1985-06-15", 1),
    (2, "张小明", "男", "北京市朝阳区",  "2018-03-10", 1),  # 张三的大宝
    (3, "张小红", "女", "北京市朝阳区",  "2020-07-22", 1),  # 张三的二宝
    (4, "李四",   "男", "北京市海淀区",  "1990-11-05", 1),
]

# 就诊状态数据：(ps_ID, ps_Name, ps_Remark)
PATIENT_STATUS = [(1, "初诊", ""), (2, "复诊", ""), (3, "急诊", "")]

# 家属关系数据：(fr_id, user_id, patient_id, relation_type, alias)
# alias：用户在 App 里给家属起的昵称，LLM 会用昵称来定位患者
FAMILY_RELATIONS = [
    (1, 1, 1, "自己", "我"),       # 张三自己
    (2, 1, 2, "儿子", "大宝"),     # 张三的儿子
    (3, 1, 3, "女儿", "二宝"),     # 张三的女儿
    (4, 2, 4, "自己", "我"),       # 李四自己
]


def run_sql_file(conn, filepath: Path):
    """
    通过 subprocess 调用 mysql 命令行客户端执行 DDL 建表脚本。

    为什么用 subprocess 而不是 Python 内解析 SQL？
      schema_mysql.sql 里包含 DELIMITER 指令、触发器等，
      用 Python split(';') 逐句执行容易出错；调用 mysql 客户端最可靠。

    参数：
      conn     : 已建立的 pymysql 连接（本函数其实不用它，只为接口一致）
      filepath : schema_mysql.sql 的 Path 对象
    """
    import subprocess   # 标准库：运行外部子进程（此处调用 mysql 命令行）
    # 从 src.config 模块引入数据库连接参数（已从 .env 加载）
    from src.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
    # 拼接 mysql 客户端命令参数，通过 stdin 传入 SQL 文件内容
    result = subprocess.run(
        ["mysql", f"-u{DB_USER}", f"-p{DB_PASSWORD}",
         f"-h{DB_HOST}", f"-P{DB_PORT}", DB_NAME],
        stdin=open(filepath, encoding="utf-8"),   # 把 SQL 文件内容作为标准输入
        capture_output=True, text=True,            # 捕获 stdout/stderr，以文本模式返回
    )
    # returncode != 0 且 stderr 含 "ERROR" 时表示执行失败
    if result.returncode != 0 and "ERROR" in result.stderr:
        raise RuntimeError(f"Schema 执行失败: {result.stderr}")
    print(f"  [OK] 执行 {filepath.name}")


def generate_schedules(date_start: date, date_end: date) -> list:
    """
    根据出诊规律（DOCTOR_PATTERN）生成日期范围内所有排班记录。

    参数：
      date_start : 起始日期（含）
      date_end   : 结束日期（含）

    返回：
      list，每条记录为 (d_id, sch_date, sch_time_slot, sch_type, sch_available, sch_total)
      sch_available：可用号数。过去日期设为 0（模拟已用完），未来保留原始配额。
    """
    rows = []
    cur_date = date_start   # 从起始日期开始逐天迭代
    while cur_date <= date_end:
        weekday = cur_date.weekday()   # 0=周一, 6=周日
        # 遍历所有医生的出诊规律
        for d_id, (workdays, slots, types) in DOCTOR_PATTERN.items():
            if weekday not in workdays:  # 该医生当天不出诊，跳过
                continue
            for slot in slots:           # 遍历该医生当天所有时间段
                for stype in types:      # 遍历该时间段所有号源类型（专家/普通）
                    total = QUOTA[stype]  # 该类型号源总数
                    # 过去日期号源清零，模拟历史数据已挂完；未来日期保留初始配额
                    available = total if cur_date >= date.today() else 0
                    rows.append((d_id, cur_date.isoformat(), slot, stype, available, total))
        cur_date += timedelta(days=1)   # 日期加一天，继续下一天
    return rows


def insert_seed(conn):
    """
    向数据库写入所有种子数据，包括：
      就诊状态 → 科室 → 医生 → 患者 → 用户账户 → 家属关系 → 排班 → 演示挂号记录

    使用 INSERT IGNORE 保证幂等（重复执行不报错，已有数据不覆盖）。
    """
    cur = conn.cursor()   # 获取游标，用于执行 SQL

    # 写入就诊状态（初诊/复诊/急诊）
    for row in PATIENT_STATUS:
        cur.execute(
            "INSERT IGNORE INTO patientstatus (ps_ID, ps_Name, ps_Remark) VALUES (%s,%s,%s)", row
        )

    # 写入科室数据
    for row in DEPARTMENTS:
        cur.execute(
            "INSERT IGNORE INTO department (dep_ID, dep_Name, dep_Address) VALUES (%s,%s,%s)", row
        )

    # 写入医生数据
    for row in DOCTORS:
        cur.execute(
            """INSERT IGNORE INTO doctor (d_ID, d_Name, d_Sex, d_Profession, dep_ID)
               VALUES (%s,%s,%s,%s,%s)""", row
        )

    # 写入患者数据
    for row in PATIENTS:
        cur.execute(
            """INSERT IGNORE INTO patient (p_ID, p_Name, p_Sex, p_Address, p_Birth, ps_ID)
               VALUES (%s,%s,%s,%s,%s,%s)""", row
        )

    # 写入用户账户数据
    for row in USERS:
        cur.execute(
            """INSERT IGNORE INTO user_account (ua_id, ua_Name, ua_Phone, ua_Email)
               VALUES (%s,%s,%s,%s)""", row
        )

    # 写入家属关系数据（用户与患者的昵称绑定）
    for row in FAMILY_RELATIONS:
        cur.execute(
            """INSERT IGNORE INTO family_relation (fr_id, user_id, patient_id, relation_type, alias)
               VALUES (%s,%s,%s,%s,%s)""", row
        )

    # ── 排班范围：上上周到未来两周，覆盖所有演示场景 ──
    today      = date.today()
    date_start = today - timedelta(days=21)   # 往前3周，确保"上周三"有排班
    date_end   = today + timedelta(days=14)   # 往后2周
    schedules  = generate_schedules(date_start, date_end)
    for row in schedules:
        cur.execute(
            """INSERT IGNORE INTO schedule (d_id, sch_date, sch_time_slot, sch_type, sch_available, sch_total)
               VALUES (%s,%s,%s,%s,%s,%s)""", row
        )
    print(f"  [OK] 生成排班记录 {len(schedules)} 条 ({date_start} ~ {date_end})")

    # ── 演示挂号记录（INSERT IGNORE 保证幂等） ──
    # 辅助函数：查某科室、某日期的一条排班 ID
    def find_sch(dept_name, d_id, target_date):
        cur.execute(
            """SELECT s.sch_id, dep.dep_ID FROM schedule s
               JOIN doctor d ON d.d_ID = s.d_id
               JOIN department dep ON dep.dep_ID = d.dep_ID
               WHERE d.d_ID = %s AND s.sch_date = %s LIMIT 1""",
            (d_id, target_date.isoformat())
        )
        return cur.fetchone()

    def insert_reg(dep_id, p_id, d_id, sch_id, reg_time_str, fee, status):
        cur.execute(
            """INSERT IGNORE INTO register
               (dep_ID, p_ID, w_ID, d_ID, sch_id, reg_Time, reg_Fee, reg_Order, reg_Status)
               VALUES (%s,%s,NULL,%s,%s,%s,%s,1,%s)""",
            (dep_id, p_id, d_id, sch_id, reg_time_str, fee, status)
        )
        if cur.rowcount > 0 and status == 1:
            cur.execute(
                "UPDATE schedule SET sch_available = sch_available-1 WHERE sch_id=%s AND sch_available>0",
                (sch_id,)
            )
        return cur.rowcount > 0

    # ── 1. 今天下午儿科专家（用于测试"帮我大宝挂儿科专家号"已有记录场景）
    #    儿科主任医师 李华 d_id=7，出诊周：0/2/4/5，今天如果是出诊日就插
    if today.weekday() in [0, 2, 4, 5]:
        s = find_sch("儿科", 7, today)
        if s:
            insert_reg(s["dep_ID"], 2, 7, s["sch_id"],
                       today.strftime("%Y-%m-%d 09:05:00"), 50, 1)

    # ── 2. 眼科专家历史记录：张建国(主任,d_id=13)，用于"再约那个专家"场景
    #    取上周一（最近一个已过去的出诊日），d_id=13 出诊周：0/2/4
    def last_weekday(wd):
        """返回最近一个已过去的周wd（0=周一），至少1天前"""
        delta = (today.weekday() - wd) % 7
        if delta == 0:
            delta = 7
        return today - timedelta(days=delta)

    eye_date = last_weekday(0)   # 上周一
    s = find_sch("眼科", 13, eye_date)
    if s:
        insert_reg(s["dep_ID"], 1, 13, s["sch_id"],
                   (eye_date - timedelta(days=3)).strftime("%Y-%m-%d 14:22:00"), 50, 1)

    # ── 3. 消化内科普通号：申请日=上周三，就诊日=未来最近一个工作日（还没到，可取消）
    #    用于"取消上周三挂的消化内科普通号"—— reg_Time 是上周三，sch_date 是未来
    #    薛涛(主治,d_id=21) 周一到周五下午都出诊，取明天/后天最近一场
    last_wed = last_weekday(2)   # 上周三（申请时间）
    # 找未来第一个薛涛出诊的工作日（周一到周五）
    future_date = None
    for offset in range(1, 8):
        cand = today + timedelta(days=offset)
        if cand.weekday() in [0, 1, 2, 3, 4]:   # 薛涛出诊日
            future_date = cand
            break
    if future_date:
        s = find_sch("消化内科", 21, future_date)
        if s:
            # reg_Time 用上周三（申请时间），sch_date 就诊日在未来
            insert_reg(s["dep_ID"], 1, 21, s["sch_id"],
                       last_wed.strftime("%Y-%m-%d 08:45:00"), 15, 1)

    # ── 4. 内科待就诊（张三本人，明天），用于统计"待就诊"数量展示
    tomorrow = today + timedelta(days=1)
    if tomorrow.weekday() in [1, 3]:   # 刘晓华 d_id=2 出诊周二/四
        s = find_sch("内科", 2, tomorrow)
        if s:
            insert_reg(s["dep_ID"], 1, 2, s["sch_id"],
                       today.strftime("%Y-%m-%d 10:10:00"), 50, 1)

    # ── 5. 已取消历史记录（上周某天，供统计"已取消"展示）
    canceled_date = last_weekday(1)   # 上周二
    s = find_sch("皮肤科", 10, canceled_date)
    if s:
        insert_reg(s["dep_ID"], 1, 10, s["sch_id"],
                   (canceled_date - timedelta(days=2)).strftime("%Y-%m-%d 16:30:00"), 50, 0)

    conn.commit()
    print("  [OK] 演示挂号记录写入完成")


def reset_data(conn):
    """清空业务数据表（保留表结构），用于重新初始化演示数据。"""
    cur = conn.cursor()
    # 按外键依赖顺序删除：先子表，再父表
    for tbl in ("register", "schedule", "family_relation",
                "patient", "user_account", "doctor",
                "department", "patientstatus"):
        cur.execute(f"DELETE FROM {tbl}")
        print(f"  [RESET] 清空 {tbl}")
    conn.commit()


def main():
    """脚本主函数：建表 → 写种子数据 → 验证记录数量。"""
    import argparse
    parser = argparse.ArgumentParser(description="工单11 数据库初始化")
    parser.add_argument("--reset", action="store_true",
                        help="清空所有演示数据后重新初始化（用于数据错乱时重置）")
    args = parser.parse_args()

    print("=== 工单11 医疗挂号Agent - 数据库初始化 ===")
    conn = pymysql.connect(**DB_CONFIG)
    try:
        print("\n[1] 执行建表 DDL ...")
        run_sql_file(conn, SCHEMA_FILE)

        if args.reset:
            print("\n[1.5] --reset 模式：清空旧数据 ...")
            reset_data(conn)

        print("\n[2] 写入种子数据 ...")
        insert_seed(conn)

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM schedule")
        n = cur.fetchone()["cnt"]
        cur.execute("SELECT COUNT(*) AS cnt FROM doctor")
        d = cur.fetchone()["cnt"]
        cur.execute("SELECT COUNT(*) AS cnt FROM register")
        r = cur.fetchone()["cnt"]
        print(f"\n[3] 验证: 医生={d}, 排班={n}, 挂号记录={r}")
        print("\n[DONE] 初始化完成！可执行 start.bat 启动服务。")
    finally:
        conn.close()


# Python 惯用写法：仅当直接运行此脚本时才执行 main()，被 import 时不执行
if __name__ == "__main__":
    main()
