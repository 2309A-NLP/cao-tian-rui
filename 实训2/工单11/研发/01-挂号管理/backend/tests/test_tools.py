"""
工单11 医疗挂号Agent — 工具函数单元测试
=========================================
测试策略说明：

  本文件测试 tools_read.py 和 tools_write.py 中的 6 个工具函数。
  由于这些函数都依赖 MySQL 数据库，我们使用 unittest.mock 模拟数据库连接，
  使测试可以在没有真实数据库的情况下运行（隔离外部依赖）。

  Mock 原理：
    @patch("src.tools_read.get_connection") 会在测试运行期间，
    把 tools_read 模块里的 get_connection 函数替换成 Mock 对象。
    我们预先配置 Mock 的返回值（fetchone/fetchall 返回假数据），
    函数就会"以为"自己真的查了数据库。

测试覆盖范围：
  1. TestHelperFunctions  → 辅助函数纯计算逻辑（_match_time_slot/_validate_dept 等）
  2. TestGetFamilyMember  → get_family_member 正常/用户不存在/家属不存在
  3. TestQuerySchedule    → query_schedule 科室非法/空结果/有结果
  4. TestGetUserHistory   → get_user_history 无家属/有历史记录
  5. TestBookAppointment  → book_appointment 正常/用户不存在/排班不存在/无余量/过期/重复
  6. TestCancelAppointment→ cancel_appointment 正常/找不到/已取消/过期
  7. TestTryParseToolCallFromText → _try_parse_tool_call_from_text 三种格式
"""

import unittest   # 标准库：Python 内置单元测试框架
from datetime import date, timedelta   # 标准库：日期操作
# MagicMock：自动模拟任意属性和方法调用的 Mock 类
# patch：装饰器，测试期间替换指定的函数/对象为 Mock
from unittest.mock import MagicMock, patch

# ─── 被测模块 ───────────────────────────────────────────────────────────────
# 注意：如果运行时提示 ModuleNotFoundError，请确认工作目录是 backend/
# 推荐运行方式：
#   cd "F:\kimi  project\医疗agent1\01-挂号管理\backend"
#   python -m pytest tests/ -v
import sys, os
# 把 backend/ 目录插入模块搜索路径，确保 `from src.xxx import yyy` 能找到模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 tools_read 导入被测函数（含私有辅助函数，加下划线前缀但仍可导入测试）
from src.tools_read import (
    get_family_member,     # 工具1：查家属信息
    query_schedule,        # 工具2：查号源
    get_user_history,      # 工具3：查历史记录
    get_doctor_schedule,   # 工具4：查医生排班
    _match_time_slot,      # 私有辅助：时间偏好 → 时间段字符串
    _validate_dept,        # 私有辅助：校验科室名
    _validate_title,       # 私有辅助：校验号源类型
)
from src.tools_write import book_appointment, cancel_appointment   # 工具5/6
from src.agent import _try_parse_tool_call_from_text               # Agent 文本解析函数


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数测试（不依赖数据库，纯计算逻辑）
# ══════════════════════════════════════════════════════════════════════════════

class TestHelperFunctions(unittest.TestCase):
    """
    测试 tools_read.py 里的私有辅助函数。
    这些函数是纯函数（只有输入输出，无副作用），无需 Mock 数据库。
    """

    def test_match_time_slot_morning(self):
        """上午（小时 < 12）→ 应映射到 09:00-11:00 时间段。"""
        self.assertEqual(_match_time_slot("09:00"), "09:00-11:00")
        self.assertEqual(_match_time_slot("11:30"), "09:00-11:00")   # 11点半也算上午

    def test_match_time_slot_afternoon(self):
        """下午（12 <= 小时 < 18）→ 应映射到 14:00-17:00 时间段。"""
        self.assertEqual(_match_time_slot("14:00"), "14:00-17:00")
        self.assertEqual(_match_time_slot("16:59"), "14:00-17:00")

    def test_match_time_slot_evening(self):
        """晚上（小时 >= 18）→ 应映射到 19:00-21:00 时间段。"""
        self.assertEqual(_match_time_slot("19:00"), "19:00-21:00")
        self.assertEqual(_match_time_slot("20:00"), "19:00-21:00")

    def test_match_time_slot_none(self):
        """time_pref 为 None 时，返回 None（表示不过滤，查全天）。"""
        self.assertIsNone(_match_time_slot(None))

    def test_validate_dept_valid(self):
        """合法科室名（在白名单中）应原样返回。"""
        from src.config import VALID_DEPARTMENTS
        for dept in VALID_DEPARTMENTS:
            self.assertEqual(_validate_dept(dept), dept)

    def test_validate_dept_invalid(self):
        """不存在的科室应返回 None。"""
        self.assertIsNone(_validate_dept("银河宇宙科"))   # 不存在的科室
        self.assertIsNone(_validate_dept(""))              # 空字符串也不合法

    def test_validate_title_valid(self):
        """合法号源类型（专家/普通）应原样返回。"""
        self.assertEqual(_validate_title("专家"), "专家")
        self.assertEqual(_validate_title("普通"), "普通")

    def test_validate_title_none(self):
        """None 透传（表示不限号源类型），不转换为其他值。"""
        self.assertIsNone(_validate_title(None))

    def test_validate_title_invalid(self):
        """非法类型应返回 None（视为不过滤，而非报错）。"""
        self.assertIsNone(_validate_title("超级VIP"))


# ══════════════════════════════════════════════════════════════════════════════
# get_family_member 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestGetFamilyMember(unittest.TestCase):
    """
    测试 get_family_member() 函数的三种路径：
      1. 正常找到家属
      2. 用户不存在
      3. 家属昵称不存在

    Mock 技术说明：
      @patch("src.tools_read.get_connection") 在测试期间把模块内的 get_connection
      函数替换为 Mock，测试函数会收到这个 Mock 作为参数 mock_get_conn。
      我们控制 mock_get_conn.return_value（即 conn 对象）的 cursor().fetchone()
      的返回值，来模拟不同的数据库查询结果。
    """

    def _make_mock_conn(self, user_row, family_row):
        """
        构造 Mock 数据库连接，按顺序返回两次 fetchone 的结果。

        参数：
          user_row   : 第一次 fetchone() 的返回值（user_account 查询结果）
          family_row : 第二次 fetchone() 的返回值（family_relation JOIN patient 结果）

        side_effect：MagicMock 的特性，传入列表时每次调用依次返回列表中的元素。
        """
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [user_row, family_row]   # 按顺序返回
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor   # conn.cursor() 返回 mock_cursor
        return mock_conn

    @patch("src.tools_read.get_connection")
    def test_success(self, mock_get_conn):
        """正常场景：用户存在，家属昵称"大宝"能找到对应 patient_id。"""
        # 配置 Mock：用户存在（返回 ua_id），家属存在（返回 patient_id 和姓名）
        mock_get_conn.return_value = self._make_mock_conn(
            user_row={"ua_id": 1},
            family_row={"patient_id": 1001, "p_Name": "张小宝"},
        )
        result = get_family_member(user_id=1, alias="大宝")
        self.assertTrue(result["ok"])                             # 成功标志
        self.assertEqual(result["patient_id"], 1001)              # 返回正确 patient_id
        self.assertEqual(result["patient_name"], "张小宝")        # 返回正确姓名

    @patch("src.tools_read.get_connection")
    def test_user_not_found(self, mock_get_conn):
        """用户不存在时，第一次 fetchone 返回 None，应返回 ok=False。"""
        mock_get_conn.return_value = self._make_mock_conn(
            user_row=None,    # None → 用户不存在
            family_row=None,  # 不会走到这里，但还是要传（side_effect 列表）
        )
        result = get_family_member(user_id=999, alias="大宝")
        self.assertFalse(result["ok"])
        self.assertIn("用户不存在", result["error"])   # 错误信息包含"用户不存在"

    @patch("src.tools_read.get_connection")
    def test_alias_not_found(self, mock_get_conn):
        """用户存在但昵称未绑定家属，第二次 fetchone 返回 None，应返回 ok=False。"""
        mock_get_conn.return_value = self._make_mock_conn(
            user_row={"ua_id": 1},  # 用户存在
            family_row=None,         # None → 该昵称无对应家属
        )
        result = get_family_member(user_id=1, alias="不存在的昵称")
        self.assertFalse(result["ok"])
        self.assertIn("未找到家属", result["error"])


# ══════════════════════════════════════════════════════════════════════════════
# query_schedule 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestQuerySchedule(unittest.TestCase):
    """测试 query_schedule() 函数的多种场景。"""

    @patch("src.tools_read.get_connection")
    def test_invalid_dept(self, mock_get_conn):
        """
        科室不在白名单中时，应在连接数据库之前就返回 ok=False。
        验证：mock_get_conn.assert_not_called() 确认没有建立数据库连接（快速失败）。
        """
        result = query_schedule(
            dept="不存在的科室",
            date_start="2099-01-01",
            date_end="2099-01-07",
        )
        self.assertFalse(result["ok"])
        self.assertIn("科室不存在", result["error"])
        # 科室校验在连接数据库之前，所以 get_connection 不应该被调用
        mock_get_conn.assert_not_called()

    @patch("src.tools_read.get_connection")
    def test_no_schedules(self, mock_get_conn):
        """数据库返回空列表时，ok=True 但 schedules 为空列表。"""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []   # 模拟没有可用号源
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        from src.config import VALID_DEPARTMENTS
        result = query_schedule(
            dept=VALID_DEPARTMENTS[0],    # 取白名单第一个科室（"内科"）
            date_start="2099-01-01",
            date_end="2099-01-07",
        )
        self.assertTrue(result["ok"])              # 查询本身成功
        self.assertEqual(result["schedules"], [])  # 但没有数据

    @patch("src.tools_read.get_connection")
    def test_returns_schedules(self, mock_get_conn):
        """有可用号源时，ok=True 且 schedules 包含正确数据。"""
        tomorrow = date.today() + timedelta(days=1)
        # 构造一条假号源数据（dict 格式，与 DictCursor 返回值一致）
        fake_schedule = {
            "sch_id": 10,
            "doctor_name": "李医生",
            "dept": "内科",
            "title": "普通",
            "date": tomorrow,            # Python date 对象（DictCursor 的真实返回类型）
            "time_slot": "09:00-11:00",
            "available": 3,
            "total": 10,
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [fake_schedule]   # 返回包含一条记录的列表
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        from src.config import VALID_DEPARTMENTS
        result = query_schedule(
            dept=VALID_DEPARTMENTS[0],
            date_start=tomorrow.isoformat(),
            date_end=(tomorrow + timedelta(days=3)).isoformat(),
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["schedules"]), 1)                     # 有一条记录
        self.assertEqual(result["schedules"][0]["doctor_name"], "李医生")  # 数据正确


# ══════════════════════════════════════════════════════════════════════════════
# get_user_history 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestGetUserHistory(unittest.TestCase):
    """测试 get_user_history() 函数。"""

    @patch("src.tools_read.get_connection")
    def test_no_family_members(self, mock_get_conn):
        """用户没有绑定任何家属时，应返回空历史记录（ok=True, history=[]）。"""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"ua_id": 1}   # 用户存在
        mock_cursor.fetchall.return_value = []              # 家属查询返回空
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        result = get_user_history(user_id=1)
        self.assertTrue(result["ok"])
        self.assertEqual(result["history"], [])

    @patch("src.tools_read.get_connection")
    def test_returns_history(self, mock_get_conn):
        """有历史记录时，ok=True 且 history 包含正确数据。"""
        fake_history = {
            "reg_id": 55,
            "doctor_name": "王大夫",
            "dept": "内科",
            "date": date.today() - timedelta(days=30),  # 一个月前的就诊记录
            "time_slot": "14:00-17:00",
            "title": "专家",
            "reg_status": 1,
            "patient_name": "张小宝",
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"ua_id": 1}   # 用户存在
        # fetchall 被调用两次：
        #   第一次：查家属列表（返回一个 patient_id）
        #   第二次：查挂号历史（返回 fake_history）
        # side_effect 列表按调用顺序依次返回
        mock_cursor.fetchall.side_effect = [
            [{"patient_id": 1001}],   # 第一次 fetchall：家属列表
            [fake_history],            # 第二次 fetchall：历史记录
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        result = get_user_history(user_id=1)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["history"][0]["doctor_name"], "王大夫")


# ══════════════════════════════════════════════════════════════════════════════
# book_appointment 测试（写操作，用 db_transaction Mock）
# ══════════════════════════════════════════════════════════════════════════════

class TestBookAppointment(unittest.TestCase):
    """
    测试 book_appointment() 函数的多种校验路径。

    db_transaction 是一个上下文管理器（使用 with 语句），Mock 时需要：
      1. 创建 MagicMock 并设置 __enter__/__exit__ 方法
      2. __enter__ 返回 mock_conn（模拟 with 块内的 conn 对象）
      3. __exit__ 返回 False（不抑制异常）

    @patch("src.tools_write.db_transaction") 替换 tools_write 模块内的
    db_transaction，测试函数收到 mock_tx 参数，设置 mock_tx.return_value
    为我们构造的 mock_ctx。
    """

    def _make_mock_ctx(self, fetchone_values: list, fetchall_values: list = None):
        """
        构造 Mock 事务上下文管理器。

        参数：
          fetchone_values  : 每次 fetchone() 按顺序返回的值列表
          fetchall_values  : 每次 fetchall() 按顺序返回的值列表（可选）

        返回：(mock_ctx, mock_cursor)
          mock_ctx    : 用于设置 mock_tx.return_value 的上下文 Mock
          mock_cursor : 用于设置额外属性（如 lastrowid、rowcount）
        """
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = fetchone_values   # 按顺序返回不同查询结果
        if fetchall_values:
            mock_cursor.fetchall.side_effect = fetchall_values
        mock_cursor.rowcount = 1   # 模拟 UPDATE 成功影响了 1 行

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # 构造上下文管理器：with db_transaction() as conn → conn = mock_conn
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)   # with 进入时返回 mock_conn
        mock_ctx.__exit__ = MagicMock(return_value=False)        # False: 不抑制异常

        return mock_ctx, mock_cursor

    @patch("src.tools_write.db_transaction")
    def test_success(self, mock_tx):
        """
        正常挂号场景：所有校验通过，最终返回 ok=True 和 reg_id。

        fetchone 依次模拟：
          1. 用户存在
          2. 患者存在
          3. 排班存在（FOR UPDATE 加锁查询）
          4. 重复挂号检查：None 表示无重复记录
        """
        tomorrow = date.today() + timedelta(days=1)
        mock_ctx, mock_cursor = self._make_mock_ctx([
            {"ua_id": 1},         # 第1次 fetchone：用户存在
            {"p_ID": 1001},        # 第2次 fetchone：患者存在
            {                      # 第3次 fetchone：排班信息（FOR UPDATE 锁定）
                "sch_id": 10, "d_id": 5,
                "sch_date": tomorrow,           # 明天，在有效范围内
                "sch_time_slot": "14:00-17:00",
                "sch_type": "普通",
                "sch_available": 3,             # 有余量
                "dep_ID": 2, "dep_Name": "内科",
                "doctor_name": "测试医生",       # 必须有此键（SQL 里 AS doctor_name）
            },
            None,                  # 第4次 fetchone：重复检查 → None 表示无重复
        ])
        mock_cursor.lastrowid = 999   # 模拟 INSERT 后自增主键（挂号编号）
        mock_tx.return_value = mock_ctx

        result = book_appointment(user_id=1, patient_id=1001, sch_id=10)
        self.assertTrue(result["ok"])
        self.assertEqual(result["reg_id"], 999)      # 挂号编号正确
        self.assertIn("挂号成功", result["msg"])      # 回复信息包含成功提示

    @patch("src.tools_write.db_transaction")
    def test_user_not_found(self, mock_tx):
        """用户不存在时（第一次 fetchone 返回 None），应返回 ok=False。"""
        mock_ctx, _ = self._make_mock_ctx([None])   # 第1次 fetchone 直接返回 None
        mock_tx.return_value = mock_ctx

        result = book_appointment(user_id=999, patient_id=1001, sch_id=10)
        self.assertFalse(result["ok"])
        self.assertIn("用户不存在", result["error"])

    @patch("src.tools_write.db_transaction")
    def test_schedule_not_found(self, mock_tx):
        """排班 ID 不存在时（FOR UPDATE 查询返回 None），应返回 ok=False。"""
        mock_ctx, _ = self._make_mock_ctx([
            {"ua_id": 1},    # 用户存在
            {"p_ID": 1001},   # 患者存在
            None,             # 排班不存在（FOR UPDATE 查不到）
        ])
        mock_tx.return_value = mock_ctx

        result = book_appointment(user_id=1, patient_id=1001, sch_id=9999)
        self.assertFalse(result["ok"])
        self.assertIn("排班不存在", result["error"])

    @patch("src.tools_write.db_transaction")
    def test_no_available_slots(self, mock_tx):
        """号源已满（sch_available=0）时，应返回 ok=False。"""
        tomorrow = date.today() + timedelta(days=1)
        mock_ctx, _ = self._make_mock_ctx([
            {"ua_id": 1},
            {"p_ID": 1001},
            {
                "sch_id": 10, "d_id": 5,
                "sch_date": tomorrow, "sch_time_slot": "14:00-17:00",
                "sch_type": "普通",
                "sch_available": 0,   # ← 关键：号源已满
                "dep_ID": 2, "dep_Name": "内科",
            },
        ])
        mock_tx.return_value = mock_ctx

        result = book_appointment(user_id=1, patient_id=1001, sch_id=10)
        self.assertFalse(result["ok"])
        self.assertIn("号源已满", result["error"])

    @patch("src.tools_write.db_transaction")
    def test_expired_date(self, mock_tx):
        """排班日期是昨天（过去的日期），应返回 ok=False。"""
        yesterday = date.today() - timedelta(days=1)
        mock_ctx, _ = self._make_mock_ctx([
            {"ua_id": 1},
            {"p_ID": 1001},
            {
                "sch_id": 10, "d_id": 5,
                "sch_date": yesterday,    # ← 关键：过去的日期
                "sch_time_slot": "14:00-17:00",
                "sch_type": "普通", "sch_available": 3,
                "dep_ID": 2, "dep_Name": "内科",
            },
        ])
        mock_tx.return_value = mock_ctx

        result = book_appointment(user_id=1, patient_id=1001, sch_id=10)
        self.assertFalse(result["ok"])
        self.assertIn("过去日期", result["error"])

    @patch("src.tools_write.db_transaction")
    def test_duplicate_booking(self, mock_tx):
        """同一患者同一排班已挂过（重复挂号检查命中），应返回 ok=False。"""
        tomorrow = date.today() + timedelta(days=1)
        mock_ctx, _ = self._make_mock_ctx([
            {"ua_id": 1},
            {"p_ID": 1001},
            {
                "sch_id": 10, "d_id": 5,
                "sch_date": tomorrow, "sch_time_slot": "14:00-17:00",
                "sch_type": "普通", "sch_available": 3,
                "dep_ID": 2, "dep_Name": "内科",
            },
            {"reg_ID": 55},   # ← 关键：重复检查返回已有记录（非 None），说明已挂过
        ])
        mock_tx.return_value = mock_ctx

        result = book_appointment(user_id=1, patient_id=1001, sch_id=10)
        self.assertFalse(result["ok"])
        self.assertIn("重复挂号", result["error"])


# ══════════════════════════════════════════════════════════════════════════════
# cancel_appointment 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestCancelAppointment(unittest.TestCase):
    """测试 cancel_appointment() 函数的多种校验路径。"""

    def _make_mock_ctx(self, fetchone_value):
        """
        构造简化版 Mock 事务上下文（只有一次 fetchone）。
        适用于 cancel_appointment，它只需要查一次挂号记录。
        """
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = fetchone_value   # 单次返回值

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        return mock_ctx

    @patch("src.tools_write.db_transaction")
    def test_success_by_reg_id(self, mock_tx):
        """通过 reg_id 精确取消，正常场景，应返回 ok=True。"""
        tomorrow = date.today() + timedelta(days=1)
        mock_tx.return_value = self._make_mock_ctx({
            "reg_ID": 55, "sch_id": 10,
            "reg_Status": 1,       # 1=有效，可以取消
            "sch_date": tomorrow,  # 就诊日期是未来，允许取消
            "p_ID": 1001,
        })
        result = cancel_appointment(user_id=1, cond={"reg_id": 55})
        self.assertTrue(result["ok"])
        self.assertEqual(result["reg_id"], 55)
        self.assertIn("已取消", result["msg"])

    @patch("src.tools_write.db_transaction")
    def test_not_found(self, mock_tx):
        """找不到挂号记录时（fetchone 返回 None），应返回 ok=False。"""
        mock_tx.return_value = self._make_mock_ctx(None)
        result = cancel_appointment(user_id=1, cond={"reg_id": 9999})
        self.assertFalse(result["ok"])
        self.assertIn("未找到", result["error"])

    @patch("src.tools_write.db_transaction")
    def test_already_cancelled(self, mock_tx):
        """挂号记录已经是取消状态（reg_Status=0），不能再次取消，应返回 ok=False。"""
        tomorrow = date.today() + timedelta(days=1)
        mock_tx.return_value = self._make_mock_ctx({
            "reg_ID": 55, "sch_id": 10,
            "reg_Status": 0,       # ← 关键：0=已取消
            "sch_date": tomorrow,
            "p_ID": 1001,
        })
        result = cancel_appointment(user_id=1, cond={"reg_id": 55})
        self.assertFalse(result["ok"])
        self.assertIn("已取消", result["error"])

    @patch("src.tools_write.db_transaction")
    def test_expired_appointment(self, mock_tx):
        """就诊日期已过（昨天），不能取消，应返回 ok=False。"""
        yesterday = date.today() - timedelta(days=1)
        mock_tx.return_value = self._make_mock_ctx({
            "reg_ID": 55, "sch_id": 10,
            "reg_Status": 1,
            "sch_date": yesterday,   # ← 关键：就诊日已过
            "p_ID": 1001,
        })
        result = cancel_appointment(user_id=1, cond={"reg_id": 55})
        self.assertFalse(result["ok"])
        self.assertIn("过期", result["error"])


# ══════════════════════════════════════════════════════════════════════════════
# Agent 纯逻辑测试（不依赖 LLM 和数据库）
# ══════════════════════════════════════════════════════════════════════════════

class TestTryParseToolCallFromText(unittest.TestCase):
    """
    测试 _try_parse_tool_call_from_text() 函数。
    这是一个纯文本解析函数，无数据库或 LLM 依赖，无需任何 Mock。
    主要验证它能正确从不同格式的文本中提取工具调用 JSON。
    """

    def test_plain_json(self):
        """
        直接是 JSON 格式（某些模型的输出格式）。
        文本本身就是合法的工具调用 JSON，不需要额外解析。
        """
        text = '{"name": "query_schedule", "arguments": {"dept": "内科", "date_start": "2026-07-10", "date_end": "2026-07-17"}}'
        result = _try_parse_tool_call_from_text(text)
        self.assertIsNotNone(result)   # 应该能成功解析
        name, args = result
        self.assertEqual(name, "query_schedule")
        self.assertEqual(args["dept"], "内科")

    def test_markdown_code_block(self):
        """
        Markdown 代码块格式（LLM 有时会把 JSON 放进 ```json ... ``` 里）。
        函数应识别并提取代码块内的 JSON。
        """
        text = '''我来帮你查询号源。
```json
{"name": "get_family_member", "arguments": {"alias": "大宝"}}
```'''
        result = _try_parse_tool_call_from_text(text)
        self.assertIsNotNone(result)
        name, args = result
        self.assertEqual(name, "get_family_member")
        self.assertEqual(args["alias"], "大宝")

    def test_with_comment_prefix(self):
        """
        带 // 注释前缀的 JSON（少数模型的输出格式）。
        函数应去掉注释行后成功解析 JSON。
        """
        text = '// 调用查询工具\n{"name": "get_user_history", "arguments": {}}'
        result = _try_parse_tool_call_from_text(text)
        self.assertIsNotNone(result)
        name, args = result
        self.assertEqual(name, "get_user_history")

    def test_none_input(self):
        """输入为 None 或空字符串时，应返回 None（不抛异常）。"""
        self.assertIsNone(_try_parse_tool_call_from_text(None))
        self.assertIsNone(_try_parse_tool_call_from_text(""))

    def test_no_tool_call(self):
        """普通自然语言文本（不含工具调用），应返回 None。"""
        text = "您好，请问您想挂哪个科室的号？"
        self.assertIsNone(_try_parse_tool_call_from_text(text))

    def test_invalid_json(self):
        """格式错误的 JSON（如缺少引号），应返回 None 而非抛 JSONDecodeError。"""
        text = '{"name": "query_schedule", "arguments": {invalid json}}'
        self.assertIsNone(_try_parse_tool_call_from_text(text))


# ──────────────────────────────────────────────────────────────────────────────
# 程序入口（直接 python tests/test_tools.py 运行时生效）
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # verbosity=2：显示每个测试用例的名称和结果，方便调试
    unittest.main(verbosity=2)
