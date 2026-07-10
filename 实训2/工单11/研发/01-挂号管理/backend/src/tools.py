"""
工单11 医疗挂号Agent — 工具函数统一入口（向后兼容导出）
==========================================================
本文件已重构为"导出汇总"模块。

重构背景：
  原 tools.py 包含 6 个工具函数，合计 505 行，超出单文件建议长度。
  按读写职责拆分为：
    - tools_read.py  → 4 个查询类工具（只读 SELECT）
    - tools_write.py → 2 个写入类工具（INSERT / UPDATE，带事务）

向后兼容：
  其他模块仍可使用 `from src import tools; tools.query_schedule(...)` 调用，
  无需修改调用方代码。新代码推荐直接导入 tools_read / tools_write。
"""

# 将两个模块的所有公开函数重新导出到 tools 命名空间
# noqa: F401 告诉 flake8/ruff 忽略"导入但未使用"的 lint 警告
# （这些函数虽然在本文件中不直接调用，但被 import 后再导出给调用方用）
from src.tools_read import (  # noqa: F401
    get_family_member,    # 工具1：根据昵称查家属 patient_id
    query_schedule,       # 工具2：查询可预约号源
    get_user_history,     # 工具3：查询用户历史挂号记录
    get_doctor_schedule,  # 工具4：查询医生完整排班时间表
)
from src.tools_write import (  # noqa: F401
    book_appointment,    # 工具5：执行挂号（写库，带事务）
    cancel_appointment,  # 工具6：取消挂号（写库，带事务）
)
