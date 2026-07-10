"""
pytest 全局 fixture 配置文件。

作用：在 pytest 收集测试模块之前，把生产环境才有的数据库驱动打桩，
让 src/database.py 的顶层 import 不报 ModuleNotFoundError。

为什么要这样做？
  database.py 在模块级做了：
    import pymysql
    from dbutils.pooled_db import PooledDB
  只要 import src.tools_read（或任何 src 模块），就会触发这两行。
  测试环境可能没有安装 pymysql / dbutils，而我们的测试根本不需要真正连库，
  只需要 Mock 连接对象。所以用 sys.modules 预先插入假模块，
  让 Python 的 import 机制认为它们"已经导入过了"，跳过真正的安装包查找。

注意：这里只打桩 import，不改变任何测试逻辑。
      具体测试用例里仍然用 @patch("src.tools_read.get_connection") 等来控制行为。
"""
import sys   # 标准库：sys.modules 是 Python 已导入模块的全局注册表
# unittest.mock.MagicMock：Python 标准库的 Mock 类
# MagicMock 会自动创建任意属性访问和方法调用，不会抛 AttributeError，
# 适合替换任何我们不想实际执行的对象
from unittest.mock import MagicMock

# ── 打桩 pymysql ──────────────────────────────────────────────────────────────
# database.py 用到：pymysql.cursors.DictCursor
# 打桩目的：避免 pytest 启动时因缺少 pymysql 包而 ImportError
if "pymysql" not in sys.modules:
    _pymysql = MagicMock()                       # 伪造 pymysql 模块对象
    _pymysql.cursors = MagicMock()               # pymysql.cursors 子模块
    _pymysql.cursors.DictCursor = MagicMock()    # pymysql.cursors.DictCursor 类
    # 将伪造模块注册到 sys.modules，Python import 时优先查这里，找到就不再去硬盘找
    sys.modules["pymysql"] = _pymysql
    sys.modules["pymysql.cursors"] = _pymysql.cursors

# ── 打桩 dbutils ──────────────────────────────────────────────────────────────
# database.py 用到：from dbutils.pooled_db import PooledDB
# 打桩目的：避免因缺少 dbutils 包而 ImportError
if "dbutils" not in sys.modules:
    _dbutils = MagicMock()                       # 伪造 dbutils 包对象
    _pooled = MagicMock()                        # 伪造 dbutils.pooled_db 子模块
    _pooled.PooledDB = MagicMock()               # 伪造 PooledDB 类（连接池）
    sys.modules["dbutils"] = _dbutils
    sys.modules["dbutils.pooled_db"] = _pooled   # 注册子模块路径
