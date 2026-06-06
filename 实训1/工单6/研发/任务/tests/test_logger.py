"""
测试：logger.py 日志管理模块
"""
import os
import sys
import logging
import tempfile
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from logger import LoggerManager, get_logger, log_exception


class TestLoggerManager(unittest.TestCase):
    """测试 LoggerManager"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        # 清空实例缓存，确保测试隔离
        LoggerManager._instances.clear()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_logger_returns_logger(self):
        """测试 get_logger 返回 logging.Logger 实例"""
        logger = LoggerManager.get_logger("test_logger", log_dir=self.tmp_dir)
        self.assertIsInstance(logger, logging.Logger)

    def test_get_logger_same_name(self):
        """测试相同名称返回相同的 logger"""
        logger1 = LoggerManager.get_logger("same_logger", log_dir=self.tmp_dir)
        logger2 = LoggerManager.get_logger("same_logger", log_dir=self.tmp_dir)
        self.assertIs(logger1, logger2)

    def test_get_logger_different_names(self):
        """测试不同名称返回不同的 logger"""
        logger1 = LoggerManager.get_logger("logger_a", log_dir=self.tmp_dir)
        logger2 = LoggerManager.get_logger("logger_b", log_dir=self.tmp_dir)
        self.assertIsNot(logger1, logger2)

    def test_logger_creates_log_file(self):
        """测试日志文件被创建"""
        logger = LoggerManager.get_logger("file_test", log_dir=self.tmp_dir)
        logger.info("测试日志消息")

        # 检查日志文件是否存在
        log_file = os.path.join(self.tmp_dir, "file_test.log")
        self.assertTrue(os.path.exists(log_file))

    def test_logger_creates_error_log_file(self):
        """测试错误日志文件被创建"""
        logger = LoggerManager.get_logger("error_test", log_dir=self.tmp_dir)
        logger.error("测试错误消息")

        error_log = os.path.join(self.tmp_dir, "error_test_error.log")
        self.assertTrue(os.path.exists(error_log))

    def test_logger_writes_content(self):
        """测试日志内容写入文件"""
        logger = LoggerManager.get_logger("content_test", log_dir=self.tmp_dir)
        logger.info("这是一条测试日志")

        log_file = os.path.join(self.tmp_dir, "content_test.log")
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("测试日志", content)

    def test_error_log_separate_file(self):
        """测试错误日志写入独立的 error 文件"""
        logger = LoggerManager.get_logger("sep_test", log_dir=self.tmp_dir)
        logger.info("普通消息")
        logger.error("错误消息")

        main_log = os.path.join(self.tmp_dir, "sep_test.log")
        error_log = os.path.join(self.tmp_dir, "sep_test_error.log")

        with open(main_log, "r", encoding="utf-8") as f:
            main_content = f.read()
        with open(error_log, "r", encoding="utf-8") as f:
            error_content = f.read()

        # 普通日志应在 main 中
        self.assertIn("普通消息", main_content)
        # 错误日志应在 error 文件中
        self.assertIn("错误消息", error_content)

    def test_get_logger_shortcut(self):
        """测试快捷函数 get_logger"""
        LoggerManager._instances.clear()
        logger = get_logger("shortcut_test")
        self.assertIsInstance(logger, logging.Logger)

    def test_console_output(self):
        """测试控制台输出（默认开启）"""
        LoggerManager._instances.clear()
        logger = LoggerManager.get_logger(
            "console_test", log_dir=self.tmp_dir, console_output=True
        )
        # 应该有 3 个 handler: file, error_file, console
        self.assertEqual(len(logger.handlers), 3)

    def test_no_console_output(self):
        """测试关闭控制台输出"""
        LoggerManager._instances.clear()
        logger = LoggerManager.get_logger(
            "noconsole_test", log_dir=self.tmp_dir, console_output=False
        )
        # 应该有 2 个 handler: file, error_file
        self.assertEqual(len(logger.handlers), 2)


class TestLogException(unittest.TestCase):
    """测试 log_exception 函数"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        LoggerManager._instances.clear()
        self.logger = LoggerManager.get_logger("exc_test", log_dir=self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_log_exception_with_exception(self):
        """测试记录异常信息"""
        try:
            raise ValueError("测试异常")
        except ValueError as e:
            log_exception(self.logger, "自定义错误消息", e)

        log_file = os.path.join(self.tmp_dir, "exc_test_error.log")
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("自定义错误消息", content)
        self.assertIn("ValueError", content)
        self.assertIn("测试异常", content)

    def test_log_exception_without_exception(self):
        """测试无异常时的 log_exception（从 sys.exc_info 获取）"""
        log_exception(self.logger, "无异常测试")

        log_file = os.path.join(self.tmp_dir, "exc_test_error.log")
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("无异常测试", content)

    def test_log_exception_no_exc_info(self):
        """测试既无参数也无当前异常的情况"""
        # 确保当前没有异常上下文
        log_exception(self.logger, "无异常信息")

        log_file = os.path.join(self.tmp_dir, "exc_test_error.log")
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("无异常信息", content)

    def test_log_exception_chained_exceptions(self):
        """测试异常链记录"""
        try:
            try:
                raise ValueError("原始错误")
            except ValueError as cause:
                raise RuntimeError("包装错误") from cause
        except RuntimeError as e:
            log_exception(self.logger, "链式异常测试", e)

        log_file = os.path.join(self.tmp_dir, "exc_test_error.log")
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("链式异常测试", content)
        self.assertIn("RuntimeError", content)


if __name__ == "__main__":
    unittest.main()
