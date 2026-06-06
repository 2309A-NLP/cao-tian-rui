"""
测试：config.py 配置管理模块
"""
import os
import sys
import json
import tempfile
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import AppConfig


class TestAppConfig(unittest.TestCase):
    """测试 AppConfig 配置管理"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="rag_test_config_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_default_config(self):
        """测试默认配置的创建"""
        config = AppConfig()
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.db_port, 3307)
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.embedding_device, "auto")
        self.assertEqual(config.milvus_dim, 1024)
        self.assertFalse(config.embedding_model_path == "")
        # project_root 应自动填充
        self.assertTrue(len(config.project_root) > 0)

    def test_save_and_load(self):
        """测试配置的保存和加载"""
        config_path = os.path.join(self.tmp_dir, "config.json")
        config = AppConfig()
        config.chunk_size = 200
        config.top_k = 3
        config.llm_provider = "mock"
        config.log_level = "INFO"
        config.save(config_path)

        # 重新加载
        loaded = AppConfig.load(config_path)
        self.assertEqual(loaded.chunk_size, 200)
        self.assertEqual(loaded.top_k, 3)
        self.assertEqual(loaded.llm_provider, "mock")
        self.assertEqual(loaded.log_level, "INFO")

    def test_load_nonexistent_returns_default(self):
        """测试加载不存在的配置文件返回默认配置"""
        fake_path = os.path.join(self.tmp_dir, "nonexistent.json")
        config = AppConfig.load(fake_path)
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.chunk_size, 512)

    def test_load_partial_config(self):
        """测试只保存部分字段时能正确加载"""
        config_path = os.path.join(self.tmp_dir, "partial.json")
        partial = {"chunk_size": 128, "top_k": 3}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(partial, f)
        config = AppConfig.load(config_path)
        self.assertEqual(config.chunk_size, 128)
        self.assertEqual(config.top_k, 3)
        # 未保存的字段应使用默认值
        self.assertTrue(len(config.embedding_model_path) > 0)

    def test_to_dict_includes_all_fields(self):
        """测试 to_dict 包含所有 dataclass 字段"""
        config = AppConfig()
        d = config.to_dict()
        self.assertIn("chunk_size", d)
        self.assertIn("top_k", d)
        self.assertIn("llm_provider", d)
        self.assertIn("embedding_model_path", d)
        self.assertIn("project_root", d)
        self.assertIn("output_dir", d)

    def test_environment_override(self):
        """测试环境变量覆盖配置"""
        # env_map 中存在的变量才测试
        old_db_port = os.environ.pop("APP_DB_PORT", None)
        old_log_level = os.environ.pop("APP_LOG_LEVEL", None)

        try:
            os.environ["APP_DB_PORT"] = "9999"
            os.environ["APP_LOG_LEVEL"] = "WARNING"

            config = AppConfig()
            self.assertEqual(config.db_port, 9999)
            self.assertEqual(config.log_level, "WARNING")
        finally:
            if old_db_port is not None:
                os.environ["APP_DB_PORT"] = old_db_port
            else:
                os.environ.pop("APP_DB_PORT", None)
            if old_log_level is not None:
                os.environ["APP_LOG_LEVEL"] = old_log_level
            else:
                os.environ.pop("APP_LOG_LEVEL", None)

    def test_path_resolution(self):
        """测试相对路径解析为绝对路径"""
        config = AppConfig()
        config.project_root = self.tmp_dir
        config._resolve_paths()
        self.assertTrue(os.path.isabs(config.output_dir))
        self.assertTrue(os.path.isabs(config.log_dir))
        self.assertTrue(os.path.isabs(config.knowledge_base_dir))

    def test_knowledge_base_default(self):
        """测试知识库目录默认值"""
        self.assertEqual(AppConfig.knowledge_base_dir, "knowledge_base")
        # 实例化后 __post_init__ 调用了 _resolve_paths，所以实例属性变成了绝对路径
        config = AppConfig()
        self.assertTrue(config.knowledge_base_dir.endswith("knowledge_base"))


if __name__ == "__main__":
    unittest.main()
