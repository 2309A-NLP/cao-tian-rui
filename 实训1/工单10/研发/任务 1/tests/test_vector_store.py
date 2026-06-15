"""
测试：vector_store.py 向量存储模块
测试 BGE-M3 嵌入模型封装 + VectorStore（Mock Milvus 连接）
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vector_store import BGEM3Embedding, VectorStore


class TestBGEM3Embedding(unittest.TestCase):
    """测试 BGEM3Embedding 类（不加载真实模型）"""

    def setUp(self):
        self.emb = BGEM3Embedding(
            model_path="/fake/path",
            device="cpu",
            max_length=8192,
        )

    def test_init_defaults(self):
        """测试初始化默认值"""
        self.assertEqual(self.emb.model_path, "/fake/path")
        self.assertEqual(self.emb.device, "cpu")
        self.assertEqual(self.emb.max_length, 8192)
        self.assertEqual(self.emb.dimension, 1024)
        self.assertTrue(self.emb.normalize)
        self.assertEqual(self.emb._actual_device, "cpu")
        self.assertIsNone(self.emb.tokenizer)
        self.assertIsNone(self.emb.model)

    def test_actual_device_property(self):
        """测试 actual_device 属性"""
        self.assertEqual(self.emb.actual_device, "cpu")

    def test_load_nonexistent_path(self):
        """测试加载不存在的路径"""
        result = self.emb.load()
        self.assertFalse(result)

    def test_encode_query_instructions(self):
        """测试 encode_query 添加 instruction 前缀"""
        query = "军用领域收入占比"
        prefixed = f"为这个句子生成表示以用于检索相关文章：{query}"
        self.assertIn(query, prefixed)
        self.assertIn("为这个句子生成表示以用于检索相关文章", prefixed)


class TestVectorStore(unittest.TestCase):
    """测试 VectorStore（Mock 外部依赖）"""

    def setUp(self):
        self.vs = VectorStore(
            model_path="/fake/path",
            milvus_host="127.0.0.1",
            milvus_port=19530,
            collection_name="test_collection",
            device="cpu",
        )

    def test_init_defaults(self):
        """测试初始化默认值"""
        self.assertEqual(self.vs.model_path, "/fake/path")
        self.assertEqual(self.vs.milvus_host, "127.0.0.1")
        self.assertEqual(self.vs.milvus_port, 19530)
        self.assertEqual(self.vs.collection_name, "test_collection")
        self.assertEqual(self.vs.dimension, 1024)
        self.assertFalse(self.vs.connected)
        self.assertIsNone(self.vs.model)
        self.assertIsNone(self.vs.collection)

    def test_load_model_nonexistent_path(self):
        """测试加载不存在的模型路径"""
        result = self.vs.load_model()
        self.assertFalse(result)

    @patch("vector_store.connections")
    @patch("vector_store.utility")
    def test_connect_milvus_creates_collection(self, mock_utility, mock_connections):
        """测试连接 Milvus 时自动创建集合"""
        mock_utility.has_collection.return_value = False

        with patch.object(self.vs, "_create_collection") as mock_create:
            result = self.vs.connect_milvus()
            self.assertTrue(result)
            self.assertTrue(self.vs.connected)
            mock_create.assert_called_once()

    @patch("vector_store.connections")
    @patch("vector_store.utility")
    def test_connect_milvus_existing(self, mock_utility, mock_connections):
        """测试连接已有集合"""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        with patch("vector_store.Collection", return_value=mock_collection):
            result = self.vs.connect_milvus()
            self.assertTrue(result)
            self.assertTrue(self.vs.connected)

    @patch("vector_store.connections")
    @patch("vector_store.utility")
    def test_connect_milvus_failure(self, mock_utility, mock_connections):
        """测试连接失败返回 False"""
        from pymilvus import exceptions
        mock_connections.connect.side_effect = Exception("连接超时")
        result = self.vs.connect_milvus()
        self.assertFalse(result)
        self.assertFalse(self.vs.connected)

    def test_document_exists_not_connected(self):
        """测试未连接时 document_exists 返回 False"""
        result = self.vs.document_exists("test.pdf")
        self.assertFalse(result)

    def test_search_not_connected(self):
        """测试未连接时 search 返回空列表"""
        result = self.vs.search("test query")
        self.assertEqual(result, [])

    def test_count_not_connected(self):
        """测试未连接时 count 返回 0"""
        count = self.vs.count()
        self.assertEqual(count, 0)

    def test_list_documents_not_connected(self):
        """测试未连接时 list_documents 返回空列表"""
        docs = self.vs.list_documents()
        self.assertEqual(docs, [])

    @patch("vector_store.connections")
    @patch("vector_store.utility")
    def test_drop_collection_not_connected_auto_connects(
        self, mock_utility, mock_connections
    ):
        """测试未连接时 drop_collection 自动连接"""
        self.vs.connected = False
        mock_utility.has_collection.return_value = True
        self.vs.drop_collection()
        # 应自动连接
        mock_connections.connect.assert_called_once()

    @patch("vector_store.connections")
    @patch("vector_store.utility")
    def test_drop_collection_nonexistent(self, mock_utility, mock_connections):
        """测试删除不存在的集合"""
        self.vs.connected = True
        mock_utility.has_collection.return_value = False
        self.vs.drop_collection()
        # 不应调用 drop_collection
        mock_utility.drop_collection.assert_not_called()

    def test_context_manager(self):
        """测试上下文管理器断开连接"""
        self.vs.connected = True
        with patch.object(self.vs, "disconnect") as mock_disconnect:
            with self.vs:
                pass
            mock_disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
