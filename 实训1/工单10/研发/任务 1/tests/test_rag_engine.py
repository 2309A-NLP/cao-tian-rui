"""
测试：rag_engine.py RAG 引擎模块
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rag_engine import ChatSession, RAGEngine


class TestChatSession(unittest.TestCase):
    """测试 ChatSession 对话管理器"""

    def setUp(self):
        self.session = ChatSession(user_id=1, session_id="test_session")

    def test_initial_state(self):
        """测试初始状态"""
        self.assertEqual(self.session.user_id, 1)
        self.assertEqual(self.session.session_id, "test_session")
        self.assertEqual(self.session.mode, "rag")
        self.assertEqual(len(self.session.history), 0)
        self.assertEqual(self.session.max_history, 20)

    def test_add_message(self):
        """测试添加消息"""
        self.session.add_message("user", "你好")
        self.session.add_message("assistant", "你好！有什么可以帮你的？")
        self.assertEqual(len(self.session.history), 2)
        self.assertEqual(self.session.history[0]["role"], "user")
        self.assertEqual(self.session.history[1]["role"], "assistant")

    def test_history_truncation(self):
        """测试历史消息裁剪"""
        session = ChatSession(user_id=1, max_history=3)
        for i in range(5):
            session.add_message("user", f"问题{i}")
            session.add_message("assistant", f"回答{i}")
        # 最多保留 3 条（3 对，但 max_history 是消息数，非轮数）
        self.assertLessEqual(len(session.history), 3)

    def test_set_mode(self):
        """测试模式切换"""
        self.assertEqual(self.session.mode, "rag")
        self.session.set_mode("direct")
        self.assertEqual(self.session.mode, "direct")
        self.session.set_mode("rag")
        self.assertEqual(self.session.mode, "rag")

    def test_set_invalid_mode(self):
        """测试设置无效模式"""
        self.session.set_mode("invalid_mode")
        self.assertEqual(self.session.mode, "rag")  # 应保持不变

    def test_clear_history(self):
        """测试清除历史"""
        self.session.add_message("user", "你好")
        self.session.clear()
        self.assertEqual(len(self.session.history), 0)

    def test_get_context_empty(self):
        """测试空历史时 get_context 返回空"""
        ctx = self.session.get_context()
        self.assertEqual(ctx, "")

    def test_get_context_with_history(self):
        """测试有历史时 get_context 返回格式化上下文"""
        self.session.add_message("user", "第一轮问题")
        self.session.add_message("assistant", "第一轮回答")
        self.session.add_message("user", "第二轮问题")
        ctx = self.session.get_context()
        self.assertIn("历史对话", ctx)
        self.assertIn("用户", ctx)
        self.assertIn("助手", ctx)

    def test_session_id_auto_generate(self):
        """测试 session_id 自动生成"""
        session = ChatSession(user_id=2)
        self.assertTrue(len(session.session_id) == 8)


class TestRAGEngine(unittest.TestCase):
    """测试 RAGEngine 核心逻辑"""

    def setUp(self):
        # 创建模拟的 vector_store 和 llm_provider
        self.mock_vec = MagicMock()
        self.mock_vec.search.return_value = [
            {"text": "报告期内，公司来自军用领域的收入占比分别为82.10%、97.31%和94.84%。",
             "page": 217, "chunk_id": 0, "score": 0.85},
            {"text": "武汉兴图新科电子股份有限公司是一家专注于视频指挥和视频监控领域的高新技术企业。",
             "page": 1, "chunk_id": 2, "score": 0.72},
        ]

        self.mock_llm = MagicMock()
        self.mock_llm.ask.return_value = "根据招股说明书，公司注册资本为5,325万元[第1页]。"
        self.mock_llm.ask_stream.return_value = iter(["根据", "招股", "说明书", "，", "注册", "资本", "为", "5,325", "万元", "[第1页]。"])

        self.engine = RAGEngine(
            vector_store=self.mock_vec,
            llm_provider=self.mock_llm,
            top_k=5,
            similarity_threshold=0.0,
        )
        self.session = ChatSession(user_id=1)

    def test_rag_answer_basic(self):
        """测试 RAG 模式基本流程"""
        self.session.set_mode("rag")
        result = self.engine.answer("公司的注册资本是多少？", self.session)

        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("retrieval_time_ms", result)
        self.assertIn("llm_time_ms", result)
        self.assertIn("total_time_ms", result)
        self.assertIn("mode", result)
        self.assertEqual(result["mode"], "rag")
        self.assertEqual(len(result["sources"]), 2)

    def test_rag_answer_direct_mode(self):
        """测试 direct 模式"""
        self.session.set_mode("direct")
        result = self.engine.answer("你好", self.session)

        self.assertEqual(result["mode"], "direct")
        self.assertEqual(len(result["sources"]), 0)
        self.mock_llm.ask.assert_called_once()

    def test_rag_answer_saves_history(self):
        """测试回答后保存到历史"""
        self.session.set_mode("rag")
        self.engine.answer("注册资本？", self.session)
        # 应该有 user 和 assistant 两条消息
        self.assertEqual(len(self.session.history), 2)
        self.assertEqual(self.session.history[0]["role"], "user")
        self.assertEqual(self.session.history[1]["role"], "assistant")

    def test_rag_answer_stream_rag_mode(self):
        """测试流式回答 RAG 模式"""
        self.session.set_mode("rag")
        events = list(self.engine.answer_stream("注册资本？", self.session))

        types = [e["type"] for e in events]
        self.assertIn("retrieval", types)
        self.assertIn("token", types)
        self.assertIn("done", types)

        # 检查是否包含检索事件
        retrieval_events = [e for e in events if e["type"] == "retrieval"]
        self.assertEqual(len(retrieval_events), 1)
        self.assertEqual(len(retrieval_events[0]["sources"]), 2)

    def test_rag_answer_stream_direct_mode(self):
        """测试流式回答 direct 模式"""
        self.session.set_mode("direct")
        events = list(self.engine.answer_stream("你好", self.session))

        types = [e["type"] for e in events]
        self.assertIn("retrieval", types)
        self.assertIn("token", types)
        self.assertIn("done", types)

        # direct 模式 sources 应为空
        retrieval_events = [e for e in events if e["type"] == "retrieval"]
        self.assertEqual(len(retrieval_events[0]["sources"]), 0)

    def test_answer_handles_error(self):
        """测试异常处理返回错误信息"""
        self.mock_llm.ask.side_effect = RuntimeError("LLM 调用失败")
        self.session.set_mode("direct")
        result = self.engine.answer("测试", self.session)
        self.assertIn("系统错误", result["answer"])

    def test_extract_question_json(self):
        """测试 JSON query 清洗"""
        query = '{"question": "注册资本是多少？"}'
        extracted = RAGEngine._extract_question(query)
        self.assertEqual(extracted, "注册资本是多少？")

    def test_extract_question_plain_text(self):
        """测试纯文本 query 不改变"""
        query = "注册资本是多少？"
        extracted = RAGEngine._extract_question(query)
        self.assertEqual(extracted, query)

    def test_build_context(self):
        """测试上下文构建"""
        sources = [
            {"text": "公司注册资本为5,325万元。", "page": 1, "score": 0.9},
            {"text": "军用领域收入占比82.10%。", "page": 217, "score": 0.8},
        ]
        context = self.engine._build_context(sources)
        self.assertIn("参考资料 1", context)
        self.assertIn("5,325", context)
        self.assertIn("第 217 页", context)

    def test_build_context_empty(self):
        """测试空 sources 构建上下文"""
        context = self.engine._build_context([])
        self.assertEqual(context, "")

    def test_build_context_removes_header(self):
        """测试上下文去掉已知页眉"""
        sources = [
            {"text": "武汉兴图新科电子股份有限公司\n公司注册资本为5,325万元。", "page": 1, "score": 0.9},
        ]
        context = self.engine._build_context(sources)
        # 应该去掉了页眉
        self.assertNotIn("武汉兴图新科电子股份有限公司\n", context)

    def test_rag_prompt_building(self):
        """测试 RAG prompt 构建"""
        prompt = self.engine._build_rag_prompt("注册资本？", "知识库内容...", self.session)
        self.assertIn("注册资本", prompt)
        self.assertIn("知识库", prompt)

    def test_direct_prompt_building(self):
        """测试 Direct prompt 构建"""
        self.session.add_message("user", "之前的对话")
        prompt = self.engine._build_direct_prompt("新问题", self.session)
        self.assertIn("新问题", prompt)
        self.assertIn("之前的对话", prompt)


if __name__ == "__main__":
    unittest.main()
