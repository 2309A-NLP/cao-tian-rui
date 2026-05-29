# -*- coding: utf-8 -*-
"""
单元测试 - RAG角色扮演系统（适配新架构）
运行方式: python -m unittest test_rag_system.py -v

⚠️ 常改动的地方：
1. 新增功能模块后，应添加对应的测试类和测试方法
2. 修改现有函数签名或行为时，需同步更新相关测试用例
3. 测试数据（如法律问题、文本内容）可根据业务变化调整
4. 期望值（如关键词数量、长度限制）可根据实际修改

⚠️ 注意事项：
1. 测试类继承 unittest.TestCase，每个测试方法以 test_ 开头
2. 部分测试使用了 Mock 和 MagicMock（当前未完全使用，可扩展）
3. 涉及外部依赖（如数据库、LLM）的测试应使用 Mock 避免实际调用
4. 运行全部测试：python -m unittest test_rag_system.py -v
5. 运行单个测试类：python -m unittest test_rag_system.TestDataCleaner
"""

import unittest
import sys
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# 添加项目路径，确保可以从项目根目录导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestDataCleaner(unittest.TestCase):
    """数据清洗器测试（processor.data_processor.DataCleaner）"""

    def setUp(self):
        """每个测试方法运行前的初始化"""
        from processor.data_processor import DataCleaner
        self.cleaner = DataCleaner()

    def test_clean_text_removes_special_chars(self):
        """测试：移除特殊字符（如 @#$%^&*）"""
        text = "你好！@#$%^&*()这是测试。"
        cleaned = self.cleaner.clean_text(text)
        self.assertNotIn("@", cleaned)
        self.assertIn("你好", cleaned)

    def test_clean_text_removes_extra_spaces(self):
        """测试：移除多余空格（多个连续空格变一个）"""
        text = "你好    这是    测试   文本"
        cleaned = self.cleaner.clean_text(text)
        self.assertNotIn("  ", cleaned)

    def test_clean_text_handles_empty(self):
        """测试：处理空文本，应返回空字符串"""
        cleaned = self.cleaner.clean_text("")
        self.assertEqual(cleaned, "")

    def test_filter_valid_texts(self):
        """测试：过滤短文本（长度小于 min_length 的会被丢弃）"""
        texts = ["这是一个足够长的测试文本", "短", "这也是一个有效的测试内容"]
        filtered = self.cleaner.filter_valid_texts(texts, min_length=10)
        self.assertEqual(len(filtered), 2)


class TestQueryRewriter(unittest.TestCase):
    """查询改写器测试（core.query_rewriter.QueryRewriter）"""

    def setUp(self):
        from core.query_rewriter import QueryRewriter
        self.rewriter = QueryRewriter()

    def test_rewrite_query(self):
        """测试：查询标准化（口语替换）"""
        query = "捡到别人的失物怎么办"
        result = self.rewriter.rewrite(query)
        self.assertIsNotNone(result)

    def test_expand_query(self):
        """测试：查询扩写（同义词替换生成变体）"""
        query = "离婚"
        variants = self.rewriter.expand(query)
        self.assertIsInstance(variants, list)
        self.assertGreaterEqual(len(variants), 1)

    def test_optimize_returns_dict(self):
        """测试：优化返回字典结构，包含 original、rewritten、variants、keywords"""
        query = "捡到失物"
        result = self.rewriter.optimize(query)
        self.assertIsInstance(result, dict)
        self.assertIn("original", result)
        self.assertIn("variants", result)


class TestPromptManager(unittest.TestCase):
    """提示词管理器测试（core.prompt_manager.PromptManager）"""

    def setUp(self):
        from core.prompt_manager import prompt_manager
        self.pm = prompt_manager

    def test_get_role_prompt(self):
        """测试：获取各角色提示词，应非空"""
        roles = ["friend", "doctor", "lawyer", "psychologist", "tcm", "finance"]
        for role in roles:
            prompt = self.pm.get_role_prompt(role)
            self.assertIsNotNone(prompt)
            self.assertNotEqual(prompt, "")

    def test_lawyer_prompt_has_rules(self):
        """测试：律师提示词中包含【重要规则】部分"""
        prompt = self.pm.get_role_prompt("lawyer")
        self.assertIn("规则", prompt)

    def test_is_law_question(self):
        """测试：法律问题检测功能，应返回布尔值"""
        law_questions = ["离婚需要什么条件？", "捡到别人的失物怎么办？", "民法典对合同有什么规定？"]
        for q in law_questions:
            result = self.pm.is_law_question(q)
            self.assertIsInstance(result, bool)

    def test_build_law_prompt(self):
        """测试：构建法律专用提示词，有/无知识库两种场景"""
        from core.prompt_manager import prompt_manager

        role_prompt = "你是律师"
        context = "民法典第三百一十四条：拾得遗失物应当返还权利人"
        long_memories = []
        history_text = ""
        message = "捡到失物怎么办"

        # 有知识库的情况
        prompt = prompt_manager.build_law_prompt(
            role_prompt, context, long_memories, history_text, message, has_knowledge=True
        )
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)

        # 无知识库的情况
        prompt2 = prompt_manager.build_law_prompt(
            role_prompt, "", long_memories, history_text, message, has_knowledge=False
        )
        self.assertIsInstance(prompt2, str)


class TestHelpers(unittest.TestCase):
    """辅助函数测试（utils.helpers）"""

    def test_hash_password(self):
        """测试：密码哈希一致性（SHA256）"""
        from utils.helpers import hash_password
        password = "test123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)

    def test_extract_keywords(self):
        """测试：中文关键词提取，返回列表"""
        from utils.helpers import extract_keywords
        text = "离婚需要满足夫妻双方自愿的条件"
        keywords = extract_keywords(text, max_words=3)
        self.assertIsInstance(keywords, list)


class TestStringMethods(unittest.TestCase):
    """基础字符串示例测试（占位/验证 unittest 工作）"""

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])


def run_tests():
    """运行所有测试类，输出测试报告"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestStringMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCleaner))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryRewriter))
    suite.addTests(loader.loadTestsFromTestCase(TestPromptManager))
    suite.addTests(loader.loadTestsFromTestCase(TestHelpers))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("单元测试总结")
    print("=" * 60)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    return result


if __name__ == "__main__":
    run_tests()