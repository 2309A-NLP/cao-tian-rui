"""
测试：llm_provider.py LLM 提供商模块
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from llm_provider import (
    MockProvider, OpenAICompatibleProvider, LLMFactory,
)


class TestMockProvider(unittest.TestCase):
    """测试 MockProvider"""

    def setUp(self):
        self.provider = MockProvider()

    def test_name(self):
        """测试返回正确的名称"""
        self.assertEqual(self.provider.name, "mock")

    def test_ask_returns_string(self):
        """测试 ask 返回非空字符串"""
        result = self.provider.ask("测试问题", "测试系统提示")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_ask_stream_yields_content(self):
        """测试 ask_stream 返回生成器"""
        tokens = list(self.provider.ask_stream("测试问题", "测试系统提示"))
        self.assertTrue(len(tokens) > 0)
        self.assertIsInstance(tokens[0], str)


class TestOpenAICompatibleProvider(unittest.TestCase):
    """测试 OpenAICompatibleProvider（使用 mock 网络请求）"""

    def setUp(self):
        self.provider = OpenAICompatibleProvider(
            api_key="sk-test-key",
            base_url="https://api.test.com/v1",
            model="test-model",
        )

    def test_name_contains_model(self):
        """测试 name 包含模型名"""
        self.assertIn("test-model", self.provider.name)

    def test_init_empty_key_raises(self):
        """测试 key 为空时应抛出异常"""
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider(api_key="")

    def test_ask_success(self):
        """测试 ask 成功返回响应（mock requests.post 全局）"""
        import requests
        original_post = requests.post
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "这是测试回答"}}]
            }
            requests.post = MagicMock(return_value=mock_response)

            result = self.provider.ask("你好", "你是一个助手")
            self.assertEqual(result, "这是测试回答")
            requests.post.assert_called_once()
        finally:
            requests.post = original_post

    def test_ask_stream_success(self):
        """测试 ask_stream 成功流式返回（mock requests.post 全局）"""
        import requests
        original_post = requests.post
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = [
                'data: {"choices":[{"delta":{"content":"你好"}}]}',
                'data: {"choices":[{"delta":{"content":"世界"}}]}',
                "data: [DONE]",
            ]
            requests.post = MagicMock(return_value=mock_response)

            tokens = list(self.provider.ask_stream("你好"))
            self.assertEqual(tokens, ["你好", "世界"])
        finally:
            requests.post = original_post

    def test_ask_network_error_returns_error_message(self):
        """测试网络错误时返回错误信息而非抛出"""
        import requests
        original_post = requests.post
        try:
            requests.post = MagicMock(side_effect=requests.exceptions.ConnectionError("连接失败"))

            result = self.provider.ask("你好")
            self.assertTrue(result.startswith("【错误】"))
        finally:
            requests.post = original_post

    def test_session_and_system_prompt(self):
        """测试 system_prompt 被正确传递"""
        import requests
        original_post = requests.post
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            requests.post = MagicMock(return_value=mock_response)

            self.provider.ask("问题", "你是一个法律顾问")

            call_kwargs = requests.post.call_args[1]
            payload = call_kwargs["json"]
            self.assertEqual(len(payload["messages"]), 2)
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertEqual(payload["messages"][0]["content"], "你是一个法律顾问")
            self.assertEqual(payload["messages"][1]["role"], "user")
        finally:
            requests.post = original_post


class TestLLMFactory(unittest.TestCase):
    """测试 LLMFactory"""

    def test_create_mock_without_key(self):
        """测试无 API key 时返回 MockProvider"""
        provider = LLMFactory.create(provider="openai", api_key="")
        self.assertIsInstance(provider, MockProvider)

    def test_create_mock_explicit(self):
        """测试显式指定 mock"""
        provider = LLMFactory.create(provider="mock")
        self.assertIsInstance(provider, MockProvider)

    def test_create_deepseek_with_key(self):
        """测试 deepseek 提供商"""
        provider = LLMFactory.create(
            provider="deepseek",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertIn("deepseek-chat", provider.name)

    def test_create_oneapi_with_key(self):
        """测试 oneapi 提供商"""
        provider = LLMFactory.create(
            provider="oneapi",
            api_key="sk-test",
            base_url="http://127.0.0.1:3000/v1",
            model="gpt-4o",
        )
        self.assertIsInstance(provider, OpenAICompatibleProvider)

    def test_create_unknown_provider_returns_mock(self):
        """测试未知提供商返回 MockProvider"""
        provider = LLMFactory.create(provider="nonexistent_provider")
        self.assertIsInstance(provider, MockProvider)


if __name__ == "__main__":
    unittest.main()
