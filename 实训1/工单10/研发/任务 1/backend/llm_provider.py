"""
LLM 提供商层
支持 mock / openai / ollama 三种模式，可扩展
"""
import time
import json
import hashlib
from typing import Optional, Generator
from abc import ABC, abstractmethod

from logger import get_logger, log_exception

logger = get_logger("llm")


class BaseLLMProvider(ABC):
    """LLM 抽象基类"""
    
    @abstractmethod
    def ask(self, prompt: str, system_prompt: str = "") -> str:
        """发送提示并获取完整响应"""
        pass

    def chat(self, prompt: str, system_prompt: str = "",
             session=None, model: str = None) -> str:
        """兼容 rag_engine.py 的 chat 接口"""
        return self.ask(prompt, system_prompt=system_prompt)

    @abstractmethod
    def ask_stream(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        """流式响应生成器"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class MockProvider(BaseLLMProvider):
    """Mock 模式 - 不依赖任何 API"""
    
    @property
    def name(self) -> str:
        return "mock"
    
    def ask(self, prompt: str, system_prompt: str = "") -> str:
        logger.debug(f"[Mock] 收到提示 (长度={len(prompt)})")
        return "这是一个模拟回答。当前系统处于 Mock 模式，没有连接真实 LLM API。请配置 config.json 中的 llm_api_key 来启用真实模型。"
    
    def ask_stream(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        yield "这是一个模拟回答。当前系统处于 Mock 模式，没有连接真实 LLM API。请配置 config.json 中的 llm_api_key 来启用真实模型。"


class OpenAICompatibleProvider(BaseLLMProvider):
    """兼容 OpenAI API 的 LLM 提供商"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", 
                 model: str = "gpt-4o", timeout: int = 60):
        if not api_key:
            raise ValueError("OpenAI API Key 不能为空")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return f"openai({self.model})"
    
    def _call_api(self, messages: list[dict], stream: bool = False) -> dict:
        """调用 OpenAI 兼容 API"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": stream,
        }
        
        url = f"{self.base_url}/chat/completions"
        
        try:
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=self.timeout, stream=stream,
            )
            resp.raise_for_status()
            
            if stream:
                return resp  # 返回原始响应对象用于流式解析
            return resp.json()
        except requests.exceptions.Timeout:
            logger.error(f"LLM API 请求超时 ({self.timeout}s)")
            raise
        except requests.exceptions.RequestException as e:
            log_exception(logger, f"LLM API 请求失败", e)
            raise
    
    def ask(self, prompt: str, system_prompt: str = "", max_tokens: Optional[int] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            payload = self._call_api(messages, stream=False)
            # 如果传了 max_tokens，在调用后根据返回内容截断
            content = payload["choices"][0]["message"]["content"]
            logger.debug(f"LLM 响应成功 (输出长度={len(content)})")
            return content
        except Exception as e:
            log_exception(logger, "LLM 调用失败", e)
            return f"【错误】LLM 调用失败: {str(e)}"
    
    def ask_stream(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._call_api(messages, stream=True)
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if choices := data.get("choices"):
                                if delta := choices[0].get("delta", {}):
                                    if content := delta.get("content"):
                                        yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            log_exception(logger, "LLM 流式调用失败", e)
            yield f"【错误】LLM 调用失败: {str(e)}"


class LLMFactory:
    """LLM 工厂"""
    
    @staticmethod
    def create(provider: str = "mock", api_key: str = "", 
               base_url: str = "", model: str = "mock") -> BaseLLMProvider:
        """创建 LLM 实例，支持多种提供商（deepseek, qwen3, openai 等）"""
        if provider == "mock" or not api_key:
            logger.info("使用 Mock LLM 提供商")
            return MockProvider()
        elif provider in ("openai", "oneapi", "deepseek", "siliconflow"):
            if not base_url:
                base_urls = {
                    "openai": "https://api.openai.com/v1",
                    "deepseek": "https://api.deepseek.com/v1",
                    "oneapi": "http://127.0.0.1:3000/v1",
                    "siliconflow": "https://api.siliconflow.cn/v1",
                }
                # Qwen3 默认配置
                qwen3_urls = {
                    "qwen3": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                }
                base_url = base_urls.get(provider, base_url)
                base_url = qwen3_urls.get(provider, base_url)
                
                base_url = base_urls.get(provider, "https://api.openai.com/v1")
            if not model:
                model = provider if provider != "openai" else "gpt-4o"
            logger.info(f"使用 LLM: {provider} / {model} / {base_url}")
            return OpenAICompatibleProvider(api_key, base_url, model)
        elif provider == "ollama":
            base_url = base_url or "http://127.0.0.1:11434"
            model = model or "qwen2:7b"
            from llm_provider_ollama import OllamaProvider
            return OllamaProvider(base_url, model)
        else:
            logger.warning(f"未知 LLM 提供商：{provider}，使用 Mock")
            return MockProvider()
        logger.info(f"使用 LLM: {provider} / {model} / {base_url}")
        return OpenAICompatibleProvider(api_key, base_url, model)
