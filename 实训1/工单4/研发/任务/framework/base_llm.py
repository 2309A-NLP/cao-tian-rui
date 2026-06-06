"""
base_llm.py — LLM 提供商抽象层

核心思想：
  策略模式 + 抽象工厂模式。
  无论底层是 OpenAI、Ollama、Claude 还是自定义模型，
  对外暴露统一的 ask() / ask_stream() / count_tokens() 接口。

复用方式：
  1. 继承 BaseLLMProvider，实现 ask() 和 count_tokens()
  2. 用 LLMFactory 注册你的 Provider 类（装饰器）
  3. 调用 LLMFactory.create("openai", **kwargs) 获取实例
  4. 业务代码只依赖 BaseLLMProvider，永远不知道具体实现
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterator


# ──────────────────────────────────────────────
# 抽象基类
# ──────────────────────────────────────────────

class BaseLLMProvider(ABC):
    """
    LLM 提供商抽象基类。
    
    所有 LLM 调用都走这三个接口：
      ask(text)        — 同步非流式，返回完整文本
      ask_stream(text) — 流式，按 token 逐个 yield
      count_tokens()   — 返回模型的最大上下文长度（不是实际用量）
    """

    def __init__(self, model: str = "default", **kwargs):
        self.model = model
        self._extra = kwargs  # 额外的配置参数（api_key, base_url 等）

    @abstractmethod
    def ask(self, text: str, system: Optional[str] = None) -> str:
        """
        向 LLM 发送消息并获取完整回复。
        
        参数:
            text: 用户消息 / prompt
            system: 系统提示词（可选）
        
        返回:
            LLM 的完整回复文本
        """
        ...

    def ask_stream(self, text: str, system: Optional[str] = None) -> Iterator[str]:
        """
        流式调用 LLM，按 token 逐个 yield。
        
        默认实现为调用 ask() 后一次 yield 全部。
        支持流式的子类请覆盖此方法实现真正的 SSE 流式输出。
        """
        yield self.ask(text, system)

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """估算文本的 token 数量（近似值也可）"""
        ...

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """该模型的最大上下文长度（输入 + 输出）"""
        ...

    # ── 便捷方法 ──

    def truncate(self, text: str, limit: Optional[int] = None) -> str:
        """
        将文本截断到指定 token 数以内。
        limit 为 None 时使用模型最大上下文长度（建议保留一半给输出）。
        """
        limit = limit or (self.max_tokens // 2)
        while self.count_tokens(text) > limit:
            text = text[:int(len(text) * 0.9)]
        return text

    def ask_with_history(
        self,
        messages: list[dict],
        system: Optional[str] = None,
    ) -> str:
        """
        带对话历史的调用（默认使用简单拼接）。
        
        参数:
            messages: [{"role": "user"/"assistant", "content": "..."}, ...]
            system: 系统提示词
        
        返回:
            LLM 的回复
        """
        # 默认实现：拼成纯文本
        parts = []
        if system:
            parts.append(f"[系统]: {system}")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"[用户]: {content}")
            elif role == "assistant":
                parts.append(f"[助手]: {content}")
            else:
                parts.append(f"[{role}]: {content}")
        prompt = "\n\n".join(parts)
        return self.ask(prompt)


# ──────────────────────────────────────────────
# 异常定义
# ──────────────────────────────────────────────

class LLMError(Exception):
    """LLM 调用异常"""
    pass


class LLMConfigError(LLMError):
    """LLM 配置缺失或错误"""
    pass


class LLMRateLimitError(LLMError):
    """触发限流"""
    pass


# ──────────────────────────────────────────────
# 工厂（抽象工厂模式）
# ──────────────────────────────────────────────

class LLMFactory:
    """
    LLM 工厂 —— 注册 + 创建 Provider。
    
    用法:
        @LLMFactory.register("my_llm")
        class MyLLM(BaseLLMProvider):
            ...
        
        provider = LLMFactory.create("my_llm", model="gpt-4", api_key="...")
    """

    _registry: dict[str, type[BaseLLMProvider]] = {}

    @classmethod
    def register(cls, name: str):
        """
        装饰器：将 Provider 类注册到工厂。
        
        用法:
            @LLMFactory.register("openai")
            class OpenAIProvider(BaseLLMProvider):
                ...
        """
        def wrapper(provider_cls: type[BaseLLMProvider]):
            cls._registry[name] = provider_cls
            return provider_cls
        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseLLMProvider:
        """
        创建 Provider 实例。
        
        参数:
            name: 注册时的名称（如 "openai"、"ollama"）
            kwargs: 传入 Provider 构造函数的参数
        
        返回:
            BaseLLMProvider 实例
        
        异常:
            LLMConfigError: 未注册的名称
        """
        if name not in cls._registry:
            raise LLMConfigError(
                f"未知的 LLM 提供商: {name}。"
                f" 已注册: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供商名称"""
        return list(cls._registry.keys())
