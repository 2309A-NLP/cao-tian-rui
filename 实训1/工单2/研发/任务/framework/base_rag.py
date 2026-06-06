"""
base_rag.py — RAG 引擎基类

核心思想：
  模板方法模式。定义 RAG 的固定三步流程：
    1. _build_context(query) — 检索相关文档
    2. _build_prompt(query, context) — 构造 prompt
    3. generate(query) — 组合以上两步，调用 LLM 生成答案
  
  子类只需重写 _build_context 和 _build_prompt，
  不需要关心 generate 的整体控制流程。

复用方式：
  1. 继承 BaseRAGEngine
  2. 实现 _build_context() 和 _build_prompt()
  3. 调用 engine.generate("你的问题")
  4. 对话管理由 ChatSession 处理
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Iterator


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class RAGResult:
    """
    RAG 生成结果。
    
    属性:
        answer: 最终生成的回答
        sources: 检索到的来源文档列表
        context: 构建的上下文文本（调试用）
        prompt: 发送给 LLM 的完整 prompt（调试用）
    """
    answer: str
    sources: list[dict] = field(default_factory=list)
    context: str = ""
    prompt: str = ""

    def __str__(self) -> str:
        return self.answer


@dataclass
class ChatMessage:
    """单条对话消息"""
    role: str       # "user" | "assistant" | "system"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# ──────────────────────────────────────────────
# RAG 引擎基类
# ──────────────────────────────────────────────

class BaseRAGEngine(ABC):
    """
    RAG 引擎抽象基类。
    
    模板方法三步：
      1. _build_context(query)  — 检索相关文档
      2. _build_prompt(query, context) — 组装 prompt
      3. generate(query) — 调用 LLM
    
    子类必须实现:
      _build_context() — 返回检索到的文档片段列表
      _build_prompt()  — 返回发给 LLM 的完整 prompt
    
    可选覆写:
      generate()  — 如果你想改变完整的生成流程
    """

    # ── 需要子类实现的模板步骤 ──

    @abstractmethod
    def _build_context(self, query: str, top_k: int = 5) -> list[dict]:
        """
        根据查询检索相关文档上下文。
        
        参数:
            query: 用户问题
            top_k: 检索数量
        
        返回:
            [{"text": "...", "score": 0.95, "source": "..."}, ...]
        """
        ...

    @abstractmethod
    def _build_prompt(self, query: str, context: list[dict]) -> str:
        """
        构造发送给 LLM 的完整 prompt。
        
        参数:
            query: 用户问题
            context: _build_context 返回的文档列表
        
        返回:
            完整的 prompt 字符串
        """
        ...

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """
        实际调用 LLM 生成回复。
        
        参数:
            prompt: _build_prompt 构造好的完整 prompt
        
        返回:
            LLM 的回答文本
        """
        ...

    # ── 对外核心方法 ──

    def generate(self, query: str, top_k: int = 5) -> RAGResult:
        """
        完整的 RAG 生成流程。
        
        这是模板方法，子类一般不需要覆盖。
        
        参数:
            query: 用户问题
            top_k: 检索文档数
        
        返回:
            RAGResult 包含回答、来源、上下文
        """
        context = self._build_context(query, top_k=top_k)
        prompt = self._build_prompt(query, context)
        answer = self._call_llm(prompt)
        return RAGResult(
            answer=answer,
            sources=[{"text": c["text"], "score": c.get("score", 0)} for c in context],
            context="\n---\n".join(c["text"] for c in context),
            prompt=prompt,
        )

    def generate_stream(
        self,
        query: str,
        top_k: int = 5,
    ) -> Iterator[RAGResult | str]:
        """
        流式 RAG 生成。
        
        先 yield RAGResult（包含 context 和 prompt，answer 为空），
        再逐个 yield answer 的 token。
        
        子类可覆盖以实现真正的流式 LLM 调用。
        """
        context = self._build_context(query, top_k=top_k)
        prompt = self._build_prompt(query, context)

        # 先发送元数据
        yield RAGResult(
            answer="",
            sources=[{"text": c["text"], "score": c.get("score", 0)} for c in context],
            context="\n---\n".join(c["text"] for c in context),
            prompt=prompt,
        )

        # 再流式输出答案
        answer = self._call_llm(prompt)
        # 按句号分块模拟流式
        for sentence in answer.replace("。", "。\n").replace("！", "！\n").split("\n"):
            sentence = sentence.strip()
            if sentence:
                yield sentence + "\n"

    # ── 便捷方法 ──

    def ask(self, query: str, top_k: int = 5) -> str:
        """单纯的文字回答（不返回元数据）"""
        return self.generate(query, top_k=top_k).answer


# ──────────────────────────────────────────────
# 对话会话管理器
# ──────────────────────────────────────────────

class ChatSession:
    """
    对话会话管理器。
    维护多轮对话历史，自动截断过长的历史。
    
    用法:
        session = ChatSession()
        session.add_user("什么是 RAG？")
        session.add_assistant("RAG 是检索增强生成...")
        
        history = session.get_history()  # 返回消息列表
        context = session.to_text()      # 返回纯文本拼接
    
    可选与 RAG 引擎配合:
        engine = YourRAGEngine(...)
        session = ChatSession(system_prompt="你是一个 AI 助手")
        
        # 用户发消息
        user_msg = "解释一下注意力机制"
        session.add_user(user_msg)
        
        # 生成 RAG 回答
        result = engine.generate(user_msg)
        session.add_assistant(result.answer)
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_turns: int = 20,
    ):
        """
        参数:
            system_prompt: 系统提示词（如 "你是一个知识渊博的助手"）
            max_turns: 保留的最大轮数（超出则丢弃最早的历史）
        """
        self.messages: list[ChatMessage] = []
        self.max_turns = max_turns
        if system_prompt:
            self.messages.append(ChatMessage(role="system", content=system_prompt))

    def add_user(self, content: str) -> None:
        """添加用户消息"""
        self.messages.append(ChatMessage(role="user", content=content))
        self._trim()

    def add_assistant(self, content: str) -> None:
        """添加助手回复"""
        self.messages.append(ChatMessage(role="assistant", content=content))

    def add_system(self, content: str) -> None:
        """添加或替换系统提示词"""
        # 如果已有 system 消息，替换它
        for msg in self.messages:
            if msg.role == "system":
                msg.content = content
                return
        self.messages.insert(0, ChatMessage(role="system", content=content))

    def get_history(self, include_system: bool = True) -> list[dict]:
        """
        获取对话历史（用于 LLM API 调用）。
        
        返回: [{"role": "user", "content": "..."}, ...]
        """
        msgs = self.messages if include_system else [
            m for m in self.messages if m.role != "system"
        ]
        return [m.to_dict() for m in msgs]

    def to_text(self, separator: str = "\n\n") -> str:
        """将对话历史拼成纯文本"""
        return separator.join(
            f"[{m.role}]: {m.content}" for m in self.messages
        )

    def clear(self) -> None:
        """清空历史（保留 system prompt）"""
        system_msgs = [m for m in self.messages if m.role == "system"]
        self.messages = system_msgs

    @property
    def turn_count(self) -> int:
        """当前对话轮数（不计 system）"""
        return len([m for m in self.messages if m.role != "system"])

    def _trim(self) -> None:
        """超出 max_turns 时丢弃最旧的消息（保留 system）"""
        non_system = [m for m in self.messages if m.role != "system"]
        if len(non_system) > self.max_turns * 2:  # user + assistant = 1 turn
            excess = len(non_system) - self.max_turns * 2
            removed = 0
            new_messages = []
            for m in self.messages:
                if m.role != "system" and removed < excess:
                    removed += 1
                    continue
                new_messages.append(m)
            self.messages = new_messages

    def last_user_message(self) -> Optional[str]:
        """获取最后一条用户消息"""
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return None

    def last_assistant_message(self) -> Optional[str]:
        """获取最后一条助手回复"""
        for m in reversed(self.messages):
            if m.role == "assistant":
                return m.content
        return None
