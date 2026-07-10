"""
LLMClient —— 硅基流动 Qwen 封装，单轮对话补全。

本模块封装了对硅基流动（SiliconFlow）平台上 Qwen 模型的调用。
硅基流动提供与 OpenAI API 兼容的接口，因此可直接使用 openai SDK，
只需将 base_url 指向硅基流动的 API 端点即可。
"""
# __future__.annotations：延迟类型注解求值
from __future__ import annotations

# logging：标准库，用于记录 LLM 调用耗时和 token 使用量
import logging
# os：标准库，读取环境变量（SILICONFLOW_* 系列配置）
import os
# time：标准库，高精度计时，统计 LLM 请求耗时
import time

# openai：第三方包（openai Python SDK），
# 官方用于调用 OpenAI API，但因硅基流动提供兼容接口，
# 可通过修改 base_url 和 api_key 复用此 SDK 调用硅基流动上的 Qwen 等模型
# OpenAI：SDK 主类，通过它创建 chat.completions.create() 调用
from openai import OpenAI

# 当前模块专属 logger，日志前缀 "agent22.llm"
logger = logging.getLogger("agent22.llm")


class LLMClient:
    """硅基流动 LLM 调用封装类。

    通过 OpenAI 兼容接口调用硅基流动平台上的 Qwen 模型，
    封装单轮 chat.completions 调用，隐藏底层 SDK 细节。
    """

    def __init__(self) -> None:
        """初始化 LLM 客户端。

        从环境变量读取模型名称、API Key 和 Base URL，
        构造 OpenAI 客户端实例（指向硅基流动 API 端点）。
        """
        # 从环境变量读取模型名称，默认使用 Qwen2.5-72B-Instruct（72B 参数版本，综合能力较强）
        self.model = os.getenv("SILICONFLOW_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

        # 创建 OpenAI 客户端，通过 base_url 指向硅基流动的 API 服务
        # api_key：硅基流动的 API 密钥（从环境变量 SILICONFLOW_API_KEY 读取）
        # base_url：硅基流动 OpenAI 兼容端点（默认 https://api.siliconflow.cn/v1）
        self.client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),                         # 硅基流动 API 密钥
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),  # API 端点
        )

    def chat(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        """执行单轮对话补全，返回模型生成的文本。

        使用标准的 chat.completions.create() 接口，构造 system + user 两条消息，
        调用 LLM 生成回复。适合单次问答场景（无多轮历史传递）。

        Args:
            system:      系统提示词（定义 LLM 角色和行为规则）。
            user:        用户侧输入（已注入历史记忆和本轮问题的完整提示）。
            temperature: 生成随机性，0.3 偏保守（降低幻觉风险），范围 [0, 2]。

        Returns:
            str: LLM 生成的回复文本（已去除首尾空白字符）。
        """
        t0 = time.perf_counter()  # 记录请求开始时间

        # client.chat.completions.create：OpenAI SDK 的核心调用方法
        # model：指定调用的模型（硅基流动上的 Qwen）
        # messages：对话历史列表，此处只传 system + user 两条（单轮对话）
        # temperature：生成温度，越低越保守（接近 0 时几乎确定性输出）
        resp = self.client.chat.completions.create(
            model=self.model,           # 模型名称（如 "Qwen/Qwen2.5-72B-Instruct"）
            messages=[
                {"role": "system", "content": system},  # 系统提示：定义助手角色
                {"role": "user",   "content": user},    # 用户输入：含历史记忆和本轮问题
            ],
            temperature=temperature,  # 生成随机性（0.3 = 偏确定性）
        )

        # 计算请求耗时（秒 → 毫秒）
        ms = (time.perf_counter() - t0) * 1000

        # 提取生成内容：resp.choices[0].message.content 是 LLM 的回复文本
        # or ""：防止极端情况下 content 为 None（确保返回类型始终为 str）
        content = resp.choices[0].message.content or ""

        # 记录性能日志：耗时（毫秒）、输入 token 数、输出 token 数
        # resp.usage：token 使用统计（若 API 未返回则为 None，用三元表达式兜底返回 -1）
        logger.info("[llm] %.0fms in=%d out=%d",
                    ms,
                    resp.usage.prompt_tokens if resp.usage else -1,       # 输入 token 数
                    resp.usage.completion_tokens if resp.usage else -1)   # 输出 token 数

        # .strip()：去除首尾空白字符（LLM 有时会在回复前后添加换行符）
        return content.strip()
