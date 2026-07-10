"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
VLM 客户端封装：硅基流动 Qwen3-VL（OpenAI 兼容协议）。

VLM（Vision Language Model）：视觉语言模型，可以同时理解图像和文字
本模块封装了与硅基流动（SiliconFlow）API 的通信：
- 支持 image + text 多模态输入（图文理解）
- 内置指数退避重试（处理 429 限流 / 超时）
- 纯文本 LLM 复用同一客户端（构建知识库描述时使用）

技术细节：
- 使用 OpenAI SDK（硅基流动提供 OpenAI 兼容接口）
- 图片通过 data URI（base64 编码）发送给模型
- 指数退避：第1次失败后等待 2s，第2次失败后等待 4s，第3次失败后等待 8s
"""

# base64：Python 内置模块，用于将二进制数据编码为 ASCII 字符串
# 图片必须转为 base64 字符串才能嵌入 JSON 请求体发送给 API
import base64

# time：Python 内置模块，用于实现重试时的等待（time.sleep）
import time

# Optional：类型提示，表示可以为 None
from typing import Optional

# openai：OpenAI 官方 Python SDK
# 硅基流动提供与 OpenAI API 兼容的接口，可直接使用此 SDK
# OpenAI：客户端类（通过修改 base_url 可以接入兼容接口）
# APIError：API 返回非 2xx 状态码时抛出的基础异常
# APITimeoutError：请求超时时抛出的异常（连接超时或读取超时）
# RateLimitError：API 返回 429 Too Many Requests 时抛出的异常
# 安装方式：pip install openai
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

# 导入 VLM/LLM 相关配置
from .config import (
    SILICONFLOW_API_KEY,    # 硅基流动 API 密钥
    SILICONFLOW_BASE_URL,   # API 基础 URL（OpenAI 兼容端点）
    VLM_MODEL,              # 视觉语言模型名称
    LLM_MODEL,              # 纯文本语言模型名称
    VLM_TIMEOUT,            # 单次请求超时时间（秒）
    VLM_MAX_RETRIES,        # 最大重试次数
)

# 导入日志记录器
from .logger import get_logger

# 获取本模块专用的日志记录器
logger = get_logger("wt13.vlm")


class VLMError(Exception):
    """
    VLM 调用最终失败（重试耗尽）时抛出的自定义异常。

    携带错误码，便于 API 层映射到对应 HTTP 状态码：
    - "VLM_TIMEOUT"：所有重试均超时 → HTTP 504
    - "VLM_UNAVAILABLE"：API 错误或限流 → HTTP 503
    - "VLM_EMPTY"：模型返回空内容 → HTTP 503
    """

    def __init__(self, message: str, code: str = "VLM_UNAVAILABLE"):
        """
        参数：
            message (str)：人类可读的错误描述
            code (str)：机器可读的错误码，默认 "VLM_UNAVAILABLE"
        """
        super().__init__(message)  # 调用父类 Exception 初始化
        self.code = code           # 存储错误码


class VLMClient:
    """
    硅基流动 VLM/LLM 客户端。

    封装了图文多模态调用（chat_vision）和纯文本调用（chat_text），
    两者都通过内部的 _call_with_retry 方法实现带重试的 API 调用。
    """

    def __init__(self):
        """
        初始化 VLM 客户端，创建 OpenAI 兼容客户端实例。
        """
        # 若 API Key 未配置，记录警告（后续调用会失败，但不在此处直接抛出）
        if not SILICONFLOW_API_KEY:
            logger.warning("SILICONFLOW_API_KEY 未设置，VLM 调用会失败", extra={"payload": {"backend": "siliconflow"}})

        # 创建 OpenAI 客户端实例，修改 base_url 指向硅基流动的兼容接口
        self.client = OpenAI(
            api_key=SILICONFLOW_API_KEY or "sk-placeholder",  # 未配置时用占位符（避免 SDK 初始化失败）
            base_url=SILICONFLOW_BASE_URL,  # 硅基流动的 OpenAI 兼容 API 地址
            timeout=VLM_TIMEOUT,            # 连接和读取超时时间（秒）
        )
        self.vlm_model = VLM_MODEL  # 视觉语言模型名称（含图片理解能力）
        self.llm_model = LLM_MODEL  # 纯文本语言模型名称

    def chat_vision(
        self,
        image_b64: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        image_format: str = "jpeg",
    ) -> str:
        """
        调用视觉语言模型（VLM），输入图片 + 文字，输出文字回答。

        参数：
            image_b64 (str)：图片的 base64 编码字符串（不含 data:image 前缀）
            prompt (str)：用户问题或指令文本
            system (str)：系统提示词（设定模型角色和行为规范），默认为空
            max_tokens (int)：生成回答的最大 token 数
            temperature (float)：采样温度，0~1，越低回答越确定性，越高越随机
            image_format (str)：图片格式（"jpeg" 或 "png"），用于构建 data URI

        返回值：
            str：模型生成的文字回答

        异常：
            VLMError：重试耗尽后抛出
        """
        # 构建图片的 data URI 格式（RFC 2397）：data:image/jpeg;base64,<base64_data>
        # 这是将图片嵌入 JSON 请求体的标准方式
        image_url = f"data:image/{image_format};base64,{image_b64}"

        # 构建消息列表
        messages: list[dict] = []
        if system:
            # 若提供了系统提示词，将其作为 system 角色的消息
            messages.append({"role": "system", "content": system})

        # 用户消息使用多模态格式：内容是一个列表，包含图片和文字两部分
        messages.append(
            {
                "role": "user",
                "content": [
                    # 图片部分：type="image_url"，url 使用 data URI
                    {"type": "image_url", "image_url": {"url": image_url}},
                    # 文字部分：type="text"，text 是用户问题
                    {"type": "text", "text": prompt},
                ],
            }
        )

        # 调用带重试的通用请求方法
        return self._call_with_retry(
            model=self.vlm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat_text(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """
        调用纯文本语言模型（LLM），仅输入文字，输出文字回答。

        参数：
            prompt (str)：用户问题或指令文本
            system (str)：系统提示词，默认为空
            max_tokens (int)：生成回答的最大 token 数
            temperature (float)：采样温度

        返回值：
            str：模型生成的文字回答

        异常：
            VLMError：重试耗尽后抛出
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})  # 系统消息

        # 纯文本模式：用户消息的 content 直接是字符串（不是列表）
        messages.append({"role": "user", "content": prompt})

        # 使用纯文本 LLM 模型（而非 VLM 视觉模型）
        return self._call_with_retry(
            model=self.llm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _call_with_retry(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        带指数退避重试的 API 调用方法（内部方法，不对外暴露）。

        重试策略：
        - 最多重试 VLM_MAX_RETRIES 次（默认 3 次）
        - 每次失败后等待 2^(attempt-1) * 2 秒（指数退避）
          第 1 次失败后等 2s，第 2 次失败后等 4s，第 3 次失败后等 8s
        - 超时和限流可重试，API 错误可重试，其他异常也可重试

        参数：
            model (str)：模型名称（VLM 或 LLM）
            messages (list[dict])：消息列表（OpenAI Chat 格式）
            max_tokens (int)：最大生成 token 数
            temperature (float)：采样温度

        返回值：
            str：模型返回的文字内容（已去除首尾空白）

        异常：
            VLMError：所有重试耗尽后抛出，携带具体错误码
        """
        last_err: Optional[Exception] = None  # 记录最后一次异常（用于错误消息）

        # 重试循环：attempt 从 1 开始到 VLM_MAX_RETRIES
        for attempt in range(1, VLM_MAX_RETRIES + 1):
            try:
                # 调用 OpenAI SDK 的 Chat Completions 接口
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # 提取模型生成的文字内容
                # resp.choices[0]：取第一个（通常也是唯一一个）候选回答
                # .message.content：获取消息内容字符串
                # or ""：若为 None 则转为空字符串
                # .strip()：去除首尾空白字符
                content = (resp.choices[0].message.content or "").strip()

                # 若模型返回空内容，视为失败并抛出 VLMError
                if not content:
                    raise VLMError("VLM 返回空内容", code="VLM_EMPTY")
                return content  # 成功返回

            # ── 超时异常处理 ──
            except (APITimeoutError,) as e:
                last_err = e
                logger.warning("VLM 超时", extra={"payload": {"attempt": attempt, "model": model, "error": str(e)}})
                # 最后一次重试仍然超时，抛出 VLMError
                if attempt >= VLM_MAX_RETRIES:
                    raise VLMError(f"VLM 超时（{VLM_MAX_RETRIES}次重试耗尽）", code="VLM_TIMEOUT") from e

            # ── 限流异常处理（API 返回 429 Too Many Requests）──
            except (RateLimitError,) as e:
                last_err = e
                logger.warning("VLM 限流", extra={"payload": {"attempt": attempt, "model": model, "error": str(e)}})
                # 最后一次重试仍然被限流，抛出 VLMError
                if attempt >= VLM_MAX_RETRIES:
                    raise VLMError(f"VLM 限流（{VLM_MAX_RETRIES}次重试耗尽）", code="VLM_UNAVAILABLE") from e

            # ── API 错误处理（非 200 状态码，如 500、401 等）──
            except APIError as e:
                last_err = e
                logger.error("VLM API 错误", extra={"payload": {"attempt": attempt, "model": model, "error": str(e)}})
                if attempt >= VLM_MAX_RETRIES:
                    raise VLMError(f"VLM API 错误: {e}", code="VLM_UNAVAILABLE") from e

            # ── VLMError 直接向上抛（不重试，如空内容错误）──
            except VLMError:
                raise

            # ── 其他未预料异常（网络底层错误等）──
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.error("VLM 未知异常", extra={"payload": {"attempt": attempt, "model": model, "error": str(e)}})
                if attempt >= VLM_MAX_RETRIES:
                    raise VLMError(f"VLM 未知异常: {e}", code="VLM_UNAVAILABLE") from e

            # 指数退避：等待 2^(attempt-1) * 2 秒后重试
            # attempt=1 → 2s，attempt=2 → 4s，attempt=3 → 8s
            time.sleep(2 ** (attempt - 1) * 2)

        # 理论上不会执行到这里（循环体内必然 return 或 raise）
        raise VLMError(f"VLM 调用失败: {last_err}", code="VLM_UNAVAILABLE")


def image_bytes_to_b64(data: bytes) -> str:
    """
    将图片字节数据编码为 base64 ASCII 字符串。

    参数：
        data (bytes)：原始图片字节数据

    返回值：
        str：base64 编码的 ASCII 字符串（不含 data URI 前缀）
    """
    # base64.b64encode：将字节数据编码为 base64 格式的字节串
    # .decode("ascii")：将字节串解码为 ASCII 字符串（base64 只含 ASCII 字符）
    return base64.b64encode(data).decode("ascii")


# ── 全局 VLMClient 单例缓存 ──
_client: Optional[VLMClient] = None


def get_vlm_client() -> VLMClient:
    """
    获取（或延迟创建）VLMClient 全局单例。

    使用单例模式避免重复创建 OpenAI 客户端对象（每次创建都有初始化开销）。

    返回值：
        VLMClient：全局唯一的 VLM 客户端实例
    """
    global _client
    if _client is None:
        _client = VLMClient()  # 首次调用时创建实例
    return _client
