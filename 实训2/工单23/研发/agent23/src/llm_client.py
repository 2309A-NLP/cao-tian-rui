"""
LLM 客户端：唯一对接阿里云百炼 Qwen 的地方。
其他模块只调 chat()，不关心底层 API 细节。

设计原则：
- 单一职责：只负责发送消息和处理重试，不涉及业务逻辑
- 弹性重试：限流/超时自动指数退避重试，最多 3 次
- 优雅降级：主模型失败后自动切换备用模型（qwen-plus）
- 异常隔离：将 API 错误转为 LLMUnavailableError，上层统一处理
"""
import time    # 标准库：时间函数，用于重试等待（指数退避）
import logging # 标准库：日志记录

# openai：第三方包（阿里云百炼与 OpenAI 接口兼容，使用 openai SDK 调用）
# OpenAI       - 客户端类，传入 api_key 和 base_url 即可对接百炼
# APIError     - API 返回的通用错误（含鉴权失败、模型不存在等）
# APITimeoutError - 请求超时错误
# RateLimitError  - 触发限流错误（超过每分钟/每天调用频率限制）
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from config import Config  # 全局配置（API Key、模型名、温度等）

logger = logging.getLogger(__name__)  # 当前模块日志记录器

# 初始化 OpenAI 兼容客户端，指向阿里云百炼的 compatible-mode 端点
# api_key   ：百炼 API Key（从 .env 读取）
# base_url  ：百炼 OpenAI 兼容接口地址（非 api.openai.com）
_client = OpenAI(
    api_key=Config.DASHSCOPE_API_KEY,
    base_url=Config.DASHSCOPE_BASE_URL,
)


class LLMUnavailableError(Exception):
    """
    LLM 不可用异常。
    当主模型和备用模型都重试 3 次失败后抛出，让上层（react_loop）决定如何降级处理。
    继承自 Exception，无需额外属性。
    """
    pass


def chat(messages: list[dict], model: str | None = None, **kwargs) -> str:
    """
    发送消息列表给 Qwen，返回生成的文本字符串。

    重试策略：
    - RateLimitError（限流）：指数退避，间隔 2/4/8 秒
    - APITimeoutError（超时）：指数退避，间隔 1/2/4 秒
    - APIError（其他 API 错误）：立即失败，不重试（如鉴权失败重试无意义）
    - 3 次失败后降级到备用模型（qwen-plus）

    :param messages: OpenAI 格式消息列表
                     例：[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    :param model:    覆盖默认模型（不传则使用 Config.LLM_MODEL）
    :param kwargs:   其他参数覆盖（如 temperature=0.5、max_tokens=256）
    :return:         LLM 生成的纯文本字符串（已去首尾空格）
    :raises LLMUnavailableError: 主模型和备用模型都重试 3 次仍失败
    """
    use_model = model or Config.LLM_MODEL  # 未指定模型时使用配置的默认模型

    # 构建请求参数，支持外部 kwargs 覆盖默认值
    params = {
        "temperature": Config.LLM_TEMPERATURE,  # 生成温度（低温=确定性高）
        "max_tokens": Config.LLM_MAX_TOKENS,     # 最大生成 token 数
        **kwargs,   # 外部传入的参数（如 max_tokens=100）优先级更高，覆盖默认值
    }

    last_err = None  # 记录最后一次错误，用于构造最终异常消息

    for attempt in range(3):  # 最多重试 3 次（attempt: 0, 1, 2）
        try:
            # 调用 OpenAI 兼容接口
            resp = _client.chat.completions.create(
                model=use_model,
                messages=messages,
                **params,
            )

            # 百炼响应兼容处理：
            # - 标准 OpenAI 格式：resp.choices[0].message.content
            # - 百炼专属域名有时返回 choices=None，答案在 resp.text
            if resp.choices:
                text = resp.choices[0].message.content or ""  # 正常情况取 choices[0]
            else:
                text = getattr(resp, "text", "") or ""  # 百炼特殊格式兜底

            # 记录调试日志（模型名、当前重试次数、消耗的 token 数）
            logger.debug("LLM [%s] attempt=%d tokens=%d",
                         use_model, attempt + 1,
                         resp.usage.total_tokens if resp.usage else -1)

            return text.strip()  # 去首尾空格后返回

        except RateLimitError as e:
            # 触发限流：等待指数退避时间后重试
            # attempt=0 → 等2秒，attempt=1 → 等4秒，attempt=2 → 等8秒
            wait = 2 ** attempt * 2
            logger.warning("Rate limit on attempt %d, wait %ds: %s", attempt + 1, wait, e)
            time.sleep(wait)   # 等待后继续循环重试
            last_err = e       # 记录错误供最终失败时使用

        except APITimeoutError as e:
            # 请求超时：等待更短的时间后重试
            # attempt=0 → 等1秒，attempt=1 → 等2秒，attempt=2 → 等4秒
            wait = 2 ** attempt
            logger.warning("Timeout on attempt %d, wait %ds: %s", attempt + 1, wait, e)
            time.sleep(wait)   # 等待后继续循环重试
            last_err = e

        except APIError as e:
            # 其他 API 错误（鉴权失败 401、模型不存在 404 等），重试无意义，立刻抛出
            logger.error("API error (non-retryable): %s", e)
            raise LLMUnavailableError(str(e)) from e  # 包装为 LLMUnavailableError 向上抛

    # ── 3 次重试全部耗尽 ──────────────────────────────────────────────────────
    # 尝试降级到备用模型（qwen-plus），避免因主模型临时不可用而整个请求失败
    if use_model != Config.LLM_MODEL_FALLBACK:
        logger.warning("Primary model %s failed, falling back to %s",
                       use_model, Config.LLM_MODEL_FALLBACK)
        # 递归调用自身，使用备用模型（备用模型同样会重试3次）
        return chat(messages, model=Config.LLM_MODEL_FALLBACK, **kwargs)

    # 主模型和备用模型都失败了，抛出最终异常
    raise LLMUnavailableError(f"LLM unavailable after 3 retries: {last_err}")
