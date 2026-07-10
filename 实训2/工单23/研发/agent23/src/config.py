"""
全局配置中心。
所有魔法数字、API 端点、环境变量读取都在这里，其他模块只 import Config。

设计原则：
- 所有可变配置通过 .env 文件注入，不硬编码在代码里
- 所有默认值都有合理的兜底（保证即使缺少环境变量也能启动）
- validate_config() 在启动时集中检查，方便排查配置问题
"""
import os  # 标准库：读取环境变量和构造文件路径

# python-dotenv：第三方包，用于从 .env 文件加载环境变量到 os.environ
# 避免把 API Key 等敏感信息硬编码在代码里或提交到 Git
from dotenv import load_dotenv

# 加载项目根目录（src/ 的上级目录）的 .env 文件
# override=True：如果环境变量已存在，.env 文件中的值会覆盖它（方便本地调试覆盖系统变量）
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


class Config:
    """
    全局配置类（所有属性均为类属性，无需实例化）。
    读取 .env 中的环境变量，提供默认值兜底。
    """

    # ── 模型配置 ──────────────────────────────────────────────────
    # 主用 LLM 模型，优先从环境变量读取，默认使用阿里云 qwen-max
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-max")
    # 备用 LLM 模型，主模型重试3次失败后切换
    LLM_MODEL_FALLBACK: str = os.getenv("LLM_MODEL_FALLBACK", "qwen-plus")
    LLM_TEMPERATURE: float = 0.1      # 生成温度：低温使答案更确定，避免随机发挥（0=完全确定，1=高随机性）
    LLM_MAX_TOKENS: int = 1024        # 单次生成最大 token 数，1024 足够完整回答

    # 阿里云百炼（DashScope）API 配置
    # 百炼提供与 OpenAI 兼容的接口，可直接用 openai SDK 调用
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")  # 必填，空字符串时 validate_config 会报警
    DASHSCOPE_BASE_URL: str = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼 OpenAI 兼容模式端点
    )

    # ── ReAct 循环参数 ────────────────────────────────────────────
    MAX_ROUNDS: int = 3               # 最多搜索 3 轮（轮数越多越慢，3轮够覆盖多步推理题）
    ROUND_TIMEOUT_S: int = 20         # 单轮最大耗时（秒），超时强制退出当前轮
    TOTAL_TIMEOUT_S: int = 80         # 整题最大耗时（秒），防止单题卡死整个评测

    # ── 搜索工具配置 ──────────────────────────────────────────────
    SEARCH_TOP_K: int = 5             # 每次搜索返回的结果条数，5条够提供足够上下文
    SEARCH_TIMEOUT_S: int = 5         # 单次搜索请求超时（秒），快速失败切换备用引擎

    # 阿里云 IQS（智能搜索服务）——优先使用，国内访问速度快
    IQS_API_KEY: str = os.getenv("IQS_API_KEY", "")    # IQS 认证 Key
    IQS_APP_ID: str = os.getenv("IQS_APP_ID", "")      # IQS 应用 ID
    IQS_ENDPOINT: str = "https://iqs.aliyuncs.com"      # IQS 服务端点（固定）

    # Bing Web Search API——备用搜索引擎，IQS 失败时切换
    BING_API_KEY: str = os.getenv("BING_API_KEY", "")
    BING_ENDPOINT: str = "https://api.bing.microsoft.com/v7.0/search"  # Bing API v7 端点

    # SerpAPI——最后备用（调用 Google 搜索），成本较高但覆盖面广
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")

    # ── Web Fetch 配置 ─────────────────────────────────────────────
    FETCH_TIMEOUT_S: int = 8          # 抓取网页的超时时间（秒），比搜索宽松一些
    FETCH_MAX_CHARS: int = 3000       # 抓取网页正文的最大字符数，避免上下文过长

    # ── 日志配置 ──────────────────────────────────────────────────
    # 日志目录路径：相对于 src/ 目录的 ../logs/
    LOG_DIR: str = os.path.join(os.path.dirname(__file__), "..", "logs")

    # ── 评测配置 ──────────────────────────────────────────────────
    EVAL_CONCURRENCY: int = 3         # 批量评测时的并发线程数，太高容易触发 API 限流


def validate_config() -> list[str]:
    """
    启动时检查关键配置项是否已正确设置。
    返回缺失项的描述列表，供调用方决定是否阻止启动。

    :return: 缺失配置项的描述列表；列表为空表示配置完整
    """
    missing = []

    # 检查百炼 API Key（必填，所有 LLM 调用都依赖它）
    if not Config.DASHSCOPE_API_KEY:
        missing.append("DASHSCOPE_API_KEY（百炼 API Key，必须）")

    # 检查搜索 Key（三个搜索引擎至少配置一个）
    if not Config.IQS_API_KEY and not Config.BING_API_KEY and not Config.SERPAPI_KEY:
        missing.append("搜索 Key（IQS_API_KEY / BING_API_KEY / SERPAPI_KEY，至少一个）")

    return missing  # 空列表表示配置完整，非空列表表示有缺失
