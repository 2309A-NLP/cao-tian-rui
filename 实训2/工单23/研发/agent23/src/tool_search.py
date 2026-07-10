"""
Web Search 工具：IQS → Bing → SerpAPI 三级降级链。

设计原则：
- 三个搜索后端按优先级依次尝试：IQS（阿里云，国内优先）→ Bing → SerpAPI（Google）
- 任何一级超时或失败，自动切换到下一级，保证搜索可用性
- 全部失败时返回空列表，不抛异常（ReAct 循环会根据空结果决策）
- 每个后端的 API Key 缺失时直接跳过（避免调用无效请求）

后端说明：
  IQS   : 阿里云智能搜索服务，国内访问快，支持中英文搜索，主力后端
  Bing  : 微软必应搜索 API v7，国际化好，备用后端
  SerpAPI: 封装了 Google 搜索的第三方服务，覆盖面最广，成本最高，最后备用

依赖：
  requests：Python HTTP 客户端库（第三方包）
"""
import logging    # 标准库：日志记录
# requests：第三方 HTTP 客户端库
# 用于向搜索 API 发送 GET/POST 请求，处理响应
import requests
from dataclasses import dataclass, field  # 标准库：数据类工具
from config import Config                  # 全局配置（API Key、超时等）

logger = logging.getLogger(__name__)  # 当前模块日志记录器


@dataclass
class SearchResult:
    """
    搜索结果数据类。

    Attributes:
        results: 搜索结果列表，每项为字典 {"title": str, "snippet": str, "url": str}
        source:  实际使用的搜索后端名称（"iqs"/"bing"/"serpapi"/"none"），用于监控
    """
    results: list[dict] = field(default_factory=list)  # 默认空列表
    source: str = "none"   # 记录用了哪个后端（方便统计各后端使用率）


def web_search(query: str, top_k: int | None = None) -> SearchResult:
    """
    按 IQS → Bing → SerpAPI 顺序搜索网络，返回第一个成功的结果。

    :param query:  搜索关键词字符串
    :param top_k:  希望返回的结果条数（默认使用 Config.SEARCH_TOP_K）
    :return:       SearchResult，results 每项含 title/snippet/url 三个字段
    """
    k = top_k or Config.SEARCH_TOP_K  # 未指定时使用配置的默认值（通常为5）

    # 按优先级依次尝试各搜索后端
    for backend_fn, name in [
        (_search_iqs, "iqs"),         # 优先：阿里云 IQS
        (_search_bing, "bing"),       # 备用：Bing
        (_search_serpapi, "serpapi"), # 最后备用：SerpAPI（Google）
    ]:
        try:
            results = backend_fn(query, k)  # 调用后端，可能抛出 ValueError 或 requests 异常
            if results:  # 有结果才返回（空结果时继续尝试下一个后端）
                logger.debug("Search [%s] query=%r hits=%d", name, query, len(results))
                return SearchResult(results=results, source=name)
        except Exception as e:
            # 后端失败时记录警告，继续尝试下一个（不中断 ReAct 循环）
            logger.warning("Search backend [%s] failed: %s", name, e)

    # 所有后端都失败，记录错误并返回空结果
    logger.error("All search backends failed for query: %r", query)
    return SearchResult(results=[], source="none")


# ── 后端实现 ──────────────────────────────────────────────────────────────────

def _search_iqs(query: str, top_k: int) -> list[dict]:
    """
    调用阿里云 IQS（智能搜索服务）联网搜索。
    接口地址：https://cloud-iqs.aliyuncs.com/search/unified

    返回格式：每项含 title/mainText（或 snippet）/link

    :param query:  搜索关键词
    :param top_k:  希望返回的条数
    :return:       标准化的搜索结果列表（含 title/snippet/url）
    :raises ValueError: API Key 未设置或包含非 ASCII 字符（占位符未替换）
    :raises requests.RequestException: HTTP 请求失败
    """
    # 检查 API Key 是否已配置
    if not Config.IQS_API_KEY:
        raise ValueError("IQS_API_KEY not set")
    # 检查是否含非 ASCII 字符（说明是未替换的占位符，如 "${IQS_API_KEY}"）
    if not Config.IQS_API_KEY.isascii():
        raise ValueError("IQS_API_KEY contains non-ASCII characters (placeholder not replaced)")

    # 发送 POST 请求到 IQS 搜索接口
    resp = requests.post(
        "https://cloud-iqs.aliyuncs.com/search/unified",
        headers={
            "Authorization": f"Bearer {Config.IQS_API_KEY}",  # Bearer Token 鉴权
            "Content-Type": "application/json",
        },
        json={
            "query": query,             # 搜索词
            "engineType": "Generic",    # 通用搜索引擎
            "numResults": top_k,        # 返回条数
            "contents": {
                "mainText": True,       # 返回正文摘要（比 snippet 更完整）
                "markdownText": False,  # 不需要 Markdown 格式
                "richMainBody": False,  # 不需要富文本格式
                "summary": True,        # 返回 AI 生成的摘要
                "rerankScore": False,   # 不返回重排序分数
            },
        },
        timeout=Config.SEARCH_TIMEOUT_S,  # 请求超时时间
    )
    resp.raise_for_status()  # 非 2xx 状态码抛出 HTTPError

    data = resp.json()  # 解析 JSON 响应
    items = data.get("pageItems", [])  # 取结果列表（IQS 返回字段名为 pageItems）

    # 标准化为统一格式：{title, snippet, url}
    return [
        {
            "title": item.get("title", ""),
            # mainText 比 snippet 更详细，优先使用；mainText 没有时回退到 snippet
            "snippet": item.get("mainText") or item.get("snippet", ""),
            "url": item.get("link", ""),  # IQS 使用 "link" 字段存储 URL
        }
        for item in items[:top_k]  # 截断确保不超过 top_k 条
    ]


def _search_bing(query: str, top_k: int) -> list[dict]:
    """
    调用微软必应搜索 API v7。
    接口地址：https://api.bing.microsoft.com/v7.0/search

    鉴权方式：请求头 Ocp-Apim-Subscription-Key

    :param query:  搜索关键词
    :param top_k:  希望返回的条数
    :return:       标准化的搜索结果列表（含 title/snippet/url）
    :raises ValueError: API Key 未设置或包含非 ASCII 字符
    :raises requests.RequestException: HTTP 请求失败
    """
    if not Config.BING_API_KEY:
        raise ValueError("BING_API_KEY not set")
    if not Config.BING_API_KEY.isascii():
        raise ValueError("BING_API_KEY contains non-ASCII characters (placeholder not replaced)")

    # 发送 GET 请求（Bing API 使用 GET + 查询参数）
    resp = requests.get(
        Config.BING_ENDPOINT,  # https://api.bing.microsoft.com/v7.0/search
        headers={"Ocp-Apim-Subscription-Key": Config.BING_API_KEY},  # Bing 专用鉴权头
        params={
            "q": query,         # 搜索词
            "count": top_k,     # 返回条数
            "mkt": "zh-CN",     # 市场地区：中文中国（搜索结果偏向中文内容）
        },
        timeout=Config.SEARCH_TIMEOUT_S,
    )
    resp.raise_for_status()

    data = resp.json()
    # Bing 响应结构：{"webPages": {"value": [...]}}
    items = data.get("webPages", {}).get("value", [])

    # 标准化为统一格式
    return [
        {
            "title": item.get("name", ""),       # Bing 用 "name" 存标题
            "snippet": item.get("snippet", ""),  # 摘要
            "url": item.get("url", ""),           # URL
        }
        for item in items[:top_k]
    ]


def _search_serpapi(query: str, top_k: int) -> list[dict]:
    """
    调用 SerpAPI 进行 Google 搜索（第三方服务，封装了 Google 搜索 API）。
    接口地址：https://serpapi.com/search

    特点：能访问 Google 搜索结果，但需要付费，成本最高，作为最后备用。

    :param query:  搜索关键词
    :param top_k:  希望返回的条数
    :return:       标准化的搜索结果列表（含 title/snippet/url）
    :raises ValueError: API Key 未设置或包含非 ASCII 字符
    :raises requests.RequestException: HTTP 请求失败
    """
    if not Config.SERPAPI_KEY:
        raise ValueError("SERPAPI_KEY not set")
    if not Config.SERPAPI_KEY.isascii():
        raise ValueError("SERPAPI_KEY contains non-ASCII characters (placeholder not replaced)")

    # 发送 GET 请求（SerpAPI 使用 GET + api_key 参数鉴权）
    resp = requests.get(
        "https://serpapi.com/search",
        params={
            "q": query,                      # 搜索词
            "api_key": Config.SERPAPI_KEY,   # SerpAPI 鉴权 Key
            "num": top_k,                    # 返回结果数
        },
        timeout=Config.SEARCH_TIMEOUT_S,
    )
    resp.raise_for_status()

    data = resp.json()
    # SerpAPI 响应结构：{"organic_results": [...]}（有机搜索结果，排除广告）
    items = data.get("organic_results", [])

    # 标准化为统一格式
    return [
        {
            "title": item.get("title", ""),    # 标题
            "snippet": item.get("snippet", ""), # 摘要
            "url": item.get("link", ""),         # SerpAPI 用 "link" 存 URL
        }
        for item in items[:top_k]
    ]
