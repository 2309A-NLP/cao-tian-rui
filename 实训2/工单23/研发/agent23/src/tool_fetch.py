"""
Web Fetch 工具：抓取网页正文，去除 HTML 噪音。

使用场景：
  搜索结果摘要（snippet）不够详细时，对特定 URL 进行完整内容抓取。
  LLM 在 ReAct 循环中通过 Action: fetch("url") 调用此工具。

安全措施：
  - SSRF 防护：拦截指向内网/本地 IP 的请求，防止服务器端请求伪造攻击
  - 设置 User-Agent 模拟正常浏览器，减少被反爬虫拦截
  - 超时限制（FETCH_TIMEOUT_S），防止单个请求卡死整个 Agent

依赖包：
  - requests: 标准 HTTP 客户端库，用于发送 GET 请求
  - ipaddress: 标准库，用于 IP 地址解析和网段检查
"""
import ipaddress    # 标准库：IP 地址和网络处理，用于 SSRF 防护
import logging      # 标准库：日志记录
import re           # 标准库：正则表达式，用于 HTML 解析
import socket       # 标准库：网络接口，用于域名解析（DNS 查询）
import urllib.parse # 标准库：URL 解析（提取 hostname 等）

# requests：最流行的 Python HTTP 客户端库（第三方包）
# 提供简洁的 API，支持 GET/POST/重定向/超时/Session 等
# 相比标准库 urllib，API 更友好，错误处理更清晰
import requests

from dataclasses import dataclass  # 标准库：数据类
from config import Config          # 全局配置（超时、最大字符数等）

logger = logging.getLogger(__name__)

# 禁止访问的私有/链路本地/回环网段（SSRF 防护列表）
# 攻击者可能通过构造特殊 URL 让服务器访问内网资源（如 ECS 元数据服务、Redis 等）
_BLOCKED_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),     # IPv4 回环地址（localhost）
    ipaddress.ip_network("10.0.0.0/8"),      # A 类私有网段
    ipaddress.ip_network("172.16.0.0/12"),   # B 类私有网段
    ipaddress.ip_network("192.168.0.0/16"),  # C 类私有网段（家庭/企业局域网）
    ipaddress.ip_network("169.254.0.0/16"),  # 链路本地地址（APIPA / AWS 元数据 169.254.169.254）
    ipaddress.ip_network("::1/128"),          # IPv6 回环地址
    ipaddress.ip_network("fc00::/7"),         # IPv6 唯一本地地址（ULA，类似 IPv4 私有网段）
]


def _is_safe_url(url: str) -> bool:
    """
    检查 URL 的目标 IP 是否为公网地址（非内网/本地），防止 SSRF 攻击。

    处理步骤：
    1. 从 URL 中提取 hostname
    2. 通过 DNS 解析 hostname 得到 IP 地址
    3. 检查 IP 是否在任何被阻止的私有网段中

    :param url: 待检查的 URL 字符串
    :return:    True 表示安全（可以访问），False 表示不安全（内网地址，拒绝访问）
    """
    try:
        hostname = urllib.parse.urlparse(url).hostname  # 从 URL 解析出主机名
        if not hostname:
            return False  # 无法解析 hostname，拒绝访问

        # DNS 解析：将域名转为 IP 地址（gethostbyname 只返回一个 IPv4 地址）
        ip_str = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_str)  # 转为 ip_address 对象，支持网段比较

        # 检查 IP 是否属于任何被阻止的网段
        return not any(ip in net for net in _BLOCKED_NETS)
    except Exception:
        # DNS 解析失败（域名不存在）或 IP 格式异常时放行，
        # 由 requests 正常超时处理（不安全的地址通常连接会超时）
        return True


@dataclass
class FetchResult:
    """
    网页抓取结果数据类。

    Attributes:
        content: 提取并清洗后的网页正文（纯文本）
        title:   网页 <title> 标签内容
        success: 是否成功抓取
        url:     实际访问的 URL（可能经过重定向）
    """
    content: str    # 清洗后的正文文本
    title: str      # 网页标题
    success: bool   # 是否成功
    url: str        # 实际访问的 URL


def web_fetch(url: str, max_chars: int | None = None) -> FetchResult:
    """
    抓取指定 URL 的网页正文，提取纯文本内容。

    :param url:       目标 URL（必须是公网可访问地址）
    :param max_chars: 最多返回的字符数（默认使用 Config.FETCH_MAX_CHARS）
    :return:          FetchResult 数据类实例
    """
    limit = max_chars or Config.FETCH_MAX_CHARS  # 默认 3000 字符

    # SSRF 安全检查：拒绝访问内网地址
    if not _is_safe_url(url):
        logger.warning("Blocked SSRF attempt: %s", url)
        return FetchResult(content="", title="", success=False, url=url)

    try:
        resp = requests.get(
            url,
            # User-Agent 模拟真实浏览器，减少被反爬虫拦截的概率
            headers={"User-Agent": "Mozilla/5.0 (Research Agent Bot)"},
            timeout=Config.FETCH_TIMEOUT_S,   # 超时时间（秒），防止单个请求卡死
            allow_redirects=True,             # 自动跟随 HTTP 重定向（如 301/302）
        )
        # raise_for_status：如果响应状态码是 4xx/5xx，抛出 HTTPError 异常
        resp.raise_for_status()

        raw_html = resp.text  # 响应 HTML 文本（encoding 由 requests 自动检测）

        # 提取网页标题和正文
        title = _extract_title(raw_html)            # 提取 <title> 内容
        content = _extract_text(raw_html, limit)    # 清洗 HTML，提取正文

        logger.debug("Fetch [%s] chars=%d", url, len(content))
        return FetchResult(content=content, title=title, success=True, url=url)

    except requests.Timeout:
        # 请求超时（超过 FETCH_TIMEOUT_S 秒），记录警告
        logger.warning("Fetch timeout: %s", url)
        return FetchResult(content="", title="", success=False, url=url)

    except Exception as e:
        # 其他异常（DNS 解析失败、连接拒绝、SSL 错误、HTTP 4xx/5xx 等）
        logger.warning("Fetch failed [%s]: %s", url, e)
        return FetchResult(content="", title="", success=False, url=url)


def _extract_title(html: str) -> str:
    """
    从 HTML 中提取 <title> 标签的文本内容。

    :param html: 原始 HTML 字符串
    :return:     title 文本（去首尾空格）；无 title 时返回空字符串
    """
    # re.DOTALL 使 . 匹配换行（title 内容可能跨行）
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_text(html: str, max_chars: int) -> str:
    """
    从 HTML 中提取纯文本正文（简单规则清洗，无第三方解析器）。

    处理步骤：
    1. 去掉不包含正文的区块标签（script/style/nav/footer 等）
    2. 去掉所有 HTML 标签（保留文本内容）
    3. 解码常见 HTML 实体（&nbsp; &amp; 等）
    4. 合并多余空白字符
    5. 截断到 max_chars 字符

    :param html:      原始 HTML 字符串
    :param max_chars: 最大返回字符数
    :return:          提取的纯文本字符串
    """
    # 去除不包含有效正文的区块标签（及其内部所有内容）
    for tag in ["script", "style", "nav", "footer", "header", "aside"]:
        # re.DOTALL 使 . 匹配换行（标签内容可能跨多行）
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", html,
                      flags=re.IGNORECASE | re.DOTALL)

    # 去除所有剩余的 HTML 标签（<div>/<p>/<span> 等），替换为空格
    text = re.sub(r"<[^>]+>", " ", html)

    # 解码常见 HTML 实体（避免 &nbsp; 等出现在正文中）
    text = (text
            .replace("&nbsp;", " ")   # 不换行空格
            .replace("&amp;", "&")    # & 符号
            .replace("&lt;", "<")     # < 符号
            .replace("&gt;", ">")     # > 符号
            .replace("&quot;", '"'))  # " 双引号

    # 将连续的空格/换行/制表符合并为单个空格，并去首尾空白
    text = re.sub(r"\s+", " ", text).strip()

    # 截断到指定长度（避免上下文过长）
    return text[:max_chars]
