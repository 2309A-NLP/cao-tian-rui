"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-MCP

voice_mcp —— 实时语音识别 MCP Server（占位）
-------------------------------------------
上游工单05（实时语音识别，端口8015）尚未开工。本文件先定义接口签名，
待上游 FastAPI 完成后，自动切换到真实转发（探测 /health）。

约定的上游接口（供工单05参考）：
  POST /transcribe  { audio_url|audio_base64, lang? } → { text, segments[], duration_ms }

Tools:
  - transcribe    语音转文字
"""
import os  # 标准库：读取环境变量
import sys  # 标准库：sys.stderr 输出启动日志
from typing import Any, Optional  # 标准库：类型注解

# httpx 包：异步 HTTP 客户端，用于探测上游健康端点及转发请求
import httpx
from dotenv import load_dotenv  # python-dotenv：加载 .env 文件
# mcp.server.fastmcp.FastMCP：MCP SDK，@mcp.tool() 注册工具
from mcp.server.fastmcp import FastMCP

# 解析 04-MCP/ 根目录
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

# 工单05 语音识别服务地址（默认 localhost:8015）
VOICE_URL = os.getenv("VOICE_API_URL", "http://localhost:8015")
# API Key（若工单05 启用鉴权）
API_KEY = os.getenv("API_KEY", "")
# 语音推理较慢，超时设置为 60s
TIMEOUT = float(os.getenv("MCP_UPSTREAM_TIMEOUT", "60.0"))

# 创建名为 "voice" 的 MCP Server 实例
mcp = FastMCP("voice")


def _headers() -> dict:
    """构造请求头（含可选 API Key 鉴权头）。"""
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _not_implemented(tool: str) -> dict[str, Any]:
    """
    生成"工具未实现"的标准返回结构。
    路由 Agent 识别 status="not_implemented" 后告知用户功能尚未开放。

    Args:
        tool: 工具名称

    Returns:
        标准未实现错误 dict
    """
    return {
        "ok": False,
        "status": "not_implemented",  # 路由 Agent 据此判断并礼貌回复用户
        "tool": tool,
        "reason": "工单05（实时语音识别）",
        "upstream": VOICE_URL,  # 上游地址，方便排查
    }


async def _upstream_alive() -> bool:
    """
    快速探测上游语音识别服务是否已启动。
    超时 0.3s，未开工时快速失败不阻塞 Agent。

    Returns:
        True 表示服务在线；False 表示未启动或超时
    """
    try:
        async with httpx.AsyncClient(timeout=0.3) as client:
            r = await client.get(f"{VOICE_URL}/health")  # 健康检查端点
            return r.status_code == 200
    except Exception:
        # 连接失败/超时等均视为未在线
        return False


@mcp.tool()
async def transcribe(
    audio_url: Optional[str] = None,    # 音频 URL（与 audio_base64 二选一）
    audio_base64: Optional[str] = None, # 音频 base64 编码（wav/mp3 等格式）
    lang: str = "zh",                   # 语言代码，默认中文
) -> dict[str, Any]:
    """
    语音转文字。

    Args:
        audio_url: 音频 URL
        audio_base64: 音频 base64（wav/mp3 等）
        lang: 语言代码，默认 "zh"（可选 "en"、"auto"）

    Returns:
        上游可用时: { ok, text, segments[], duration_ms }
        上游未开工: { ok: False, status: "not_implemented", ... }
    """
    # 音频输入必须至少提供一种
    if not audio_url and not audio_base64:
        return {"ok": False, "error": "audio_url 或 audio_base64 必须提供其一"}

    # 探测上游是否在线；未开工时快速返回占位响应
    if not await _upstream_alive():
        return _not_implemented("transcribe")

    # 构造请求体
    payload = {"lang": lang}
    if audio_url:
        payload["audio_url"] = audio_url
    if audio_base64:
        payload["audio_base64"] = audio_base64

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # POST /transcribe 调用上游语音转文字接口
            r = await client.post(f"{VOICE_URL}/transcribe", json=payload, headers=_headers())
            r.raise_for_status()
            return {"ok": True, **r.json()}  # 展开上游返回字段
    except Exception as e:
        return {"ok": False, "error": f"上游调用失败: {e}"}


if __name__ == "__main__":
    # 打印启动信息，提示当前是占位模式
    print(f"[voice_mcp] 上游: {VOICE_URL}（未开工时返回 not_implemented）", file=sys.stderr)
    # 以 stdio 传输方式启动 MCP Server
    mcp.run(transport="stdio")
