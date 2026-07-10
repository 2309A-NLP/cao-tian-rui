"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-MCP

imaging_mcp —— 医学影像分析 MCP Server（占位）
--------------------------------------------
上游工单03（影像分析，端口8013）尚未开工。本文件先定义接口签名，
待上游 FastAPI 完成后，切换到 HTTP 代理即可，无需改动 MCP Client 侧。

约定的上游接口（供工单03参考）：
  POST /analyze  { image_url|image_base64, question? } → { report, findings[], confidence }
  POST /report   { image_url|image_base64, template } → { report_html, report_markdown }

Tools:
  - analyze_image      影像分析（VQA/病灶识别）
  - generate_report    影像报告生成（MRG）
"""
import os  # 标准库：读取环境变量
import sys  # 标准库：sys.stderr 输出日志
from typing import Any, Optional  # 标准库：类型注解

# httpx 包：异步 HTTP 客户端，用于探测上游服务 /health 端点及转发请求
import httpx
from dotenv import load_dotenv  # python-dotenv：加载 .env 文件
# mcp.server.fastmcp.FastMCP：MCP SDK 高级 Server 封装，@mcp.tool() 注册工具
from mcp.server.fastmcp import FastMCP

# 解析 04-MCP/ 根目录
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

# 影像分析上游服务地址（工单03 FastAPI，端口 8013）
IMAGING_URL = os.getenv("IMAGING_API_URL", "http://localhost:8013")
# API Key（若工单03 启用鉴权）
API_KEY = os.getenv("API_KEY", "")
# 影像推理较慢，超时设置为 60s（比其他工具长）
TIMEOUT = float(os.getenv("MCP_UPSTREAM_TIMEOUT", "60.0"))

# 创建名为 "imaging" 的 MCP Server 实例
mcp = FastMCP("imaging")


def _headers() -> dict:
    """
    构造 HTTP 请求头，若配置了 API_KEY 则附加鉴权头。

    Returns:
        请求头字典
    """
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY  # 传递给上游服务的鉴权头
    return h


def _not_implemented(tool: str) -> dict[str, Any]:
    """
    生成"工具未实现"的标准返回结构。
    当上游服务未启动时，所有工具调用均返回此结构。
    路由 Agent 识别 status="not_implemented" 后会告知用户功能尚未开放。

    Args:
        tool: 工具名称

    Returns:
        标准未实现错误 dict
    """
    return {
        "ok": False,
        "status": "not_implemented",  # 路由 Agent 通过此字段识别占位状态
        "tool": tool,
        "reason": "工单03（影像分析）尚未开工，此工具暂不可用",
        "upstream": IMAGING_URL,  # 上游地址，方便排查
    }


async def _upstream_alive() -> bool:
    """
    快速探测上游影像分析服务是否已启动。
    超时设置极短（0.3s），保证未开工时工具调用快速失败，不阻塞 Agent。

    Returns:
        True 表示上游服务在线；False 表示未启动或超时
    """
    try:
        async with httpx.AsyncClient(timeout=0.3) as client:
            r = await client.get(f"{IMAGING_URL}/health")  # 探测健康检查端点
            return r.status_code == 200
    except Exception:
        # 连接拒绝/超时等均视为未在线
        return False


@mcp.tool()
async def analyze_image(
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    question: Optional[str] = None,
) -> dict[str, Any]:
    """
    医学影像分析（VQA/病灶识别）。

    Args:
        image_url: 影像的 URL（与 image_base64 二选一）
        image_base64: 影像的 base64 编码
        question: 可选问题，如"这张片子有什么异常？"

    Returns:
        上游可用时: { ok, report, findings[], confidence }
        上游未开工: { ok: False, status: "not_implemented", ... }
    """
    # 两种图片输入必须至少提供一种
    if not image_url and not image_base64:
        return {"ok": False, "error": "image_url 或 image_base64 必须提供其一"}

    # 探测上游是否在线；未开工时快速返回 not_implemented
    if not await _upstream_alive():
        return _not_implemented("analyze_image")

    # 构造请求体
    payload = {"question": question}
    if image_url:
        payload["image_url"] = image_url
    if image_base64:
        payload["image_base64"] = image_base64

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # 调用上游影像分析接口
            r = await client.post(f"{IMAGING_URL}/analyze", json=payload, headers=_headers())
            r.raise_for_status()  # HTTP 非 200 抛异常
            return {"ok": True, **r.json()}  # 将上游返回字段展开合并
    except Exception as e:
        return {"ok": False, "error": f"上游调用失败: {e}"}


@mcp.tool()
async def generate_report(
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    template: str = "standard",
) -> dict[str, Any]:
    """
    医学影像报告生成（MRG）。

    Args:
        image_url: 影像 URL
        image_base64: 影像 base64
        template: 报告模板类型，默认 "standard"（可选 detailed/brief）

    Returns:
        上游可用时: { ok, report_html, report_markdown }
        上游未开工: { ok: False, status: "not_implemented", ... }
    """
    # 图片输入校验
    if not image_url and not image_base64:
        return {"ok": False, "error": "image_url 或 image_base64 必须提供其一"}

    # 上游未在线则返回占位响应
    if not await _upstream_alive():
        return _not_implemented("generate_report")

    # 构造请求体
    payload = {"template": template}
    if image_url:
        payload["image_url"] = image_url
    if image_base64:
        payload["image_base64"] = image_base64

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # 调用上游报告生成接口
            r = await client.post(f"{IMAGING_URL}/report", json=payload, headers=_headers())
            r.raise_for_status()
            return {"ok": True, **r.json()}
    except Exception as e:
        return {"ok": False, "error": f"上游调用失败: {e}"}


if __name__ == "__main__":
    # 打印上游地址，提示当前是占位模式
    print(f"[imaging_mcp] 上游: {IMAGING_URL}（未开工时返回 not_implemented）", file=sys.stderr)
    # 以 stdio 传输方式启动 MCP Server
    mcp.run(transport="stdio")
