"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-MCP

registration_mcp —— 挂号管理 MCP Server
---------------------------------------
代理工单01（挂号管理）的 FastAPI 服务（http://localhost:8011），
将其 4 个业务接口封装为 MCP tool。

传输方式：stdio

Tools:
  - chat_registration       端到端挂号对话（挂号/取消/查询全走这里）
  - query_slots             号源筛选查询（结构化参数）
  - get_appointments        我的预约列表 + 统计
  - get_doctor_schedule     医生排班查询
"""
import os  # 标准库：读取环境变量
import sys  # 标准库：sys.stderr 输出启动日志
from typing import Any, Optional  # 标准库：类型注解，Optional[X] = X 或 None

# httpx 包：异步 HTTP 客户端，代理调用工单01 FastAPI 接口
import httpx
from dotenv import load_dotenv  # python-dotenv：加载 .env 文件
# mcp.server.fastmcp.FastMCP：MCP SDK，@mcp.tool() 注册工具
from mcp.server.fastmcp import FastMCP

# 解析 04-MCP/ 根目录
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

# 工单01 挂号管理服务地址（默认 localhost:8011）
REGISTRATION_URL = os.getenv("REGISTRATION_API_URL", "http://localhost:8011")
# API Key（若工单01 启用鉴权）
API_KEY = os.getenv("API_KEY", "")
# 请求超时，挂号操作涉及数据库写入，设 30s
TIMEOUT = float(os.getenv("MCP_UPSTREAM_TIMEOUT", "30.0"))

# 创建名为 "registration" 的 MCP Server 实例
mcp = FastMCP("registration")


def _headers() -> dict:
    """构造请求头（含可选 API Key 鉴权头）。"""
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


@mcp.tool()
async def chat_registration(user_id: int, query: str) -> dict[str, Any]:
    """
    挂号对话入口：挂号、取消、查询预约等全部自然语言操作。
    对应工单01 POST /chat 接口。

    Args:
        user_id: 用户 ID（整数），挂号必须有 user_id
        query: 用户请求，例如"帮我挂儿科明天上午的号"、"取消我下周三的预约"

    Returns:
        { "ok": bool, "reply": str, "trace_id": str }
        失败时 { "ok": False, "error": str }
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # POST /chat 发起对话式挂号请求
            r = await client.post(
                f"{REGISTRATION_URL}/chat",
                json={"user_id": user_id, "query": query},
                headers=_headers(),
            )
            r.raise_for_status()
            return {"ok": True, **r.json()}  # 展开工单01返回字段
    except httpx.HTTPStatusError as e:
        # 上游 HTTP 错误（如 400 参数错误、500 内部错误）
        return {"ok": False, "error": f"上游返回 {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.RequestError as e:
        # 网络连接失败（工单01 未启动）
        return {"ok": False, "error": f"无法连接挂号服务 {REGISTRATION_URL}: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"未知错误: {e}"}


@mcp.tool()
async def query_slots(
    dept: Optional[str] = None,        # 科室名，如"儿科"
    date_start: Optional[str] = None,  # 开始日期 YYYY-MM-DD
    date_end: Optional[str] = None,    # 结束日期 YYYY-MM-DD
    slot_type: Optional[str] = None,   # 号源类型："专家" 或 "普通"
    doctor: Optional[str] = None,      # 医生姓名（模糊匹配）
) -> dict[str, Any]:
    """
    号源筛选查询（结构化参数）。
    对应工单01 GET /slots 接口。

    Args:
        dept: 科室名，如"儿科"、"内科"
        date_start: 起始日期 YYYY-MM-DD（缺省=今天）
        date_end: 截止日期 YYYY-MM-DD（缺省=起始日期+7天）
        slot_type: 号源类型，"专家" 或 "普通"
        doctor: 医生姓名（模糊匹配）

    Returns:
        { "ok": bool, "slots": [...], "total": int }
    """
    # 过滤掉值为 None 的参数，避免传空参数给工单01
    params = {k: v for k, v in {
        "dept": dept, "date_start": date_start, "date_end": date_end,
        "slot_type": slot_type, "doctor": doctor,
    }.items() if v is not None}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # GET /slots 查询号源（URL 查询参数）
            r = await client.get(f"{REGISTRATION_URL}/slots", params=params, headers=_headers())
            r.raise_for_status()
            return {"ok": True, **r.json()}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"上游返回 {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.RequestError as e:
        return {"ok": False, "error": f"无法连接挂号服务: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"未知错误: {e}"}


@mcp.tool()
async def get_appointments(
    user_id: int,              # 用户 ID，必填
    dept: Optional[str] = None,    # 可选科室筛选
    status: Optional[int] = None,  # 1=正常 0=已取消
) -> dict[str, Any]:
    """
    查询指定用户的预约列表。
    对应工单01 GET /appointments/{user_id} 接口。

    Args:
        user_id: 用户 ID
        dept: 可选科室筛选
        status: 1=正常 0=已取消

    Returns:
        { "ok": bool, "appointments": [...], "stats": {pending, total, canceled} }
    """
    # 只传非 None 的筛选参数
    params = {k: v for k, v in {"dept": dept, "status": status}.items() if v is not None}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # GET /appointments/{user_id} 查询预约列表
            r = await client.get(
                f"{REGISTRATION_URL}/appointments/{user_id}",
                params=params, headers=_headers(),
            )
            r.raise_for_status()
            return {"ok": True, **r.json()}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"上游返回 {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.RequestError as e:
        return {"ok": False, "error": f"无法连接挂号服务: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"未知错误: {e}"}


@mcp.tool()
async def get_doctor_schedule(
    doctor: str,                        # 医生姓名，必填
    date_start: Optional[str] = None,  # 起始日期
    date_end: Optional[str] = None,    # 截止日期
) -> dict[str, Any]:
    """
    查询指定医生的排班。
    对应工单01 GET /schedule 接口。

    Args:
        doctor: 医生姓名（支持模糊匹配）
        date_start: 起始日期 YYYY-MM-DD（缺省=今天）
        date_end: 截止日期 YYYY-MM-DD（缺省=起始+13天）

    Returns:
        { "ok": bool, "doctor": {...}, "schedules": [...] }
    """
    # 医生名是必填参数，日期为可选
    params = {"doctor": doctor}
    if date_start:
        params["date_start"] = date_start
    if date_end:
        params["date_end"] = date_end
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # GET /schedule 查询医生排班
            r = await client.get(f"{REGISTRATION_URL}/schedule", params=params, headers=_headers())
            r.raise_for_status()
            return {"ok": True, **r.json()}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"上游返回 {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.RequestError as e:
        return {"ok": False, "error": f"无法连接挂号服务: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"未知错误: {e}"}


if __name__ == "__main__":
    # 打印上游地址到 stderr 确认配置
    print(f"[registration_mcp] 上游: {REGISTRATION_URL}", file=sys.stderr)
    # 以 stdio 传输方式启动 MCP Server
    mcp.run(transport="stdio")
