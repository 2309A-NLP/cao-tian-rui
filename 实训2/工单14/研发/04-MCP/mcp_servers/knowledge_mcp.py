"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-MCP

knowledge_mcp —— 健康咨询 MCP Server
------------------------------------
代理工单02（健康咨询）的 FastAPI 服务（http://localhost:8012），
将其 /chat 和 /stats 接口封装为 MCP tool。

传输方式：stdio（由 MCP Client 拉起子进程）

Tools:
  - health_consultation(query, patient_id)  端到端健康咨询（症状/疾病/科室/饮食/药物）
  - graph_stats()                            图谱节点统计（Disease/Symptom/Drug/...）
"""
import os  # 标准库：读取环境变量
import sys  # 标准库：sys.stderr 输出启动日志
from typing import Any  # 标准库：类型注解

# httpx 包：异步 HTTP 客户端，用于代理调用工单02（健康咨询）FastAPI 服务
import httpx
from dotenv import load_dotenv  # python-dotenv：加载 .env 文件
# mcp.server.fastmcp.FastMCP：MCP SDK 高级 Server，@mcp.tool() 注册工具
from mcp.server.fastmcp import FastMCP

# .env 位于 04-MCP/ 根目录
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

# 工单02 健康咨询服务的地址（默认 localhost:8012）
KNOWLEDGE_URL = os.getenv("KNOWLEDGE_API_URL", "http://localhost:8012")
# API Key（若工单02 启用鉴权）
API_KEY = os.getenv("API_KEY", "")
# 健康咨询涉及 LLM 推理，超时设置为 30s
TIMEOUT = float(os.getenv("MCP_UPSTREAM_TIMEOUT", "30.0"))

# 创建名为 "knowledge" 的 MCP Server 实例
mcp = FastMCP("knowledge")


def _headers() -> dict:
    """
    构造带可选鉴权头的请求头。

    Returns:
        包含 Content-Type 和可选 X-API-Key 的字典
    """
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY  # 若工单02 开启鉴权则传递
    return h


@mcp.tool()
async def health_consultation(query: str, patient_id: str = "default_patient") -> dict[str, Any]:
    """
    健康咨询：症状分析、疾病百科、科室推荐、饮食建议、药物用法等。

    通过 POST /chat 调用工单02（Neo4j 知识图谱 + LLM）实现。

    Args:
        query: 用户问题，例如"百日咳有什么症状？"、"我头疼该挂哪个科？"
        patient_id: 患者标识（用于记忆功能，可用默认值）

    Returns:
        {
            "ok": bool,
            "reply": str,           # 自然语言回答
            "intent": str,          # 识别意图，如 disease_to_symptom
            "entity": str,          # 抽取的实体，如"百日咳"
            "graph_data": list,     # 图谱可视化数据（前端可选用）
            "elapsed_ms": int,
        }
        失败时返回 { "ok": False, "error": str }
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # POST /chat 发起咨询请求
            r = await client.post(
                f"{KNOWLEDGE_URL}/chat",
                json={"query": query, "patient_id": patient_id},  # 请求体
                headers=_headers(),
            )
            r.raise_for_status()  # HTTP 非 200 时抛出 HTTPStatusError
            data = r.json()
            return {"ok": True, **data}  # 将工单02返回字段展开合并
    except httpx.HTTPStatusError as e:
        # 上游返回 4xx/5xx 错误
        return {"ok": False, "error": f"上游返回 {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.RequestError as e:
        # 网络连接失败（工单02 未启动、端口不通等）
        return {"ok": False, "error": f"无法连接健康咨询服务 {KNOWLEDGE_URL}: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"未知错误: {e}"}


@mcp.tool()
async def graph_stats() -> dict[str, Any]:
    """
    获取医疗知识图谱节点统计（前端侧边栏展示用）。

    通过 GET /stats 调用工单02，返回各类节点数量。

    Returns:
        {
            "ok": bool,
            "Disease": int,     # 疾病节点数
            "Symptom": int,     # 症状节点数
            "Drug": int,        # 药物节点数
            "Department": int,  # 科室节点数
            "Food": int,        # 食物节点数
            "Transmission": int,# 传播途径节点数
            "relations": int    # 关系总数
        }
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # GET /stats 获取图谱统计
            r = await client.get(f"{KNOWLEDGE_URL}/stats", headers=_headers())
            r.raise_for_status()
            return {"ok": True, **r.json()}
    except httpx.RequestError as e:
        return {"ok": False, "error": f"无法连接健康咨询服务: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"未知错误: {e}"}


if __name__ == "__main__":
    # 打印上游地址到 stderr，确认配置
    print(f"[knowledge_mcp] 上游: {KNOWLEDGE_URL}", file=sys.stderr)
    # 以 stdio 传输方式启动 MCP Server
    mcp.run(transport="stdio")
