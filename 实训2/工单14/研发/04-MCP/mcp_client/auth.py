"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-MCP

API Key 校验中间件。
.env 中 API_KEY 留空 = 开发模式（不校验）；填值 = 生产模式（校验 X-API-Key header）。
"""
import os  # 标准库：读取环境变量 API_KEY

from fastapi import Header, HTTPException, status
# fastapi.Header: 声明从 HTTP 请求头读取参数的依赖项注入标记
# fastapi.HTTPException: 抛出带状态码的 HTTP 错误
# fastapi.status: 提供 HTTP 状态码常量（如 HTTP_401_UNAUTHORIZED = 401）


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    FastAPI 依赖函数：校验请求头 X-API-Key。

    工作逻辑：
      - 若 .env 中 API_KEY 未配置（空字符串）：开发模式，直接放行
      - 若已配置 API_KEY：
          - 请求头未携带 X-API-Key 或值不匹配 → 抛出 401 Unauthorized
          - 值匹配 → 放行（返回 None，FastAPI 不阻断后续处理）

    Args:
        x_api_key: FastAPI 自动从请求头 "X-API-Key" 提取，未提供时为 None

    Raises:
        HTTPException(401): API Key 无效或缺失时抛出
    """
    # 读取服务端期望的 API Key（从环境变量，默认空字符串）
    expected = os.getenv("API_KEY", "")
    if not expected:
        # 开发模式：API_KEY 未配置，跳过鉴权直接返回
        return
    # 生产模式：请求头缺失或与期望值不符，返回 401
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,  # HTTP 401
            detail="Invalid or missing X-API-Key",  # 错误说明（返回给调用方）
        )
