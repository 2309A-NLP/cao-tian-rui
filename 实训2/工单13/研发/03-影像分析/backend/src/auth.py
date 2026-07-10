"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
API Key 鉴权。

本模块实现 HTTP 请求头鉴权逻辑：
- API_KEY 未配置（空）→ 开发模式，不校验，方便本地调试
- API_KEY 已配置 → 请求头必须携带 X-API-Key: <key>，否则返回 401

使用方式：在 FastAPI 路由中通过 Depends(verify_api_key) 注入。
"""

# os：Python 内置模块，用于读取操作系统环境变量
import os

# HTTPException：FastAPI 提供的 HTTP 异常类，抛出后 FastAPI 自动返回对应状态码响应
# Security：FastAPI 依赖注入系统的安全版本，用于声明安全依赖项
from fastapi import HTTPException, Security

# APIKeyHeader：FastAPI 提供的安全工具，用于从 HTTP 请求头中提取 API Key
from fastapi.security import APIKeyHeader

# 从环境变量读取 API Key，若未配置则默认为空字符串
# 空字符串表示开发模式，不进行鉴权
_API_KEY = os.getenv("API_KEY", "")

# 定义从请求头 "X-API-Key" 提取密钥的方案
# auto_error=False：若请求头不存在，不自动抛出 403，而是返回 None（由下方逻辑处理）
_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_header_scheme)) -> None:
    """
    API Key 校验依赖函数，由 FastAPI 的依赖注入系统自动调用。

    参数：
        api_key (str)：从请求头 X-API-Key 中提取的值，若请求头不存在则为 None

    返回值：
        None（校验通过时不返回任何值）

    异常：
        HTTPException(401)：API Key 配置了但请求中的值不匹配时抛出
    """
    # 若环境变量 API_KEY 未配置（为空），则跳过校验（开发模式）
    if not _API_KEY:
        return  # dev mode: 未配置密钥则跳过鉴权

    # API_KEY 已配置时，校验请求头中的值是否匹配
    # 若不匹配（包括请求头不存在、api_key 为 None 或值错误），返回 401
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="无效的 API Key")
