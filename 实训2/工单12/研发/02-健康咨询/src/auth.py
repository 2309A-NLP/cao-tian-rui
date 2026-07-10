"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-健康咨询

API Key 鉴权模块。
- API_KEY 未配置（空）→ 开发模式，不校验，本地调试方便
- API_KEY 已配置 → 请求头必须携带 X-API-Key: <key>
"""
import os  # 标准库：读取环境变量 API_KEY

# fastapi 包：HTTPException 用于返回 HTTP 错误响应；Security 是依赖注入的安全专用版本
from fastapi import HTTPException, Security
# APIKeyHeader：FastAPI 内置的 API Key 请求头提取器
# 会自动从请求头中读取指定字段名（X-API-Key）的值
from fastapi.security import APIKeyHeader

# 从环境变量读取 API Key，未配置时为空字符串（开发模式）
_API_KEY = os.getenv("API_KEY", "")

# 创建 API Key 请求头提取方案：
# name="X-API-Key"：从此请求头读取密钥
# auto_error=False：找不到请求头时不自动报错，而是传入 None，让 verify_api_key 手动处理
_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_header_scheme)) -> None:
    """
    FastAPI 依赖项函数：校验请求头中的 API Key。

    通过 Depends(verify_api_key) 挂载到端点后，每个请求都会先执行此函数。

    参数:
        api_key (str): 由 _header_scheme 自动从请求头 X-API-Key 中提取的值；
                       未携带该请求头时为 None

    异常:
        HTTPException(401): API_KEY 已配置但请求未携带正确的密钥

    开发模式（API_KEY 为空）：
        直接 return，不做任何校验
    """
    if not _API_KEY:
        return  # 开发模式：API_KEY 未配置则跳过校验，方便本地调试

    # 生产模式：严格比对请求头中的密钥
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="无效的 API Key")
