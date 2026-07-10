"""
API Key 鉴权模块。

工作逻辑：
  - API_KEY 未配置（空字符串）→ 开发模式，跳过校验，方便本地调试
  - API_KEY 已配置         → 客户端请求头必须携带 X-API-Key: <key>，否则返回 401

依赖的第三方库说明：
  - fastapi：高性能 Python Web 框架，基于 Starlette，支持异步和类型注解
  - fastapi.security.APIKeyHeader：FastAPI 内置的 API Key 提取工具，
    从 HTTP 请求头中读取指定名称的字段值
"""
import os   # 标准库：读取环境变量 API_KEY

# FastAPI：流行的 Python Web 框架，用于快速构建 RESTful API
# HTTPException：抛出 HTTP 错误响应（如 401 Unauthorized）
# Security：FastAPI 的依赖注入标记，专门用于安全/鉴权类依赖
from fastapi import HTTPException, Security
# APIKeyHeader：从 HTTP 请求头中提取 API Key 的安全方案类
from fastapi.security import APIKeyHeader

# 从环境变量读取预设的 API Key，若未配置则为空字符串（开发模式）
_API_KEY = os.getenv("API_KEY", "")

# 声明从请求头 "X-API-Key" 字段读取 key
# auto_error=False：若请求头中没有该字段，不自动抛 401，而是返回 None（由业务逻辑决定如何处理）
_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_header_scheme)) -> None:
    """
    FastAPI 依赖注入函数，验证请求携带的 API Key 是否有效。

    用法：在路由装饰器中声明 dependencies=[Depends(verify_api_key)]，
    FastAPI 会在处理每个请求前自动调用此函数。

    参数：
      api_key : 从请求头 X-API-Key 提取的值（由 _header_scheme 自动解析）

    返回：None（校验通过时无返回值）
    抛出：HTTPException(401) 校验失败时
    """
    if not _API_KEY:
        return  # 开发模式：未配置密钥，直接放行所有请求（不做鉴权）
    # 生产模式：对比请求携带的 key 与服务器预设的 key
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="无效的 API Key")  # 返回 401 Unauthorized
