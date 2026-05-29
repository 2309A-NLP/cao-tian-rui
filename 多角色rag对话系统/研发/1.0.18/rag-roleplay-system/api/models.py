# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
请求/响应数据模型

定义所有 API 接口的请求体和响应体结构

⚠️ 常改动的地方：
1. 如果接口需要新增字段（如添加用户头像、对话标签等），在此处添加模型字段
2. 字段的默认值（如 ChatRequest 中 role 默认为 "friend"）可根据业务需求调整
3. 字段的可选性（Optional）如需修改，请同步调整前端调用

⚠️ 注意事项：
1. 使用 Pydantic BaseModel，自动进行类型校验和文档生成
2. 字段类型提示有助于 FastAPI 自动生成 API 文档
3. 与 storage/mysql_client 中返回的字典结构需保持一致
"""

from typing import Optional, List, Dict
from pydantic import BaseModel


class RegisterRequest(BaseModel):
    """注册请求模型"""
    username: str                       # 用户名，必填
    password: str                       # 密码（明文，前端应做好哈希，但本系统在后端哈希）
    email: Optional[str] = None         # 邮箱，可选


class LoginRequest(BaseModel):
    """登录请求模型"""
    username: str                       # 用户名
    password: str                       # 密码（后端会进行哈希比对）


class ChatRequest(BaseModel):
    """
    对话请求模型
    ⚠️ 常改动：
        1. role 的默认值 "friend" 可改为其他角色或从 config 读取
        2. 若需增加温度参数或检索参数，可在此扩展字段
    """
    user_id: int                        # 用户 ID
    username: str                       # 用户名（用于日志/记忆）
    message: str                        # 用户输入的消息
    role: str = "friend"                # 角色类型：friend, doctor, lawyer, tcm, psychologist, finance
    conversation_id: Optional[str] = None   # 对话 ID，若为空则后端生成


class SourceItem(BaseModel):
    """
    来源项模型（用于前端展示引用来源）
    """
    file: str                           # 来源文件名，格式如《民法典》
    preview: str                        # 内容预览（前200字符）


class ChatResponse(BaseModel):
    """
    对话响应模型（非流式接口）
    ⚠️ 常改动：如需返回其他元信息（如检索耗时、记忆条数），可在此添加字段
    """
    code: int = 200                     # 状态码，200 表示成功
    answer: str = ""                    # 模型生成的回答
    sources: List[SourceItem] = []      # 引用来源列表
    conversation_id: str = ""           # 对话 ID（用于后续关联）


class LoginResponse(BaseModel):
    """登录响应模型"""
    code: int                           # 状态码，200/401/400
    message: str                        # 提示信息
    user: Optional[Dict] = None         # 用户信息字典（id, username, nickname, email, status）


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str                         # "ok" 或 "error"