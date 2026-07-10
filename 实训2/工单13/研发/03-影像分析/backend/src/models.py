"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
Pydantic 数据模型：请求/响应/错误。

本模块定义了整个 API 的数据结构，使用 Pydantic 库实现：
- 自动数据校验（类型不符时自动报错）
- 自动生成 OpenAPI/Swagger 文档
- 自动序列化为 JSON

Pydantic：Python 最流行的数据校验库，通过类型注解实现运行时数据验证
安装方式：pip install pydantic
"""

# Enum：Python 内置枚举基类，用于定义一组命名常量
from enum import Enum

# Optional：类型提示，表示该字段可以为 None
from typing import Optional

# BaseModel：Pydantic 的模型基类，继承它即可获得数据校验和 JSON 序列化能力
# Field：用于为模型字段添加描述、默认值、验证规则等元信息
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """
    任务类型枚举，定义本系统支持（或预留）的所有影像分析任务。

    继承自 str 使得枚举值可以直接当字符串使用（如 task.value == "vqa"）。
    """
    VQA = "vqa"             # Visual Question Answering：视觉问答，回答关于影像的问题
    MRG = "mrg"             # Medical Report Generation：医疗影像报告生成
    RAG = "rag"             # Retrieval Augmented Generation：检索增强生成，结合知识库回答
    # 以下三种为 CV（计算机视觉）方向预留，本工单（NLP）不实现，返回 501
    CLASSIFICATION = "classification"  # 影像分类（预留）
    DETECTION = "detection"            # 目标检测（预留）
    SEGMENTATION = "segmentation"      # 语义分割（预留）


class RefDoc(BaseModel):
    """
    RAG 检索召回的单条参考文档模型。

    当 RAG 任务在向量库中检索到相关文档时，以此格式返回每条文档信息。
    """
    doc_id: str    # 文档在向量库中的唯一标识符
    title: str = ""  # 文档标题（如 "CT Chest - abnormality"），默认空字符串
    snippet: str   # 文档的文本片段（用于展示给用户的召回内容）
    score: float   # 相似度分值，范围 0~1，越高表示与查询越相关


class ReportSchema(BaseModel):
    """
    MRG 任务输出的结构化影像诊断报告模型。

    参照标准放射科报告格式，分为四个部分。
    """
    chief_complaint: str = Field(default="", description="主诉")
    # 主诉：患者的主要症状或检查目的（如"胸痛2天"）

    findings: str = Field(default="", description="影像所见")
    # 影像所见：对影像中可观察到的解剖结构、信号、异常的客观描述

    impression: str = Field(default="", description="印象")
    # 印象（结论）：基于影像所见的诊断意见（如"考虑肺炎，待排除肿瘤"）

    recommendation: str = Field(default="", description="建议")
    # 建议：后续临床处置建议（如"建议增强 CT 扫描"）


class AnalyzeResponse(BaseModel):
    """
    /api/analyze 接口成功时的响应模型。

    根据不同任务类型，部分字段可能为空：
    - VQA：answer 有值，report 为 None，references 为空列表
    - MRG：answer 为格式化文本，report 为结构化报告对象
    - RAG：answer 有值，references 包含召回的参考文档
    """
    request_id: str        # 请求唯一标识符（UUID hex），用于日志追踪
    task: TaskType          # 任务类型枚举值
    answer: str = ""        # 模型回答文本（所有任务都会有）
    references: list[RefDoc] = Field(default_factory=list)
    # 参考文档列表（仅 RAG 任务有值，其他任务为空列表）
    # default_factory=list 确保每个实例都有独立的列表对象（避免共享可变默认值）

    report: Optional[ReportSchema] = None
    # 结构化报告（仅 MRG 任务有值，其他任务为 None）
    # Optional[X] 等价于 Union[X, None]

    latency_ms: int = 0    # 总处理耗时（毫秒），含图像读取、VLM 调用等全部时间
    model: str = ""         # 实际使用的模型名称（来自 VLM_MODEL 配置）
    disclaimer: str = ""    # 医疗免责声明文本


class ErrorResponse(BaseModel):
    """
    /api/analyze 接口出错时的响应模型。

    用于 4xx/5xx 状态码的统一错误格式。
    """
    request_id: str    # 请求唯一标识符，便于用户反馈问题时定位日志
    error_code: str    # 错误代码字符串（如 "INVALID_TASK"、"IMAGE_TOO_LARGE"）
    message: str       # 人类可读的错误描述（中文）
    latency_ms: int = 0  # 发生错误时已消耗的时间（毫秒）


class HealthResponse(BaseModel):
    """
    /health 接口的响应模型，用于服务健康检查和状态监控。
    """
    status: str          # 服务状态，正常时为 "ok"
    workorder_id: str    # 工单标识符
    vlm_backend: str     # VLM 后端名称（如 "siliconflow"）
    vlm_model: str       # 当前配置的 VLM 模型名称
    kb_docs: int = 0     # 向量知识库中的文档总数，0 表示知识库未建立或为空
    version: str = "0.1.0"  # API 版本号
