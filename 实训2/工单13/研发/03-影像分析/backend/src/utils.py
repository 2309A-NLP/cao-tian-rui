"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
通用工具：图像校验、请求ID生成。

本模块提供两类工具函数：
1. 图像校验：验证上传图片的大小、格式和分辨率是否符合要求
2. 请求ID生成：为每个 API 请求生成全局唯一的追踪 ID
"""

# io：Python 内置模块，提供内存中的字节流操作
# io.BytesIO 可以将 bytes 包装成类文件对象，供 PIL 读取
import io

# uuid：Python 内置模块，用于生成全局唯一标识符（UUID）
import uuid

# Optional：类型提示，表示可以为 None
from typing import Optional

# PIL（Pillow）：Python 最流行的图像处理库
# PIL 是原始库名，Pillow 是其维护良好的分叉版本
# Image：核心图像操作类（打开、验证、读取格式/尺寸等）
# UnidentifiedImageError：PIL 无法识别文件格式时抛出的异常
# 安装方式：pip install Pillow
from PIL import Image, UnidentifiedImageError

# 导入图像校验相关配置常量
from .config import (
    MAX_IMAGE_SIZE_BYTES,      # 最大文件大小（字节）
    MIN_IMAGE_RESOLUTION,      # 最小分辨率（像素）
    SUPPORTED_IMAGE_FORMATS,   # 支持的图片格式集合
)


class ImageValidationError(Exception):
    """
    图像校验失败时抛出的自定义异常。

    携带额外信息：
    - code：机器可读的错误码字符串（如 "IMAGE_TOO_LARGE"）
    - status：对应的 HTTP 状态码（如 413、415、422）

    这样 API 层可以直接从异常中取出 HTTP 状态码和错误码，构造标准化错误响应。
    """

    def __init__(self, message: str, code: str, status: int = 422):
        """
        参数：
            message (str)：人类可读的错误描述（中文）
            code (str)：机器可读的错误码（大写下划线风格，如 "IMAGE_TOO_LARGE"）
            status (int)：对应的 HTTP 状态码，默认 422（Unprocessable Entity）
        """
        super().__init__(message)  # 调用父类 Exception 的初始化，设置异常消息
        self.code = code           # 错误码
        self.status = status       # HTTP 状态码


def new_request_id() -> str:
    """
    生成一个全局唯一的请求标识符（UUID v4 的十六进制字符串）。

    返回值：
        str：32 位十六进制字符串（如 "a3f8c2d1..."），不含连字符
    """
    # uuid4()：基于随机数生成 UUID，碰撞概率极低
    # .hex：去掉连字符后的 32 位十六进制字符串
    return uuid.uuid4().hex


def validate_image(data: bytes, filename: Optional[str] = None) -> tuple[str, tuple[int, int]]:
    """
    校验图片字节数据的合法性，返回图片格式和尺寸信息。

    校验顺序：
    1. 文件非空
    2. 文件大小不超过上限
    3. PIL 可以解析（格式合法）
    4. 格式在白名单内（JPEG/PNG）
    5. 分辨率不低于最小要求

    参数：
        data (bytes)：从 HTTP 请求中读取的图片字节数据
        filename (Optional[str])：原始文件名（目前仅用于错误日志，不影响校验逻辑）

    返回值：
        tuple[str, tuple[int, int]]：(格式小写字符串, (宽度, 高度))
        例如：("jpeg", (1024, 768))

    异常：
        ImageValidationError：任意校验条件不满足时抛出，携带对应错误码和 HTTP 状态码
    """
    # 校验1：文件不能为空
    if not data:
        raise ImageValidationError("图片文件为空", code="MISSING_FIELD", status=400)

    # 校验2：文件大小不能超过上限（默认 20MB）
    if len(data) > MAX_IMAGE_SIZE_BYTES:
        raise ImageValidationError(
            f"图片过大，最大 {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB，当前 {len(data) / (1024 * 1024):.1f}MB",
            code="IMAGE_TOO_LARGE",
            status=413,  # 413 Payload Too Large
        )

    # 校验3：使用 PIL 验证图片数据完整性
    # Image.open()：打开图片，不立即完全解码（懒加载）
    # io.BytesIO(data)：将字节数据包装为内存文件对象，供 PIL 读取
    # img.verify()：验证图片文件头和数据完整性，不完全解码
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # verify() 会检查文件完整性，但执行后图像对象不可再用
    except (UnidentifiedImageError, Exception) as e:  # noqa: BLE001
        # UnidentifiedImageError：PIL 无法识别文件格式（非图片文件伪装成图片）
        # Exception：其他解析错误（文件损坏等）
        raise ImageValidationError(f"图片解析失败: {e}", code="IMAGE_INVALID", status=422) from e

    # 注意：verify() 执行后图像对象不可再使用，必须重新打开才能读取格式和尺寸
    # 使用 with 语句确保文件对象正确关闭
    with Image.open(io.BytesIO(data)) as img:
        # 获取 PIL 识别的图片格式（大写，如 "JPEG"、"PNG"）
        fmt = (img.format or "").upper()

        # 校验4：格式必须在白名单内（SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "JPG"}）
        if fmt not in SUPPORTED_IMAGE_FORMATS:
            raise ImageValidationError(
                f"不支持的图片格式: {fmt}，仅支持 {'/'.join(sorted(SUPPORTED_IMAGE_FORMATS))}",
                code="UNSUPPORTED_FORMAT",
                status=415,  # 415 Unsupported Media Type
            )

        # 获取图片宽高（像素）
        w, h = img.size  # img.size 返回 (width, height) 元组

    # 校验5：分辨率不能过低（过低的图片无法用于分析）
    if w < MIN_IMAGE_RESOLUTION or h < MIN_IMAGE_RESOLUTION:
        raise ImageValidationError(
            f"图片分辨率过低：{w}x{h}，要求 ≥ {MIN_IMAGE_RESOLUTION}x{MIN_IMAGE_RESOLUTION}",
            code="IMAGE_INVALID",
            status=422,
        )

    # 格式标准化：将 PIL 返回的格式统一为小写，并将 "JPG" 转为 "jpeg"
    # （HTTP Content-Type 和 data URI 中使用 "jpeg" 而非 "jpg"）
    fmt_lower = "jpeg" if fmt == "JPG" else fmt.lower()
    return fmt_lower, (w, h)
