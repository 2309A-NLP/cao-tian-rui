"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
API 端点单元测试（TestClient，不需要真实 VLM 或网络）

关键设计：
- 用 patcher.start() / stop() 确保整个测试方法期间 mock 生效
- 用最小合法 JPEG 作为上传图片，通过 validate_image 校验
- 不依赖任何外部服务（Neo4j / ChromaDB / SiliconFlow）

运行：cd 13--工单13 && python -m pytest tests/test_api.py -v
"""

# io：Python 内置模块，io.BytesIO 用于将字节数据包装为内存文件对象（multipart 上传所需）
import io

# json：Python 内置模块（此文件间接使用，保留）
import json

# sys：Python 内置模块，用于修改模块搜索路径（让 pytest 能找到 src 包）
import sys

# pathlib.Path：面向对象的文件路径操作
from pathlib import Path

# unittest.mock：Python 内置 mock 测试工具
# MagicMock：自动创建 mock 对象（模拟任意对象的属性和方法）
# patch：上下文管理器/装饰器，临时替换模块中的对象为 mock
from unittest.mock import MagicMock, patch

# pytest：Python 最流行的测试框架，提供参数化、fixture、断言失败信息等增强功能
# 安装方式：pip install pytest
import pytest

# 将项目根目录（backend/）添加到模块搜索路径，确保 "from src.xxx import" 能正常工作
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 最小合法 JPEG（PIL 可解析，validate_image 可通过）────────────────────────
# 这是一个 1×1 白色像素的 JPEG 文件，base64 编码
# 用于单元测试中需要上传图片但不关心图片内容的场景
# 1×1 像素满足 MIN_IMAGE_RESOLUTION=32... 注意：实际上 1<32，
# 所以在 test 中我们同时 mock 了 validate_image，这个 JPEG 只用于通过 PIL 解析
import base64
_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoH"
    "BwYIDAoMCwsKCwsNCxAQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/wAARC"
    "AABAAEDASIA AhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
    "AAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAA"
    "AAAAAAAA/9oADAMBAAIRAxEAPwCwABmX/9k="
)
# base64.b64decode：将 base64 字符串解码为原始字节数据
# replace 是为了去掉换行和空格（base64 数据中不应有这些字符）
_MIN_JPEG = base64.b64decode(_JPEG_B64.replace("\n", "").replace(" ", ""))

# ── 1×1 透明 PNG（备用）─────────────────────────────────────────────────────
# 手工构造的最小 PNG 文件字节序列，包含 PNG 签名、IHDR、IDAT、IEND 四个数据块
# 每个十六进制字节注释说明其含义
_MIN_PNG = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG 签名（固定 8 字节）
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk：长度(13) + 类型("IHDR")
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 宽度=1, 高度=1（各 4 字节大端序）
    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,  # 位深=8, 颜色类型=2(RGB) + CRC 前缀
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk：长度(12) + 类型("IDAT")
    0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,  # IDAT 数据（zlib 压缩的像素数据）
    0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,  # IDAT 数据续 + CRC
    0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk：长度(0) + 类型("IEND")
    0x44, 0xAE, 0x42, 0x60, 0x82,                     # IEND CRC（固定值）
])


def _jpeg_file(name="test.jpg"):
    """
    构造 multipart 上传所需的三元组：(文件名, BytesIO对象, Content-Type)。

    参数：
        name (str)：上传时使用的文件名

    返回值：
        tuple：(文件名字符串, io.BytesIO, MIME类型字符串)
    """
    return (name, io.BytesIO(_MIN_JPEG), "image/jpeg")


# ════════════════════════════════════════════════
# /health 端点测试
# ════════════════════════════════════════════════

class TestHealthEndpoint:
    """测试 /health 端点的基本功能。"""

    def test_health_returns_200(self):
        """
        health() 内部 try/except 保证 rag_store 不可用时也能返回 200。
        不需要 mock — 直接请求，let the handler default kb=0。

        验证：即使 ChromaDB 已初始化，也能通过 mock 正常返回健康状态。
        """
        # FastAPI 内置的测试客户端，不需要启动真实 HTTP 服务器
        from fastapi.testclient import TestClient
        from src.api import app

        # mock rag_store.count_docs 避免 ChromaDB 连接失败导致测试崩溃
        # patch 语法：patch("模块路径.函数名", return_value=返回值)
        with patch("src.rag_store.count_docs", return_value=5):
            # raise_server_exceptions=False：服务器端异常不会在客户端直接抛出
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")

        # 断言 HTTP 状态码为 200
        assert resp.status_code == 200
        data = resp.json()
        # 断言必要字段存在
        assert data["status"] == "ok"
        assert "workorder_id" in data
        assert "vlm_backend" in data

    def test_health_kb_docs_is_int(self):
        """
        kb_docs 字段应为整数，即使 rag_store 不可用也应默认为 0（health 方法内有 try/except）。
        """
        from fastapi.testclient import TestClient
        from src.api import app

        # 模拟 rag_store.count_docs 抛出异常（如 ChromaDB 未安装）
        # side_effect=Exception("...")：调用时抛出指定异常
        with patch("src.rag_store.count_docs", side_effect=Exception("no chroma")):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")

        assert resp.status_code == 200
        # isinstance 检查类型：kb_docs 应为 int（健康检查 catch 了异常后 default 为 0）
        assert isinstance(resp.json()["kb_docs"], int)


# ════════════════════════════════════════════════
# 基类：管理 handler mock 的生命周期
# ════════════════════════════════════════════════

class _BaseHandlerTest:
    """
    Handler 测试基类，提供 mock Handler 的创建和清理机制。

    使用 patcher.start() / teardown_method 中 stop()，
    确保 mock 在整个测试方法执行期间生效（包括 FastAPI 内部异步调度）。
    子类在 setup_method 中调用 _start_mock_handlers()，
    在 teardown_method 中调用 _stop_mock_handlers()。
    """

    def _start_mock_handlers(self, vqa_answer=None, mrg_result=None, rag_result=None):
        """
        创建 mock Handler 对象并启动 patch，注册到 self.client。

        参数：
            vqa_answer (str | None)：VQA handler.run() 的返回值，默认为固定答案字符串
            mrg_result (tuple | None)：MRG handler.run() 的返回值 (ReportSchema, str)
            rag_result (tuple | None)：RAG handler.run() 的返回值 (str, list[RefDoc])
        """
        from src.models import ReportSchema
        from fastapi.testclient import TestClient

        # 创建 VQA Mock Handler
        self._mock_vqa = MagicMock()
        # return_value：设置 mock 方法被调用时的返回值
        self._mock_vqa.run.return_value = vqa_answer or "这是一张胸部 X 光影像，未见明显异常。"

        # 创建 MRG Mock Handler（返回值是 (ReportSchema, 文本) 元组）
        default_report = ReportSchema(
            chief_complaint="测试主诉",
            findings="双肺纹理清晰",
            impression="未见急性病变",
            recommendation="建议随访复查",
        )
        self._mock_mrg = MagicMock()
        if mrg_result is not None:
            self._mock_mrg.run.return_value = mrg_result  # 使用自定义返回值
        else:
            self._mock_mrg.run.return_value = (
                default_report,
                "## 医疗影像诊断报告\n**影像所见**\n双肺纹理清晰",
            )

        # 创建 RAG Mock Handler（返回值是 (回答文本, [RefDoc列表]) 元组）
        self._mock_rag = MagicMock()
        self._mock_rag.run.return_value = rag_result or (
            "根据影像判断，未见明显异常。本回答仅供参考。", []
        )

        # 保存 mock 对象的本地引用（供内部闭包使用）
        mock_vqa = self._mock_vqa
        mock_mrg = self._mock_mrg
        mock_rag = self._mock_rag

        def _patched_get_handler(task):
            """替换 api.get_handler 的函数：根据任务类型返回对应 mock Handler。"""
            from src.models import TaskType
            from fastapi import HTTPException

            mapping = {TaskType.VQA: mock_vqa, TaskType.MRG: mock_mrg, TaskType.RAG: mock_rag}

            # 未知任务类型抛出 HTTPException（与真实 get_handler 行为一致）
            if task not in mapping:
                raise HTTPException(
                    status_code=501,
                    detail={"error_code": "NOT_IMPLEMENTED", "message": f"任务 {task} 未实现"},
                )
            return mapping[task]

        # patch get_handler：将 src.api 模块中的 get_handler 替换为 _patched_get_handler
        self._patcher = patch("src.api.get_handler", side_effect=_patched_get_handler)

        # patch validate_image：让图像校验直接返回合法结果，不实际解析图片字节
        # 这样单元测试只关心 API 逻辑层，不关心图片真实性
        self._patcher_img = patch(
            "src.api.validate_image", return_value=("jpeg", (640, 480))
        )

        # start()：启动 patch（返回 mock 对象，此处不需要返回值）
        self._patcher.start()
        self._patcher_img.start()

        from src.api import app
        # 创建测试客户端（此时 get_handler 和 validate_image 已被替换为 mock）
        self.client = TestClient(app, raise_server_exceptions=False)

    def _stop_mock_handlers(self):
        """
        停止所有 patch，恢复被替换的函数（测试结束后必须调用）。
        不调用 stop() 会导致 mock 持续影响后续测试。
        """
        if hasattr(self, "_patcher"):
            self._patcher.stop()
        if hasattr(self, "_patcher_img"):
            self._patcher_img.stop()

    def _post(self, task, query=None, image=None, content_type="image/jpeg"):
        """
        发送一次 /api/analyze 请求的辅助方法，封装了 multipart 构造。

        参数：
            task (str)：任务类型（"vqa"/"mrg"/"rag"/其他）
            query (str | None)：用户问题，None 表示不传
            image (bytes | None)：图片字节数据，None 时使用默认最小 JPEG
            content_type (str)：图片的 MIME 类型，默认 "image/jpeg"

        返回值：
            requests.Response：HTTP 响应对象（可调用 .status_code 和 .json()）
        """
        img_bytes = image or _MIN_JPEG  # 默认使用最小 JPEG
        files = {"image": ("test.jpg", io.BytesIO(img_bytes), content_type)}
        data = {"task": task}
        if query is not None:
            data["query"] = query
        return self.client.post("/api/analyze", data=data, files=files)


# ════════════════════════════════════════════════
# /api/analyze — 输入校验测试
# ════════════════════════════════════════════════

class TestAnalyzeInputValidation(_BaseHandlerTest):
    """测试 /api/analyze 接口的输入校验逻辑。"""

    def setup_method(self):
        """每个测试方法执行前调用，初始化 mock Handler。"""
        self._start_mock_handlers()

    def teardown_method(self):
        """每个测试方法执行后调用，清理 mock Handler（防止影响后续测试）。"""
        self._stop_mock_handlers()

    def test_invalid_task_returns_422(self):
        """task 参数不在 TaskType 枚举内 → 应返回 422 INVALID_TASK。"""
        resp = self._post("not_a_task", query="问题")
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "INVALID_TASK"

    def test_vqa_missing_query_returns_400(self):
        """VQA 任务不传 query 参数 → 应返回 400 MISSING_QUERY。"""
        resp = self._post("vqa")  # 不传 query
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "MISSING_QUERY"

    def test_rag_missing_query_returns_400(self):
        """RAG 任务不传 query 参数 → 应返回 400 MISSING_QUERY。"""
        resp = self._post("rag")  # 不传 query
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "MISSING_QUERY"

    def test_vqa_empty_query_returns_400(self):
        """VQA 任务 query 为纯空白字符串 → 应返回 400 MISSING_QUERY（strip 后为空）。"""
        resp = self._post("vqa", query="   ")  # 三个空格，strip 后为空字符串
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "MISSING_QUERY"

    def test_unsupported_mime_returns_415(self):
        """上传 image/gif 格式（不在白名单内）→ 应返回 415 UNSUPPORTED_FORMAT。"""
        # content_type 设为 image/gif，API 会在 validate_image 之前先检查 MIME
        resp = self._post("vqa", query="问题", content_type="image/gif")
        assert resp.status_code == 415
        assert resp.json()["error_code"] == "UNSUPPORTED_FORMAT"

    def test_mrg_no_query_is_ok(self):
        """MRG 任务不需要 query，不传 query 时应正常返回 200（不报 MISSING_QUERY）。"""
        resp = self._post("mrg")  # MRG 不传 query
        assert resp.status_code == 200

    def test_classification_returns_501(self):
        """CV 预留任务 classification → 应返回 501 NOT_IMPLEMENTED。"""
        # classification 在枚举中存在，但 get_handler 会抛 HTTPException(501)
        resp = self._post("classification", query="问题")
        assert resp.status_code == 501


# ════════════════════════════════════════════════
# /api/analyze — 正常路径测试
# ════════════════════════════════════════════════

class TestAnalyzeHappyPath(_BaseHandlerTest):
    """测试 /api/analyze 接口在正常输入下的响应格式和内容。"""

    def setup_method(self):
        """每个测试方法执行前初始化 mock Handler。"""
        self._start_mock_handlers()

    def teardown_method(self):
        """每个测试方法执行后清理 mock Handler。"""
        self._stop_mock_handlers()

    def test_vqa_returns_answer(self):
        """VQA 正常请求 → HTTP 200，响应包含非空 answer，task 字段正确。"""
        resp = self._post("vqa", query="这张影像是什么？")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task"] == "vqa"       # task 字段应为 "vqa"
        assert body["answer"] != ""        # answer 不能为空
        assert "request_id" in body        # 每次请求必须有 request_id
        assert "latency_ms" in body        # 必须包含耗时信息

    def test_mrg_returns_structured_report(self):
        """MRG → HTTP 200，响应包含 report 对象且含四个必要字段。"""
        resp = self._post("mrg")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task"] == "mrg"

        rpt = body.get("report")
        assert rpt is not None, "report 字段缺失"  # assert 失败时打印自定义消息

        # 逐一检查四个结构化报告字段
        for field in ("chief_complaint", "findings", "impression", "recommendation"):
            assert field in rpt, f"report.{field} 缺失"

    def test_rag_returns_answer_and_references(self):
        """RAG → HTTP 200，响应包含 answer 字符串和 references 列表。"""
        resp = self._post("rag", query="肺部有异常吗？")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task"] == "rag"
        assert "answer" in body
        assert isinstance(body["references"], list)  # isinstance：检查类型是否为 list

    def test_response_has_disclaimer(self):
        """所有任务的响应中都应包含非空的 disclaimer 字段（医疗免责声明）。"""
        resp = self._post("vqa", query="问题")
        body = resp.json()
        assert "disclaimer" in body
        assert body["disclaimer"] != ""

    def test_request_id_unique_per_call(self):
        """两次独立请求的 request_id 应不同（UUID 唯一性验证）。"""
        r1 = self._post("mrg").json()
        r2 = self._post("mrg").json()
        assert r1["request_id"] != r2["request_id"]  # 两次请求的 ID 不能相同

    def test_vqa_handler_called_once(self):
        """VQA 请求应恰好调用一次 handler.run()（验证不会多次调用）。"""
        self._post("vqa", query="检查问题")
        # assert_called_once()：断言 mock 方法被调用了恰好一次
        self._mock_vqa.run.assert_called_once()

    def test_mrg_handler_passes_extra_context(self):
        """MRG 携带 query 时，API 应将 query 作为 extra_context 传给 handler.run()。"""
        self._post("mrg", query="68岁男性，肺结节随访")

        # call_args：最后一次调用的参数，包含 args（位置参数）和 kwargs（关键字参数）
        call_kwargs = self._mock_mrg.run.call_args

        # 兼容两种调用方式：
        # 1. handler.run(image_b64=..., extra_context="68岁男性，肺结节随访", ...)
        # 2. handler.run(image_b64, "68岁男性，肺结节随访", ...)
        extra = (
            call_kwargs.kwargs.get("extra_context")
            or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
        )
        assert extra == "68岁男性，肺结节随访"


# ════════════════════════════════════════════════
# /api/analyze — VLMError 模拟测试
# ════════════════════════════════════════════════

class TestAnalyzeVLMErrors:
    """测试 VLMError 异常应正确映射到对应 HTTP 状态码。"""

    def _post_with_vlm_error(self, error_code: str):
        """
        构造一个会抛出指定 error_code 的 VLMError 的测试请求。

        参数：
            error_code (str)：VLMError 的错误码（如 "VLM_TIMEOUT"/"VLM_API_ERROR"）

        返回值：
            requests.Response：测试 HTTP 响应
        """
        from fastapi.testclient import TestClient
        from src.vlm_client import VLMError

        # 创建会抛出 VLMError 的 mock VQA Handler
        mock_vqa = MagicMock()
        # side_effect：设置调用时抛出的异常
        mock_vqa.run.side_effect = VLMError(f"模拟: {error_code}", code=error_code)

        def _patched(task):
            """仅处理 VQA 任务的 mock get_handler。"""
            from src.models import TaskType
            if task == TaskType.VQA:
                return mock_vqa
            raise NotImplementedError

        # 启动两个 patch
        patcher = patch("src.api.get_handler", side_effect=_patched)
        patcher_img = patch("src.api.validate_image", return_value=("jpeg", (640, 480)))
        patcher.start()
        patcher_img.start()
        try:
            from src.api import app
            client = TestClient(app, raise_server_exceptions=False)
            files = {"image": ("t.jpg", io.BytesIO(_MIN_JPEG), "image/jpeg")}
            resp = client.post(
                "/api/analyze", data={"task": "vqa", "query": "测试"}, files=files
            )
        finally:
            # 无论测试是否成功，都必须停止 patch（try/finally 确保清理）
            patcher.stop()
            patcher_img.stop()
        return resp

    def test_vlm_timeout_returns_504(self):
        """VLM 超时错误（VLM_TIMEOUT）→ 应映射到 HTTP 504 Gateway Timeout。"""
        resp = self._post_with_vlm_error("VLM_TIMEOUT")
        assert resp.status_code == 504
        assert resp.json()["error_code"] == "VLM_TIMEOUT"

    def test_vlm_api_error_returns_503(self):
        """VLM API 错误（非超时）→ 应映射到 HTTP 503 Service Unavailable。"""
        resp = self._post_with_vlm_error("VLM_API_ERROR")
        assert resp.status_code == 503
        assert resp.json()["error_code"] == "VLM_API_ERROR"

    def test_error_response_has_request_id(self):
        """即使 VLM 出错，错误响应中也应包含 request_id（便于日志追踪）。"""
        resp = self._post_with_vlm_error("VLM_TIMEOUT")
        assert "request_id" in resp.json()
