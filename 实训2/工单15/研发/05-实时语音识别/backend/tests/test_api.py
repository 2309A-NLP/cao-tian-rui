"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-实时语音识别、翻译与会议概要

MOCK 模式下的 API 集成测试。

测试策略：
  - 所有测试在 MOCK_MODE=true 下运行，不连接任何外部服务（讯飞/硅基流动）
  - 覆盖 REST API 全部路由的正常路径和异常路径
  - WS 实时流使用专项 test_ws_mock_stream 测试（通过内置 TestClient WS 支持）

运行方式：
  set MOCK_MODE=true   # Windows
  pytest tests/ -v
"""
import os  # 标准库：在 import app 之前设置测试环境变量

# 在 import src.app 之前设置环境变量，确保 config.py 读到正确配置
os.environ.setdefault("MOCK_MODE", "true")    # 强制 Mock 模式，不连外网
os.environ.setdefault("ASR_ENGINE", "mock")   # 显式指定 ASR 引擎为 mock

# pytest 包：Python 标准测试框架，提供测试发现、夹具（fixture）、参数化等功能
# 安装：pip install pytest
import pytest

# fastapi.testclient.TestClient：
# 基于 httpx（底层 ASGI 驱动），允许在同步测试函数中直接调用 FastAPI 应用
# 无需真正启动 HTTP 服务器，直接在进程内模拟请求/响应（类似 Django 的 Client）
# 安装：pip install httpx（TestClient 依赖 httpx）
from fastapi.testclient import TestClient

# 导入 FastAPI 应用实例（在设置环境变量之后导入，确保 config 读取到 MOCK_MODE=true）
from src.app import app

# 创建 TestClient 实例（全局共享，避免每次测试重新初始化 lifespan）
client = TestClient(app)


# ────────── 基础健康检查 ──────────

def test_health():
    """
    测试 GET /health 接口：
    - 状态码为 200
    - mock_mode 字段为 True（确认测试环境正确）
    - asr 字段为 "mock"
    """
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["mock_mode"] is True  # 确认 MOCK_MODE 已生效
    assert data["asr"] == "mock"      # 确认 ASR 引擎为 mock


# ────────── Agent 标准时序：start → stop → poll ──────────

def test_session_start():
    """
    测试 POST /api/session/start 接口：
    - 返回 task_id（格式：wt15-<12位hex>）
    - 返回 meeting_join_url（WebSocket 地址，以 ws:// 开头）
    - 初始 status 为 "CREATED"
    """
    r = client.post("/api/session/start", json={"lang": "zh_cn"})
    assert r.status_code == 200
    data = r.json()
    assert "task_id" in data
    assert data["task_id"].startswith("wt15-")        # 服务端生成的 task_id 前缀
    assert "meeting_join_url" in data
    assert data["meeting_join_url"].startswith("ws://")  # 测试环境为 HTTP，对应 ws://
    assert data["status"] == "CREATED"


def test_session_stop():
    """
    测试 POST /api/session/{task_id}/stop 接口：
    - 先创建会话
    - 调用 stop 后返回 status=stopped
    """
    r = client.post("/api/session/start", json={"lang": "zh_cn"})
    task_id = r.json()["task_id"]

    r2 = client.post(f"/api/session/{task_id}/stop")
    assert r2.status_code == 200
    assert r2.json()["status"] == "stopped"


def test_session_stop_unknown_returns_404():
    """
    测试对未知 session ID 调用 stop 接口：
    应返回 404（而非 500 或 200）
    """
    r = client.post("/api/session/no-such-id/stop")
    assert r.status_code == 404


# ────────── 回调接口测试 ──────────

def test_callback_success():
    """
    测试正常回调流程（CALLBACK_SECRET 为空，无需鉴权）：
    - 先创建会话
    - POST 回调包含完整 Result（Chapters + Summarization）
    - 返回 code="0"
    """
    r = client.post("/api/session/start", json={})
    task_id = r.json()["task_id"]

    payload = {
        "TaskId": task_id,
        "TaskStatus": "COMPLETED",
        "Result": {
            "Chapters": [{"Headline": "测试章节", "BeginTime": 0}],
            "Summarization": {"Paragraph": "测试摘要", "ActionItems": ["测试待办"]},
        },
    }
    # CALLBACK_SECRET 为空时无需鉴权（测试环境默认行为）
    r2 = client.post("/api/callback", json=payload)
    assert r2.status_code == 200
    assert r2.json()["code"] == "0"


def test_callback_rejects_wrong_secret(monkeypatch):
    """
    L-NEW-5：配置了 CALLBACK_SECRET 后，提供错误鉴权头应返回 401。
    monkeypatch：pytest 内置夹具，用于在测试中临时替换模块属性/函数，
    测试结束后自动恢复原值（无需手动清理）。
    """
    monkeypatch.setattr("src.config.CALLBACK_SECRET", "correct-secret")
    import importlib, src.app as _app
    importlib.reload(_app)  # 重新加载 app，让其读取修改后的 config.CALLBACK_SECRET

    r = client.post(
        "/api/callback",
        json={"TaskId": "x", "TaskStatus": "COMPLETED"},
        headers={"X-Callback-Secret": "wrong-secret"},  # 提供错误的 Secret
    )
    assert r.status_code == 401  # 应拒绝未授权请求


def test_callback_accepts_correct_secret(monkeypatch):
    """
    L-NEW-5：配置了 CALLBACK_SECRET 后，提供正确鉴权头应返回 200。
    """
    monkeypatch.setattr("src.config.CALLBACK_SECRET", "correct-secret")

    r = client.post("/api/session/start", json={})
    task_id = r.json()["task_id"]
    payload = {"TaskId": task_id, "TaskStatus": "COMPLETED", "Result": {}}
    r2 = client.post(
        "/api/callback",
        json=payload,
        headers={"X-Callback-Secret": "correct-secret"},  # 提供正确的 Secret
    )
    assert r2.status_code == 200


def test_callback_missing_task_id():
    """
    测试回调缺少 TaskId 字段：
    应返回 400（缺少必填字段）
    """
    r = client.post("/api/callback", json={"TaskStatus": "COMPLETED"})
    assert r.status_code == 400


def test_callback_invalid_payload():
    """
    测试回调 body 非 JSON 格式：
    应返回 400（无效请求体）
    """
    r = client.post("/api/callback", data="not-json")  # 发送非 JSON 字符串
    assert r.status_code == 400


# ────────── 轮询接口测试 ──────────

def test_poll_mock_returns_completed():
    """
    测试 GET /api/session/{task_id}/poll 接口：
    - 已创建的 session 应返回 TaskStatus 和 Result 字段（通义听悟兼容格式）
    """
    r = client.post("/api/session/start", json={"lang": "zh_cn"})
    task_id = r.json()["task_id"]

    r2 = client.get(f"/api/session/{task_id}/poll")
    assert r2.status_code == 200
    data = r2.json()
    # 验证通义听悟风格字段存在（大写格式）
    assert "TaskStatus" in data
    assert "Result" in data


def test_poll_unknown_session_mock_still_ok():
    """
    MOCK 模式下对未知 session ID 轮询：
    应返回 200 + TaskStatus=COMPLETED（固定 Mock 数据，方便前端开发测试）
    """
    r = client.get("/api/session/nonexistent/poll")
    assert r.status_code == 200
    data = r.json()
    assert data["TaskStatus"] == "COMPLETED"  # MOCK 模式下未知 session 也返回完成状态


# ────────── 摘要视图接口测试 ──────────

def test_summary_endpoint():
    """
    测试 GET /api/session/{task_id}/summary 接口：
    - 已创建的 session 应返回 task_id 和 transcript 字段
    """
    r = client.post("/api/session/start", json={})
    task_id = r.json()["task_id"]

    r2 = client.get(f"/api/session/{task_id}/summary")
    assert r2.status_code == 200
    data = r2.json()
    assert data["task_id"] == task_id  # 返回的 task_id 应与请求一致
    assert "transcript" in data        # 转写列表字段必须存在（可为空列表）


def test_summary_unknown_returns_404():
    """
    测试对未知 session ID 请求摘要：
    应返回 404（与 start 行为对称）
    """
    r = client.get("/api/session/no-such-id/summary")
    assert r.status_code == 404


# ────────── 健康咨询对接（工单12）测试 ──────────

def test_consult_no_transcript():
    """
    测试 POST /api/session/{task_id}/consult：
    session 存在但 transcript 为空时，应返回 {"error": "no transcript yet"}
    """
    r = client.post("/api/session/start", json={})
    task_id = r.json()["task_id"]

    r2 = client.post(f"/api/session/{task_id}/consult")
    assert r2.status_code == 200
    # 无转写内容时返回错误提示（而非 404，session 存在）
    assert r2.json().get("error") == "no transcript yet"


def test_consult_unknown_session():
    """
    测试对未知 session ID 发起咨询：
    应返回 404
    """
    r = client.post("/api/session/no-such-id/consult")
    assert r.status_code == 404


# ────────── 翻译降级逻辑单元测试 ──────────

def test_translate_empty_text_returns_empty():
    """
    空文本不应发起 LLM 请求，应直接返回空串。
    验证 translate_sentence 的输入校验逻辑。
    asyncio.run 用于在同步测试函数中运行异步函数。
    """
    import asyncio
    from src.xfyun import translate_sentence
    result = asyncio.run(translate_sentence("", "en"))  # 空文本
    assert result == ""


def test_translate_no_api_key_returns_empty(monkeypatch):
    """
    未配置 SILICONFLOW_API_KEY 时，不发起网络请求，直接返回空串（静默降级）。
    monkeypatch.setattr 将模块变量 SILICONFLOW_API_KEY 替换为 ""。
    """
    import asyncio
    # 替换 xfyun 模块内引用的 SILICONFLOW_API_KEY（非 config 模块的，已在 import 时复制）
    monkeypatch.setattr("src.xfyun.SILICONFLOW_API_KEY", "")
    from src.xfyun import translate_sentence
    result = asyncio.run(translate_sentence("测试", "en"))
    assert result == ""


def test_translate_http_error_returns_empty(monkeypatch):
    """
    LLM HTTP 请求失败时，应静默降级返回空串（不抛出异常）。
    使用 monkeypatch 替换 httpx.AsyncClient 为模拟的"网络断连"客户端。
    这是 Mock 第三方 HTTP 客户端的标准测试模式：
      1. 定义一个假的 Client 类，实现 async with 协议（__aenter__/__aexit__）
      2. 在 post() 方法中抛出网络异常
      3. monkeypatch 将其替换到被测模块
    """
    import asyncio
    import httpx as _httpx  # 导入别名，便于在下面的 Fake 类中引用真实异常类

    class _BadClient:
        """模拟"网络断连"的假 httpx.AsyncClient"""
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self  # 支持 async with 语法
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw):
            raise _httpx.ConnectError("fake network down")  # 模拟网络连接失败

    monkeypatch.setattr("src.xfyun.SILICONFLOW_API_KEY", "sk-fake-for-test")  # 有效的（假）Key
    monkeypatch.setattr("src.xfyun.httpx.AsyncClient", _BadClient)             # 替换 HTTP 客户端

    from src.xfyun import translate_sentence
    result = asyncio.run(translate_sentence("测试文本", "en"))
    assert result == ""  # 网络异常时应静默降级，不抛出异常
