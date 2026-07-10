"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-实时语音识别、翻译与会议概要

FastAPI 主应用，端口 8015。

暴露两条对接路径：
  1) 前端浏览器直连（简化路径）
     - WS  /ws/stream                浏览器推 PCM → 讯飞IAT → 结果推回
     - GET /api/session/{id}/summary 会议纪要（转写全文+章节+要点）
     - GET /api/session/{id}/poll    别名，兼容轮询接口命名
     - POST /api/session/{id}/consult 把转写转发给工单12健康咨询Agent

  2) 外部 Agent 标准时序（对齐 PDF §2 交互流程图）
     - POST /api/session/start        CreateTask，返回 task_id + meeting_join_url
     - POST /api/session/{id}/stop    停止会话
     - POST /api/callback             通义听悟风格回调（用于Agent异步收结果）

技术要点（对应产出物"技术总结文档"）：
  - WebSocket vs HTTP: WS 长连接全双工，用于实时音频推流+识别事件下推；
    HTTP 短连接，用于会话管理、结果轮询、回调接收。
  - 异步 asyncio: 用 asyncio.gather 同时跑上行(browser→xf)/下行(xf→browser)两个协程。
  - 回调 Callback: 通义听悟异步任务完成后 POST 回 CALLBACK_BASE_URL，
    我们同样提供 /api/callback 接口对齐该模式。
  - 轮询 Polling: 客户端 3 秒轮询一次 /poll，直到 TaskStatus=COMPLETED。
"""
import asyncio  # 标准库：异步事件循环，asyncio.create_task 用于后台任务
import json  # 标准库：JSON 解析（WebSocket 消息）
import logging  # 标准库：结构化日志，方便生产环境收集
import time  # 标准库：time.time() 记录会话创建时间，用于过期清理
import uuid  # 标准库：生成全局唯一会话 ID（task_id）
from contextlib import asynccontextmanager  # 标准库：用于定义 FastAPI lifespan 上下文
from pathlib import Path  # 标准库：跨平台路径操作
from typing import Any  # 标准库：类型注解

# httpx 包：异步 HTTP 客户端，用于调用工单12（健康咨询）FastAPI 接口
import httpx
# fastapi 包：高性能异步 Python Web 框架
# FastAPI: 应用主类；WebSocket/WebSocketDisconnect: WebSocket 支持；Request: 访问原始请求
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
# CORSMiddleware: 跨域资源共享中间件，控制浏览器跨域访问
from fastapi.middleware.cors import CORSMiddleware
# HTMLResponse: 返回 HTML 内容；JSONResponse: 返回 JSON 内容
from fastapi.responses import HTMLResponse, JSONResponse
# StaticFiles: 挂载静态文件目录（前端 HTML/JS/CSS）
from fastapi.staticfiles import StaticFiles

from src import config  # 本工单配置模块（读取 .env）
# generate_summary: 调用硅基流动 LLM 生成会议纪要
# run_iat_session: 将浏览器 WS 音频桥接到讯飞 IAT 的核心函数
from src.xfyun import generate_summary, run_iat_session

logger = logging.getLogger("wt15")  # 模块级 logger，日志前缀为 "wt15"
logging.basicConfig(level=logging.INFO)  # 配置日志级别（INFO 及以上）

SESSION_TTL = 3600  # 会话最长保留时长（秒）—— 超时的会话由后台清理任务回收


# ---------------------------------------------------------------------------
# 后台定期清理：防止 _sessions 内存无限增长（L4 安全需求）
# ---------------------------------------------------------------------------

async def _periodic_cleanup():
    """
    每 10 分钟扫描并清除超过 SESSION_TTL 的过期会话。
    作为 asyncio Task 运行，在应用 lifespan 启动时创建，关闭时取消。
    """
    while True:
        await asyncio.sleep(600)  # 每 10 分钟检查一次（600 秒）
        now = time.time()
        # 找出所有超期会话的 key
        expired = [
            k for k, v in list(_sessions.items())
            if now - v.get("created_at", now) > SESSION_TTL
        ]
        for k in expired:
            _sessions.pop(k, None)  # 从字典中移除（pop 防止 KeyError）
        if expired:
            logger.info(f"[cleanup] 清理过期会话 {len(expired)} 个")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理器。
    asynccontextmanager 将普通生成器转为异步上下文管理器。
    yield 之前：启动后台清理任务；yield 之后：取消清理任务。
    """
    task = asyncio.create_task(_periodic_cleanup())  # 启动后台清理协程
    yield  # FastAPI 在此暂停，直到应用关闭
    task.cancel()  # 关闭时取消后台任务，防止 "Task was destroyed but pending" 警告


# ---------------------------------------------------------------------------
# FastAPI 应用实例
# ---------------------------------------------------------------------------

app = FastAPI(
    title="医疗智能体-实时语音识别、翻译与会议纪要",
    version="3.1",
    description="讯飞IAT实时识别 + 硅基流动翻译/摘要 + 通义听悟风格 REST 兼容层",
    lifespan=lifespan,  # 关联生命周期管理器（替代旧版 on_event）
)

# S3 安全需求：CORS 收窄到配置的来源列表，不再全开 ["*"]
# allow_origins: 允许哪些前端域名发跨域请求
# allow_methods: 只允许 GET/POST（不开 PUT/DELETE 等）
# allow_headers: 允许所有请求头（前端会带 Content-Type 等）
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,  # 从 .env 读取，生产必须收窄
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_ROOT = Path(__file__).parent.parent  # backend/ 目录
_STATIC = _ROOT.parent / "frontend"  # frontend/ 与 backend/ 同级（挂载前端静态资源）

if _STATIC.exists():
    # 将 frontend/ 目录挂载到 /static 路径（若存在）
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# ---------------------------------------------------------------------------
# 内存会话存储
# ---------------------------------------------------------------------------

# key: task_id（如 "wt15-abc123def456"）
# value: 会话状态 dict（status / transcript / result / consult_result / created_at）
_sessions: dict[str, dict[str, Any]] = {}


def _new_session(lang: str = "zh_cn") -> str:
    """
    创建新会话，返回 task_id。
    task_id 由服务端生成（S2 安全需求：禁止客户端指定，防会话劫持）。

    Args:
        lang: 识别语言代码，默认中文

    Returns:
        task_id 字符串（格式：wt15-<12位hex>）
    """
    task_id = f"wt15-{uuid.uuid4().hex[:12]}"  # 前缀 + 12 位随机十六进制
    _sessions[task_id] = {
        "status": "CREATED",      # 初始状态
        "lang": lang,
        "transcript": [],          # 转写结果列表（每句话一条记录）
        "result": None,            # 会议纪要/章节/摘要（识别完成后填入）
        "consult_result": None,    # 工单12健康咨询结果
        "created_at": time.time(), # 会话创建时间（用于 L4 过期清理）
    }
    return task_id


# ---------------------------------------------------------------------------
# 前端页面
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    """返回前端主页 HTML，前端不存在时显示引导说明。"""
    html_path = _STATIC / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    # 前端未构建时的占位页面
    return HTMLResponse(
        "<h1>wt15 语音助手</h1>"
        "<p>前端未构建。请检查 frontend/index.html 是否存在。</p>"
    )


@app.get("/health")
def health():
    """
    健康检查接口。供 MCP Server（voice_mcp.py）探测本服务是否在线。
    返回当前 ASR 引擎状态和翻译配置。
    """
    return {
        "status": "ok",
        # 展示实际运行的 ASR 引擎类型
        "mock_mode": config.MOCK_MODE,
        "asr": (
            "mock" if config.ASR_ENGINE == "mock"
            else ("xfyun" if config.XF_APP_ID else "unconfigured")  # 有 APP_ID 才算配置完成
        ),
        "translate_enabled": config.TRANSLATE_ENABLED,
        "translate_langs": config.TRANSLATE_LANGUAGES,
    }


# ---------------------------------------------------------------------------
# WebSocket 主入口（浏览器直连简化路径）
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """
    浏览器 WebSocket 实时语音识别主接口。

    连接协议：
    1. 浏览器先发 JSON 初始化消息：{"lang": "zh_cn", "translate_lang": "en"}
    2. 服务端创建 session，返回 {"type": "started", "task_id": "..."}
    3. 浏览器持续发送 PCM 音频二进制帧
    4. 服务端推送识别片段（transcription）和句子结束（sentence_end）事件
    5. 浏览器发送 {"type": "stop"} 结束录音
    6. 服务端生成会议纪要后推送 {"type": "completed"}
    """
    await ws.accept()  # 接受 WebSocket 握手
    task_id: str | None = None

    try:
        # 接收初始化消息（JSON）
        init_raw = await ws.receive_text()
        init_msg = json.loads(init_raw)
        lang = init_msg.get("lang", "zh_cn")

        # S2 安全需求：task_id 永远由服务端生成，禁止客户端指定（防会话劫持）
        task_id = _new_session(lang)
        _sessions[task_id]["status"] = "streaming"  # 更新状态为流式传输中

        # L2 + S-NEW-1：白名单校验翻译目标语言，防止 Prompt Injection
        _ALLOWED_LANGS = {"en", "ja", "ko", "zh", "fr", "de", "es"}
        translate_lang = init_msg.get("translate_lang", "en")
        if translate_lang not in _ALLOWED_LANGS:
            translate_lang = "en"  # 非法语言码退回英文

        # 通知浏览器会话已建立，返回 task_id 供轮询使用
        await ws.send_json({"type": "started", "task_id": task_id})

        if config.ASR_ENGINE == "mock":
            # Mock 模式：不连接讯飞，返回预设医患对话数据（用于开发/演示/CI）
            await _mock_stream(ws, task_id)
        else:
            # 实际模式：将浏览器 WS 桥接到讯飞 IAT
            await run_iat_session(ws, task_id, _sessions, lang, translate_lang)

    except WebSocketDisconnect:
        # 浏览器主动断开（刷新页面、关闭 Tab 等）
        pass
    except Exception as exc:
        logger.exception("ws_stream 异常")
        try:
            # 尝试推送错误信息给浏览器
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass  # WS 已断开则忽略
    finally:
        # 无论何种退出原因，确保会话状态不残留为 "streaming"
        if task_id:
            sess = _sessions.setdefault(task_id, {})
            if sess.get("status") != "COMPLETED":
                sess["status"] = "stopped"


# ---------------------------------------------------------------------------
# Mock 模式：模拟医患对话流
# ---------------------------------------------------------------------------

async def _mock_stream(ws: WebSocket, task_id: str):
    """
    在 Mock 模式下模拟一段医患对话的实时识别流程。
    逐字增量推送（模拟真实 ASR 的增量识别），结束后推送固定会议纪要。

    Args:
        ws: 已接受的 WebSocket 连接
        task_id: 会话 ID
    """
    # 预设医患对话：(发言人, 文本)
    mock_sentences = [
        ("发言人1", "医生您好，我最近头疼发烧已经两天了。"),
        ("发言人2", "我看一下，体温多少？有没有咳嗽？"),
        ("发言人1", "体温 38.5 度，有轻微咳嗽。"),
        ("发言人2", "先做个血常规，可能是病毒性感冒，注意休息多喝水。"),
    ]

    async def _drain_browser():
        """消耗浏览器发来的所有消息（音频帧或 stop 命令），收到 stop 后退出。"""
        while True:
            msg = await ws.receive()
            if "text" in msg:
                data = json.loads(msg["text"])
                if data.get("type") == "stop":
                    break

    # 后台消耗浏览器上行消息（避免 WS 写缓冲区满导致阻塞）
    drain_task = asyncio.create_task(_drain_browser())

    for i, (speaker, text) in enumerate(mock_sentences):
        await asyncio.sleep(1.5)  # 模拟每句话之间的停顿
        # 逐字增量推送（分 3 段），模拟 ASR 的中间结果（is_final=False）
        for j in range(3):
            await asyncio.sleep(0.3)
            await ws.send_json({
                "type": "transcription",
                "text": text[: (j + 1) * len(text) // 3],  # 每次推送更多字
                "is_final": False,
                "speaker": speaker,
            })
        # 整句识别完成：追加到 transcript，推送 sentence_end 事件
        _sessions[task_id]["transcript"].append({
            "text": text, "speaker": speaker,
            "begin_ms": i * 4000, "end_ms": (i + 1) * 4000,
        })
        await ws.send_json({
            "type": "sentence_end",
            "text": text,
            "speaker": speaker,
            "begin_ms": i * 4000,
            "end_ms": (i + 1) * 4000,
        })
        # 模拟翻译推送（仅 Mock 数据，实际走 LLM 翻译）
        if config.TRANSLATE_ENABLED:
            fake_trans = {
                "医生您好，我最近头疼发烧已经两天了。": "Hello doctor, I've had headache and fever for two days.",
                "我看一下，体温多少？有没有咳嗽？": "Let me check. What's your temperature? Any cough?",
                "体温 38.5 度，有轻微咳嗽。": "Temperature 38.5°C, slight cough.",
                "先做个血常规，可能是病毒性感冒，注意休息多喝水。":
                    "Do a blood test first, likely viral flu. Rest and drink more water.",
            }
            await ws.send_json({
                "type": "translation",
                "text": fake_trans.get(text, ""),
                "target_lang": "en",
                "sentence_index": i,
            })

    # 停止消耗上行消息的后台任务
    drain_task.cancel()
    try:
        await drain_task  # L-NEW-7：等待取消信号真正生效，避免 "Task was destroyed but pending" 警告
    except asyncio.CancelledError:
        pass

    # 推送固定会议纪要结果
    _sessions[task_id]["result"] = {
        "Chapters": [{
            "ChapterId": 1, "Headline": "就诊主诉", "BeginTime": 0,
            "Summary": "患者主诉头痛发烧持续两天，医生问诊体温及症状。",
        }],
        "Summarization": {
            "Paragraph": "患者陈述头痛发烧两天，医生询问体温和伴随症状，"
                         "初步判断为病毒性感冒，建议血常规检查、休息、多喝水。",
            "ActionItems": ["做血常规检查", "多喝水休息", "如症状加重立即就医"],
            "KeyInformation": ["症状：头痛发烧", "持续时间：两天", "体温：38.5度",
                               "初步诊断：病毒性感冒"],
        },
    }
    _sessions[task_id]["status"] = "COMPLETED"
    await ws.send_json({"type": "completed", "task_id": task_id})


# ---------------------------------------------------------------------------
# REST：Agent 标准时序（对齐 PDF 交互流程图）
# ---------------------------------------------------------------------------

@app.post("/api/session/start")
async def session_start(request: Request):
    """
    CreateTask —— 对齐通义听悟 1.1 步骤。
    外部 Agent 调用此接口初始化会话，获取 task_id 和 WebSocket 地址。

    Returns:
        {task_id, meeting_join_url（WebSocket 地址）, status}
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}  # 无 body 或非 JSON 时兜底空 dict
    lang = payload.get("lang", "zh_cn")
    task_id = _new_session(lang)

    # L-NEW-4：从请求 Host 头动态构造 WS URL，避免非本机部署时返回错误地址
    host = request.headers.get("host", f"localhost:{config.PORT}")
    # HTTP 请求对应 ws://，HTTPS 请求对应 wss://
    ws_scheme = "wss" if request.url.scheme == "https" else "ws"
    return {
        "task_id": task_id,
        "meeting_join_url": f"{ws_scheme}://{host}/ws/stream",
        "status": "CREATED",
    }


@app.post("/api/session/{task_id}/stop")
async def session_stop(task_id: str):
    """
    停止会话。若 WebSocket 仍连接中，此调用只标记状态；
    实际停流由 WS 内 {"type": "stop"} 消息触发。

    Returns:
        {task_id, status: "stopped"} 或 404
    """
    session = _sessions.get(task_id)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)
    # 仅对未完成的会话更新状态（已完成的不回退）
    if session["status"] != "COMPLETED":
        session["status"] = "stopped"
    return {"task_id": task_id, "status": "stopped"}


# ---------------------------------------------------------------------------
# REST：结果查询（轮询 + 回调）
# ---------------------------------------------------------------------------

@app.get("/api/session/{task_id}/poll")
async def session_poll(task_id: str):
    """
    轮询接口（客户端每 3 秒调一次，直到 TaskStatus=COMPLETED）。
    返回通义听悟风格字段（TaskStatus/Result）+ 兼容小写字段（旧前端）。
    未知 session 也返回 200，避免前端轮询死循环报错。

    Returns:
        {TaskId, TaskStatus, Result, task_id, status, transcript, chapters, summarization}
    """
    session = _sessions.get(task_id)
    if not session:
        if config.ASR_ENGINE == "mock":
            # MOCK 模式：未初始化的 session 也返回 COMPLETED + 固定摘要（方便测试）
            return {
                "TaskId": task_id, "TaskStatus": "COMPLETED",
                "Result": _mock_result(),
            }
        # 非 Mock 模式：未知 session 返回 UNKNOWN 状态
        return {"TaskId": task_id, "TaskStatus": "UNKNOWN", "Result": {}}

    result = session.get("result") or {}
    return {
        # 通义听悟风格字段（外部 Agent 对接用）
        "TaskId": task_id,
        "TaskStatus": session.get("status", "UNKNOWN"),
        "Result": result,
        # 兼容旧前端小写字段
        "task_id": task_id,
        "status": session.get("status", "UNKNOWN"),
        "transcript": session.get("transcript", []),
        "chapters": result.get("Chapters", []),
        "summarization": result.get("Summarization", {}),
    }


@app.get("/api/session/{task_id}/summary")
async def get_summary(task_id: str):
    """
    会议纪要详细视图：转写全文 + 章节 + 摘要 + 健康建议。

    Returns:
        {task_id, status, transcript, chapters, summarization, consult_result} 或 404
    """
    session = _sessions.get(task_id)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)

    result = session.get("result") or {}
    return {
        "task_id": task_id,
        "status": session.get("status"),
        "transcript": session.get("transcript", []),       # 原始转写句子列表
        "chapters": result.get("Chapters", []),             # 章节列表
        "summarization": result.get("Summarization", {}),  # 摘要/要点/待办
        "consult_result": session.get("consult_result"),   # 工单12健康建议
    }


@app.post("/api/callback")
async def tingwu_callback(request: Request):
    """
    通义听悟风格回调接收端。
    工单要求"回调 URL 和轮询都要求实现"，此接口即回调路径。
    通义听悟异步任务完成后 POST 完整结果到此处；
    本地讯飞路径也可主动 POST 触发（调试用）。

    S1 安全需求：若配置了 CALLBACK_SECRET，校验 X-Callback-Secret 请求头。

    Returns:
        {"code": "0", "message": "ok"} 或 401/400
    """
    # S1：若配置了 CALLBACK_SECRET，则校验请求头中的鉴权令牌
    secret = config.CALLBACK_SECRET
    if secret:
        provided = request.headers.get("X-Callback-Secret", "")
        if provided != secret:
            logger.warning("[callback] 拒绝未授权请求")
            return JSONResponse({"code": "Unauthorized"}, status_code=401)

    try:
        payload = await request.json()  # 解析回调 body
    except Exception:
        return JSONResponse({"code": "InvalidPayload"}, status_code=400)

    # 支持大写 TaskId（通义听悟标准）和小写 task_id（内部兼容）
    task_id = payload.get("TaskId") or payload.get("task_id")
    if not task_id:
        return JSONResponse({"code": "MissingTaskId"}, status_code=400)

    # S-NEW-2：补 created_at，防止外部 callback 创建的 session 绕过过期清理
    session = _sessions.setdefault(task_id, {
        "status": "CREATED", "transcript": [], "result": None,
        "created_at": time.time(),
    })
    session["status"] = payload.get("TaskStatus", "COMPLETED")
    session["result"] = payload.get("Result", session.get("result"))

    logger.info(f"[callback] task_id={task_id} status={session['status']}")

    # 回调完成后，异步触发工单12健康咨询（后台任务，不阻塞回调响应）
    if session["status"] == "COMPLETED":
        asyncio.create_task(_auto_consult(task_id))

    return {"code": "0", "message": "ok"}


async def _auto_consult(task_id: str):
    """
    回调完成后自动调用工单12（后台任务）。
    失败不影响主流程，仅记录日志。

    Args:
        task_id: 会话 ID
    """
    try:
        session = _sessions.get(task_id)
        if not session or not session.get("transcript"):
            return  # 无转写内容，跳过咨询
        await consult_health(task_id)
    except Exception:
        logger.exception(f"[callback] auto_consult failed task_id={task_id}")


# ---------------------------------------------------------------------------
# REST：对接工单12健康咨询
# ---------------------------------------------------------------------------

@app.post("/api/session/{task_id}/consult")
async def consult_health(task_id: str):
    """
    将本次会话的医患对话转写文本发送给工单12健康咨询 Agent，
    获取症状分析和健康建议，写回 session["consult_result"]。

    工单12不可达时静默降级（不影响主识别流程）。

    Returns:
        {task_id, health_reply} 或 {"error": "..."} 或 404
    """
    session = _sessions.get(task_id)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)

    transcript = session.get("transcript", [])
    if not transcript:
        return {"error": "no transcript yet"}  # 尚无转写内容

    # 将转写列表拼成对话文本，发给工单12
    full_text = "\n".join(f"{s['speaker']}：{s['text']}" for s in transcript)
    query = f"以下是医患对话记录，请分析症状并提供健康建议：\n\n{full_text}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # POST 到工单12健康咨询 FastAPI 的 /chat 接口
            resp = await client.post(f"{config.WT12_URL}/chat", json={"query": query})
            result = resp.json()
    except Exception as exc:
        # 工单12不可达时不影响本工单主流程，记录警告
        logger.warning(f"consult wt12 unreachable: {exc}")
        result = {"error": f"wt12 unreachable: {exc}"}

    session["consult_result"] = result
    # S5 安全需求：不回传 query（完整对话文本），避免敏感数据重复暴露
    return {
        "task_id": task_id,
        "health_reply": result.get("reply", result),  # 优先取 reply 字段，否则返回完整 result
    }


# ---------------------------------------------------------------------------
# Mock 数据生成
# ---------------------------------------------------------------------------

def _mock_result() -> dict:
    """
    返回 Mock 模式下的固定会议纪要，供 /poll 接口在未初始化 session 时返回。

    Returns:
        通义听悟兼容格式的摘要 dict
    """
    return {
        "Chapters": [{
            "ChapterId": 1, "Headline": "MOCK示例", "BeginTime": 0,
            "Summary": "MOCK模式下的固定示例摘要。",
        }],
        "Summarization": {
            "Paragraph": "MOCK模式：这是用于开发和测试的默认摘要。",
            "ActionItems": ["示例待办1", "示例待办2"],
            "KeyInformation": ["示例关键词"],
        },
    }
