"""
PAI-EAS 服务入口（Flask Web 服务）。

接口规范：
  POST /  或  POST /api/v1/chat
  Request:  {"question": "..."}
  Response: {"answer": "..."}          ← evaluator 只读这个字段
            {"answer": "...",
             "trace": [...],            ← 调试用，evaluator 忽略
             "rounds_used": N,
             "exit_reason": "...",
             "elapsed_ms": N}

任何异常都返回 {"answer": "Unknown"}，HTTP 200（保证接口健壮性）。

辅助接口：
  GET /                          → 返回前端 index.html
  GET /api/v1/status/<req_id>    → 前端轮询进度（每1.5s一次）
  GET /health                    → 健康检查
"""
import json       # 标准库：JSON 序列化/反序列化
import logging    # 标准库：日志记录
import os         # 标准库：操作系统接口（路径、环境变量）
import sys        # 标准库：Python 解释器相关（修改模块搜索路径）
import time       # 标准库：时间相关函数（计时、sleep）
import uuid       # 标准库：生成唯一标识符（本文件实际未直接用，保留供扩展）
from threading import Lock  # 标准库 threading：线程锁，保护共享状态字典的并发安全

# flask：轻量级 Python Web 框架，用于快速搭建 HTTP 服务
# Flask    - 应用工厂类
# request  - 获取当前请求的数据（JSON body、查询参数等）
# jsonify  - 将 Python dict 序列化为 JSON HTTP 响应
# send_from_directory - 从指定目录安全地返回静态文件
from flask import Flask, request, jsonify, send_from_directory

# 将当前文件所在目录加入 Python 模块搜索路径
# 这样可以直接 import config / preprocessor 等本地模块，无需相对导入
sys.path.insert(0, os.path.dirname(__file__))

from config import Config, validate_config          # 全局配置类 + 配置校验函数
from preprocessor import preprocess                 # 问题预处理：语言检测、格式提取、搜索词生成
from react_loop import run_react                    # ReAct 循环控制器：核心搜索推理引擎
from answer_generator import generate_answer        # 答案生成器：归一化 ReAct 原始输出
from answer_validator import validate as validate_answer  # 答案自检器：最后一道格式卫兵

# ── 日志配置 ──────────────────────────────────────────────────────────────────
# 创建日志目录（logs/），exist_ok=True 表示目录已存在时不报错
os.makedirs(Config.LOG_DIR, exist_ok=True)

# 配置全局日志：JSON 格式方便日志平台解析
logging.basicConfig(
    level=logging.INFO,   # 只输出 INFO 及以上级别的日志
    # 日志格式：JSON 字符串，包含时间、级别、模块名、消息
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}',
    handlers=[
        logging.StreamHandler(sys.stdout),   # 同时输出到控制台（stdout）
        # 同时写入文件，utf-8 编码防止中文乱码
        logging.FileHandler(os.path.join(Config.LOG_DIR, "agent.log"), encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# ── 启动时检查配置 ─────────────────────────────────────────────────────────────
_missing = validate_config()  # 检查必要的 API Key 是否已配置
if _missing:
    # 配置缺失时只警告，不终止（允许部分功能降级运行）
    logger.warning("Missing config: %s", _missing)

# 创建 Flask 应用实例
app = Flask(__name__)

# ── 状态存储（用于前端轮询进度）─────────────────────────────────────────────────
# 字典格式：{req_id: {"round": N, "action": "search", "query": "...", "done": bool}}
# 前端每 1.5s 调用 /api/v1/status/<req_id> 获取当前处理进度
_status_store: dict = {}
_status_lock = Lock()  # 线程锁：保护 _status_store 在多线程并发时的数据安全


def _set_status(req_id: str, **kwargs):
    """
    更新指定请求 ID 的处理状态，供前端进度轮询接口使用。

    :param req_id:  请求唯一标识（前端传入）
    :param kwargs:  状态字段，如 round=1, action="search", query="...", done=False
    """
    if not req_id:  # req_id 为空时跳过（评测脚本不传 req_id）
        return
    with _status_lock:  # 加锁，防止多线程并发写入时数据竞争
        _status_store[req_id] = kwargs  # 覆盖写入最新状态


def _clear_status(req_id: str):
    """
    清除指定请求 ID 的状态记录，释放内存。

    :param req_id: 请求唯一标识
    """
    with _status_lock:  # 加锁保证线程安全
        _status_store.pop(req_id, None)  # 键不存在时不报错


@app.after_request
def add_cors(resp):
    """
    在每个响应中添加 CORS 头，允许浏览器跨域访问。
    前端页面可能部署在不同域名，需要此配置。
    """
    resp.headers["Access-Control-Allow-Origin"] = "*"   # 允许任意来源
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"  # 允许 Content-Type 请求头
    return resp


@app.get("/")
def index():
    """
    返回前端首页 HTML 文件。
    前端目录位于 src/ 同级的 frontend/ 文件夹中。
    """
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    return send_from_directory(frontend_dir, "index.html")  # 安全返回静态文件，防路径遍历


# ── 状态轮询接口（前端每 1.5s 调一次，获取真实进度）──────────────────────────────
@app.get("/api/v1/status/<req_id>")
def get_status(req_id):
    """
    返回指定请求 ID 的当前处理状态。
    前端根据此接口显示实时搜索进度（正在搜索哪一轮、查哪个词）。

    :param req_id: URL 路径参数，前端发起请求时生成的唯一 ID
    :return:       JSON 格式的状态字典，如 {"round": 2, "action": "search", "query": "xxx", "done": false}
    """
    with _status_lock:  # 加锁读取，防止读到脏数据
        status = _status_store.get(req_id, {})  # 键不存在时返回空字典
    return jsonify(status)


def process_question(question: str, req_id: str = "") -> dict:
    """
    核心处理逻辑：预处理 → ReAct 循环 → 答案生成 → 答案自检。
    封装为独立函数，方便评测脚本（run_eval.py）直接调用。

    :param question: 原始问题字符串
    :param req_id:   请求 ID（可选），有则更新前端状态
    :return:         包含 answer 和调试信息的字典
    """
    try:
        # 阶段 0：更新状态为"预处理中"
        _set_status(req_id, round=0, action="preprocessing", query="", done=False)

        # 问题预处理：检测语言、提取格式约束、生成初始搜索词
        meta = preprocess(question)

        # 阶段 1：更新状态为"搜索中"，显示初始搜索词
        _set_status(req_id, round=1, action="searching", query=meta.search_query, done=False)

        # 执行 ReAct 循环：Think → Search/Fetch/Calculate → Observe，最多 MAX_ROUNDS 轮
        react_result = run_react(meta, req_id=req_id, set_status=_set_status)

        # 阶段 N：更新状态为"生成答案中"
        _set_status(req_id, round=react_result.rounds_used, action="generating", query="", done=False)

        # 从 ReAct 结果中提取并归一化最终答案
        answer = generate_answer(react_result, meta)

        # M4：答案自检（截断解释句、去噪、去多余引号）
        vr = validate_answer(answer, format_hint=meta.format_hint, lang=meta.lang)
        answer = vr.answer  # 使用自检后的答案

        # 更新状态为"已完成"
        _set_status(req_id, round=react_result.rounds_used, action="done", query="", done=True)

        # 将 trace 序列化为 JSON 友好格式（observation 截断避免响应体过大）
        trace_data = [
            {
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation[:500],  # 截断到 500 字符避免响应过大
            }
            for step in react_result.trace
        ]

        return {
            "answer": answer or "Unknown",  # 空答案时兜底返回 "Unknown"
            "trace": trace_data,            # 调试信息：搜索轨迹
            "rounds_used": react_result.rounds_used,  # 实际使用的搜索轮数
            "exit_reason": react_result.exit_reason,  # 退出原因（final_answer/max_rounds/timeout等）
            "evidence_urls": react_result.evidence_urls[:10],  # 最多返回10个来源URL
        }
    except Exception as e:
        # 捕获所有未预期异常，记录详细错误信息（exc_info=True 输出堆栈）
        logger.error("process_question error: %s", e, exc_info=True)
        # 更新状态为"出错"，将错误信息前100字写入状态供前端展示
        _set_status(req_id, round=0, action="error", query=str(e)[:100], done=True)
        # 返回兜底结果，保证接口始终返回有效响应
        return {"answer": "Unknown", "trace": [], "rounds_used": 0, "exit_reason": "error", "evidence_urls": []}


@app.post("/")
@app.post("/api/v1/chat")
def chat():
    """
    主问答接口，支持两个路由（根路径和 RESTful 路径）。

    Request Body (JSON):
        question (str): 用户问题
        req_id   (str): 前端生成的请求 ID，用于状态轮询（可选）

    Response (JSON):
        answer      (str): 最终答案
        trace       (list): 搜索轨迹（调试用）
        rounds_used (int): 使用的搜索轮数
        exit_reason (str): 退出原因
        elapsed_ms  (int): 本次请求总耗时（毫秒）
    """
    t0 = time.time()  # 记录请求开始时间，用于计算总耗时

    try:
        # force=True：即使 Content-Type 不是 application/json 也尝试解析
        body = request.get_json(force=True)
        question = body.get("question", "").strip()  # 取问题并去首尾空格
        req_id = body.get("req_id", "")              # 取请求 ID（可选）
    except Exception:
        # JSON 解析失败时降级处理（如请求体为空或格式错误）
        question = ""
        req_id = ""

    # 空问题直接返回，不进入处理流程
    if not question:
        return jsonify({"answer": "Unknown", "error": "empty question"}), 200

    # 调用核心处理逻辑
    result = process_question(question, req_id=req_id)

    # 计算总耗时（毫秒）
    elapsed_ms = int((time.time() - t0) * 1000)
    result["elapsed_ms"] = elapsed_ms

    # 写结构化日志（JSON 格式），方便后续统计分析
    logger.info(json.dumps({
        "question": question[:80],           # 问题截断到80字，防日志过长
        "answer": result["answer"][:80],     # 答案同样截断
        "rounds": result["rounds_used"],
        "exit": result["exit_reason"],
        "elapsed_ms": elapsed_ms,
    }, ensure_ascii=False))  # ensure_ascii=False 保留中文字符

    # 请求结束后延迟3秒清理状态（让前端最后一次轮询能拿到 done=True）
    def _delayed_clear():
        time.sleep(3)          # 等待3秒，确保前端轮询到最终状态
        _clear_status(req_id)  # 清理状态，释放内存

    if req_id:
        import threading  # 延迟导入，仅在有 req_id 时才需要
        # daemon=True：主线程退出时此线程自动结束，不阻塞服务关闭
        threading.Thread(target=_delayed_clear, daemon=True).start()

    return jsonify(result)  # 返回 JSON 响应


@app.get("/health")
def health():
    """
    健康检查接口，供 PAI-EAS 或 K8s 探针使用。
    返回服务状态和当前使用的模型名称。
    """
    return jsonify({"status": "ok", "model": Config.LLM_MODEL})


if __name__ == "__main__":
    # 直接运行此文件时启动开发服务器
    port = int(os.getenv("PORT", 8023))  # 从环境变量读取端口，默认 8023
    logger.info("Starting Research Agent on port %d", port)
    app.run(
        host="0.0.0.0",   # 监听所有网卡（包括外部访问）
        port=port,
        debug=False        # 生产模式关闭 debug（debug=True 会暴露代码和自动重启）
    )
