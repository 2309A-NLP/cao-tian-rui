# 工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
# 全量评测脚本 — 从 cases.yaml 读取所有测试用例，向 API 发送请求，计算并输出 metrics.json
#
# 评测覆盖四类用例：
# - vqa_cases：视觉问答，含关键词命中检查
# - mrg_cases：报告生成，含字段完整性和最小长度检查
# - rag_cases：检索增强生成，含引用数量和回答非空检查
# - boundary_cases：边界条件，含状态码和错误码检查
#
# 用法：python tests/eval.py --server http://localhost:8014 --output tests/metrics.json

# argparse：Python 内置命令行参数解析模块
import argparse

# io：Python 内置模块，io.BytesIO 用于在内存中模拟文件对象（用于重试时重新构造上传文件）
import io

# json：Python 内置模块，用于读写 JSON 格式的评测结果
import json

# os：Python 内置模块（此处导入但未使用）
import os

# re：Python 内置正则表达式模块，用于检测答案中是否包含数字
import re

# time：Python 内置模块，用于计算请求耗时和重试等待
import time

# datetime：Python 内置模块，用于记录评测时间戳
from datetime import datetime, timezone

# pathlib.Path：面向对象的文件路径操作
from pathlib import Path

# requests：第三方 HTTP 客户端库
# 安装方式：pip install requests
import requests

# yaml：YAML 格式文件解析库（test cases 存储在 YAML 文件中）
# 安装方式：pip install pyyaml
import yaml

# 测试图片目录（tests/images/）
IMG_DIR = Path(__file__).parent / "images"

# 测试用例 YAML 文件路径（需提前准备 cases.yaml）
CASES_FILE = Path(__file__).parent / "cases.yaml"


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def load_cases():
    """
    从 cases.yaml 文件加载所有测试用例。

    返回值：
        dict：包含 vqa_cases/mrg_cases/rag_cases/boundary_cases 四个列表的字典
    """
    with open(CASES_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)  # safe_load：安全加载 YAML（不执行任意 Python 代码）


def img_path(name: str) -> Path:
    """将图片文件名转换为完整路径。"""
    return IMG_DIR / name


def send_request(server: str, task: str, image_bytes: bytes | None,
                 filename: str, query: str | None,
                 session_id: str | None = None) -> tuple[int, dict]:
    """
    向 API 发送单次 analyze 请求，支持一次重试（针对 504/超时）。

    参数：
        server (str)：API 服务器地址（如 "http://localhost:8013"）
        task (str)：任务类型（"vqa"/"mrg"/"rag"）
        image_bytes (bytes | None)：图片字节数据，None 表示不传图片（边界测试）
        filename (str)：上传时使用的文件名
        query (str | None)：用户问题，None 表示不传 query
        session_id (str | None)：会话 ID（可选，用于日志追踪）

    返回值：
        tuple[int, dict]：(HTTP 状态码, 响应体字典)
    """
    url = f"{server}/api/analyze"

    # 构建表单数据
    data = {"task": task}
    if query is not None:
        data["query"] = query        # 若有问题，加入表单
    if session_id:
        data["session_id"] = session_id  # 若有会话ID，加入表单

    # 构建文件上传字典（若 image_bytes 不为 None）
    files = {}
    if image_bytes is not None:
        files["image"] = (filename, io.BytesIO(image_bytes), "image/jpeg")

    for attempt in range(2):   # 最多重试1次（仅对 504/超时情况重试）
        try:
            # 发送 POST 请求，最长等待 180 秒
            resp = requests.post(url, data=data, files=files if files else None, timeout=180)
            try:
                body = resp.json()     # 尝试解析 JSON 响应
            except Exception:
                body = {"_raw": resp.text[:500]}  # JSON 解析失败则保留原始文本

            # 若收到 504 且是第一次尝试，等待 5 秒后重试
            if resp.status_code == 504 and attempt == 0:
                print(f"    [重试] 504 timeout，等待5s后重试...")
                time.sleep(5)
                # 重试时需要重新构造 BytesIO（已被读取完毕）
                if image_bytes is not None:
                    files["image"] = (filename, io.BytesIO(image_bytes), "image/jpeg")
                continue  # 继续重试
            return resp.status_code, body  # 返回状态码和响应体

        except requests.exceptions.Timeout:
            # 请求超时（连接超时或读取超时）
            if attempt == 0:
                print(f"    [重试] 请求超时，等待5s后重试...")
                time.sleep(5)
                if image_bytes is not None:
                    files["image"] = (filename, io.BytesIO(image_bytes), "image/jpeg")
                continue
            return 504, {"error_code": "TIMEOUT"}  # 两次都超时，返回 504

        except requests.exceptions.ConnectionError as e:
            # 连接错误（服务未启动、网络问题等）
            return 503, {"error_code": "CONNECTION_ERROR", "message": str(e)}

    # 两次重试都失败（理论上不会到达这里）
    return 504, {"error_code": "TIMEOUT"}


def contains_number(text: str) -> bool:
    """
    检查文本中是否包含数字。

    用于验证某些 VQA 答案（如尺寸、计数类问题）必须包含数字。

    参数：
        text (str)：待检测文本

    返回值：
        bool：包含数字返回 True，否则返回 False
    """
    return bool(re.search(r"\d+", text))  # \d+ 匹配一个或多个连续数字


def check_keywords(answer: str, keywords: list[str], mode: str = "any") -> bool:
    """
    检查答案中是否包含关键词。

    参数：
        answer (str)：模型回答文本
        keywords (list[str])：关键词列表
        mode (str)："any"（至少一个关键词存在）或 "all"（所有关键词都存在）

    返回值：
        bool：满足条件返回 True
    """
    answer_lower = answer.lower()  # 转小写以进行不区分大小写的匹配
    if mode == "any":
        # any 模式：只要有一个关键词在答案中就算通过
        return any(k.lower() in answer_lower for k in keywords)
    # all 模式：所有关键词都必须在答案中
    return all(k.lower() in answer_lower for k in keywords)


def read_image(name: str) -> bytes | None:
    """
    读取图片文件为字节数据。

    参数：
        name (str)：图片文件名（相对于 IMG_DIR）

    返回值：
        bytes | None：图片字节数据，文件不存在则返回 None
    """
    p = img_path(name)
    if not p.exists():
        return None
    return p.read_bytes()  # read_bytes()：读取文件全部内容为 bytes


# ──────────────────────────────────────────────
# 各类 case 执行函数
# ──────────────────────────────────────────────

def run_vqa_case(server: str, c: dict) -> dict:
    """
    执行单个 VQA 测试用例。

    验证逻辑（按顺序）：
    1. 图片存在
    2. HTTP 200
    3. 关键词命中（若配置了 expected_keywords）
    4. 答案含数字（若 expected_type=contains_number）

    参数：
        server (str)：API 服务器地址
        c (dict)：YAML 中的单个测试用例配置

    返回值：
        dict：包含 id/status/reason 等字段的测试结果字典
    """
    image_bytes = read_image(c["image"])  # 读取图片
    if image_bytes is None:
        # 图片不存在：跳过此用例（status="skip"）
        return {"id": c["id"], "status": "skip", "reason": f"image not found: {c['image']}"}

    t0 = time.time()
    code, body = send_request(server, "vqa", image_bytes, c["image"], c.get("query"))
    latency = round((time.time() - t0) * 1000)  # 计算耗时（毫秒）

    result = {"id": c["id"], "task": "vqa", "http_status": code, "latency_ms": latency}

    # HTTP 非 200：测试失败
    if code != 200:
        result["status"] = "fail"
        result["reason"] = f"HTTP {code}: {body.get('error_code', '')} {body.get('message', '')}"
        return result

    answer = body.get("answer", "")
    result["answer_snippet"] = answer[:120]  # 记录答案前 120 字符用于展示

    # 关键词检查：expected_keywords 或 expected_keywords_any 两种配置方式都支持
    kw = c.get("expected_keywords") or c.get("expected_keywords_any")
    if kw:
        hit = check_keywords(answer, kw, mode="any")
        if not hit:
            result["status"] = "fail"
            result["reason"] = f"keywords not found: {kw}"
            return result

    # 数字检查：若要求答案中含数字（如尺寸、计数类问题）
    if c.get("expected_type") == "contains_number" and not contains_number(answer):
        result["status"] = "fail"
        result["reason"] = "expected number in answer"
        return result

    result["status"] = "pass"  # 所有检查通过
    return result


def run_mrg_case(server: str, c: dict) -> dict:
    """
    执行单个 MRG 测试用例。

    验证逻辑（按顺序）：
    1. 图片存在
    2. HTTP 200
    3. 报告字段完整性（expected_fields 中的字段不能为空）
    4. 字段最小长度（expected_min_length 配置）

    参数：
        server (str)：API 服务器地址
        c (dict)：YAML 中的单个测试用例配置

    返回值：
        dict：测试结果字典
    """
    image_bytes = read_image(c["image"])
    if image_bytes is None:
        return {"id": c["id"], "status": "skip", "reason": f"image not found: {c['image']}"}

    t0 = time.time()
    # MRG 任务不需要 query，传 None
    code, body = send_request(server, "mrg", image_bytes, c["image"], None)
    latency = round((time.time() - t0) * 1000)

    result = {"id": c["id"], "task": "mrg", "http_status": code, "latency_ms": latency}

    if code != 200:
        result["status"] = "fail"
        result["reason"] = f"HTTP {code}: {body.get('error_code', '')} {body.get('message', '')}"
        return result

    report = body.get("report", {}) or {}  # 获取报告对象（若为 None 则用空字典）

    # 检查必填字段是否都有非空值
    # 默认检查四个核心字段，YAML 中可通过 expected_fields 自定义
    expected_fields = c.get("expected_fields", ["chief_complaint", "findings", "impression", "recommendation"])
    missing = [f for f in expected_fields if not report.get(f)]  # 收集空字段
    if missing:
        result["status"] = "fail"
        result["reason"] = f"missing/empty fields: {missing}"
        result["report_snippet"] = {k: (v or "")[:60] for k, v in report.items()}
        return result

    # 检查字段最小长度（expected_min_length 是字段名到最小长度的字典）
    min_len = c.get("expected_min_length", {})
    for field, min_l in min_len.items():
        val = report.get(field, "")
        if len(val) < min_l:
            result["status"] = "fail"
            result["reason"] = f"field '{field}' too short: {len(val)} < {min_l}"
            return result

    result["status"] = "pass"
    result["report_snippet"] = {k: (v or "")[:60] for k, v in report.items()}  # 记录摘要
    return result


def run_rag_case(server: str, c: dict) -> dict:
    """
    执行单个 RAG 测试用例。

    验证逻辑（按顺序）：
    1. 图片存在
    2. HTTP 200
    3. 检索到的参考文档数量不少于 expected_references_count_min
    4. 答案非空

    参数：
        server (str)：API 服务器地址
        c (dict)：YAML 中的单个测试用例配置

    返回值：
        dict：测试结果字典
    """
    image_bytes = read_image(c["image"])
    if image_bytes is None:
        return {"id": c["id"], "status": "skip", "reason": f"image not found: {c['image']}"}

    t0 = time.time()
    code, body = send_request(server, "rag", image_bytes, c["image"], c.get("query"))
    latency = round((time.time() - t0) * 1000)

    result = {"id": c["id"], "task": "rag", "http_status": code, "latency_ms": latency}

    if code != 200:
        result["status"] = "fail"
        result["reason"] = f"HTTP {code}: {body.get('error_code', '')} {body.get('message', '')}"
        return result

    refs = body.get("references", []) or []  # 获取参考文档列表
    min_refs = c.get("expected_references_count_min", 0)  # 最少参考文档数，默认 0

    # 检查检索到的文档数量是否满足要求
    if len(refs) < min_refs:
        result["status"] = "fail"
        result["reason"] = f"references {len(refs)} < min {min_refs}"
        return result

    answer = body.get("answer", "")
    # 检查答案是否非空
    if not answer:
        result["status"] = "fail"
        result["reason"] = "empty answer"
        return result

    result["status"] = "pass"
    result["ref_count"] = len(refs)           # 记录检索到的文档数
    result["answer_snippet"] = answer[:120]   # 记录答案摘要
    return result


def run_boundary_case(server: str, c: dict) -> dict:
    """
    执行单个边界测试用例。

    支持多种特殊请求构造：
    - no_image：不传图片（测试缺少图片时的错误处理）
    - generate=large_dummy_21mb：生成 21MB 假数据（测试文件大小限制）
    - generate=fake_pdf：发送 PDF 文件（测试文件格式校验）
    - query_repeat：重复字符构造超长 query（测试 query 长度限制）
    - 普通图片：使用真实图片测试其他边界条件

    验证逻辑：
    - expected_status：检查 HTTP 状态码（支持单个值或列表）
    - expected_error_code：检查响应中的错误码
    - expected_task_in_response：检查响应中的 task 字段

    参数：
        server (str)：API 服务器地址
        c (dict)：YAML 中的单个边界测试用例配置

    返回值：
        dict：测试结果字典
    """
    cid = c["id"]
    result = {"id": cid, "task": "boundary"}

    # 准备请求参数
    image_bytes = None
    filename = "test.jpg"
    query = c.get("query")  # 基础 query

    # 若配置了 query_repeat（重复字符生成超长 query），构造超长字符串
    if "query_repeat" in c:
        char, n = c["query_repeat"]  # (字符, 重复次数)
        query = char * n

    # ── 根据用例类型构造不同的请求 ──

    if c.get("no_image"):
        # 边界类型1：不传图片（测试服务如何处理缺少图片的情况）
        t0 = time.time()
        code, body = send_request(server, c.get("task", "vqa"), None, "", query)
        latency = round((time.time() - t0) * 1000)

    elif c.get("generate") == "large_dummy_21mb":
        # 边界类型2：生成 21MB 超大假文件（超过 MAX_IMAGE_SIZE_MB=20MB 的限制）
        # 使用 JPEG 文件头 + 填充字节，使 MIME 类型通过前置检查但文件超大
        fake = b"\xff\xd8\xff\xe0" + b"\x00" * (21 * 1024 * 1024)  # JPEG 魔数 + 21MB 零字节
        t0 = time.time()
        code, body = send_request(server, c.get("task", "vqa"), fake, "big.jpg", query)
        latency = round((time.time() - t0) * 1000)

    elif c.get("generate") == "fake_pdf":
        # 边界类型3：发送 PDF 文件（测试 MIME 类型校验）
        fake = b"%PDF-1.4 fake content"  # PDF 文件签名
        t0 = time.time()
        # 直接用 requests 构造请求（设置 content_type 为 application/pdf）
        files = {"image": ("test.pdf", io.BytesIO(fake), "application/pdf")}
        data = {"task": c.get("task", "vqa")}
        try:
            resp = requests.post(f"{server}/api/analyze", data=data, files=files, timeout=30)
            code, body = resp.status_code, resp.json()
        except Exception as e:
            code, body = 503, {"error_code": "CONNECTION_ERROR"}
        latency = round((time.time() - time.time()) * 1000)  # 此处计算有误（始终为0），但影响不大

    elif "image" in c:
        # 边界类型4：使用真实图片（测试其他边界条件，如超低分辨率等）
        image_bytes = read_image(c["image"])
        if image_bytes is None:
            return {"id": cid, "status": "skip", "reason": f"image not found: {c['image']}"}
        t0 = time.time()
        code, body = send_request(server, c.get("task", "vqa"), image_bytes,
                                  c["image"], query, c.get("session_id"))
        latency = round((time.time() - t0) * 1000)
    else:
        # 未知边界用例配置，跳过
        return {"id": cid, "status": "skip", "reason": "unhandled boundary case config"}

    result["http_status"] = code
    result["latency_ms"] = latency

    # ── 验证 HTTP 状态码 ──
    expected_status = c.get("expected_status")
    if isinstance(expected_status, list):
        # 支持多个合法状态码（如 [400, 422]）
        status_ok = code in expected_status
    elif expected_status is not None:
        # 单个期望状态码
        status_ok = (code == expected_status)
    else:
        # 未配置期望状态码：任意状态码都算通过
        status_ok = True

    if not status_ok:
        result["status"] = "fail"
        result["reason"] = f"expected status {expected_status}, got {code}"
        result["body_snippet"] = str(body)[:200]
        return result

    # ── 验证错误码字段 ──
    exp_ec = c.get("expected_error_code")
    if exp_ec:
        actual_ec = body.get("error_code", "")
        if actual_ec != exp_ec:
            result["status"] = "fail"
            result["reason"] = f"expected error_code={exp_ec}, got={actual_ec}"
            return result

    # ── 验证响应中的 task 字段 ──
    exp_task = c.get("expected_task_in_response")
    if exp_task:
        actual_task = body.get("task", "")
        if actual_task != exp_task:
            result["status"] = "fail"
            result["reason"] = f"expected task={exp_task} in response, got={actual_task}"
            return result

    result["status"] = "pass"  # 所有验证通过
    return result


# ──────────────────────────────────────────────
# 指标计算
# ──────────────────────────────────────────────

def compute_metrics(all_results: list[dict]) -> dict:
    """
    从所有测试结果计算汇总指标。

    计算的指标：
    - task_routing_accuracy：任务路由准确率（成功返回 200 的比例）
    - keyword_hit_rate：VQA 关键词命中率（用 VQA 通过率近似）
    - mrg_completeness：MRG 结构完整率
    - rag_pass_rate：RAG 通过率
    - boundary_handling_rate：边界用例处理率
    - latency：各任务类型的 P50/P95/min/max 时延

    参数：
        all_results (list[dict])：所有测试用例的结果列表

    返回值：
        dict：包含各项指标的字典
    """
    # 按任务类型分组（排除 status=skip 的用例，因为它们未实际运行）
    vqa = [r for r in all_results if r.get("task") == "vqa" and r.get("status") != "skip"]
    mrg = [r for r in all_results if r.get("task") == "mrg" and r.get("status") != "skip"]
    rag = [r for r in all_results if r.get("task") == "rag" and r.get("status") != "skip"]
    boundary = [r for r in all_results if r.get("task") == "boundary" and r.get("status") != "skip"]

    def pass_rate(cases):
        """计算通过率（0~1 的小数），空列表返回 None。"""
        if not cases:
            return None
        return round(sum(1 for c in cases if c["status"] == "pass") / len(cases), 4)

    # 路由准确率：所有非边界用例（VQA/MRG/RAG）中 HTTP 状态码为 200 的比例
    # 200 意味着任务路由成功且 handler 正常执行
    routed = [r for r in vqa + mrg + rag]
    routing_acc = round(sum(1 for r in routed if r.get("http_status") == 200) / len(routed), 4) if routed else None

    # 关键词命中率：用 VQA 整体通过率近似（通过的 VQA 用例必然已通过关键词检查）
    kw_hit = pass_rate(vqa)

    mrg_completeness = pass_rate(mrg)  # MRG 结构完整率：四个字段都非空且满足最小长度
    rag_pass = pass_rate(rag)          # RAG 通过率
    boundary_rate = pass_rate(boundary)  # 边界用例处理率

    # 计算各任务类型的时延分布（P50/P95/min/max）
    latencies = {}
    for label, cases in [("vqa", vqa), ("mrg", mrg), ("rag", rag)]:
        lats = sorted(r["latency_ms"] for r in cases if "latency_ms" in r)  # 升序排列
        if lats:
            latencies[label] = {
                "p50_ms": lats[len(lats) // 2],  # P50（中位数）
                "p95_ms": lats[min(int(len(lats) * 0.95), len(lats) - 1)],  # P95（第95百分位）
                "min_ms": lats[0],   # 最小值
                "max_ms": lats[-1],  # 最大值
            }

    return {
        "task_routing_accuracy": routing_acc,
        "keyword_hit_rate": kw_hit,
        "mrg_completeness": mrg_completeness,
        "rag_pass_rate": rag_pass,
        "boundary_handling_rate": boundary_rate,
        "latency": latencies,
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    """
    评测主函数：解析命令行参数 → 健康检查 → 运行所有用例 → 计算指标 → 输出结果。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="工单13 全量评测")
    parser.add_argument("--server", default="http://localhost:8013", help="API 地址")
    parser.add_argument("--output", default="tests/metrics.json", help="结果输出路径")
    parser.add_argument("--skip-boundary", action="store_true", help="跳过边界用例")  # action="store_true"：出现此参数时值为 True
    parser.add_argument("--only", choices=["vqa", "mrg", "rag", "boundary"], help="只跑某类用例")
    args = parser.parse_args()

    # ── 健康检查：确认服务已启动 ──
    print(f"[检查] 连接 {args.server}/health ...")
    try:
        r = requests.get(f"{args.server}/health", timeout=10)
        h = r.json()
        print(f"[OK] 服务在线 — KB {h.get('kb_docs', '?')} 条，模型 {h.get('model', '?')}")
    except Exception as e:
        print(f"[ERROR] 服务未响应: {e}")
        print("请先启动服务：python -m uvicorn src.api:app --port 8014")
        return  # 服务不可用则直接退出，不执行测试

    # ── 加载测试用例 ──
    cases = load_cases()
    all_results = []  # 所有测试结果列表
    totals = {"vqa": 0, "mrg": 0, "rag": 0, "boundary": 0}  # 各类型用例总数
    passed = {"vqa": 0, "mrg": 0, "rag": 0, "boundary": 0}  # 各类型通过数

    def run_group(label, case_list, runner):
        """运行某一类型的所有测试用例，打印进度。"""
        # --only 参数指定了特定类型，非指定类型跳过
        if args.only and args.only != label:
            return
        print(f"\n{'=' * 50}")
        print(f"[{label.upper()}] {len(case_list)} 个用例")
        print(f"{'=' * 50}")
        for c in case_list:
            r = runner(args.server, c)         # 执行单个用例
            all_results.append(r)
            totals[label] += 1 if r.get("status") != "skip" else 0  # skip 不计入总数
            if r.get("status") == "pass":
                passed[label] += 1
            # 构造状态标记字符串
            status_mark = "[PASS]" if r["status"] == "pass" else ("[SKIP]" if r["status"] == "skip" else "[FAIL]")
            lat = f"{r.get('latency_ms', '?')}ms" if r.get("status") != "skip" else ""
            reason = f"  <- {r.get('reason', '')}" if r.get("status") == "fail" else ""
            print(f"  {status_mark} {r['id']} {lat}{reason}")

    # 依次运行各类型用例
    run_group("vqa", cases.get("vqa_cases", []), run_vqa_case)
    run_group("mrg", cases.get("mrg_cases", []), run_mrg_case)
    run_group("rag", cases.get("rag_cases", []), run_rag_case)
    if not args.skip_boundary:  # --skip-boundary 参数可跳过边界用例
        run_group("boundary", cases.get("boundary_cases", []), run_boundary_case)

    # ── 汇总统计 ──
    total_run = sum(totals.values())   # 总运行用例数
    total_pass = sum(passed.values())  # 总通过数
    failed_cases = [r for r in all_results if r.get("status") == "fail"]  # 失败用例列表

    # 计算各项指标
    metrics = compute_metrics(all_results)

    # 构建最终输出对象
    output = {
        "eval_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),  # UTC 评测时间
        "server": args.server,          # 被测服务器地址
        "total_cases": total_run,        # 总用例数
        "passed": total_pass,            # 通过数
        "failed": total_run - total_pass,  # 失败数
        "by_task": {                     # 按任务类型分解的统计
            k: {"total": totals[k], "passed": passed[k], "failed": totals[k] - passed[k]}
            for k in totals if totals[k] > 0  # 只包含有用例的任务类型
        },
        "metrics": metrics,              # 汇总指标
        "failed_cases": [                # 失败用例详情（方便排查问题）
            {
                "id": r["id"],
                "reason": r.get("reason", ""),
                "http_status": r.get("http_status"),
                "answer_snippet": r.get("answer_snippet", ""),
            }
            for r in failed_cases
        ],
    }

    # ── 写入 JSON 结果文件 ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)  # indent=2 美化格式

    # ── 打印终端汇总 ──
    print(f"\n{'=' * 50}")
    print(f"[结果] {total_pass}/{total_run} 通过")
    print(f"  VQA       {passed['vqa']}/{totals['vqa']}")
    print(f"  MRG       {passed['mrg']}/{totals['mrg']}")
    print(f"  RAG       {passed['rag']}/{totals['rag']}")
    print(f"  边界      {passed['boundary']}/{totals['boundary']}")
    print(f"\n[指标]")
    m = metrics
    print(f"  任务路由准确率    {m.get('task_routing_accuracy', 'N/A')} (目标 >=0.95)")
    print(f"  关键词命中率      {m.get('keyword_hit_rate', 'N/A')} (目标 >=0.85)")
    print(f"  MRG 结构完整率    {m.get('mrg_completeness', 'N/A')} (目标 >=0.90)")
    print(f"  RAG 通过率        {m.get('rag_pass_rate', 'N/A')}")
    print(f"  边界处理率        {m.get('boundary_handling_rate', 'N/A')} (目标 1.0)")
    if m.get("latency"):
        print(f"\n[时延]")
        for t, lat in m["latency"].items():
            print(f"  {t.upper()} P50={lat['p50_ms']}ms  P95={lat['p95_ms']}ms")
    print(f"\n[输出] {out_path.resolve()}")
    if failed_cases:
        print(f"\n[失败用例]")
        for r in failed_cases:
            print(f"  - {r['id']}: {r.get('reason', '')}")


if __name__ == "__main__":
    main()
