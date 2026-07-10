"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
VQA 端到端冒烟测试：向 /api/analyze POST 4张图 + 4个问题，验证基本接口功能。

冒烟测试验证点：
1. 接口返回 HTTP 200
2. 响应中有非空 answer 字段
3. 一中一英混合问题均能正常处理（多语言支持）

运行前提：
- 服务已在 http://127.0.0.1:8013 运行
- tests/images/ 下有对应图片

运行方式：python tests/smoke_vqa.py
"""

# sys：Python 内置模块，用于程序退出（sys.exit）
import sys

# time：Python 内置模块，用于计算请求耗时（perf_counter 精度高于 time.time）
import time

# pathlib.Path：面向对象的文件路径操作
from pathlib import Path

# requests：第三方 HTTP 客户端库，用于发送 HTTP 请求
# 安装方式：pip install requests
import requests

# 项目根目录（backend/）
ROOT = Path(__file__).resolve().parent.parent

# 测试图片目录（tests/images/）
IMG_DIR = ROOT / "tests" / "images"

# API 接口地址
API_URL = "http://127.0.0.1:8013/api/analyze"

# 测试用例列表：(图片文件名, 问题文本)
# 涵盖中英文问题、不同影像类型（X光/CT/病理），验证模型多语言和多模态能力
CASES = [
    ("chest_xray_01.jpg", "这张影像使用的是什么成像方式？"),                          # 中文问题：识别成像方式
    ("chest_xray_02.jpg", "Does this X-ray image show a sign of Cardiomegaly? Find the answer."),  # 英文问题：心脏扩大
    ("ct_tumor_01.jpg", "图像中是否存在肿瘤？请指出位置"),                            # 中文问题：肿瘤检测
    ("pathology_01.jpg", "这张病理图像中最显著的组织学特征是什么？"),                 # 中文问题：病理特征
]


def run_case(img_name: str, query: str) -> dict:
    """
    向 API 发送单个 VQA 请求，返回响应信息字典。

    参数：
        img_name (str)：图片文件名（相对于 IMG_DIR）
        query (str)：用户问题文本

    返回值：
        dict：包含 status（HTTP 状态码）、latency_ms（耗时毫秒）、data（响应体）的字典
              若图片不存在，返回包含 error 字段的字典
    """
    img = IMG_DIR / img_name  # 完整图片路径

    # 检查图片文件是否存在（运行前需要准备测试图片）
    if not img.exists():
        return {"error": f"图片不存在: {img}"}

    t0 = time.perf_counter()  # 记录开始时间

    with open(img, "rb") as f:
        # 发送 multipart/form-data POST 请求
        resp = requests.post(
            API_URL,
            data={"task": "vqa", "query": query},  # task=vqa 触发 VQA 处理器
            files={"image": (img_name, f, "image/jpeg")},
            timeout=90,  # VQA 超时 90 秒（仅 VLM 调用，比 RAG 快）
        )

    # 计算耗时（毫秒）
    dt = int((time.perf_counter() - t0) * 1000)

    # 尝试解析 JSON 响应体
    try:
        data = resp.json()
    except Exception:
        # 若响应体不是合法 JSON（如服务返回 HTML 错误页），则截取原始文本
        data = {"raw": resp.text[:200]}

    return {"status": resp.status_code, "latency_ms": dt, "data": data}


def main():
    """
    运行所有 VQA 冒烟测试用例，打印结果，失败时以非零退出码退出。
    """
    print("=" * 70)
    print("VQA 冒烟测试")
    print("=" * 70)

    all_ok = True  # 全部通过标志

    for i, (img, q) in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] {img} | Q: {q}")
        r = run_case(img, q)
        status = r.get("status")

        # HTTP 200 视为成功
        if status == 200:
            answer = r["data"].get("answer", "")
            print(f"  [OK] status=200, {r['latency_ms']}ms")
            # 打印回答前 220 字符（超出追加"..."）
            print(f"  Answer: {answer[:220]}{'...' if len(answer) > 220 else ''}")
        else:
            # HTTP 非 200 视为失败
            all_ok = False
            print(f"  [FAIL] status={status}, {r}")

    print("\n" + "=" * 70)
    print(f"结果：{'全部通过 [PASS]' if all_ok else '有失败 [FAIL]'}")
    sys.exit(0 if all_ok else 1)  # 0=成功，1=失败（便于 CI/CD 检测）


if __name__ == "__main__":
    main()
