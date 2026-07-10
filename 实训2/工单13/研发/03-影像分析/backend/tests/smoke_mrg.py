"""MRG 端到端冒烟测试：向运行中的服务发送 4 张 SLAKE 图，各生成一份结构化报告。

冒烟测试（Smoke Test）是一种快速验证服务基本功能是否正常的测试，
不追求完整覆盖，只验证"主流程不冒烟（不崩溃）"。

运行前提：服务已在 http://127.0.0.1:8013 运行，且 tests/images/ 下有对应图片。
运行方式：python tests/smoke_mrg.py
"""

# sys：Python 内置模块，用于程序退出（sys.exit）和路径操作
import sys

# time：Python 内置模块，用于计算请求耗时（perf_counter 精度更高）
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

# 测试用例列表：(图片文件名, 可选的临床背景上下文)
# 临床背景通过 query 参数传入（MRG 任务的 extra_context）
CASES = [
    ("chest_xray_01.jpg", ""),              # 胸部 X 光，无临床背景
    ("chest_xray_02.jpg", "45岁女性，胸闷 3 月"),  # 胸部 X 光，有临床背景
    ("ct_tumor_01.jpg", "68岁男性，体检发现肺部结节"),  # CT 影像，有临床背景
    ("pathology_01.jpg", ""),              # 病理图像，无临床背景
]


def run(img_name: str, ctx: str) -> dict:
    """
    向 API 发送单个 MRG 请求，返回响应信息字典。

    参数：
        img_name (str)：图片文件名（相对于 IMG_DIR）
        ctx (str)：临床背景信息（可为空字符串）

    返回值：
        dict：包含 status（HTTP 状态码）、latency_ms（耗时毫秒）、data（响应体）的字典
    """
    img = IMG_DIR / img_name       # 完整图片路径
    t0 = time.perf_counter()       # 记录开始时间（perf_counter 比 time.time 精度更高）

    with open(img, "rb") as f:     # 以二进制模式打开图片文件
        data = {"task": "mrg"}     # 基础请求参数：任务类型为 mrg
        if ctx:
            data["query"] = ctx    # 若有临床背景，通过 query 参数传入
        # 发送 multipart/form-data POST 请求
        # files 参数将图片以表单文件形式上传
        resp = requests.post(
            API_URL, data=data,
            files={"image": (img_name, f, "image/jpeg")},  # (文件名, 文件对象, MIME类型)
            timeout=120,  # 超时 120 秒（MRG 任务可能较慢）
        )
    # 计算耗时（毫秒）
    dt = int((time.perf_counter() - t0) * 1000)
    return {"status": resp.status_code, "latency_ms": dt, "data": resp.json()}


def main():
    """
    运行所有 MRG 冒烟测试用例，打印结果，失败时以非零退出码退出。
    """
    print("=" * 70)
    print("MRG 冒烟测试")
    print("=" * 70)

    all_ok = True  # 全部通过标志，任一用例失败则改为 False

    for i, (img, ctx) in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] {img} | 背景: {ctx or '(无)'}")
        r = run(img, ctx)

        # 检查 HTTP 状态码
        if r["status"] != 200:
            all_ok = False
            print(f"  [FAIL] status={r['status']}, {r['data']}")
            continue  # 失败后继续下一个用例

        d = r["data"]
        rpt = d.get("report") or {}  # 获取报告字段（若不存在则用空字典）

        print(f"  [OK] {r['latency_ms']}ms")
        # 截取各字段前 80/120 个字符打印，避免输出过长
        print(f"  主诉:   {rpt.get('chief_complaint','')[:80]}")
        print(f"  所见:   {rpt.get('findings','')[:120]}")
        print(f"  印象:   {rpt.get('impression','')[:120]}")
        print(f"  建议:   {rpt.get('recommendation','')[:120]}")

        # 检查 findings 是否为空（空 findings 表示可能触发了降级）
        if not rpt.get("findings"):
            print("  [WARN] findings 为空，可能解析降级")

    print("\n" + "=" * 70)
    print(f"结果：{'全部通过 [PASS]' if all_ok else '有失败 [FAIL]'}")
    sys.exit(0 if all_ok else 1)  # 0=成功，1=失败（便于 CI/CD 检测）


if __name__ == "__main__":
    main()
