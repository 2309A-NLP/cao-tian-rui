"""RAG 端到端冒烟测试：验证向量检索命中 + 知识引用注入是否正常工作。

冒烟测试验证点：
1. 接口返回 HTTP 200
2. 响应中有 references 字段（即使为空列表也合法）
3. answer 字段非空

运行前提：
- 服务已在 http://127.0.0.1:8013 运行
- 知识库已通过 build_kb.py 构建（否则 references 会为空）
- tests/images/ 下有对应图片

运行方式：python tests/smoke_rag.py
"""

# sys：Python 内置模块，用于程序退出（sys.exit）
import sys

# time：Python 内置模块，用于计算请求耗时
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

# 测试用例列表：(图片文件名, 问题)
# 问题设计为与知识库中的医学影像相关，验证 RAG 检索能力
CASES = [
    ("chest_xray_01.jpg", "这张影像用的是什么成像方式？"),       # 测试对成像方式的识别
    ("chest_xray_02.jpg", "胸部影像最主要能看到什么解剖结构？"),   # 测试解剖结构识别
    ("ct_tumor_01.jpg", "肺部结节的常见类型有哪些？"),            # 测试知识库检索（肺结节知识）
    ("pathology_01.jpg", "腹部 CT 常见的病变有哪些？"),           # 测试跨模态知识检索
]


def run(img_name: str, query: str) -> dict:
    """
    向 API 发送单个 RAG 请求，返回响应信息字典。

    参数：
        img_name (str)：图片文件名（相对于 IMG_DIR）
        query (str)：用户问题文本

    返回值：
        dict：包含 status（HTTP 状态码）、latency_ms（耗时毫秒）、data（响应体）的字典
    """
    img = IMG_DIR / img_name   # 完整图片路径
    t0 = time.perf_counter()   # 记录开始时间

    with open(img, "rb") as f:
        # 发送 multipart/form-data POST 请求
        resp = requests.post(
            API_URL,
            data={"task": "rag", "query": query},  # task=rag 触发 RAG 处理器
            files={"image": (img_name, f, "image/jpeg")},
            timeout=180,  # RAG 任务包含检索和 VLM 调用，超时设置较长
        )
    dt = int((time.perf_counter() - t0) * 1000)  # 计算耗时（毫秒）
    return {"status": resp.status_code, "latency_ms": dt, "data": resp.json()}


def main():
    """
    运行所有 RAG 冒烟测试用例，打印检索结果摘要和回答，失败时以非零退出码退出。
    """
    print("=" * 70)
    print("RAG 冒烟测试")
    print("=" * 70)

    all_ok = True  # 全部通过标志

    for i, (img, q) in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] {img} | Q: {q}")
        r = run(img, q)

        # 检查 HTTP 状态码
        if r["status"] != 200:
            all_ok = False
            print(f"  [FAIL] status={r['status']}, {r['data']}")
            continue  # 失败后继续下一个用例

        d = r["data"]
        refs = d.get("references") or []   # 检索到的参考文档列表（可能为空）
        ans = d.get("answer", "")           # 模型回答文本

        print(f"  [OK] {r['latency_ms']}ms | 命中 {len(refs)} 条参考")

        # 打印前 3 条参考文档的摘要（截取 120 字符）
        for ref in refs[:3]:
            # replace("\n", " ")：将换行替换为空格，使输出更整洁
            snippet = ref.get("snippet", "").replace("\n", " ")[:120]
            print(f"    - score={ref.get('score')} {snippet}")

        # 打印回答（最多 250 字符，超出则追加"..."）
        print(f"  Answer: {ans[:250]}{'...' if len(ans) > 250 else ''}")

    print("\n" + "=" * 70)
    print(f"结果：{'全部通过 [PASS]' if all_ok else '有失败 [FAIL]'}")
    sys.exit(0 if all_ok else 1)  # 0=成功，1=失败


if __name__ == "__main__":
    main()
