"""
快速验证 /chat 接口的脚本，无需 pytest 环境。

直接使用标准库 urllib 发起 HTTP 请求，适合在无 pytest 的环境中
快速验证服务是否正常响应。
运行前请确保服务已启动：uvicorn src.app:app --port 8012
"""

import urllib.request  # 标准库：HTTP 客户端，发送 POST 请求
import json            # 标准库：序列化请求体、解析响应 JSON
import time            # 标准库：控制请求间隔，防止 API 限流
import sys             # 标准库：通过退出码反映测试结果（暂未用，保留以便扩展）

# 服务基础 URL
BASE = "http://127.0.0.1:8012"

# ──────────────────────────────────────────────────────
# 测试用例列表：(名称, 问题, 关键词列表)
# 关键词列表中任一关键词出现在回复里即视为 PASS
# ──────────────────────────────────────────────────────
TESTS = [
    # 并发症测试：回复中应提到"支气管肺炎"或"肺不张"
    ("并发症", "百日咳最常见的严重并发症是什么？", ["支气管肺炎", "肺不张"]),
    # 饮食忌口测试：回复中应提到海鲜类食物
    ("饮食忌口", "百日咳患者应避免食用哪类食物？", ["海鲜", "螃蟹", "海虾"]),
    # 血常规测试：回复中应提到白细胞/淋巴细胞异常
    ("血常规", "百日咳患者的血常规检查会呈现什么特征？", ["白细胞", "淋巴细胞"]),
    # 隔离期测试：回复中应包含"40天"相关表述
    ("隔离期", "百日咳患者的隔离期应持续多久？", ["40天", "40 天", "四十天"]),
]

passed = 0  # 累计通过用例数

for name, q, keywords in TESTS:
    # 将问题编码为 UTF-8 JSON 字节串
    data = json.dumps({"query": q}, ensure_ascii=False).encode("utf-8")

    # 构造 HTTP POST 请求，指向非流式 /chat 接口
    req = urllib.request.Request(
        f"{BASE}/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        # 发送请求，30 秒超时；urlopen 返回响应对象
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())  # 读取响应体并解析为 dict

        # 检查回复中是否包含任一预期关键词
        ok = any(kw in d["reply"] for kw in keywords)
        status = "PASS" if ok else "FAIL"

        if ok:
            passed += 1  # 通过则计数

        # 打印单题结果：状态 / 意图 / 实体 / 耗时 / 回复摘要
        print(f"[{status}] {name} | intent={d['intent']} entity={d['entity']} {d['elapsed_ms']}ms")
        print(f"       reply: {d['reply'][:100]}")  # 只打印前 100 字，避免输出过长

    except Exception as e:
        # 网络异常或服务未启动时捕获，不中断后续用例
        print(f"[ERR ] {name} | {e}")

    time.sleep(1)  # 每题间隔 1 秒，避免触发 API 限流

# 打印汇总结果
print(f"\n结果: {passed}/{len(TESTS)} 通过")
