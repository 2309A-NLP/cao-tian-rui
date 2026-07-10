"""
流式接口烟测（Smoke Test）：验证首字节延迟 + 规则命中 + 缓存。

对 /chat/stream 端点发送 5 条测试问题，覆盖：
  - 规则命中（3 题）：关键词直接触发意图，首字节应 < 500ms
  - LLM 兜底（1 题）：复杂句式，转交 LLM 分析
  - 缓存复测（1 题）：与规则题重复，期望 cache_hit=True
运行前请确保服务已启动：uvicorn src.app:app --port 8012
"""

import json            # 标准库：序列化请求体、解析 SSE 事件 JSON
import time            # 标准库：计时（首字节延迟 / 总耗时）与限流间隔
import urllib.request  # 标准库：HTTP 客户端，无需第三方包

# 服务基础 URL
BASE = "http://127.0.0.1:8012"

# ──────────────────────────────────────────────────────
# 测试用例：(场景名称, 问题文本)
# ──────────────────────────────────────────────────────
TESTS = [
    ("并发症（规则命中）", "百日咳的并发症有哪些？"),   # 含"并发症"关键词 → 规则
    ("传播（规则命中）",   "乙肝怎么传播？"),           # 含"怎么传播"关键词 → 规则
    ("饮食（规则命中）",   "糖尿病不能吃什么？"),       # 含"不能吃什么"关键词 → 规则
    ("模糊（LLM 兜底）",   "百日咳最具特征性的临床表现是什么？"),  # 含"最具特征性"限定 → LLM
    ("缓存复测",           "百日咳的并发症有哪些？"),   # 与第1题相同，应命中缓存
]


def stream_once(name, q):
    """
    向 /chat/stream 发送一次 SSE 流式请求，解析并打印性能指标。

    参数:
        name (str): 场景名称（用于打印标识）
        q (str): 查询问题文本
    """
    # 构造 JSON 请求体
    body = json.dumps({"query": q}, ensure_ascii=False).encode("utf-8")

    # 构造 HTTP 请求，Accept: text/event-stream 表示接受 SSE 格式响应
    req = urllib.request.Request(
        f"{BASE}/chat/stream", data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",  # 告知服务端期望 SSE 流
        },
        method="POST",
    )

    t0 = time.perf_counter()  # 请求发出时刻（高精度单调时钟，不受系统时间调整影响）
    first_byte = None          # 首字节时刻（第一个非空 chunk 到达时赋值）
    meta = None                # meta 事件内容（含 intent/entity/intent_source）
    reply_len = 0              # token 事件字符数累计
    done = None                # done 事件内容（含 cache_hit/elapsed_ms）

    try:
        # 建立连接，60 秒超时
        with urllib.request.urlopen(req, timeout=60) as r:
            buf = b""  # 字节缓冲区，用于处理 TCP 分片数据

            while True:
                # read1() 一次读取可用数据（非阻塞），read(4096) 是后备方案
                chunk = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
                if not chunk:
                    break  # 连接已关闭，流结束

                # 第一个非空 chunk 到达 → 记录首字节延迟
                if first_byte is None:
                    first_byte = round((time.perf_counter() - t0) * 1000)

                buf += chunk  # 追加到缓冲区

                # SSE 事件以 "\n\n" 作为分隔符
                while b"\n\n" in buf:
                    evt_raw, buf = buf.split(b"\n\n", 1)  # 分割出一个完整事件块

                    for line in evt_raw.split(b"\n"):       # 事件块内逐行处理
                        if line.startswith(b"data:"):       # 只关心 data: 行
                            payload = line[5:].strip()      # 去掉 "data:" 前缀
                            if not payload:
                                continue  # 空 data 行（心跳）跳过

                            evt = json.loads(payload.decode("utf-8"))  # 解析 JSON

                            # 根据事件类型分别处理
                            if evt.get("type") == "meta":
                                # meta：首条事件，包含意图识别结果
                                meta = evt
                            elif evt.get("type") == "token":
                                # token：流式文字片段，累加字符数统计回复长度
                                reply_len += len(evt.get("content", ""))
                            elif evt.get("type") == "done":
                                # done：流结束事件，包含 cache_hit 标志
                                done = evt

    except Exception as e:
        # 捕获所有异常（网络超时、连接拒绝等），打印错误后返回
        print(f"[ERR ] {name}: {e}")
        return

    # 计算总耗时（从请求发出到最后一个 chunk 接收完毕）
    total = round((time.perf_counter() - t0) * 1000)

    # 从解析结果中提取关键指标
    src    = meta.get("intent_source") if meta else "?"   # 意图来源（rule/llm）
    hit    = done.get("cache_hit") if done else False      # 缓存命中标志
    intent = meta.get("intent") if meta else "?"           # 意图分类结果
    entity = meta.get("entity") if meta else "?"           # 提取的医学实体

    # 打印单条测试结果（对齐格式便于比对）
    print(f"[OK ] {name:24} | intent={intent:26} entity={entity:8} "
          f"src={src:5} first={first_byte}ms total={total}ms len={reply_len} cache={hit}")


# ──────────────────────────────────────────────────────
# 逐题执行，每题间隔 0.5 秒（防限流）
# ──────────────────────────────────────────────────────
for name, q in TESTS:
    stream_once(name, q)
    time.sleep(0.5)  # 短暂等待，避免请求过于密集
