"""
性能基准测试脚本：分场景统计首字节延迟、总耗时、规则命中率、缓存命中。

本脚本直接调用运行中的 FastAPI 服务（端口 8012），通过 SSE 流式接口逐事件解析，
覆盖五类场景：规则命中 / LLM 兜底 / 缓存复测 / 容错 / 生僻疾病。
运行前请确保服务已启动：uvicorn src.app:app --port 8012
"""

import json          # 标准库：JSON 编解码，用于构造请求体和解析 SSE 事件
import time          # 标准库：高精度计时，用于测量首字节延迟和总耗时
import urllib.request  # 标准库：HTTP 请求客户端，无需安装第三方库

# 服务基础 URL，指向本地运行的 FastAPI 健康咨询 Agent
BASE = "http://127.0.0.1:8012"

# ──────────────────────────────────────────────────────
# 测试用例集（label 标识场景类型，q 是问题文本）
# 共 5 大类 26 题
# ──────────────────────────────────────────────────────
CASES = [
    # ── 规则命中场景（14 题）：关键词直接触发意图，0ms 意图识别 ──
    ("rule", "百日咳的并发症有哪些？"),       # 并发症类关键词
    ("rule", "百日咳怎么传播？"),             # 传播类关键词
    ("rule", "百日咳能吃什么？"),             # 饮食类关键词
    ("rule", "百日咳挂什么科？"),             # 科室类关键词
    ("rule", "糖尿病用什么药？"),             # 用药类关键词
    ("rule", "高血压不能吃什么？"),           # 饮食禁忌类关键词
    ("rule", "流感怎么预防？"),               # 预防类关键词
    ("rule", "乙肝的病因是什么？"),           # 病因类关键词
    ("rule", "肺炎的血常规有什么特征？"),     # disease_info 类（含"血常规"）
    ("rule", "肾结石怎么治疗？"),             # 治疗类关键词
    ("rule", "阑尾炎的症状有哪些？"),         # 症状类关键词
    ("rule", "哮喘的护理需要注意什么？"),     # 护理类关键词
    ("rule", "头痛可能是什么病？"),           # 症状→疾病反查
    ("rule", "偏头痛怎么预防发作？"),         # 预防类关键词

    # ── LLM 兜底场景（5 题）：句式复杂/多限定，规则无法高置信度命中 ──
    ("llm",  "百日咳最具特征性的临床表现是什么？"),   # 含"最具特征性"等限定
    ("llm",  "护理百日咳患儿时需特别注意防范什么紧急情况？"),  # 复杂句式
    ("llm",  "百日咳西医治疗首选的抗生素是什么？"),   # 含"西医"限定
    ("llm",  "小儿感冒发烧时家长该做什么？"),         # 含"小儿""家长"等限定
    ("llm",  "中医治疗痉咳期百日咳的主方是什么？"),   # 含"中医""痉咳期"双限定

    # ── 缓存复测（3 题）：与规则场景重复，应命中 LRU 缓存 ──
    ("cache", "百日咳的并发症有哪些？"),   # 第二次请求，期望 cache_hit=True
    ("cache", "糖尿病用什么药？"),         # 同上
    ("cache", "肾结石怎么治疗？"),         # 同上

    # ── 容错场景（4 题）：无效输入，期望返回 error 事件而不崩溃 ──
    ("error", ""),          # 空字符串
    ("error", "   "),       # 纯空白
    ("error", "!!!???"),    # 纯符号（无中文/字母）
    ("error", "啊" * 600),  # 超长（600 > 最大 500 字）
]


def stream_once(q):
    """
    向 /chat/stream 发送一次流式请求，解析 SSE 事件流，返回性能指标字典。

    参数:
        q (str): 问题文本（可为空或无效，用于容错测试）

    返回:
        dict: {
            "first_byte_ms": 首字节耗时毫秒（None 表示超时前无数据）,
            "total_ms": 总耗时毫秒,
            "reply_len": token 事件中 content 的总字符数,
            "intent": meta 事件中的意图字段,
            "source": 意图来源（"rule"/"llm"/"error"），
            "cache_hit": 是否命中缓存（bool）,
            "error": 错误信息字符串或 None
        }
    """
    # 将查询体序列化为 JSON 字节串（ensure_ascii=False 保留中文）
    body = json.dumps({"query": q}, ensure_ascii=False).encode("utf-8")

    # 构造 HTTP POST 请求对象，指定 Content-Type
    req = urllib.request.Request(
        f"{BASE}/chat/stream", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )

    t0 = time.perf_counter()  # 记录请求开始时间（高精度单调时钟）
    first = None               # 首字节时刻（延迟到第一个数据块到达时赋值）
    meta, done = None, None    # 分别存储 meta 事件和 done 事件的内容
    reply_len = 0              # 累计 token 字符数
    err_msg = None             # 错误信息（校验失败或 API 异常）

    try:
        # 发起 HTTP 请求，60 秒超时；with 块结束时自动关闭连接
        with urllib.request.urlopen(req, timeout=60) as r:
            buf = b""          # 字节缓冲区，用于拼接分片数据

            while True:
                # read1() 是非阻塞单次读取（如果响应对象支持）；否则退回 read(4096)
                chunk = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
                if not chunk:
                    break  # 服务端关闭连接，流结束

                # 记录首字节延迟（只在第一个非空 chunk 时赋值）
                if first is None:
                    first = round((time.perf_counter() - t0) * 1000)

                buf += chunk  # 追加到缓冲区

                # SSE 事件以 "\n\n" 分隔，逐事件解析
                while b"\n\n" in buf:
                    ev_raw, buf = buf.split(b"\n\n", 1)  # 分割出一个完整事件

                    for line in ev_raw.split(b"\n"):  # 事件内可能有多行
                        if line.startswith(b"data:"):  # 只处理 data: 行
                            # 去掉 "data:" 前缀后解析 JSON
                            evt = json.loads(line[5:].strip().decode("utf-8"))

                            if evt.get("type") == "meta":
                                # meta 事件：包含 intent/entity/intent_source
                                meta = evt
                            elif evt.get("type") == "token":
                                # token 事件：流式文字片段，累加字符数
                                reply_len += len(evt.get("content", ""))
                            elif evt.get("type") == "done":
                                # done 事件：最终汇总（含 cache_hit）
                                done = evt
                            elif evt.get("type") == "error":
                                # error 事件：校验失败或 Agent 异常
                                err_msg = evt.get("message", "")

    except urllib.error.HTTPError as e:
        # HTTP 422：Pydantic 校验失败（如超长 query 被拒绝）
        first = round((time.perf_counter() - t0) * 1000)  # 仍记录首字节（错误响应）
        err_msg = f"HTTP {e.code}"

    # 计算总耗时（毫秒）
    total = round((time.perf_counter() - t0) * 1000)

    return {
        "first_byte_ms": first,
        "total_ms": total,
        "reply_len": reply_len,
        "intent": meta.get("intent") if meta else None,        # 意图字符串
        "source": meta.get("intent_source") if meta else ("error" if err_msg else None),  # 意图来源
        "cache_hit": done.get("cache_hit", False) if done else False,  # 缓存命中标志
        "error": err_msg,
    }


# ──────────────────────────────────────────────────────
# 清空缓存：确保规则/LLM 场景是"冷启动"，消除上次测试缓存污染
# ──────────────────────────────────────────────────────
urllib.request.urlopen(
    urllib.request.Request(
        f"{BASE}/cache/clear",       # 清缓存端点
        method="POST",
        headers={"Content-Type": "application/json"},
        data=b"{}",                  # 空 JSON 请求体
    ),
).read()  # 读取响应（丢弃），确保请求完成

# ──────────────────────────────────────────────────────
# 逐题执行，收集结果
# ──────────────────────────────────────────────────────
results = []
for label, q in CASES:
    r = stream_once(q)          # 发起流式请求并解析
    r["label"] = label          # 附加场景标签
    # 截断过长的查询文本，便于打印
    r["query"] = q if len(q) < 40 else q[:37] + "..."
    results.append(r)
    time.sleep(0.6)             # 防止请求过于密集，触发 API 限流


# ──────────────────────────────────────────────────────
# 统计函数：按 label 筛选，计算首字节延迟和总耗时的 min/avg/max
# ──────────────────────────────────────────────────────
def stat(results, label):
    """
    计算指定 label 场景的性能统计数据。

    参数:
        results (list): 所有测试结果列表
        label (str): 场景标签（"rule"/"llm"/"cache"/"error"）

    返回:
        dict: 包含 count / first_byte_min / first_byte_avg / first_byte_max / total_avg，
              或 None（该 label 无数据）
    """
    xs = [r for r in results if r["label"] == label]  # 筛选目标 label 的数据
    if not xs:
        return None  # 无数据时返回 None，跳过打印

    # 过滤掉 None（超时前无数据的场景），避免 min/sum 出错
    fbs = [r["first_byte_ms"] for r in xs if r["first_byte_ms"] is not None]
    tot = [r["total_ms"] for r in xs if r["total_ms"] is not None]

    return {
        "count": len(xs),
        "first_byte_min": min(fbs) if fbs else None,                          # 最快首字节
        "first_byte_avg": round(sum(fbs) / len(fbs)) if fbs else None,        # 平均首字节
        "first_byte_max": max(fbs) if fbs else None,                          # 最慢首字节
        "total_avg": round(sum(tot) / len(tot)) if tot else None,             # 平均总耗时
    }


# ──────────────────────────────────────────────────────
# 打印逐题明细表格
# ──────────────────────────────────────────────────────
print("\n" + "=" * 96)
print(f"{'类别':<8} {'意图源':<8} {'首字节ms':<10} {'总耗时ms':<10} {'字数':<6} {'缓存':<6} {'查询'}")
print("=" * 96)

for r in results:
    src   = r["source"] or "-"                             # 意图来源（rule/llm/error/-）
    fb    = r["first_byte_ms"] or "-"                      # 首字节延迟
    tot   = r["total_ms"] or "-"                           # 总耗时
    cache = "Yes" if r["cache_hit"] else ""               # 缓存命中标记
    err   = f" [ERROR: {r['error']}]" if r.get("error") else ""  # 错误信息后缀

    print(f"{r['label']:<8} {src:<8} {str(fb):<10} {str(tot):<10} "
          f"{r['reply_len']:<6} {cache:<6} {r['query']}{err}")

# ──────────────────────────────────────────────────────
# 打印汇总统计
# ──────────────────────────────────────────────────────
print("\n" + "=" * 96)
print("汇总统计")
print("=" * 96)

for label in ["rule", "llm", "cache", "error"]:
    s = stat(results, label)
    if s:
        print(f"{label:<8} 用例数={s['count']:<3} 首字节: min={s['first_byte_min']}ms "
              f"avg={s['first_byte_avg']}ms max={s['first_byte_max']}ms  总耗时 avg={s['total_avg']}ms")

# ──────────────────────────────────────────────────────
# 规则命中率（排除容错场景后计算）
# ──────────────────────────────────────────────────────
non_err = [r for r in results if r["label"] != "error" and r["source"]]  # 非容错且有来源
rule_hit = sum(1 for r in non_err if r["source"] == "rule")               # 规则命中数量
print(f"\n规则命中率（非容错场景）: {rule_hit}/{len(non_err)} = {round(rule_hit / len(non_err) * 100, 1)}%")
print(f"缓存命中数: {sum(1 for r in results if r['cache_hit'])}")          # 缓存复测命中总数
