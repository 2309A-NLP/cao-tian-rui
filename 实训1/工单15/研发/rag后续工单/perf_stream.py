# -*- coding: utf-8 -*-
# 流式首字延迟(TTFT)测试：衡量"系统多快开始回应"。
import json, time, urllib.request
TOKEN = "ragflow-reparse-tmp"
CHAT = "40d4a4e468ab11f1868a810dbccbbded"
BASE = "http://localhost:9380/api/v1/chats/" + CHAT
H = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}

def post_json(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=H, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=120))

qs = [
    ("Q1", "根据文本信息，该静电除尘器的发明人是？"),
    ("Q2", "该静电除尘器的管状入口具有怎样的结构特征？"),
    ("Q3", "在第7页图片中，部件4相对于部件5在图中的位置关系是？"),
    ("Q4", "在第7页图片中，尺寸X1、X2、X3分别代表什么部件之间的间隔距离？"),
    ("Q5", "根据第7页图示，气流方向(7)首先经过哪个部件？紧接着经过哪个部件？"),
    ("Q6", "根据第7页图示，已知外壳直径D，h1和h2的尺寸可以用来计算/确定什么？"),
]

ttfts, totals = [], []
print("=== 流式首字延迟(TTFT) + 整段完成 ===")
for tag, q in qs:
    sid = post_json("/sessions", {"name": tag})["data"]["id"]
    body = json.dumps({"question": q, "session_id": sid, "stream": True}).encode()
    req = urllib.request.Request(BASE + "/completions", data=body, headers=H, method="POST")
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "true":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            ans = obj.get("data", {})
            txt = ans.get("answer", "") if isinstance(ans, dict) else ""
            if ttft is None and txt.strip():
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    ttfts.append(ttft if ttft else total)
    totals.append(total)
    print(f"{tag}: TTFT={ttfts[-1]:5.2f}s {'✔<3s' if ttfts[-1]<3 else '✘≥3s'}  整段={total:5.2f}s")

def pct(a, p):
    a = sorted(a); return a[min(len(a)-1, int(round(p*(len(a)-1))))]
print("\n=== 汇总 ===")
print(f"TTFT  <3s: {sum(1 for x in ttfts if x<3)}/6   min/p50/p95/max = {min(ttfts):.2f}/{pct(ttfts,.5):.2f}/{pct(ttfts,.95):.2f}/{max(ttfts):.2f}s")
print(f"整段  <3s: {sum(1 for x in totals if x<3)}/6   min/p50/p95/max = {min(totals):.2f}/{pct(totals,.5):.2f}/{pct(totals,.95):.2f}/{max(totals):.2f}s")
