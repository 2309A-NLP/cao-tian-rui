# -*- coding: utf-8 -*-
# 性能验证：6问单问延迟 + 命中校验。需临时 api_token 绑 dialog，跑完记得删。
import json, time, urllib.request
TOKEN = "ragflow-reparse-tmp"
CHAT = "40d4a4e468ab11f1868a810dbccbbded"
BASE = "http://localhost:9380/api/v1/chats/" + CHAT
H = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=H, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=120))

qs = [
    ("Q1/A", "根据文本信息，该静电除尘器的发明人是？", "吉特勒"),
    ("Q2/C", "该静电除尘器的管状入口具有怎样的结构特征？", "80至95%"),
    ("Q3/A", "在第7页图片中，部件4相对于部件5在图中的位置关系是？", "左"),
    ("Q4/A", "在第7页图片中，尺寸X1、X2、X3分别代表什么部件之间的间隔距离？", "6"),
    ("Q5/C", "根据第7页图示，气流方向(7)首先经过哪个部件？紧接着经过哪个部件？", "6″"),
    ("Q6/B", "根据第7页图示，已知外壳直径D，h1和h2的尺寸可以用来计算/确定什么？", "位置"),
]

lats, hits = [], 0
print("=== 单问延迟 + 命中（stream=False，计时仅 /completions）===")
for tag, q, key in qs:
    s = post("/sessions", {"name": tag})
    sid = s["data"]["id"]
    t0 = time.perf_counter()
    r = post("/completions", {"question": q, "session_id": sid, "stream": False})
    dt = time.perf_counter() - t0
    lats.append(dt)
    ans = r.get("data", {}).get("answer", "") if r.get("code") == 0 else "ERR:" + str(r.get("message"))[:80]
    hit = key in ans
    hits += 1 if hit else 0
    flag = "✅" if hit else "❓"
    p3 = "✔<3s" if dt < 3 else "✘≥3s"
    print(f"{tag} {flag} {dt:5.2f}s {p3}  | {ans[:70].strip()}")

lats_sorted = sorted(lats)
n = len(lats_sorted)
p50 = lats_sorted[n // 2]
p95 = lats_sorted[min(n - 1, int(round(0.95 * (n - 1))))]
under3 = sum(1 for x in lats if x < 3)
print("\n=== 汇总 ===")
print(f"命中: {hits}/6   <3s: {under3}/6")
print(f"延迟 min/p50/p95/max = {min(lats):.2f}/{p50:.2f}/{p95:.2f}/{max(lats):.2f}s")
