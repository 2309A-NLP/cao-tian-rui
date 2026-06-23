# -*- coding: utf-8 -*-
# 高并发稳定性测试：N 路并发问答，统计成功率 + 延迟分布。
import json, time, urllib.request, threading
TOKEN = "ragflow-reparse-tmp"
CHAT = "40d4a4e468ab11f1868a810dbccbbded"
BASE = "http://localhost:9380/api/v1/chats/" + CHAT
H = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=H, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=180))

QS = [
    "根据文本信息，该静电除尘器的发明人是？",
    "该静电除尘器的管状入口具有怎样的结构特征？",
    "在第7页图片中，部件4相对于部件5在图中的位置关系是？",
    "在第7页图片中，尺寸X1、X2、X3分别代表什么部件之间的间隔距离？",
    "根据第7页图示，气流方向(7)首先经过哪个部件？紧接着经过哪个部件？",
    "根据第7页图示，已知外壳直径D，h1和h2的尺寸可以用来计算/确定什么？",
]

import sys
CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 12   # 并发路数
results = []        # (ok, latency, err)
lock = threading.Lock()

def worker(i):
    q = QS[i % len(QS)]
    try:
        sid = post("/sessions", {"name": f"conc{i}"})["data"]["id"]
        t0 = time.perf_counter()
        r = post("/completions", {"question": q, "session_id": sid, "stream": False})
        dt = time.perf_counter() - t0
        ok = r.get("code") == 0 and bool(r.get("data", {}).get("answer"))
        with lock:
            results.append((ok, dt, "" if ok else str(r.get("message"))[:60]))
    except Exception as e:
        with lock:
            results.append((False, None, repr(e)[:80]))

print(f"=== 高并发：{CONCURRENCY} 路同时发起 ===")
wall0 = time.perf_counter()
ts = [threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)]
for t in ts: t.start()
for t in ts: t.join()
wall = time.perf_counter() - wall0

ok_n = sum(1 for ok, _, _ in results if ok)
lats = sorted(d for ok, d, _ in results if ok and d is not None)
def pct(a, p):
    return a[min(len(a)-1, int(round(p*(len(a)-1))))] if a else 0
print(f"成功: {ok_n}/{CONCURRENCY}   总墙钟: {wall:.2f}s   吞吐≈{ok_n/wall:.2f} req/s")
if lats:
    print(f"单请求延迟 min/p50/p95/max = {lats[0]:.2f}/{pct(lats,.5):.2f}/{pct(lats,.95):.2f}/{lats[-1]:.2f}s")
for ok, d, err in results:
    if not ok:
        print("  FAIL:", err)
