# -*- coding: utf-8 -*-
"""工单15 答案评测：用 OpenAI 兼容端点对6问取答案，判对错，结果写 UTF-8 文件。
用法: python t15_answers.py <dialog_id> <out_tag>
"""
import sys, json, urllib.request

BASE = "http://localhost:9380"
TOKEN = "ragflow-t15diag0001"
dialog = sys.argv[1] if len(sys.argv) > 1 else "40d4a4e468ab11f1868a810dbccbbded"
tag = sys.argv[2] if len(sys.argv) > 2 else "baseline"

qs = json.load(open("E:/rag后续工单/t15_questions.json", encoding="utf-8"))

# 判对规则：答案关键词
JUDGE = {
    1: ["块状散料"],
    2: ["链条"],
    3: ["12之内", "12的部件之内", "之内", "内部", "里面", "保护管"],
    4: ["顶部", "上部", "上方", "顶端"],
    5: ["13", "链条"],
    6: ["11", "紧固"],
}

def ask(q):
    body = {"model": "model", "messages": [{"role": "user", "content": q}], "stream": False}
    data = json.dumps(body).encode("utf-8")
    url = f"{BASE}/api/v1/chats_openai/{dialog}/chat/completions"
    req = urllib.request.Request(url, data=data,
        headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"]

out = [f"==== 工单15 答案评测 [{tag}] dialog={dialog[:8]} ===="]
correct = 0
for item in qs:
    ans = ask(item["q"])
    # 去掉 <think>
    if "</think>" in ans:
        ans = ans.split("</think>")[-1]
    ans_clean = ans.strip()
    keys = JUDGE[item["id"]]
    ok = any(k in ans_clean for k in keys)
    # Q5 特判: 必须含13且不是仅含14
    if item["id"] == 5:
        ok = ("13" in ans_clean)
    if item["id"] == 6:
        ok = ("11" in ans_clean) or ("紧固" in ans_clean)
    correct += int(ok)
    mark = "✓" if ok else "✗"
    out.append(f"\n[Q{item['id']}|{item['type']}] {mark} 期望={item['answer']}")
    out.append(f"  问: {item['q']}")
    out.append(f"  答: {ans_clean[:300]}")

out.append(f"\n==== 准确率: {correct}/6 ====")
txt = "\n".join(out)
open(f"E:/rag后续工单/t15_result_{tag}.txt", "w", encoding="utf-8").write(txt)
print(f"DONE {tag}: {correct}/6  -> t15_result_{tag}.txt")
