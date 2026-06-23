# -*- coding: utf-8 -*-
"""工单15 检索诊断：对6问做 /api/v1/retrieval，打印 Top-K 命中的文档/页/片段。
用法: python t15_retrieval_diag.py [vector_weight] [sim_threshold] [rerank_id] [doc_scope]
  doc_scope: "auto" = 按问题里的 document 字段(本工单都是 CN100347506C) 锁文档; "none" = 全库检索(基线)
"""
import sys, json, urllib.request

BASE = "http://localhost:9380"
TOKEN = "ragflow-t15diag0001"
KB = "b9efcbe8689611f1868a810dbccbbded"
DOC_506 = "543d61d6695c11f199469d7f2aa2a4ae"   # CN100347506C (本工单)
DOC_976 = "0242e1a0689711f1868a810dbccbbded"   # CN100342976C (工单14, 干扰项)

vw   = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
sim  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
rid  = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else ""
scope= sys.argv[4] if len(sys.argv) > 4 else "none"

qs = json.load(open("E:/rag后续工单/t15_questions.json", encoding="utf-8"))

def docname(did):
    return "506(本)" if did == DOC_506 else ("976(干扰)" if did == DOC_976 else did[:6])

def is_fig3(c):
    """该片段是否与图3相关(含图3部件描述)"""
    t = c.get("content", "")
    keys = ["图3", "图 3", "落料架 10", "落料架10", "保护管", "紧固机构", "输入管 14", "链条 13", "链条13"]
    return any(k in t for k in keys)

def retrieve(q, docids):
    body = {"dataset_ids": [KB], "question": q, "page": 1, "page_size": 5,
            "similarity_threshold": sim, "vector_similarity_weight": vw, "top_k": 1024}
    if docids:
        body["document_ids"] = docids
    if rid:
        body["rerank_id"] = rid
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE + "/api/v1/retrieval", data=data,
        headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

print(f"=== 检索诊断 vw={vw} sim={sim} rerank={'on' if rid else 'off'} scope={scope} ===\n")
fig3_hit_top1 = fig3_hit_top3 = 0
for item in qs:
    docids = [DOC_506] if scope == "auto" else None
    res = retrieve(item["q"], docids)
    chunks = res.get("data", {}).get("chunks", [])
    print(f"[Q{item['id']}|{item['type']}] {item['q']}")
    f1 = f3 = False
    for i, c in enumerate(chunks[:5]):
        fig = "★图3" if is_fig3(c) else "    "
        if is_fig3(c):
            if i == 0: f1 = True
            if i < 3: f3 = True
        snip = c.get("content", "").replace("\n", " ")[:70]
        print(f"   #{i} sim={c.get('similarity',0):.3f} {docname(c.get('document_id',''))} {fig} {snip}")
    if item["type"] == "image":
        fig3_hit_top1 += int(f1); fig3_hit_top3 += int(f3)
    print()
print(f"图文4题 图3命中: Top-1={fig3_hit_top1}/4  Top-3={fig3_hit_top3}/4")
