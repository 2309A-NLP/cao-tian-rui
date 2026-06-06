# -*- coding: utf-8 -*-
"""
RAG系统自动化评估脚本
用法: python run_eval.py
输出: eval_report.md
"""

import json
import time
import requests

API_URL = "http://127.0.0.1:8010/api/ask/stream"
QUESTIONS_FILE = "test_questions.json"
REPORT_FILE = "eval_report.md"

def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def call_api(query, mode="rag"):
    """调用SSE接口，返回(retrieval_events, tokens, latency_first_token, latency_total)"""
    payload = {
        "query": query,
        "mode": mode,
        "user_id": 1,
        "session_id": f"eval_{int(time.time())}"
    }
    retrieval_events = []
    tokens = []
    first_token_time = None
    start = time.time()

    try:
        resp = requests.post(API_URL, json=payload, stream=True, timeout=60)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if not data_str:
                continue
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            etype = event.get("type", "")
            if etype == "retrieval":
                retrieval_events.append(event)
            elif etype == "token":
                if first_token_time is None:
                    first_token_time = time.time()
                tokens.append(event.get("content", ""))
            elif etype == "done":
                break
    except Exception as e:
        tokens.append(f"[ERROR: {e}]")

    latency_first = (first_token_time - start) if first_token_time else -1
    latency_total = time.time() - start
    return retrieval_events, "".join(tokens), latency_first, latency_total

def check_keywords(answer, expected_keywords):
    """检查关键词命中"""
    hits = [kw for kw in expected_keywords if kw in answer]
    return hits, len(hits) / len(expected_keywords) if expected_keywords else 0

def main():
    questions = load_questions()
    results = []

    print(f"开始评估，共 {len(questions)} 道题目...\n")

    for q in questions:
        print(f"[{q['id']}] {q['question']}")
        _, answer, lat_first, lat_total = call_api(q["question"])
        hits, hit_rate = check_keywords(answer, q["expected_keywords"])
        result = {
            **q,
            "answer": answer[:500],
            "latency_first": round(lat_first, 3),
            "latency_total": round(lat_total, 3),
            "keyword_hits": hits,
            "hit_rate": round(hit_rate * 100, 1)
        }
        results.append(result)
        print(f"  命中率: {result['hit_rate']}%  延迟: {result['latency_total']}s\n")

    # 生成报告
    avg_hit = sum(r["hit_rate"] for r in results) / len(results)
    avg_lat = sum(r["latency_total"] for r in results) / len(results)
    avg_lat_first = sum(r["latency_first"] for r in results if r["latency_first"] > 0)
    valid_first = sum(1 for r in results if r["latency_first"] > 0)
    avg_lat_first = avg_lat_first / valid_first if valid_first else 0

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# RAG系统评估报告\n\n")
        f.write(f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 总体指标\n\n")
        f.write(f"| 指标 | 值 |\n|------|----|\n")
        f.write(f"| 测试题数 | {len(results)} |\n")
        f.write(f"| 平均关键词命中率 | {avg_hit:.1f}% |\n")
        f.write(f"| 平均首Token延迟 | {avg_lat_first:.3f}s |\n")
        f.write(f"| 平均总响应时间 | {avg_lat:.3f}s |\n\n")

        f.write("## 详细结果\n\n")
        for r in results:
            f.write(f"### [{r['id']}] {r['question']}\n\n")
            f.write(f"- **分类**: {r['category']}\n")
            f.write(f"- **首Token延迟**: {r['latency_first']}s\n")
            f.write(f"- **总响应时间**: {r['latency_total']}s\n")
            f.write(f"- **关键词命中**: {r['keyword_hits']} ({r['hit_rate']}%)\n")
            f.write(f"- **期望关键词**: {r['expected_keywords']}\n")
            f.write(f"- **回答摘要**: {r['answer'][:300]}...\n\n")

    print(f"评估完成！报告已保存至 {REPORT_FILE}")
    print(f"平均关键词命中率: {avg_hit:.1f}%")
    print(f"平均总响应时间: {avg_lat:.3f}s")

if __name__ == "__main__":
    main()
