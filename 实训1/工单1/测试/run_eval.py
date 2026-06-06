#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG系统评估脚本
读取测试问题，调用RAG API，评估检索和回答质量，输出评估报告。
依赖：仅 requests（标准库 json/time/datetime）
用法：python run_eval.py
"""

import json
import time
import requests
from datetime import datetime


API_URL = "http://127.0.0.1:8010/api/ask/stream"
QUESTIONS_FILE = "test_questions.json"
REPORT_FILE = "eval_report.md"


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_api(query):
    """调用SSE流式接口，返回 (answer, sources, latency_seconds)"""
    payload = {
        "query": query,
        "mode": "default",
        "user_id": "eval_user",
        "session_id": "eval_session_" + str(int(time.time() * 1000))
    }
    start = time.time()
    answer = ""
    sources = []
    try:
        resp = requests.post(API_URL, json=payload, stream=True, timeout=60)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            # SSE格式: "data: ..."
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    # 尝试从不同字段提取内容
                    if isinstance(obj, dict):
                        # 提取answer/token
                        if "answer" in obj:
                            answer += obj["answer"]
                        elif "content" in obj:
                            answer += obj["content"]
                        elif "token" in obj:
                            answer += obj["token"]
                        elif "data" in obj:
                            answer += str(obj["data"])
                        # 提取sources
                        if "sources" in obj:
                            sources = obj["sources"]
                        elif "references" in obj:
                            sources = obj["references"]
                        elif "context" in obj:
                            sources = obj["context"] if isinstance(obj["context"], list) else []
                except json.JSONDecodeError:
                    # 非JSON的纯文本token
                    answer += data_str
            else:
                # 可能没有"data:"前缀的纯文本行
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if "answer" in obj:
                            answer += obj["answer"]
                        elif "content" in obj:
                            answer += obj["content"]
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        answer = f"[ERROR] {e}"
    latency = time.time() - start
    return answer, sources, latency


def check_keywords(answer, keywords):
    """检查答案是否包含所有关键词，返回命中率"""
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if kw in answer)
    return hits / len(keywords)


def check_source_relevance(sources, source_id):
    """检查检索来源是否包含预期的文档ID"""
    if not source_id or not sources:
        return None
    for src in sources:
        sid = ""
        if isinstance(src, dict):
            sid = str(src.get("id", src.get("doc_id", src.get("source_id", ""))))
        elif isinstance(src, str):
            sid = src
        if source_id in sid:
            return True
    return False


def main():
    questions = load_questions(QUESTIONS_FILE)
    results = []
    total_latency = 0.0
    total_keyword_rate = 0.0
    total_completeness = 0.0

    print(f"开始评估，共 {len(questions)} 个问题...\n")

    for q in questions:
        print(f"[{q['id']}] {q['question']}")
        answer, sources, latency = call_api(q["question"])
        kw_rate = check_keywords(answer, q.get("expected_keywords", []))
        source_found = check_source_relevance(sources, q.get("source_id", ""))

        completeness = 1.0 if kw_rate >= 1.0 else (0.5 if kw_rate > 0 else 0.0)
        faithfulness = 1.0 if len(answer) > 10 and "[ERROR]" not in answer else 0.0
        relevancy = 1.0 if kw_rate > 0 else 0.0

        total_latency += latency
        total_keyword_rate += kw_rate
        total_completeness += completeness

        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "sources": sources,
            "latency": latency,
            "keyword_rate": kw_rate,
            "completeness": completeness,
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "source_found": source_found,
        })

        status = "✓" if kw_rate >= 1.0 else ("△" if kw_rate > 0 else "✗")
        print(f"  {status} 关键词命中: {kw_rate:.0%} | 延迟: {latency:.2f}s | 答案长度: {len(answer)}字\n")

    n = len(results)
    avg_latency = total_latency / n
    avg_keyword = total_keyword_rate / n
    avg_completeness = total_completeness / n

    # 生成报告
    lines = []
    lines.append("# RAG系统评估报告\n")
    lines.append(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**测试问题数**: {n}\n")
    lines.append(f"**API地址**: {API_URL}\n")

    lines.append("\n## 一、整体指标汇总\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 平均端到端延迟 | {avg_latency:.2f}s |")
    lines.append(f"| 平均关键词命中率 | {avg_keyword:.1%} |")
    lines.append(f"| 平均完整性得分 | {avg_completeness:.2f} |")
    lines.append(f"| 忠实度（无报错） | {sum(1 for r in results if r['faithfulness']==1.0)/n:.1%} |")
    lines.append(f"| 相关性（有命中） | {sum(1 for r in results if r['relevancy']==1.0)/n:.1%} |")

    lines.append("\n## 二、逐题评估详情\n")
    for r in results:
        lines.append(f"### 问题 {r['id']}: {r['question']}\n")
        lines.append(f"- **延迟**: {r['latency']:.2f}s")
        lines.append(f"- **关键词命中率**: {r['keyword_rate']:.0%}")
        lines.append(f"- **完整性**: {r['completeness']:.1f}")
        lines.append(f"- **忠实度**: {r['faithfulness']:.1f}")
        lines.append(f"- **相关性**: {r['relevancy']:.1f}")
        if r['source_found'] is not None:
            sf = "是" if r['source_found'] else "否"
            lines.append(f"- **预期来源命中**: {sf}")
        lines.append(f"- **答案摘要**: {r['answer'][:200]}{'...' if len(r['answer'])>200 else ''}\n")

    lines.append("\n## 三、性能指标\n")
    latencies = [r['latency'] for r in results]
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 最小延迟 | {min(latencies):.2f}s |")
    lines.append(f"| 最大延迟 | {max(latencies):.2f}s |")
    lines.append(f"| 平均延迟 | {avg_latency:.2f}s |")
    lines.append(f"| 中位数延迟 | {sorted(latencies)[n//2]:.2f}s |")

    report = "\n".join(lines)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n评估完成！报告已保存至: {REPORT_FILE}")
    print(f"平均延迟: {avg_latency:.2f}s | 关键词命中率: {avg_keyword:.1%} | 完整性: {avg_completeness:.2f}")


if __name__ == "__main__":
    main()
