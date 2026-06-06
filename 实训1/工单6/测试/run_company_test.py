# -*- coding: utf-8 -*-
"""
公司路由测试脚本
测试不带公司名的查询是否能正确路由到对应公司的文档
招股书1(兴图新科): 页码1-170
招股书2(力源信息): 页码171-350
用法: python run_company_test.py
输出: company_routing_report.md
"""

import json
import time
import requests

API_URL = "http://127.0.0.1:8010/api/ask/stream"
REPORT_FILE = "company_routing_report.md"

# 测试问题：不带公司名，答案应只来自兴图新科(招股书1, 页码1-170)
TEST_QUERIES = [
    {
        "id": 1,
        "question": "注册资本是多少",
        "expected_company": "兴图新科",
        "expected_source_pages": (1, 170),
        "expected_keywords": ["注册资本", "万元"]
    },
    {
        "id": 2,
        "question": "前五名客户销售额占比",
        "expected_company": "兴图新科",
        "expected_source_pages": (1, 170),
        "expected_keywords": ["前五", "客户", "占比"]
    },
    {
        "id": 3,
        "question": "2018年营业收入",
        "expected_company": "兴图新科",
        "expected_source_pages": (1, 170),
        "expected_keywords": ["2018", "营业收入"]
    }
]

def call_api(query):
    """调用SSE接口，返回(answer, sources, latency)"""
    payload = {
        "query": query,
        "mode": "rag",
        "user_id": 1,
        "session_id": f"company_test_{int(time.time())}"
    }
    answer_parts = []
    sources = []
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
                docs = event.get("documents", event.get("sources", []))
                if isinstance(docs, list):
                    sources.extend(docs)
            elif etype == "token":
                answer_parts.append(event.get("content", ""))
            elif etype == "done":
                # done事件可能也包含sources
                done_sources = event.get("sources", [])
                if isinstance(done_sources, list) and done_sources:
                    sources.extend(done_sources)
                break
    except Exception as e:
        answer_parts.append(f"[ERROR: {e}]")

    latency = time.time() - start
    return "".join(answer_parts), sources, round(latency, 3)

def extract_page(sources):
    """从sources中提取页码信息，返回页码列表"""
    pages = []
    for src in sources:
        if isinstance(src, dict):
            page = src.get("page", src.get("page_num", src.get("metadata", {}).get("page", None)))
            if page is not None:
                try:
                    pages.append(int(page))
                except (ValueError, TypeError):
                    pass
        elif isinstance(src, str):
            # 尝试从字符串中提取页码
            import re
            found = re.findall(r'(?:page|页码?)[\s:]*(\d+)', src)
            pages.extend(int(p) for p in found)
    return pages

def check_routing(pages, expected_range):
    """检查所有页码是否都在期望范围内"""
    if not pages:
        return None, "未获取到页码信息"
    low, high = expected_range
    out_of_range = [p for p in pages if p < low or p > high]
    if out_of_range:
        return False, f"发现{len(out_of_range)}个页码超出范围: {out_of_range}"
    return True, f"所有{len(pages)}个页码均在范围内"

def main():
    results = []

    print("开始公司路由测试...\n")

    for q in TEST_QUERIES:
        print(f"[测试{q['id']}] {q['question']}")
        answer, sources, latency = call_api(q["question"])
        pages = extract_page(sources)
        routing_ok, routing_msg = check_routing(pages, q["expected_source_pages"])

        keyword_hits = [kw for kw in q["expected_keywords"] if kw in answer]
        hit_rate = len(keyword_hits) / len(q["expected_keywords"]) * 100 if q["expected_keywords"] else 0

        result = {
            **q,
            "answer": answer[:500],
            "latency": latency,
            "source_pages": pages,
            "routing_ok": routing_ok,
            "routing_msg": routing_msg,
            "keyword_hits": keyword_hits,
            "hit_rate": round(hit_rate, 1)
        }
        results.append(result)
        print(f"  延迟: {latency}s  路由: {routing_msg}  命中率: {hit_rate}%\n")

    # 生成报告
    routing_pass = sum(1 for r in results if r["routing_ok"] is True)
    routing_fail = sum(1 for r in results if r["routing_ok"] is False)
    routing_unknown = sum(1 for r in results if r["routing_ok"] is None)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# 公司路由测试报告\n\n")
        f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 测试背景\n\n")
        f.write("- 知识库包含2份招股书：招股书1(兴图新科, 页码1-170)、招股书2(力源信息, 页码171-350)\n")
        f.write("- 测试查询不包含公司名称，验证系统能否正确路由到对应公司文档\n\n")

        f.write("## 总体结果\n\n")
        f.write(f"| 指标 | 值 |\n|------|----|\n")
        f.write(f"| 测试数 | {len(results)} |\n")
        f.write(f"| 路由正确 | {routing_pass} |\n")
        f.write(f"| 路由错误 | {routing_fail} |\n")
        f.write(f"| 无法判断 | {routing_unknown} |\n")
        avg_hit = sum(r["hit_rate"] for r in results) / len(results) if results else 0
        avg_lat = sum(r["latency"] for r in results) / len(results) if results else 0
        f.write(f"| 平均关键词命中率 | {avg_hit:.1f}% |\n")
        f.write(f"| 平均响应时间 | {avg_lat:.3f}s |\n\n")

        f.write("## 详细结果\n\n")
        for r in results:
            status = "PASS" if r["routing_ok"] is True else ("FAIL" if r["routing_ok"] is False else "UNKNOWN")
            f.write(f"### 测试{r['id']}: {r['question']}\n\n")
            f.write(f"- **期望公司**: {r['expected_company']}\n")
            f.write(f"- **期望页码范围**: {r['expected_source_pages'][0]}-{r['expected_source_pages'][1]}\n")
            f.write(f"- **实际页码**: {r['source_pages']}\n")
            f.write(f"- **路由结果**: [{status}] {r['routing_msg']}\n")
            f.write(f"- **关键词命中**: {r['keyword_hits']} ({r['hit_rate']}%)\n")
            f.write(f"- **响应时间**: {r['latency']}s\n")
            f.write(f"- **回答摘要**: {r['answer'][:300]}...\n\n")

        f.write("## 结论\n\n")
        if routing_fail == 0 and routing_pass > 0:
            f.write("公司路由功能正常，所有查询均正确路由到目标公司文档。\n")
        elif routing_fail > 0:
            f.write(f"公司路由存在问题，{routing_fail}个查询路由到了错误公司的文档。\n")
        else:
            f.write("无法从返回数据中判断路由结果，建议检查API返回的sources格式。\n")

    print(f"测试完成！报告已保存至 {REPORT_FILE}")

if __name__ == "__main__":
    main()
