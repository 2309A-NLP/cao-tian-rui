"""
RAG vs LightRAG 对比评估脚本
使用 RAGAS 指标评估两种检索方式的回答质量

在 Windows gpu_env 运行:
  cd E:\10--agent--任务\任务 1\backend
  python rag_comparison.py --api-key YOUR_DEEPSEEK_KEY
"""
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import get_logger

logger = get_logger("comparison")


# ── 16 个测试问题 ──────────────────────────────────

TEST_QUESTIONS = [
    {
        "id": 5,
        "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
        "company": "力源信息",
    },
    {
        "id": 6,
        "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
        "company": "力源信息",
    },
    {
        "id": 1,
        "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
        "company": "力源信息",
    },
    {
        "id": 2,
        "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
        "company": "力源信息",
    },
    {
        "id": 3,
        "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
        "company": "力源信息",
    },
    {
        "id": 4,
        "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
        "company": "力源信息",
    },
    {
        "id": 260,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "company": "兴图新科",
    },
    {
        "id": 95,
        "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
        "company": "兴图新科",
    },
    {
        "id": 33,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
        "company": "兴图新科",
    },
    {
        "id": 34,
        "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
        "company": "兴图新科",
    },
    {
        "id": 957,
        "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
        "company": "兴图新科",
    },
    {
        "id": 793,
        "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
        "company": "兴图新科",
    },
    {
        "id": 795,
        "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
        "company": "兴图新科",
    },
    {
        "id": 543,
        "question": "武汉兴图新科电子股份有限公司注册资本是多少？",
        "company": "兴图新科",
    },
    {
        "id": 531,
        "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
        "company": "兴图新科",
    },
    {
        "id": 207,
        "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
        "company": "兴图新科",
    },
]


# ── RAG 查询 ──────────────────────────────────

def query_existing_rag(question: str, api_url: str = "http://localhost:8010") -> dict:
    """查询现有 RAG 系统"""
    import httpx

    try:
        with httpx.Client(timeout=60) as client:
            # 使用流式接口
            resp = client.post(
                f"{api_url}/api/ask",
                json={"query": question, "mode": "rag"},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
                "time_ms": data.get("total_time_ms", 0),
            }
    except Exception as e:
        return {"answer": f"RAG 查询失败: {e}", "sources": [], "time_ms": 0}


def query_existing_rag_direct(question: str) -> dict:
    """
    直接调用 RAG 引擎（不经过 API 服务器）
    适用于 API 服务器未启动的情况
    """
    try:
        from config import AppConfig
        from vector_store import VectorStore
        from llm_provider import BaseLLMProvider
        from rag_engine import RAGEngine, ChatSession

        cfg = AppConfig.load()

        # 初始化向量存储（加载模型 + 连接 Milvus）
        store = VectorStore(
            model_path=cfg.embedding_model_path,
            milvus_host=cfg.milvus_host,
            milvus_port=cfg.milvus_port,
            collection_name=cfg.milvus_collection,
        )
        store.load_model()
        store.connect_milvus()
        # 加载 chunks 元数据 + 构建 BM25 索引
        chunks_file = os.path.join(cfg.output_dir, "chunks", "all_chunks.json")
        if os.path.exists(chunks_file):
            with open(chunks_file, encoding="utf-8") as f:
                all_chunks = json.load(f)
            store.chunks_metadata = all_chunks
        if cfg.bm25_enabled and store.chunks_metadata:
            store.build_bm25_index(store.chunks_metadata)

        llm = BaseLLMProvider.create(
            provider=cfg.llm_provider,
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            model=cfg.llm_model,
        )

        engine = RAGEngine(
            vector_store=store,
            llm_provider=llm,
            top_k=cfg.top_k,
            similarity_threshold=cfg.similarity_threshold,
            reranker_model_path=cfg.reranker_model_path,
        )

        session = ChatSession(user_id=0, session_id="eval")

        # 收集完整回答
        answer = ""
        sources = []
        for event in engine.answer_stream(question, session):
            if event["type"] == "retrieval":
                sources = event.get("sources", [])
            elif event["type"] == "token":
                answer += event.get("content", "")
            elif event["type"] == "done":
                total_ms = event.get("total_time_ms", 0)

        return {"answer": answer, "sources": sources, "time_ms": total_ms}

    except Exception as e:
        return {"answer": f"RAG 直接调用失败: {e}", "sources": [], "time_ms": 0}


# ── RAGAS 评估 ──────────────────────────────────

def compute_ragas_metrics(
    questions: list[dict],
    rag_answers: list[dict],
    lightrag_answers: list[dict],
    api_key: str = "",
    base_url: str = "https://api.deepseek.com",
) -> dict:
    """
    使用 RAGAS 计算评估指标
    
    RAGAS 指标:
    - faithfulness: 回答是否忠实于检索到的上下文
    - answer_relevancy: 回答与问题的相关性
    - context_precision: 检索上下文的精确度
    - context_recall: 检索上下文的召回率
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        logger.warning("ragas 或 datasets 未安装，使用简化评估")
        return compute_simple_metrics(questions, rag_answers, lightrag_answers)

    # 设置 LLM 用于 RAGAS 评估
    os.environ["OPENAI_API_KEY"] = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    os.environ["OPENAI_API_BASE"] = base_url

    results = {}

    for label, answers in [("RAG", rag_answers), ("LightRAG", lightrag_answers)]:
        # 准备 RAGAS 数据集
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for q, a in zip(questions, answers):
            data["question"].append(q["question"])
            data["answer"].append(a.get("answer", ""))
            # contexts 需要是 list[str]
            ctx = a.get("context", "")
            if isinstance(ctx, str):
                data["contexts"].append([ctx[:3000]])
            elif isinstance(ctx, list):
                data["contexts"].append(ctx[:5])
            else:
                data["contexts"].append([str(ctx)[:3000]])
            data["ground_truth"].append("")  # 无标准答案时留空

        dataset = Dataset.from_dict(data)

        # 计算指标
        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            results[label] = {
                "faithfulness": round(result["faithfulness"], 4),
                "answer_relevancy": round(result["answer_relevancy"], 4),
                "context_precision": round(result["context_precision"], 4),
                "context_recall": round(result["context_recall"], 4),
            }
        except Exception as e:
            logger.error(f"{label} RAGAS 评估失败: {e}")
            results[label] = {"error": str(e)}

    return results


def compute_simple_metrics(questions, rag_answers, lightrag_answers) -> dict:
    """简化评估（不依赖 RAGAS 库）"""
    import re

    def _score_answer(answer: str, question: str) -> dict:
        if not answer or "失败" in answer or "未包含" in answer:
            return {"has_answer": 0, "has_number": 0, "has_page_ref": 0, "length": 0}

        has_number = 1 if re.search(r'[\d,.]+[%万元个条]', answer) else 0
        has_page_ref = 1 if re.search(r'第?\s*\d+\s*页', answer) else 0
        return {
            "has_answer": 1,
            "has_number": has_number,
            "has_page_ref": has_page_ref,
            "length": len(answer),
        }

    results = {}
    for label, answers in [("RAG", rag_answers), ("LightRAG", lightrag_answers)]:
        scores = [_score_answer(a.get("answer", ""), q["question"])
                  for q, a in zip(questions, answers)]
        total = len(scores)
        results[label] = {
            "回答率": sum(s["has_answer"] for s in scores) / total,
            "含数值率": sum(s["has_number"] for s in scores) / total,
            "含页码引用率": sum(s["has_page_ref"] for s in scores) / total,
            "平均回答长度": sum(s["length"] for s in scores) / total,
            "平均耗时ms": sum(a.get("time_ms", 0) for a in answers) / total,
        }

    return results


# ── 主流程 ──────────────────────────────────

def run_comparison(
    api_key: str = "",
    use_api: bool = False,
    api_url: str = "http://localhost:8010",
    lightrag_dir: str = "./lightrag_storage",
    output_path: str = "output/comparison_results.json",
) -> dict:
    """运行完整对比评估"""

    logger.info("=" * 60)
    logger.info("开始 RAG vs LightRAG 对比评估")
    logger.info("=" * 60)

    # 1. 初始化 LightRAG
    logger.info("初始化 LightRAG...")
    from lightrag_integration import create_lightrag, build_deepseek_llm_func, query_lightrag

    llm_func = build_deepseek_llm_func(api_key=api_key)

    # 尝试加载 BGE 嵌入
    embedding_func = None
    try:
        from lightrag_integration import build_bge_embedding_func
        embedding_func = build_bge_embedding_func()
    except Exception as e:
        logger.warning(f"BGE 嵌入加载失败，使用默认: {e}")

    lightrag = create_lightrag(
        working_dir=lightrag_dir,
        llm_func=llm_func,
        embedding_func=embedding_func,
    )

    # 2. 逐题查询
    rag_answers = []
    lightrag_answers = []

    for i, q in enumerate(TEST_QUESTIONS):
        qid = q["id"]
        question = q["question"]
        logger.info(f"\n[{i+1}/{len(TEST_QUESTIONS)}] ID={qid}: {question[:60]}...")

        # RAG 查询
        if use_api:
            rag_result = query_existing_rag(question, api_url=api_url)
        else:
            rag_result = query_existing_rag_direct(question)
        rag_answers.append(rag_result)
        logger.info(f"  RAG: {rag_result['time_ms']}ms, {len(rag_result.get('answer', ''))}字")

        # LightRAG 查询
        lg_result = query_lightrag(lightrag, question, mode="hybrid")
        # 获取上下文（用于 RAGAS）
        from lightrag_integration import get_lightrag_context
        ctx = get_lightrag_context(lightrag, question, mode="hybrid")
        lg_result["context"] = ctx.get("context", "")
        lightrag_answers.append(lg_result)
        logger.info(f"  LightRAG: {lg_result['time_ms']}ms, {len(lg_result.get('answer', ''))}字")

    # 3. 计算评估指标
    logger.info("\n计算评估指标...")
    metrics = compute_ragas_metrics(
        TEST_QUESTIONS, rag_answers, lightrag_answers, api_key=api_key
    )

    # 4. 汇总结果
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "details": [],
    }

    for i, q in enumerate(TEST_QUESTIONS):
        comparison["details"].append({
            "id": q["id"],
            "question": q["question"],
            "company": q["company"],
            "rag": {
                "answer": rag_answers[i].get("answer", "")[:500],
                "time_ms": rag_answers[i].get("time_ms", 0),
            },
            "lightrag": {
                "answer": lightrag_answers[i].get("answer", "")[:500],
                "time_ms": lightrag_answers[i].get("time_ms", 0),
            },
        })

    # 5. 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    logger.info(f"\n结果已保存: {output_path}")

    # 6. 打印对比表
    print_comparison_table(comparison)

    return comparison


def print_comparison_table(comparison: dict):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("  RAG vs LightRAG 评估指标对比")
    print("=" * 80)

    metrics = comparison.get("metrics", {})
    if "RAG" in metrics and "LightRAG" in metrics:
        rag_m = metrics["RAG"]
        lg_m = metrics["LightRAG"]

        if "error" not in rag_m and "error" not in lg_m:
            # RAGAS 指标
            print(f"\n{'指标':<25} {'RAG':>12} {'LightRAG':>12} {'差异':>12}")
            print("-" * 65)
            for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if key in rag_m and key in lg_m:
                    r, l = rag_m[key], lg_m[key]
                    diff = l - r
                    sign = "+" if diff > 0 else ""
                    print(f"{key:<25} {r:>12.4f} {l:>12.4f} {sign}{diff:>11.4f}")
        else:
            # 简化指标
            print(f"\n{'指标':<20} {'RAG':>12} {'LightRAG':>12}")
            print("-" * 48)
            for key in ["回答率", "含数值率", "含页码引用率", "平均回答长度", "平均耗时ms"]:
                if key in rag_m and key in lg_m:
                    r, l = rag_m[key], lg_m[key]
                    if isinstance(r, float):
                        print(f"{key:<20} {r:>12.2f} {l:>12.2f}")
                    else:
                        print(f"{key:<20} {r:>12} {l:>12}")

    # 逐题对比
    print(f"\n{'='*80}")
    print("  逐题回答对比")
    print("=" * 80)

    for d in comparison.get("details", []):
        print(f"\n[ID={d['id']}] {d['question'][:50]}...")
        print(f"  RAG ({d['rag']['time_ms']}ms): {d['rag']['answer'][:100]}...")
        print(f"  LightRAG ({d['lightrag']['time_ms']}ms): {d['lightrag']['answer'][:100]}...")


# ──────────────────────────────────────────────────────
#  CLI 入口
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG vs LightRAG 对比评估")
    parser.add_argument("--api-key", default="", help="DeepSeek API Key")
    parser.add_argument("--use-api", action="store_true", help="通过 API 调用现有 RAG（需要服务器运行）")
    parser.add_argument("--api-url", default="http://localhost:8010", help="RAG API 地址")
    parser.add_argument("--lightrag-dir", default="./lightrag_storage", help="LightRAG 存储目录")
    parser.add_argument("--output", default="output/comparison_results.json", help="输出文件路径")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run_comparison(
        api_key=args.api_key,
        use_api=args.use_api,
        api_url=args.api_url,
        lightrag_dir=args.lightrag_dir,
        output_path=args.output,
    )
