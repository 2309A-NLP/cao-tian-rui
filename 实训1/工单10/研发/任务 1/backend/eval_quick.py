"""
RAG 快速评估脚本（含 RAGAS 评分）
每次修改后运行：python eval_quick.py
自动逐题提问，输出关键字命中统计 + RAGAS 指标。
"""
import os
import sys
import json
import re
import time
import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import get_logger

logger = get_logger("eval")

# ═══════════════════════════════════════════════════════════
#  测试集：问题 + 标准答案 + 侧重点 + 关键命中词
# ═══════════════════════════════════════════════════════════

TEST_CASES = [
    # ── 力源信息 ──────────────────────────────────
    {
        "id": 1,
        "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
        "company": "力源信息",
        "focus": "文字",
        "answer_keywords": ["1,670万", "1670万", "25.04%"],
        "expected_answer": "本次发行股数为1,670万股，占发行后总股本的比例为25.04%。",
    },
    {
        "id": 2,
        "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
        "company": "力源信息",
        "focus": "文字/表格",
        "answer_keywords": ["仓储", "物流中心", "研发中心", "电子商务", "扩充产品"],
        "expected_answer": "拟投资项目包括：仓储及物流中心（3,393.40万元）、研发中心（1,526.38万元）、电子商务平台（2,492.78万元）、扩充产品种类和数量（9,000.00万元）、其他与主营业务相关的营运资金。",
    },
    {
        "id": 3,
        "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
        "company": "力源信息",
        "focus": "文字",
        "answer_keywords": ["赵马克", "Mark Zhao", "42.35%", "控股股东"],
        "expected_answer": "存在控制关系的关联方为赵马克（Mark Zhao），持股42.35%，为本公司控股股东。",
    },
    {
        "id": 4,
        "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
        "company": "力源信息",
        "focus": "文字",
        "answer_keywords": ["融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源", "力源贸易", "普芯达", "佰力电子", "盈硅电子"],
        "expected_answer": "非控制关系关联方企业包括：融冰投资、武汉博润、上海博润、听音投资、联众聚源、力源贸易、普芯达。此外，报告期内曾为关联方但已不存在控制关系的企业有佰力电子、盈硅电子。",
    },
    {
        "id": 5,
        "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
        "company": "力源信息",
        "focus": "图片/表格",
        "answer_keywords": ["四个", "渠道销售", "电话及网络销售", "大客户销售", "国际贸易", "6个销售处"],
        "expected_answer": "销售部下设4个部门：渠道销售部、电话及网络销售部、大客户销售部、国际贸易部。大客户销售部不设具体销售处，销售网络中有6个销售处（北京、广州、成都、深圳、武汉、珠海）负责为客户提供贴身服务。",
    },
    {
        "id": 6,
        "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
        "company": "力源信息",
        "focus": "图片",
        "answer_keywords": ["工业控制", "IC卡", "负增长"],
        "expected_answer": "增长率最快的是工业控制领域（增长率达10.5%）；出现负增长的是IC卡行业，主要原因是二代身份证市场萎缩。",
    },
    # ── 兴图新科 ──────────────────────────────────
    {
        "id": 260,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "company": "兴图新科",
        "focus": "文字/表格",
        "answer_keywords": ["6,464.51", "14,414.16", "18,780.67", "4,627.14"],
        "expected_answer": "报告期内（2016-2018年及2019年上半年），公司直接和间接向国防客户的销售额合计分别为6,464.51万元、14,414.16万元、18,780.67万元和4,627.14万元。",
    },
    {
        "id": 95,
        "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
        "company": "兴图新科",
        "focus": "文字",
        "answer_keywords": ["视频指挥系统", "视频技术规范", "某视频技术规范"],
        "expected_answer": "参与制定了全军第一个视频指挥系统技术标准，即《某视频技术规范1.0》。",
    },
    {
        "id": 33,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
        "company": "兴图新科",
        "focus": "文字/表格",
        "answer_keywords": ["82.10%", "97.31%", "94.84%", "94.34%"],
        "expected_answer": "2016-2018年及2019年上半年占比分别为82.10%、97.31%、94.84%和94.34%。",
    },
    {
        "id": 34,
        "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
        "company": "兴图新科",
        "focus": "文字",
        "answer_keywords": ["电子元器件", "金属壳体", "机箱"],
        "expected_answer": "电子信息行业的上游涉及电子元器件制造企业，以及机箱、机柜等金属壳体制造企业。",
    },
    {
        "id": 957,
        "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
        "company": "兴图新科",
        "focus": "文字",
        "answer_keywords": ["国防", "军队", "视频指挥", "视频"],
        "expected_answer": "公司在我国国防军队视频指挥领域已经成为重要供应商。",
    },
    {
        "id": 793,
        "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
        "company": "兴图新科",
        "focus": "文字",
        "answer_keywords": ["军队", "政府机关", "能源"],
        "expected_answer": "电子信息行业的下游为各类终端用户，覆盖范围广泛，主要包括军队、政府机关、能源等行业企业。",
    },
    {
        "id": 795,
        "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
        "company": "兴图新科",
        "focus": "文字",
        "answer_keywords": ["C4ISR", "指挥", "控制", "通信", "情报", "一体化"],
        "expected_answer": "公司参与的某情报、指挥、控制与通信网络一体化工程（C4ISR系统）荣获国家科技进步一等奖。",
    },
    {
        "id": 543,
        "question": "武汉兴图新科电子股份有限公司注册资本是多少？",
        "company": "兴图新科",
        "focus": "文字",
        "answer_keywords": ["5,520万", "5520万", "5,520.00万"],
        "expected_answer": "公司注册资本为5,520万元人民币。",
    },
    {
        "id": 531,
        "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
        "company": "兴图新科",
        "focus": "文字",
        "answer_keywords": ["程家明"],
        "expected_answer": "公司法定代表人为程家明。",
    },
    {
        "id": 207,
        "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
        "company": "兴图新科",
        "focus": "文字/表格",
        "answer_keywords": ["15,000万", "15000万", "1,810.00万", "1810万"],
        "expected_answer": "公司首次公开发行时，计划使用15,000万元募集资金用于补充流动资金，后续还从超募资金中划拨1,810.00万元永久补充流动资金。",
    },
]


def ask_rag(question: str, api_url: str = "http://localhost:8010", retrieval_mode: str | None = None) -> tuple:
    """调用 RAG API 获取回答、检索上下文和耗时信息"""
    try:
        with httpx.Client(timeout=60) as client:
            payload = {"query": question, "mode": "rag"}
            if retrieval_mode:
                payload["retrieval_mode"] = retrieval_mode
            resp = client.post(
                f"{api_url}/api/ask",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            contexts = [s.get("text", "") for s in sources if isinstance(s, dict)]
            retrieval_ms = int(data.get("retrieval_time_ms", 0))
            llm_ms = int(data.get("llm_time_ms", 0))
            total_ms = int(data.get("total_time_ms", 0))
            return answer, contexts, sources, retrieval_ms, llm_ms, total_ms
    except Exception as e:
        return f"[API错误] {e}", [], [], 0, 0, 0


def score_answer(answer: str, keywords: list[str]) -> dict:
    """判断回答是否命中关键信息。

    用 _norm 归一化后做子串匹配，消除空格/逗号/markdown 星号带来的误杀
    （如答案"5,520 万元"应能命中关键词"5,520万"）。
    注意：不再因答案中出现"知识库中未包含"就直接判 0——很多答案是
    "先给出正确信息、再补一句某细节未包含"，早退会把这类正确答案误杀。
    纯拒答自然命中不到具体关键词，会被关键词计数判为低分，无需特判。
    """
    if not answer or "[API错误]" in answer:
        return {"pass": False, "hit": 0, "total": len(keywords), "detail": "API错误或空回答"}

    na = _norm(answer)
    hits = 0
    hit_details = []
    for kw in keywords:
        if _norm(kw) in na:
            hits += 1
            hit_details.append(f"✅ {kw}")
        else:
            hit_details.append(f"❌ {kw}")

    return {
        "pass": hits >= len(keywords) * 0.5,
        "hit": hits,
        "total": len(keywords),
        "detail": " | ".join(hit_details),
    }


# ═══════════════════════════════════════════════════════════
#  纯检索指标（不依赖 LLM，快速可复现）
# ═══════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """归一化：小写 + 去空格/逗号/markdown星号，提升关键词匹配鲁棒性
    （"1,670万" vs "1670 万" vs "**1,670 万元**" 应等价匹配）"""
    return re.sub(r"[\s,，*]", "", s.lower())


def _is_relevant(text: str, keywords: list[str]) -> bool:
    """判定一个检索块是否相关：含任一答案关键词即视为相关块"""
    nt = _norm(text)
    return any(_norm(kw) in nt for kw in keywords)


def compute_retrieval_metrics(sources: list, keywords: list[str], ks=(1, 3, 5, 10)) -> dict:
    """纯检索质量指标，relevance 信号 = 检索块文本是否含答案关键词。

    - Hit@k:       top-k 中至少有一个相关块
    - Precision@k: top-k 中相关块占比
    - Recall@k:    top-k 内命中的不同关键词比例（关键词覆盖率）
    - MRR:         第一个相关块的倒数排名
    完全不依赖 LLM，跑得快、可复现，专门回答"检索有没有把答案块捞上来、排得够不够高"。
    """
    texts = [s.get("text", "") for s in sources if isinstance(s, dict)]
    rel_ranks = [i for i, t in enumerate(texts) if _is_relevant(t, keywords)]
    metrics = {}
    for k in ks:
        topk = texts[:k]
        metrics[f"hit@{k}"] = 1.0 if any(r < k for r in rel_ranks) else 0.0
        rel_in_k = sum(1 for r in rel_ranks if r < k)
        metrics[f"precision@{k}"] = rel_in_k / min(k, len(topk)) if topk else 0.0
        found = sum(1 for kw in keywords if any(_norm(kw) in _norm(t) for t in topk))
        metrics[f"recall@{k}"] = found / len(keywords) if keywords else 0.0
    metrics["mrr"] = (1.0 / (rel_ranks[0] + 1)) if rel_ranks else 0.0
    metrics["n_sources"] = len(texts)
    return metrics


def _percentile(values: list, pct: float) -> float:
    """简单分位数（线性插值），用于延迟 p50/p95"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    idx = pct / 100.0 * (len(xs) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def run_ragas(questions, answers, ground_truths, contexts):
    """运行 RAGAS 评分（用 DeepSeek 作为评判 LLM）"""
    try:
        import os
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            answer_correctness,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from embedding_provider import BGEM3LangchainEmbeddings

        # 用 DeepSeek 作为评判 LLM
        # 优先从 config.json 读取 key，fallback 到环境变量
        try:
            from config import AppConfig
            # 自动检测 config.json 位置：先看 backend/，再看上级目录
            _cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
            if not os.path.exists(_cfg_path):
                _cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
            if os.path.exists(_cfg_path):
                _cfg = AppConfig.load(_cfg_path)
                api_key = _cfg.llm_api_key
            else:
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        except Exception:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key or "..." in api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("  ⚠️ 未找到有效的 DeepSeek API Key，跳过 RAGAS 评分")
            return None
        # n=1：DeepSeek 仅支持 n=1，而 RAGAS 默认会请求 n=3 做多次采样集成，
        # 导致大量 Job 报 "Invalid n value" 而失败、分数变 nan。显式锁定 n=1。
        evaluator_llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            n=1,
            temperature=0,
        )
        # 用 RAGAS 的包装器并关闭多次采样（is_finished/n>1 的来源）
        ragas_llm = LangchainLLMWrapper(evaluator_llm)
        try:
            # 新版 RAGAS 支持在 run_config 里限制并发；同时确保底层 n=1
            ragas_llm.langchain_llm.n = 1
        except Exception:
            pass
        evaluator_emb = LangchainEmbeddingsWrapper(
            BGEM3LangchainEmbeddings()
        )

        data = {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "retrieved_contexts": contexts,
        }
        dataset = Dataset.from_dict(data)

        # answer_relevancy 默认 strictness=3 → 内部请求 n=3 → DeepSeek 报 Invalid n value。
        # 设为 1 是 "Invalid n value" 的真正修复点（n 来自 strictness，不是 LLM 实例）。
        try:
            answer_relevancy.strictness = 1
        except Exception:
            pass

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, answer_correctness],
            llm=ragas_llm,
            embeddings=evaluator_emb,
        )

        # 取分数：兼容多版本 RAGAS 的 EvaluationResult。
        # 新版没有 .get()，但通常支持 dict 下标、to_pandas()、或 ._scores_dict。
        metric_keys = ["faithfulness", "answer_relevancy", "answer_correctness"]
        raw = {}
        # 1) 优先用 to_pandas() 取每个指标的列均值（最稳，自动跳��� nan）
        try:
            import math
            df = result.to_pandas()
            for k in metric_keys:
                if k in df.columns:
                    vals = [x for x in df[k].tolist()
                            if isinstance(x, (int, float)) and not math.isnan(x)]
                    if vals:
                        raw[k] = sum(vals) / len(vals)
        except Exception:
            pass
        # 2) 回退：dict 下标 或 ._scores_dict
        if not raw:
            for k in metric_keys:
                v = None
                try:
                    v = result[k]
                except Exception:
                    sd = getattr(result, "_scores_dict", None) or getattr(result, "scores", None)
                    if isinstance(sd, dict):
                        v = sd.get(k)
                if v is not None:
                    try:
                        if isinstance(v, (list, tuple)) and v:
                            v = sum(v) / len(v)
                        raw[k] = float(v)
                    except (TypeError, ValueError):
                        pass
        scores = {k: round(float(v), 4) for k, v in raw.items()}
        return scores if scores else {"error": "RAGAS 已运行但无法解析分数（EvaluationResult 结构未知）"}
    except ImportError:
        return {"error": "请安装 ragas: pip install ragas"}
    except Exception as e:
        return {"error": str(e)}


def run_eval(api_url: str = "http://localhost:8010", with_ragas: bool = False, retrieval_mode: str | list[str] | None = None):
    if isinstance(retrieval_mode, list):
        overall_success = True
        for mode in retrieval_mode:
            success = run_eval(api_url, with_ragas=with_ragas, retrieval_mode=mode)
            overall_success = overall_success and success
        return overall_success

    print("=" * 70)
    print("  RAG 快速评估")
    print("=" * 70)
    print(f"  API: {api_url}")
    print(f"  测试数: {len(TEST_CASES)}")
    print(f"  检索模式: {retrieval_mode or '服务端默认(config.retrieval_mode)'}")
    if with_ragas:
        print(f"  RAGAS: ✅ (faithfulness + relevancy + correctness)")
    else:
        print(f"  RAGAS: ❌")
    print()

    results = []
    passed = 0
    failed = 0
    all_questions = []
    all_answers = []
    all_ground_truths = []
    all_contexts = []

    for tc in TEST_CASES:
        qid = tc["id"]
        question = tc["question"]
        focus = tc["focus"]
        keywords = tc["answer_keywords"]
        expected = tc.get("expected_answer", "")

        all_questions.append(question)
        all_ground_truths.append(expected)

        print(f"  [{qid}] ({focus}) {question[:50]}...")
        t0 = time.time()

        answer, contexts, sources, retrieval_ms, llm_ms, total_ms = ask_rag(question, api_url, retrieval_mode=retrieval_mode)
        all_answers.append(answer)
        all_contexts.append(contexts)
        elapsed = time.time() - t0

        score = score_answer(answer, keywords)

        status = "✅" if score["pass"] else "❌"
        if score["pass"]:
            passed += 1
        else:
            failed += 1

        # ── 纯检索指标（不依赖 LLM）──
        ret = compute_retrieval_metrics(sources, keywords)
        # ── 端到端 3s 预算 pass/fail ──
        under_3s = total_ms <= 3000

        budget_flag = "✅<3s" if under_3s else "⚠️>3s"
        print(f"    {status} 答案命中:{score['hit']}/{score['total']}  "
              f"检索 hit@5={ret['hit@5']:.0f} mrr={ret['mrr']:.2f} recall@10={ret['recall@10']:.2f}  "
              f"{total_ms}ms {budget_flag} (retr={retrieval_ms} llm={llm_ms})")
        if not score["pass"]:
            print(f"    回答: {answer[:120]}...")
            print("    --- top retrieval sources ---")
            for idx, src in enumerate(sources[:3], start=1):
                text = src.get("text", "")
                page = src.get("page", "?")
                chunk_id = src.get("chunk_id", "?")
                score_val = src.get("score", 0)
                snippet = text.replace("\n", " ")[:240]
                print(f"      [{idx}] page={page} chunk_id={chunk_id} score={score_val:.4f} text={snippet}...")
            print("    -----------------------------")

        results.append({
            "id": qid,
            "question": question,
            "focus": focus,
            "pass": score["pass"],
            "hit": score["hit"],
            "total": score["total"],
            "detail": score["detail"],
            "time_s": round(elapsed, 1),
            "retrieval_time_ms": retrieval_ms,
            "llm_time_ms": llm_ms,
            "total_time_ms": total_ms,
            "under_3s": under_3s,
            "retrieval": {k: round(v, 4) for k, v in ret.items()},
        })

    # 汇总
    print()
    print("=" * 70)
    print(f"  结果汇总")
    print("=" * 70)
    total = len(results)
    print(f"  通过: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"  失败: {failed}/{total} ({failed/total*100:.0f}%)")
    print()

    # 按侧重点分组
    by_focus = {}
    for r in results:
        f = r["focus"]
        if f not in by_focus:
            by_focus[f] = {"pass": 0, "total": 0}
        by_focus[f]["total"] += 1
        if r["pass"]:
            by_focus[f]["pass"] += 1

    print("  按类型:")
    for f, v in sorted(by_focus.items()):
        print(f"    {f}: {v['pass']}/{v['total']} ({v['pass']/v['total']*100:.0f}%)")

    # ── 汇总：检索质量（纯检索，不含 LLM）──
    def _avg(key):
        return sum(r["retrieval"].get(key, 0.0) for r in results) / total if total else 0.0
    agg_ret = {mk: _avg(mk) for mk in
               ["hit@1", "hit@3", "hit@5", "hit@10",
                "recall@5", "recall@10", "precision@5", "mrr"]}

    # ── 汇总：延迟 + 3s 预算 ──
    totals_ms = [r.get("total_time_ms", 0) for r in results]
    retr_ms = [r.get("retrieval_time_ms", 0) for r in results]
    llm_ms_list = [r.get("llm_time_ms", 0) for r in results]
    n_under_3s = sum(1 for r in results if r.get("under_3s"))
    under_3s_rate = n_under_3s / total if total else 0.0
    lat = {
        "retrieval_avg": sum(retr_ms) / total if total else 0,
        "llm_avg": sum(llm_ms_list) / total if total else 0,
        "total_avg": sum(totals_ms) / total if total else 0,
        "total_p50": _percentile(totals_ms, 50),
        "total_p95": _percentile(totals_ms, 95),
        "under_3s_rate": under_3s_rate,
    }

    print()
    print("  检索质量 (纯检索, 不含 LLM):")
    print(f"    Hit@1={agg_ret['hit@1']:.2f}  Hit@3={agg_ret['hit@3']:.2f}  "
          f"Hit@5={agg_ret['hit@5']:.2f}  Hit@10={agg_ret['hit@10']:.2f}")
    print(f"    Recall@5={agg_ret['recall@5']:.2f}  Recall@10={agg_ret['recall@10']:.2f}  "
          f"Precision@5={agg_ret['precision@5']:.2f}  MRR={agg_ret['mrr']:.2f}")
    print()
    print("  延迟 & 3s 预算:")
    print(f"    检索均值={lat['retrieval_avg']:.0f}ms  LLM均值={lat['llm_avg']:.0f}ms  总均值={lat['total_avg']:.0f}ms")
    print(f"    总延迟 p50={lat['total_p50']:.0f}ms  p95={lat['total_p95']:.0f}ms")
    print(f"    端到端<3s 达标率: {under_3s_rate*100:.0f}% ({n_under_3s}/{total})")

    # RAGAS 评分
    if with_ragas and all_answers:
        print()
        print("=" * 70)
        print("  RAGAS 评分")
        print("=" * 70)
        ragas_scores = run_ragas(all_questions, all_answers, all_ground_truths, all_contexts)
        if not ragas_scores:
            print("  ⚠️ RAGAS 未返回结果（缺少 API Key 或依赖）")
            ragas_scores = {}
        elif "error" in ragas_scores:
            print(f"  ❌ {ragas_scores['error']}")
        else:
            for k, v in ragas_scores.items():
                print(f"  {k:<20}: {v:.4f}")
            print()
            print("  参考范围:")
            print("    >0.9 优秀  |  >0.7 良好  |  >0.5 及格  |  <0.5 待优化")
    else:
        ragas_scores = {}

    # 保存结果（归档 + latest）
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retrieval_mode": retrieval_mode or "default",
        "total": total,
        "passed": passed,
        "failed": failed,
        "answer_pass_rate": round(passed / total, 4) if total else 0,
        "retrieval_quality": {k: round(v, 4) for k, v in agg_ret.items()},
        "latency": {k: round(v, 2) for k, v in lat.items()},
        "by_focus": {f: {"pass": v["pass"], "total": v["total"]} for f, v in by_focus.items()},
        "ragas": ragas_scores,
        "results": results,
    }
    os.makedirs("output/eval_history", exist_ok=True)
    mode_tag = (retrieval_mode or "default")
    ts_file = time.strftime("%Y%m%d_%H%M%S")
    hist_path = f"output/eval_history/eval_{ts_file}_{mode_tag}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    # 同时写一份 latest 供快速查看
    with open("output/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已归档: {hist_path}")
    print(f"  跨运行对比:   python eval_compare.py")

    return passed == total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG 快速评估")
    parser.add_argument("--api-url", default="http://localhost:8010", help="RAG API 地址")
    parser.add_argument("--ragas", action="store_true", help="额外运行 RAGAS 评分（慢且需联网，默认关闭）")
    parser.add_argument("--retrieval-mode", nargs="*", choices=["vector", "fulltext", "hybrid"], default=None,
                        help="指定检索模式：vector / fulltext / hybrid（默认 hybrid）。可指定多个模式，例如 --retrieval-mode vector fulltext")
    parser.add_argument("--testset", default=None,
                        help="外部测试集 JSON 路径（默认用内置招股书 16 题）。例如 --testset backend/ccf_testset.json")
    args = parser.parse_args()
    if args.retrieval_mode is not None and len(args.retrieval_mode) == 0:
        args.retrieval_mode = None
    if args.testset:
        with open(args.testset, encoding="utf-8") as f:
            loaded = json.load(f)
        # 原地替换，保持模块级 TEST_CASES 引用不变
        TEST_CASES.clear()
        TEST_CASES.extend(loaded)
        print(f"  已加载外部测试集: {args.testset}（{len(TEST_CASES)} 题）")
    run_eval(args.api_url, with_ragas=args.ragas, retrieval_mode=args.retrieval_mode)
