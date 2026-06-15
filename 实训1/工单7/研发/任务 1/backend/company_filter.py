"""
公司识别 + 文档路由过滤

两层策略：
  1. 查询文本显式提到公司名 → 直接过滤对手公司chunk
  2. 查询没提公司名 → 从检索结果推断目标公司（多数投票），再过滤

原理：
  - 两个招股书PDF各自独立，chunk中有些包含公司名（~40%），有些没有（~60%，目录/声明等通用内容）
  - 可识别公司名的chunk中，如果某家公司占比>=70%，认定查询目标是它
  - 过滤掉对手公司的chunk，保留目标公司专属 + 通用chunk
  - 如果可识别chunk太少或两家差不多，不过滤（可能是跨公司问题）

不需要改Milvus schema，不需要重新建索引。
"""
import re
from collections import Counter
from logger import get_logger

logger = get_logger("company_filter")


# ── 公司注册表 ──────────────────────────────────────────────

COMPANIES = {
    "兴图新科": {
        "aliases": ["兴图新科", "武汉兴图新科", "兴图新科电子", "武汉兴图新科电子"],
        "exclude": ["力源信息", "武汉力源信息", "力源信息技术"],
        "pdf_file": "招股说明书1-无水印.pdf",
    },
    "力源信息": {
        "aliases": ["力源信息", "武汉力源信息", "力源信息技术", "武汉力源信息技术"],
        "exclude": ["兴图新科", "武汉兴图新科", "兴图新科电子"],
        "pdf_file": "招股说明书2--无水印.pdf",
    },
}

# 多数投票阈值：可识别chunk中某公司占比 >= 此值才认定目标公司
MAJORITY_THRESHOLD = 0.65


def detect_company(query: str) -> str | None:
    """
    从查询文本中检测目标公司名。
    返回: 公司key ("兴图新科"/"力源信息") 或 None（未识别到）
    """
    if not query:
        return None

    for company_key, info in COMPANIES.items():
        for alias in info["aliases"]:
            if alias in query:
                logger.debug(f"公司检测(查询): '{query[:40]}...' → {company_key}")
                return company_key

    return None


def infer_company_from_results(results: list[dict]) -> str | None:
    """
    从检索结果中推断目标公司（多数投票）。
    统计结果中包含公司名的chunk，哪家公司占多数就认定目标是它。
    
    返回: 公司key 或 None（无法确定）
    """
    if not results:
        return None

    company_votes: Counter = Counter()
    
    for r in results:
        text = r.get("text", "")
        for company_key, info in COMPANIES.items():
            # 优先用【文件】前缀识别（100%准确）
            file_marker = f"【文件】{info['pdf_file']}"
            if file_marker in text:
                company_votes[company_key] += 1
                break
            # 后备：文本内容匹配公司名
            if any(alias in text for alias in info["aliases"]):
                company_votes[company_key] += 1
                break  # 一个chunk只投一票

    total_votes = sum(company_votes.values())
    if total_votes == 0:
        logger.debug("公司推断: 无可识别公司名的chunk")
        return None

    # 找得票最多的公司
    top_company, top_count = company_votes.most_common(1)[0]
    ratio = top_count / total_votes

    logger.debug(f"公司推断: {dict(company_votes)}, 总票={total_votes}, "
                 f"top={top_company}({top_count}/{total_votes}={ratio:.0%})")

    if ratio >= MAJORITY_THRESHOLD:
        logger.debug(f"公司推断结果: {top_company} (占比{ratio:.0%} >= {MAJORITY_THRESHOLD:.0%})")
        return top_company
    else:
        logger.debug(f"公司推断结果: 不确定 (占比{ratio:.0%} < {MAJORITY_THRESHOLD:.0%})")
        return None


def filter_by_company(results: list[dict], company_key: str | None) -> list[dict]:
    """
    根据目标公司过滤检索结果。
    
    规则：
      - company_key=None → 不过滤，原样返回
      - company_key="兴图新科" → 移除包含"力源信息"关键词的chunk
      - company_key="力源信息" → 移除包含"兴图新科"关键词的chunk
    
    保留没有公司名的chunk（目录页、声明页、通用法规等），这些对两家公司都适用。
    """
    if not company_key or company_key not in COMPANIES:
        return results

    exclude_keywords = COMPANIES[company_key]["exclude"]
    # 对手公司的文件名标记（header_cleaner 洗掉公司名后，文件名是唯一标识）
    opponent_keys = [k for k in COMPANIES if k != company_key]
    opponent_file_markers = [
        f"【文件】{COMPANIES[k]['pdf_file']}" for k in opponent_keys
    ]
    original_count = len(results)

    filtered = []
    demoted = []
    for r in results:
        text = r.get("text", "")
        # 如果chunk包含对手公司的文件名标记或关键词，排除
        is_opponent = (
            any(marker in text for marker in opponent_file_markers)
            or any(kw in text for kw in exclude_keywords)
        )
        if is_opponent:
            demoted.append(r)
            continue
        filtered.append(r)

    # 如果过滤后结果太少，把对手公司chunk降权而不是全丢掉
    if len(filtered) < 8 and demoted:
        logger.debug(f"公司过滤({company_key}): 过滤后仅剩{len(filtered)}条，将{len(demoted)}条对手公司chunk降权")
        for r in demoted:
            r["score"] = r.get("score", 0) * 0.3  # 对手公司chunk分数打3折
            filtered.append(r)
        # 重新排序
        filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

    removed = original_count - len(filtered)
    if removed > 0:
        logger.debug(f"公司过滤({company_key}): {original_count} → {len(filtered)} (移除{removed}条对手公司chunk)")

    return filtered


def route_and_filter(query: str, results: list[dict]) -> list[dict]:
    """
    统一入口：先从查询检测公司，检测不到则从结果推断，最后过滤。
    """
    # 第一层：从查询文本检测
    company = detect_company(query)
    
    # 第二层：查询没提到公司，从检索结果推断
    if not company:
        company = infer_company_from_results(results)
    
    # 过滤
    return filter_by_company(results, company)
