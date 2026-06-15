"""
LightRAG 集成模块
针对招股书优化的知识图谱构建 + 检索

在 Windows gpu_env 运行:
  cd E:\10--agent--任务\任务 1\backend
  python lightrag_integration.py --build
  python lightrag_integration.py --query "武汉兴图新科注册资本是多少？"
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

# 确保 backend 在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import get_logger, log_exception

logger = get_logger("lightrag")


# ──────────────────────────────────────────────────────
#  招股书专用实体类型定义
# ──────────────────────────────────────────────────────

FINANCIAL_ENTITY_TYPES = """
- Company: 公司、企业（如：武汉兴图新科电子股份有限公司、武汉力源信息技术股份有限公司）
- Person: 个人（如：法定代表人、控股股东、董事、高管，程家明、赵马克）
- Organization: 非公司组织（如：军方客户、政府部门、行业协会、保荐机构、上海证券交易所）
- FinancialProduct: 金融产品（如：首次公开发行股票、科创板上市）
- Project: 募投项目（如：基于云联邦架构的军用视频指挥平台升级及产业化项目、研发中心建设项目）
- FinancialMetric: 财务指标及数值（如：注册资本1360万元、营业收入、净利润、毛利率、募集资金40584.83万元）
- Industry: 行业领域（如：电子信息行业、军用视频指挥控制领域、集成电路行业）
- Technology: 技术标准、专利、软件著作权（如：GJB军用标准、实用新型专利）
- LegalDocument: 法律法规、规章制度（如：军工资质管理规定、公司法）
- Location: 地理位置（如：武汉市、湖北省、东湖新技术开发区）
"""

# 招股书专用示例（注入到默认 prompt 的 {examples} 占位符）
FINANCIAL_EXTRACTION_EXAMPLES = """entity<|#|>兴图新科<|#|>Company<|#|>武汉兴图新科电子股份有限公司，军队专用视频指挥控制系统提供商
entity<|#|>力源信息<|#|>Company<|#|>武汉力源信息技术股份有限公司，电子元器件分销商
entity<|#|>程家明<|#|>Person<|#|>兴图新科控股股东、实际控制人，持股比例55.85%
entity<|#|>赵马克<|#|>Person<|#|>力源信息控股股东、实际控制人
entity<|#|>云联邦架构项目<|#|>Project<|#|>基于云联邦架构的军用视频指挥平台升级及产业化项目，总投资20,658.33万元
entity<|#|>补充流动资金<|#|>Project<|#|>募集资金补充流动资金项目，金额15,000.00万元
entity<|#|>军用领域收入<|#|>FinancialMetric<|#|>兴图新科来自军用视频指挥控制领域的营业收入
entity<|#|>注册资本<|#|>FinancialMetric<|#|>兴图新科注册资本1360万元
relation<|#|>程家明<|#|>兴图新科<|#|>持股,控股<|#|>程家明持有兴图新科55.85%的股份
relation<|#|>兴图新科<|#|>云联邦架构项目<|#|>投资<|#|>兴图新科计划投资20,658.33万元用于云联邦架构项目
relation<|#|>兴图新科<|#|>补充流动资金<|#|>募集资金用途<|#|>兴图新科计划使用15,000.00万元募集资金补充流动资金
<|COMPLETE|>"""


# ──────────────────────────────────────────────────────
#  LLM 和 Embedding 函数构建
# ──────────────────────────────────────────────────────

def build_deepseek_llm_func(api_key: str = "", base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
    """构建 DeepSeek LLM 调用函数（供 LightRAG 使用）"""
    import httpx

    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("需要设置 DEEPSEEK_API_KEY 环境变量或传入 api_key 参数")

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        # kwargs 包含 LightRAG 传入的 hashing_kv 等参数，忽略即可
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    return llm_func


def build_openai_compatible_llm_func(api_key: str, base_url: str, model: str = "deepseek-chat"):
    """通用 OpenAI 兼容 LLM 函数"""
    return build_deepseek_llm_func(api_key=api_key, base_url=base_url)


def build_bge_embedding_func(model_path: str = "", device: str = "auto"):
    """构建 BGE-M3 嵌入函数（供 LightRAG 使用）"""
    from lightrag.base import EmbeddingFunc
    import numpy as np

    if not model_path:
        model_path = os.environ.get(
            "BGE_MODEL_PATH",
            r"F:\4--专业所有安装的软件及改动设置\2-3--专高3\bge-m3"
        )

    _model = None

    async def _embed(texts: list[str]) -> np.ndarray:
        nonlocal _model
        if _model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载嵌入模型: {model_path}")
            _model = SentenceTransformer(model_path, device=device)
            logger.info(f"嵌入模型加载完成, device={_model.device}")
        embeddings = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings)

    # 获取维度
    try:
        from sentence_transformers import SentenceTransformer
        tmp = SentenceTransformer(model_path, device="cpu")
        dim = tmp.get_sentence_embedding_dimension()
        del tmp
        import gc; gc.collect()
    except Exception:
        dim = 1024  # BGE-M3 默认

    return EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=_embed)


# ──────────────────────────────────────────────────────
#  LightRAG 实例创建
# ──────────────────────────────────────────────────────

def create_lightrag(
    working_dir: str = "./lightrag_storage",
    llm_func=None,
    embedding_func=None,
    llm_model_name: str = "deepseek-chat",
):
    """创建针对招股书优化的 LightRAG 实例"""
    from lightrag import LightRAG

    os.makedirs(working_dir, exist_ok=True)

    # 覆盖 LightRAG 全局 PROMPTS 中的示例（针对金融文档）
    from lightrag.prompt import PROMPTS
    PROMPTS["entity_extraction_examples"] = FINANCIAL_EXTRACTION_EXAMPLES.split("\n")

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=embedding_func,
        llm_model_name=llm_model_name,
        tiktoken_model_name="gpt-4",  # 用 cl100k_base 编码，避免下载 o200k_base
        # 分块参数
        chunk_token_size=1000,
        chunk_overlap_token_size=150,
        # 检索参数
        top_k=20,
        chunk_top_k=12,
        # 实体抽取参数
        entity_extract_max_gleaning=1,
        entity_extract_max_records=30,
        entity_extract_max_entities=20,
        # 自定义实体类型（通过 addon_params 注入）
        addon_params={
            "entity_types_guidance": FINANCIAL_ENTITY_TYPES,
            "language": "中文",
        },
    )

    # 初始化 pipeline 状态（新版 LightRAG 要求）
    import asyncio
    try:
        asyncio.run(rag.initialize_storages())
        logger.info("LightRAG 存储初始化完成")
    except RuntimeError:
        # event loop 已存在时用另一种方式
        loop = asyncio.get_event_loop()
        loop.run_until_complete(rag.initialize_storages())
        logger.info("LightRAG 存储初始化完成")

    logger.info(f"LightRAG 已创建: working_dir={working_dir}")
    return rag


# ──────────────────────────────────────────────────────
#  知识图谱构建
# ──────────────────────────────────────────────────────

def build_knowledge_graph(
    rag,
    chunks_path: str = "output/chunks/all_chunks.json",
    batch_size: int = 20,
) -> dict:
    """
    从已有 chunks 构建知识图谱
    
    Args:
        rag: LightRAG 实例
        chunks_path: chunks JSON 文件路径
        batch_size: 每批插入的 chunk 数量
    """
    start = time.time()

    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = json.load(f)

    logger.info(f"加载 {len(all_chunks)} 个 chunks, 开始构建知识图谱...")

    # 按页合并 chunks
    page_texts = {}
    for c in all_chunks:
        page = c.get("page", 0)
        if page not in page_texts:
            page_texts[page] = []
        page_texts[page].append(c.get("text", ""))

    merged_docs = []
    for page_num in sorted(page_texts.keys()):
        text = "\n".join(page_texts[page_num]).strip()
        if len(text) > 50:
            merged_docs.append(text)

    logger.info(f"合并为 {len(merged_docs)} 页文档, 开始分批插入...")

    import asyncio

    async def _insert_all():
        total = 0
        for i in range(0, len(merged_docs), batch_size):
            batch = merged_docs[i : i + batch_size]
            try:
                await rag.ainsert(batch)
                total += len(batch)
                logger.info(f"  已插入 {total}/{len(merged_docs)} 页")
            except Exception as e:
                log_exception(logger, f"  插入第 {i}~{i+len(batch)} 页失败", e)
        return total

    total_inserted = asyncio.run(_insert_all())
    elapsed = time.time() - start
    result = {
        "total_chunks": len(all_chunks),
        "total_pages": len(merged_docs),
        "inserted": total_inserted,
        "time_s": round(elapsed, 1),
    }
    logger.info(f"知识图谱构建完成: {result}")
    return result


# ──────────────────────────────────────────────────────
#  查询
# ──────────────────────────────────────────────────────

def query_lightrag(rag, question: str, mode: str = "hybrid") -> dict:
    """
    查询 LightRAG 知识图谱
    
    Args:
        rag: LightRAG 实例
        question: 问题
        mode: local/global/hybrid/naive/mix
    
    Returns:
        {"answer": str, "mode": str, "time_ms": int}
    """
    from lightrag.base import QueryParam

    start = time.time()

    param = QueryParam(
        mode=mode,
        top_k=20,
        chunk_top_k=12,
        response_type="简洁的中文段落，保留原始数据和页码",
    )

    try:
        result = rag.query(question, param=param)
        answer = result if isinstance(result, str) else ""
        elapsed_ms = int((time.time() - start) * 1000)
        return {"answer": answer, "mode": mode, "time_ms": elapsed_ms}
    except Exception as e:
        log_exception(logger, "LightRAG 查询失败", e)
        return {"answer": f"查询失败: {e}", "mode": mode, "time_ms": 0}


def get_lightrag_context(rag, question: str, mode: str = "hybrid") -> dict:
    """
    只获取 LightRAG 检索上下文（不调用 LLM 生成回答）
    用于 RAGAS 评估
    """
    from lightrag.base import QueryParam

    param = QueryParam(
        mode=mode,
        top_k=20,
        chunk_top_k=12,
        only_need_context=True,
    )

    try:
        context = rag.query(question, param=param)
        return {"context": context if isinstance(context, str) else str(context)}
    except Exception as e:
        log_exception(logger, "LightRAG 上下文获取失败", e)
        return {"context": ""}


# ──────────────────────────────────────────────────────
#  CLI 入口
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LightRAG 招股书知识图谱")
    parser.add_argument("--build", action="store_true", help="构建知识图谱")
    parser.add_argument("--query", type=str, help="查询问题")
    parser.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid", "naive", "mix"])
    parser.add_argument("--api-key", default="", help="DeepSeek API Key")
    parser.add_argument("--working-dir", default="./lightrag_storage")
    parser.add_argument("--chunks", default="output/chunks/all_chunks.json")
    parser.add_argument("--batch-size", type=int, default=10, help="每批插入页数")
    parser.add_argument("--no-embed", action="store_true", help="使用 LightRAG 默认嵌入（不用 BGE-M3）")
    args = parser.parse_args()

    # 切换到 backend 目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 构建 LLM 函数
    llm_func = build_deepseek_llm_func(api_key=args.api_key)

    # 构建嵌入函数
    embedding_func = None
    if not args.no_embed:
        try:
            embedding_func = build_bge_embedding_func()
            print("使用 BGE-M3 嵌入模型")
        except Exception as e:
            print(f"BGE-M3 加载失败，使用默认嵌入: {e}")

    # 创建 LightRAG
    rag = create_lightrag(
        working_dir=args.working_dir,
        llm_func=llm_func,
        embedding_func=embedding_func,
    )

    if args.build:
        print("=" * 60)
        print("开始构建知识图谱...")
        print("=" * 60)
        result = build_knowledge_graph(rag, chunks_path=args.chunks, batch_size=args.batch_size)
        print(f"\n构建完成: {json.dumps(result, ensure_ascii=False, indent=2)}")

    if args.query:
        print(f"\n查询: {args.query}")
        print(f"模式: {args.mode}")
        print("-" * 60)
        answer = query_lightrag(rag, args.query, mode=args.mode)
        print(f"回答 ({answer['time_ms']}ms):\n{answer['answer']}")
