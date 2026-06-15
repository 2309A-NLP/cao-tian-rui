"""
retrieval_strategy.py — 检索策略配置层

统一管理三种检索模式（向量、全文、混合）的配置与应用。
支持：
  - 检索模式切换：vector_only / fulltext_only / hybrid
  - 重排算法选择：reranker / keyword / adaptive / none
  - 权重与阈值可配
  - 查询改写、指代消解的衔接

架构：
  用户/前端 → config / API → RetrievalStrategy → vector_store.search/hybrid_search/search_bm25 → 重排 → 最终结果
"""

from enum import Enum
from typing import Optional, Callable
import time
import re
import json
import logging
from pathlib import Path

import time

from logger import get_logger, get_perf_logger, log_exception

logger = get_logger("retrieval")
perf_log = get_perf_logger()


# ──────────────────────────────────────────────
#  共享常量（关键词重排 & 强制召回共用）
# ──────────────────────────────────────────────

_STOP_WORDS = {'的', '了', '是', '在', '有', '多少', '哪些', '哪个',
               '分别', '各', '以及', '或者', '怎么', '如何', '什么',
               '谁', '为什么', '哪', '吗', '啊', '呢', '吧', '呀',
               '公司', '股份', '有限', '电子', '公司法',
               '报告期内', '本次', '中国', '根据', '包括', '其中'}

_COMPANY_NAMES = {'武汉兴图新科', '兴图新科', '武汉力源', '力源信息',
                  '电子股份有限', '股份有限公司', '有限',
                  '招股说明书', '申报稿', '招股意向书', '兴图', '新科'}

# ──────────────────────────────────────────────
#  枚举 & 数据类
# ──────────────────────────────────────────────

class RetrievalMode(str, Enum):
    VECTOR_ONLY = "vector"       # 仅向量检索
    FULLTEXT_ONLY = "fulltext"   # 仅全文检索（BM25）
    HYBRID = "hybrid"            # 向量 + 全文混合


class RerankMethod(str, Enum):
    NONE = "none"                # 不重排
    RERANKER = "reranker"        # BGE Reranker 交叉编码器
    KEYWORD = "keyword"          # 关键词命中率重排
    ADAPTIVE = "adaptive"        # 自适应：根据结果置信度自动选择


class RetrievalConfig:
    """
    检索策略配置（可序列化，可从前端传入）
    """
    def __init__(
        self,
        mode: str = "hybrid",
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        # 混合检索权重
        alpha: float = 0.4,          # BM25 权重（0-1），向量权重 = 1-alpha
        rrf_k: int = 60,             # RRF 融合常数
        # 查询改写
        query_rewrite: bool = True,
        synonym_expansion: bool = True,
        max_rewrites: int = 3,        # 从7→3：每个变体都要全量BM25(18443文档)，7变体是<3s的主要破坏者。
                                       # 保留 原查询 + 2个同义变体，兼顾词表不匹配召回与延迟。可按 Hit@5 实测回调。
        # 重排
        rerank_method: str = "adaptive",
        rerank_top_n: int = 50,       # 送给重排器的候选数（从20→50，避免过早截断丢失关键chunk）
        # 向量源数
        per_variant_k_multiplier: int = 3,
    ):
        self.mode = mode if mode in ("vector", "fulltext", "hybrid") else "hybrid"
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.query_rewrite = query_rewrite
        self.synonym_expansion = synonym_expansion
        self.max_rewrites = max_rewrites
        self.rerank_method = rerank_method if rerank_method in ("none", "reranker", "keyword", "adaptive") else "adaptive"
        self.rerank_top_n = rerank_top_n
        self.per_variant_k_multiplier = per_variant_k_multiplier

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalConfig":
        valid_keys = {k for k in cls.__init__.__code__.co_varnames[1:]}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def __repr__(self) -> str:
        return (
            f"RetrievalConfig(mode={self.mode}, top_k={self.top_k}, "
            f"alpha={self.alpha}, rerank={self.rerank_method})"
        )


# ──────────────────────────────────────────────
#  检索策略执行器
# ──────────────────────────────────────────────

class RetrievalExecutor:
    """
    检索策略执行器：整合 vector_store + 重排器 → 统一检索入口

    用法:
        executor = RetrievalExecutor(vector_store, reranker_path)
        sources = executor.search(query, config)
    """

    def __init__(
        self,
        vector_store,
        reranker_model_path: str = "",
    ):
        self.vector_store = vector_store
        self._reranker = None
        self._reranker_path = reranker_model_path
        self._reranker_tokenizer = None
        self._reranker_device = "cpu"

    # ── 查询改写 ─────────────────────────────────

    _SYNONYMS = {
        '军用领域': ['国防领域', '军工领域', '军队', '军事领域', '国防', '军方'],
        '军用': ['国防', '军工', '军队', '军事', '军品', '军方'],
        '民用领域': ['民用', '民品', '民用市场', '民用产品'],
        '民用': ['民用领域', '民品领域', '民用市场'],
        '营业收入': ['收入', '营收', '主营业务收入', '销售额', '销售收入'],
        '净利润': ['净利', '归母净利润', '归属于母公司'],
        '主营业务收入': ['收入', '营收', '营业收入', '销售额'],
        '毛利率': ['毛利', '毛利率水平'],
        '占比': ['占比', '比例', '比重', '占收入比重', '占主营业务收入比重'],
        '募集资金': ['募集资金', '募资', '募集', '募投项目', '投资项目'],
        '武汉力源信息技术': ['力源信息', '武汉力源', '力源信息技术'],
        '武汉兴图新科电子': ['兴图新科', '武汉兴图新科', '兴图新科电子'],
        '视频指挥控制': ['视频指挥控制', '视频指挥', '指挥控制'],
        '视频预警控制': ['视频预警控制', '视频预警', '预警控制'],
    }

    @staticmethod
    def _query_rewrite(query: str, config: RetrievalConfig) -> list[str]:
        """查询改写：生成多个同义/扩展查询"""
        queries = [query]
        rewrites = set()

        # 同义词扩展
        if config.synonym_expansion:
            for keyword, alt_list in RetrievalExecutor._SYNONYMS.items():
                if keyword in query:
                    for alt in alt_list:
                        new_q = query.replace(keyword, alt)
                        if new_q != query and new_q not in rewrites:
                            rewrites.add(new_q)

        # 公司全称缩短（仅当查询使用了简称时才生成全称→简称变体）
        # 条件：查询中不含完整的公司全称（>=6字）时，说明用户用了简称，
        # 此时生成变体帮 chunk 找到匹配。如果查询已含全称，向量搜索自能命中。
        company_patterns = [
            (r'武汉兴图新科电子股份有限公司', ['兴图新科', '武汉兴图新科']),
            (r'武汉兴图新科', ['兴图新科']),
            (r'武汉力源信息技术股份有限公司', ['武汉力源', '力源信息']),
            (r'武汉力源信息', ['武汉力源', '力源信息']),
        ]
        for pattern, short_list in company_patterns:
            if not re.search(pattern, query):
                continue
            # 如果查询已含完整全称（匹配到的文本长度>=6），跳过变体生成
            match = re.search(pattern, query)
            if match and len(match.group()) >= 6:
                continue
            for short in short_list:
                new_q = re.sub(pattern, short, query)
                if new_q not in rewrites:
                    rewrites.add(new_q)

        # 去除非关键限定词
        relaxations = re.sub(r'(分别|各是多少|分别是多少|分别是哪些|有哪些|都有哪些)', '', query).strip()
        if relaxations and relaxations != query and relaxations not in rewrites:
            rewrites.add(relaxations)

        # 收入类查询精简
        if '收入' in query and ('多少' in query or '哪些' in query):
            kw_q = re.sub(r'(分别是多少|是多少|分别是哪些|有哪些|都有哪些)', '', query)
            kw_q = kw_q.replace('分别', '').strip()
            if kw_q and kw_q not in rewrites:
                rewrites.add(kw_q)

        queries.extend(sorted(rewrites))
        if len(queries) > config.max_rewrites:
            queries = [queries[0]] + sorted(rewrites, key=lambda x: len(x))[:config.max_rewrites - 1]

        logger.debug(f"查询改写: '{query[:50]}' → {len(queries)} 个变体")
        return queries

    # ── 检索核心 ─────────────────────────────────

    def search(self, query: str, config: RetrievalConfig) -> list[dict]:
        """
        统一检索入口

        返回: [{"text": str, "page": int, "chunk_id": int, "score": float}, ...]
        """

        if config.mode == "fulltext":
            return self._search_fulltext(query, config)
        elif config.mode == "vector":
            return self._search_vector(query, config)
        else:  # hybrid
            return self._search_hybrid(query, config)

    def search_with_rerank(self, query: str, config: RetrievalConfig) -> list[dict]:
        """
        统一检索 + 重排入口

        按需启用组件原则：
        - 检索（核心）→ 始终执行
        - 强制召回 → 仅当检索结果不足 3 条时启用
        - 公司过滤 → 仅在问题明确提及公司名称时启用
        - 查询改写 → 仅在单轮对话且有代词时启用（已在 rag_engine 控制）
        - 查询变体 → 仅在开启了 query_rewrite 时启用
        - 重排 → 按配置的可选环节

        返回: [{"text": str, "page": int, "chunk_id": int, "score": float}, ...]
        """
        t_start = time.time()

        # ── 1. 检索（核心） ──
        t0 = time.time()
        sources = self.search(query, config)
        t_search = int((time.time() - t0) * 1000)
        n_raw = len(sources)

        # ── 1b. 强制召回（按需：检索结果不足时才执行） ──
        n_after_recall = len(sources)
        if len(sources) < 3:
            sources = self._force_recall(query, sources)
            n_after_recall = len(sources)

        # ── 1c. 公司路由过滤（按需：问题中明确提及公司名时才执行） ──
        n_after_filter = len(sources)
        if self._has_company_name(query):
            try:
                from company_filter import route_and_filter
                sources = route_and_filter(query, sources)
                n_after_filter = len(sources)
            except ImportError:
                pass  # company_filter 模块不存在时跳过

        # ── 2. 关键词重排（保底逻辑，确保含关键词的排上来） ──
        sources = self._keyword_rerank(query, sources)

        # ── 3. 裁剪到 rerank_top_n 作为重排候选 ──
        sources = sources[:config.rerank_top_n]

        # ── 4. 按选定的重排方法执行（可选） ──
        t_rerank = 0
        if config.rerank_method == "reranker":
            t0 = time.time()
            sources = self._rerank_with_model(query, sources)
            t_rerank = int((time.time() - t0) * 1000)
        elif config.rerank_method == "keyword":
            sources = self._keyword_rerank(query, sources)
        elif config.rerank_method == "adaptive":
            # adaptive：只有当 keyword 重排后 top1 分数偏低时才启用模型重排
            top1_score = sources[0].get("score", 0) if sources else 0
            if top1_score < 5.0:
                t0 = time.time()
                sources = self._rerank_with_model(query, sources)
                t_rerank = int((time.time() - t0) * 1000)

        # ── 5. 最终截取 top_k ──
        result = sources[:config.top_k]

        t_total = int((time.time() - t_start) * 1000)

        # 性能日志
        pages = sorted(set(s.get("page", 0) for s in result))
        perf_log.info(
            f"[RETRIEVAL] query=\"{query[:60]}\" "
            f"mode={config.mode} rerank={config.rerank_method} "
            f"raw={n_raw} recall={n_after_recall} filtered={n_after_filter} "
            f"final={len(result)} pages={pages} "
            f"search={t_search}ms rerank={t_rerank}ms total={t_total}ms "
            f"top1_score={result[0].get('score',0):.4f}" if result else
            f"[RETRIEVAL] query=\"{query[:60]}\" "
            f"mode={config.mode} rerank={config.rerank_method} "
            f"raw={n_raw} recall={n_after_recall} filtered={n_after_filter} "
            f"final=0 search={t_search}ms total={t_total}ms"
        )
        if result:
            perf_log.debug(f"[RETRIEVAL TOP] {[{'page': s.get('page'), 'chunk_id': s.get('chunk_id'), 'score': s.get('score'), 'source': s.get('source')} for s in result[:5]]}")

        return result

    # ── 检索子方法 ───────────────────────────────

    def _search_vector(self, query: str, config: RetrievalConfig) -> list[dict]:
        """纯向量检索"""
        if not self.vector_store or not hasattr(self.vector_store, 'search'):
            return []
        # 配合查询改写：多个变体各自检索后去重合并
        all_sources = []
        seen_keys = set()
        query_variants = [query]
        if config.query_rewrite:
            query_variants = self._query_rewrite(query, config)
        per_variant_k = max(config.top_k * config.per_variant_k_multiplier, 25)
        for vq in query_variants:
            hits = self.vector_store.search(vq, top_k=per_variant_k)
            for s in hits:
                key = (s.get("page"), s.get("chunk_id"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_sources.append(s)
        # 分数过滤
        sources = [s for s in all_sources if s.get("score", 0) >= config.similarity_threshold]
        logger.debug(f"向量检索: query='{query[:30]}', 命中={len(sources)} 条")
        return sources

    def _search_fulltext(self, query: str, config: RetrievalConfig) -> list[dict]:
        """纯全文检索（BM25）"""
        if not self.vector_store or not hasattr(self.vector_store, 'search_bm25'):
            return []
        if not getattr(self.vector_store, 'bm25_ready', False):
            logger.warning("BM25 索引未就绪，全文检索不可用")
            return []
        all_sources = []
        seen_keys = set()
        query_variants = [query]
        if config.query_rewrite:
            query_variants = self._query_rewrite(query, config)
        per_variant_k = max(config.top_k * config.per_variant_k_multiplier, 25)
        for vq in query_variants:
            hits = self.vector_store.search_bm25(vq, top_k=per_variant_k)
            for s in hits:
                key = (s.get("page"), s.get("chunk_id"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_sources.append(s)
        sources = [s for s in all_sources if s.get("score", 0) >= config.similarity_threshold]
        logger.debug(f"全文检索: query='{query[:30]}', 命中={len(sources)} 条")
        return sources

    def _search_hybrid(self, query: str, config: RetrievalConfig) -> list[dict]:
        """混合检索：向量 + 全文 BM25，RRF 融合（统一走 RRF merge，不再走 vector_store.hybrid_search）"""
        vec_sources = self._search_vector(query, config)
        ft_sources = self._search_fulltext(query, config)
        sources = self._rrf_merge(vec_sources, ft_sources, config)
        logger.debug(f"混合检索: query='{query[:30]}', 命中={len(sources)} 条, alpha={config.alpha}")
        return sources

    @staticmethod
    def _has_company_name(query: str) -> bool:
        """检查查询中是否明确提到了公司名称

        按需启用公司过滤：问题中明确提及某家公司时才过滤，
        避免不必要的过滤操作（通用问题如"行业上游有哪些"不需要）。
        """
        companies = [
            "兴图新科", "武汉兴图新科",
            "力源信息", "武汉力源",
        ]
        return any(c in query for c in companies)

    @staticmethod
    def _rrf_merge(vector_results: list[dict], bm25_results: list[dict],
                   config: RetrievalConfig) -> list[dict]:
        """RRF 融合（当 vector_store 不支持 hybrid_search 时的备用实现）"""
        if not vector_results and not bm25_results:
            return []
        if not bm25_results:
            return vector_results[:config.top_k]
        if not vector_results:
            return bm25_results[:config.top_k]

        rrf_scores = {}
        chunk_map = {}
        for rank, hit in enumerate(vector_results):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1 - config.alpha) / (config.rrf_k + rank + 1)
            chunk_map[cid] = hit
        for rank, hit in enumerate(bm25_results):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + config.alpha / (config.rrf_k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = hit

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        merged = []
        for cid, rrf_score in ranked[:config.top_k * 2]:
            if cid in chunk_map:
                item = dict(chunk_map[cid])
                orig_score = item.get("score", 0)
                item["score"] = round(rrf_score, 4)
                item["vector_score"] = orig_score
                item["bm25_score"] = next(
                    (h["score"] for h in bm25_results if h["chunk_id"] == cid), 0
                )
                if "source" not in item:
                    item["source"] = "hybrid"
                merged.append(item)
        return merged

    # ── 重排算法 ─────────────────────────────────

    def _rerank_with_model(self, query: str, sources: list[dict]) -> list[dict]:
        """BGE Reranker 交叉编码器精排"""
        if not sources:
            return sources
        reranker = self._get_reranker()
        if not reranker:
            return sources
        import torch
        try:
            to_rerank = sources[:20]  # 最多排20条
            pairs = [(query, s["text"][:1000]) for s in to_rerank]
            encoded = self._reranker_tokenizer(
                pairs, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )
            encoded = {k: v.to(self._reranker_device) for k, v in encoded.items()}
            with torch.no_grad():
                scores = reranker(**encoded).logits.squeeze(-1).cpu().tolist()
            if not isinstance(scores, list):
                scores = [scores]
            for s, score in zip(to_rerank, scores):
                s["rerank_score"] = round(score, 4)
            reranked = sorted(to_rerank, key=lambda x: x.get("rerank_score", 0), reverse=True)
            result = reranked + sources[20:]
            logger.debug(f"Reranker 重排完成: 首条 score={result[0].get('rerank_score', 'N/A')}")
            return result
        except Exception as e:
            log_exception(logger, "Reranker 推理失败，使用原始排序", e)
            return sources

    @staticmethod
    def _keyword_rerank(query: str, sources: list[dict],
                        keyword_weight: float = 0.5) -> list[dict]:
        """关键词命中率重排（增强版：精确短语匹配 + 模板页面惩罚）"""
        if not sources:
            return sources

        # 重置旧的 boost（避免多轮调用时累积）
        for s in sources:
            s["_keyword_boost"] = 0

        # 去除查询中的空格（JSON提取可能留下空格如"注 册资本"）
        clean_query = re.sub(r'\s+', '', query)

        # 提取查询中的关键词（2字以上）
        query_words = re.findall(r'[\u4e00-\u9fff]{2,}', clean_query)
        keywords = [w for w in query_words if len(w) >= 2 and w not in _STOP_WORDS]

        # 提取关键短语（3-8字的子串，用于精确匹配）
        # 过滤掉公司名相关子串——公司名应该由 intent_words 处理，不作为短语精确匹配
        key_phrases = set()
        for phrase_len in range(8, 2, -1):
            for i in range(len(clean_query) - phrase_len + 1):
                phrase = clean_query[i:i + phrase_len]
                if len(phrase) >= 3 and phrase not in _STOP_WORDS:
                    # 过滤纯公司名片段（如"武汉力源信息"、"兴图新科"等）
                    # 公司名相关子串会让封面/释义页排在数据页前面
                    is_company_fragment = any(
                        cn in phrase or phrase in cn
                        for cn in _COMPANY_NAMES
                    )
                    if is_company_fragment:
                        continue
                    # 只保留有意义的短语（不全是停用词）
                    if any(w not in _STOP_WORDS for w in re.findall(r'[\u4e00-\u9fff]{2}', phrase)):
                        key_phrases.add(phrase)
        key_phrases = sorted(key_phrases, key=len, reverse=True)  # 长短语优先

        # 查询特定关键词（排除公司名等通用词，用于识别查询意图）
        query_intent_words = [w for w in keywords
                              if w not in _COMPANY_NAMES and len(w) >= 2]

        numbers = re.findall(r'\d+', clean_query)

        # 模板页面识别（声明、目录等，查询无关）
        boilerplate_signals = ['声明', '承诺本招股说明书不存在虚假记载',
                               '确认招股说明书', '全体董事', '监事声明',
                               '保荐机构声明', '律师声明', '有关声明',
                               '招股说明书不存在虚假记载']

        for s in sources:
            text = s.get("text", "")
            boost = 0.0

            # 1. 完整查询匹配（最高权重）
            if clean_query in text or clean_query[:20] in text:
                boost += 3.0

            # 2. 精确短语匹配（长短语权重更高）
            for phrase in key_phrases:
                if phrase in text:
                    if len(phrase) >= 8:
                        boost += 1.0
                    elif len(phrase) >= 5:
                        boost += 0.7
                    else:
                        boost += 0.35

            # 3. 查询意图关键词匹配（排除公司名后的核心词）
            intent_hits = 0
            for kw in query_intent_words:
                if kw in text:
                    boost += 0.15
                    intent_hits += 1

            # 4. 多个意图关键词共现加成（关键！）
            if intent_hits >= 2:
                boost += 0.5  # 多个核心词同时出现 → 强烈相关

            # 5. 数字匹配
            for num in numbers:
                if num in text:
                    boost += 0.3

            # 6. 表格数据加成
            tab_count = text.count('\t')
            if tab_count >= 5:
                boost += 0.6
            elif tab_count >= 3:
                boost += 0.4
            elif tab_count >= 1:
                boost += 0.2
            if '【表格】' in text:
                boost += 0.4

            # 7. 模板页面惩罚（声明页、目录页等）
            is_boilerplate = any(bp in text for bp in boilerplate_signals)
            if is_boilerplate:
                boost -= 1.5  # 重罚

            s["_keyword_boost"] = boost

        for s in sources:
            base = s.get("score", 0)
            boost = s.get("_keyword_boost", 0)
            s["_combined"] = base + boost * keyword_weight

        sources.sort(key=lambda x: x["_combined"], reverse=True)
        # 将 _combined 写回 score，让下游（adaptive_rerank 等）读到正确的排序分
        for s in sources:
            s["score"] = s.get("_combined", s.get("score", 0))
        return sources

    def _force_recall(self, query: str, sources: list[dict],
                      chunk_dir: str = "output/chunks") -> list[dict]:
        """对查询中的关键短语做强制召回——从索引中 grep 含短语的 chunk 并强行加入候选池"""
        # 清除空格
        clean_query = re.sub(r'\s+', '', query)

        # 提取 2-4 字的关键词（排除公司名和停用词）
        recall_phrases = []
        query_words = re.findall(r'[\u4e00-\u9fff]{2,}', clean_query)
        for w in query_words:
            if w in _STOP_WORDS or w in _COMPANY_NAMES:
                continue
            if any(cn in w or w in cn for cn in _COMPANY_NAMES):
                continue
            if len(w) >= 2 and len(w) <= 6:
                recall_phrases.append(w)

        # 如果关键词太少，也从完整查询中提取 3-5 字子串（但排除公司名片段）
        if len(recall_phrases) < 2:
            for phrase_len in range(5, 2, -1):
                for i in range(len(clean_query) - phrase_len + 1):
                    phrase = clean_query[i:i + phrase_len]
                    # 跳过纯公司名片段的子串（如"科电子股份"、"兴图新"）
                    if any(cn in phrase for cn in _COMPANY_NAMES):
                        continue
                    if any(phrase in cn for cn in _COMPANY_NAMES):
                        continue
                    if phrase in _STOP_WORDS:
                        continue
                    # 跳过查询末端长度 <=2 字的残余
                    if len(phrase) < 3:
                        continue
                    if phrase not in recall_phrases:
                        recall_phrases.append(phrase)
                    if len(recall_phrases) >= 5:
                        break

        if not recall_phrases:
            return sources

        chunk_path = Path(chunk_dir) / "all_chunks.json"
        if not chunk_path.exists():
            return sources

        try:
            with open(chunk_path, encoding="utf-8") as f:
                all_chunks = json.load(f)
        except Exception:
            return sources

        seen_ids = {s.get("chunk_id") for s in sources}
        forced = []

        for phrase in recall_phrases[:3]:  # 最多用3个短语
            for c in all_chunks:
                cid = c.get("chunk_id")
                if cid not in seen_ids and phrase in c.get("text", ""):
                    forced.append({
                        "text": c["text"],
                        "page": c["page"],
                        "chunk_id": cid,
                        "score": 0.5,  # 基础分，关键词重排会提升它
                    })
                    seen_ids.add(cid)

        if forced:
            logger.info(f"强制召回: {len(forced)} 个含关键短语 '{recall_phrases[0]}' 的 chunk")
            sources.extend(forced)

        return sources

    def _adaptive_rerank(self, query: str, sources: list[dict]) -> list[dict]:
        """
        自适应重排：根据检索结果的置信度自动选择重排策略

        策略：
        1. 如果 top-1 的分数 >= 0.7 → 高置信度，不做额外重排（避免破坏好结果）
        2. 如果 top-1 的分数在 0.4 ~ 0.7 → 中等置信度，用 Reranker 精排
        3. 如果 top-1 的分数 < 0.4 → 低置信度，用关键词重排 + 多样性扩展（扩大候选集）
        4. 如果候选集内部多样性低（大量同一页面）→ 触发多样性重排（分散到不同页面）
        """
        if not sources:
            return sources

        top_score = sources[0].get("score", 0)
        top_is_good = top_score >= 0.7

        # 检查来源页面的多样性
        pages = [s.get("page", 0) for s in sources[:10]]
        page_diversity = len(set(pages)) / max(len(pages), 1)

        logger.debug(
            f"自适应重排: top_score={top_score:.4f}, "
            f"page_diversity={page_diversity:.2f}, count={len(sources)}"
        )

        # 高置信度 + 多样性足够 → 不做额外重排
        if top_is_good and page_diversity >= 0.3:
            logger.debug("  自适应: 高置信度+多样性足，跳过重排")
            return sources

        # 低置信度 → 先用关键词重排增加命中率
        if top_score < 0.4:
            logger.debug("  自适应: 低置信度，执行关键词重排+多样性扩展")
            sources = self._keyword_rerank(query, sources, keyword_weight=0.7)

        # 多样性不足 → 从尾部补充不同页面
        if page_diversity < 0.2 and len(sources) > 5:
            logger.debug("  自适应: 多样性不足，执行多样性扩展")
            sources = self._diversity_rerank(sources)

        # 中等置信度以上 → Reranker 精排
        if top_score >= 0.4 and self._get_reranker():
            logger.debug("  自适应: 执行 Reranker 精排")
            sources = self._rerank_with_model(query, sources)

        return sources

    @staticmethod
    def _diversity_rerank(sources: list[dict]) -> list[dict]:
        """
        多样性重排：确保结果覆盖不同页面
        每页最多保留 3 条，剩余位置从其他页面补充
        """
        if not sources:
            return sources

        page_buckets = {}
        for s in sources:
            page = s.get("page", 0)
            if page not in page_buckets:
                page_buckets[page] = []
            page_buckets[page].append(s)

        result = []
        # 轮流从每页取 1 条，直到每页最多取 3 条
        pages = list(page_buckets.keys())
        per_page_taken = {p: 0 for p in pages}
        max_per_page = 3

        # 先放 top-1 保留
        result.append(sources[0])
        per_page_taken[sources[0].get("page", 0)] = 1

        # 轮询补充
        while len(result) < len(sources):
            added = False
            for p in pages:
                if per_page_taken[p] >= max_per_page:
                    continue
                bucket = page_buckets[p]
                taken = per_page_taken[p]
                if taken < len(bucket) and bucket[taken] not in result:
                    result.append(bucket[taken])
                    per_page_taken[p] += 1
                    added = True
            if not added:
                break  # 所有页都取满了

        # 把剩余的按原始顺序追加
        seen_ids = {id(s) for s in result}
        for s in sources:
            if id(s) not in seen_ids:
                result.append(s)

        return result

    # ── Reranker 懒加载 ──────────────────────────

    def _get_reranker(self):
        """懒加载 BGE Reranker 模型"""
        if self._reranker is not None and self._reranker is not False:
            return self._reranker
        if self._reranker is False:  # 已经加载失败过
            return None
        if not self._reranker_path:
            return None
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            logger.info(f"加载 BGE Reranker: {self._reranker_path}")
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(self._reranker_path)
            self._reranker = AutoModelForSequenceClassification.from_pretrained(self._reranker_path)
            self._reranker.eval()
            self._reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._reranker_device == "cuda":
                self._reranker = self._reranker.to("cuda")
                logger.info(f"Reranker 使用 GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Reranker 使用 CPU")
            return self._reranker
        except Exception as e:
            log_exception(logger, "Reranker 加载失败", e)
            self._reranker = False
            return None
