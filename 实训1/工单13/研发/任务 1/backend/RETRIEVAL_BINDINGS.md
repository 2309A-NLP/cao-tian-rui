# 检索链路逻辑羁绊文档

## 一、完整链路图

```
用户查询
  │
  ▼
[1] 查询改写 (query_rewrite=True)
  │  └─ 公司名全称→简称变体（条件：查询不含完整全称>=6字）
  │  └─ 去除非关键限定词（"分别是多少"等）
  │  └─ 收入类精简（含"收入"+"多少"时）
  │  └─ 结果：1~3个查询变体
  │
  ▼
[2] 向量搜索 (VectorStore.search)
  │  参数: per_variant_k = max(top_k × per_variant_k_multiplier, 25)
  │  每个变体独立搜索 → 去重合并
  │
[3] BM25全文搜索 (VectorStore.search_bm25)
  │  参数: per_variant_k = max(top_k × per_variant_k_multiplier, 25)
  │  条件: bm25_ready=True（启动时从Milvus加载）
  │  每个变体独立搜索 → 去重合并
  │
  ▼
[4] RRF融合 (vector + bm25)
  │  参数: alpha=0.4, rrf_k=60
  │  结果: ~50-90条候选
  │
[5] 强制召回 (_force_recall)
  │  条件: 候选 < 3 条
  │  动作: 从chunk文件grep关键词，强行加入
  │
  ▼
[6] 公司路由过滤 (company_filter)
  │  条件: 查询含公司别名
  │  第一层: detect_company(query) → 直接指定目标
  │  第二层: infer_company_from_results() → 多数投票
  │  动作: 移除对手公司chunk | 降权(if filtered<8)
  │
  ▼
[7] 关键词重排 (_keyword_rerank)
  │  始终执行
  │  来源: key_phrases(3-8字子串,过滤公司名片段) + query_intent_words(排除公司名)
  │  权重: keyword_weight=0.5
  │  动作: score = base + keyword_boost × 0.5 → 重排序
  │
  ▼
[8] 截断 → rerank_top_n (默认50, 配置20)
  │
  ▼
[9] 自适应重排 (adaptive)
  │  条件: keyword_rerank后top1.score < 5.0
  │  动作: 调用BGE Reranker → 重新打分排序
  │  实际: reranker_model_path="" → 无操作(跳过)
  │
  ▼
[10] 最终截取 → top_k (当前15)
  │
  ▼
[11] 构建上下文 (_build_context)
  │  截断: uniq >= 18
  │  每chunk截断: 1200字符
  │
  ▼
    LLM生成答案
```

## 二、各环节触发条件与副作用

| 环节 | 触发条件 | 副作用 | 影响范围 |
|------|----------|--------|----------|
| 查询改写(公司名) | 查询匹配公司pattern AND 匹配文本长度<6字 | 增加1-2个搜索变体 | 搜索耗时×2~3 |
| 查询改写(去限定词) | 查询含"分别是多少"等 | 增加1个变体 | 搜索耗时×1.3 |
| 强制召回 | 初始候选 < 3 | 从磁盘读chunk文件grep | 仅低召回场景 |
| 公司过滤(显式) | 查询含公司别名 | 移除对手公司chunk | 候选数减少30~50% |
| 公司过滤(推断) | 候选中有公司名的chunk占比≥65% | 同上 | 同上 |
| 公司过滤(降权) | 过滤后 < 8条 | 对手chunk评分×0.3 | 保留所有chunk但降权 |
| 关键词重排 | 始终执行 | key_phrases匹配+intent_words匹配 | 改变排序，影响top-k |
| 自适应重排 | top1_score < 5.0 | 调BGE Reranker | 排序可能完全改变 |

## 三、参数联动关系

### 3.1 top_k ↔ rerank_top_n
```
rerank_top_n 必须 >= top_k
否则第9步截断已经切掉所有候选，第10步拿不到足够结果

当前: rerank_top_n=50(默认) > top_k=15 → 安全
如果未来增大 top_k，需要同步增大 rerank_top_n
推荐: rerank_top_n >= top_k × 3
```

### 3.2 top_k ↔ _build_context.uniq
```
上下文截断值必须 >= top_k
否则[10]输出N条，[11]只拿前M条(M < N)

当前: uniq=18 > top_k=15 → 安全
如果改top_k，uniq要跟着：
  uniq = top_k × 1.2 (向上取整)
```

### 3.3 per_variant_k_multiplier ↔ top_k
```
各变体搜索的候选数 = max(top_k × multiplier, 25)

multiplier=3, top_k=15 → 45候选/变体
multiplier=3, top_k=8  → 25候选/变体

增大top_k后，per_variant_k 自然增大，给RRF融合更多原始材料。
不需要手动调整multiplier。
```

### 3.4 bm25_alpha ↔ rrf_k (RRF参���)
```
alpha: BM25权重占比 (0~1)
rrf_k: 平滑参数 (推荐60)

当前: alpha=0.4, rrf_k=60
含义: BM25占40%权重，向量搜索占60%

两个值在RetrievalConfig中固定为alpha=0.4, rrf_k=60。
如果未来调整，alpha+rrf_k要一起考虑：
  - 增大alpha → BM25权重更高 → 关键词精确匹配优先
  - 减小alpha → 向量语义匹配优先
  - rrf_k越大 → RRF融合越平滑，排名靠后的也有机会进
```

### 3.5 keyword_weight ↔ rerank触发阈值
```
keyword_rerank后: score = base + boost × keyword_weight (默认0.5)
adaptive rerank触发: top1.score < 5.0

如果调整keyword_weight：
  - 增大 → boost效应更强 → top1.score可能超5.0 → 跳过adaptive rerank
  - 减小 → boost效应弱 → top1.score可能低于5.0 → 触发adaptive rerank

当前二者平衡：keyword_weight=0.5 时，top1.score大约3~6，
部分场景触发adaptive rerank（虽然实际因为无模型而跳过）
```

### 3.6 similarity_threshold ↔ 召回率
```
当前: 0.0 (不过滤)
含义: 所有搜索结果保留，不按相似度阈值裁减

如果改为正值 (如0.3)：
  - 低分chunk被移除 → BM25结果可能全丢（BM25分数范围不确定）
  - 公司过滤后候选不足 → 频繁触发降权逻辑
  - 强制召回概率增大

建议: 保持0.0，让top_k和keyword_rerank负责质量控制
```

## 四、数据依赖关系

### 4.1 chunk文本格式
```
【文件】{source_filename}
【章节】{section_title}
【子标题】{subsection_title}
【来源】pymupdf|qwen3_vl
{实际内容}

依赖方:
  - company_filter → 读【文件】前缀识别文档归属
  - keyword_rerank → 读全文做短语匹配
  - LLM prompt    → 读全文提取答案
```

### 4.2 Milvus schema
```
字段: id, chunk_id, page, field_type, text, vector
没有 source 字段 → 公司路由只能靠文本中的【文件】标记
如果未来改schema，要同步改：
  - vector_store.py: _create_collection, index_documents, _init_bm25_from_milvus
```

### 4.3 BM25生命周期
```
启动 → _init_bm25_from_milvus() → 从Milvus读全部chunk → 建BM25(内存)
  ↓
index_documents() → 追加chunks_metadata → 重建BM25(全量)
  ↓
search_bm25 → 读self.bm25(内存) → 返回结果

依赖:
  - chunks_metadata 必须是完整数据（含启动加载+新增）
  - 如果启动时Milvus无数据，BM25跳过 → 混合检索退化为纯向量
```

## 五、典型故障场景

| 症状 | 根因 | 排查方向 |
|------|------|----------|
| 检索结果都是封面/目录页 | key_phrases 没过滤公司名片段 | 检查 _COMPANY_NAMES 是否完整 |
| 对手公司chunk混入结果 | 【文件】标记缺失(旧索引) | reindex 后重试 |
| BM25搜索返回空 | bm25_ready=False | 检查启动日志是否有BM25错误 |
| 查询改写生成了多余变体 | 条件`len>=6`判断不准 | 检查 query_rewrite 日志输出 |
| 答案被截断(信息缺失) | uniq 或 1200字符上限 | 增大对应阈值 |
| 检索耗时>3s | 查询变体过多 | 检查公司名改写是否触发 |
