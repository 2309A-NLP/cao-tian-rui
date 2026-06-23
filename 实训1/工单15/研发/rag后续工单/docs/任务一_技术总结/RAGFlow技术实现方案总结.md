# RAGFlow 技术实现方案总结（任务一）

> 工单：修复低质量工业 PDF 的解析与信息丢失
> 分析对象：RAGFlow 源码（`E:/rag后续工单/ragflow`，git clone 自 infiniflow/ragflow）
> 日期：2026/06/14

本文回答验收要求的三个核心问题：
1. PDF 文档在 `parser_id` 为 `paper`/`table`/`one`/`knowledge_graph` 时，分块策略与解析任务**如何触发并放入 Redis Stream**，等待任务执行器消费。
2. `do_handle_task` 的主要逻辑与所用技术。
3. DeepDoc 深度解析模块内置了哪些解析器、能解析什么类型、**PDF 解析技术**重点。

---

## 0. 总体架构

RAGFlow 有两个核心进程，通过 **Redis Stream 消息队列**解耦：

```
                    ┌─────────────────────┐
   用户 / 前端  ───▶ │   API Server        │  ragflow_server.py
                    │  - 知识库 / 文件管理  │  提供 HTTP 接口
                    │  - queue_tasks()    │  上传后把"解析任务"入队
                    └──────────┬──────────┘
                               │ REDIS_CONN.queue_product(队列名, message=task)
                               ▼
                    ┌─────────────────────┐
                    │   Redis Stream      │  消息队列（仅放任务id等轻量信息）
                    │   rag_flow_svr_queue│  消费者组 SVR_CONSUMER_GROUP_NAME
                    └──────────┬──────────┘
                               │ collect(): queue_consumer / get_unacked_iterator
                               ▼
                    ┌─────────────────────┐
                    │   Task Executor     │  rag/svr/task_executor.py
                    │   do_handle_task()  │  解析→分块→向量化→索引
                    │     └─ DeepDoc      │  OCR / 版面分析 / 表格结构识别
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Elasticsearch /     │  docStoreConn（二选一）：存 chunk + 向量
                    │ Infinity  +  MinIO  │  MinIO 存原文件与图片切片
                    └─────────────────────┘
```

- **API Server**：`api/ragflow_server.py`，负责外部接口与平台功能（知识库、文件上传），并在文件触发"解析"时调用 `queue_tasks()` 入队。
- **Task Executor**：`rag/svr/task_executor.py`，独立进程，从 Redis Stream 消费任务并执行完整解析流水线。

---

## 1. 第一问：分块策略如何按 parser_id 切分并放入 Redis Stream

入口函数：**`queue_tasks(doc, bucket, name, priority)`**，位于
[`api/db/services/task_service.py:358`](../../ragflow/api/db/services/task_service.py)。

### 1.1 按 parser_id / 文件类型决定"任务切分粒度"

`queue_tasks` 把**一个文档**拆成**一个或多个 task**（每个 task 负责一段页码/行），核心差异如下：

| 条件 | 每个 task 粒度 | 说明 | 代码行 |
|---|---|---|---|
| PDF（默认 naive 等） | 每 **12 页** | `task_page_size` 默认 12 | L397 |
| PDF 且 `parser_id == "paper"` | 每 **22 页** | 论文结构需更大上下文 | L398-399 |
| `parser_id ∈ {"one","knowledge_graph"}`，或 `layout_recognize != "DeepDOC"`，或开启 `toc_extraction` | **整篇一个 task**（`MAXIMUM_TASK_PAGE_NUMBER`） | `one`=整文档做一块；`knowledge_graph`=要全局抽实体建图，都必须看到全文 | L400-401 |
| `parser_id == "table"`（Excel） | 每 **3000 行** | 按行切分 | L413-420 |
| 其它（docx/ppt/图片/email…） | **整篇一个 task** | `else` 分支 | L421-422 |

> **结论**：`paper`/`table`/`one`/`knowledge_graph` 的本质区别是**任务切分粒度**——
> 由文档需要的"上下文跨度"决定。这一步只决定怎么切 task，真正的分块算法在 Task Executor 里由对应 chunker 完成（见 §3 的 FACTORY）。

### 1.2 任务复用优化（digest 去重）

对每个 task 用 **xxhash** 基于「分块配置 + doc_id + 页码范围」算一个 `digest`（L429-439）。
若上次解析过、配置未变（`digest` 相同且 `progress==1.0`），则 `reuse_prev_task_chunks` 直接复用旧 chunk，跳过重复解析（L443-456）。

### 1.3 放入 Redis Stream

```python
# task_service.py L461-465
unfinished_task_array = [t for t in parse_task_array if t["progress"] < 1.0]
for unfinished_task in unfinished_task_array:
    assert REDIS_CONN.queue_product(
        settings.get_svr_queue_name(priority, suffix),   # 队列名 = f(优先级, suffix)
        message=unfinished_task
    ), "Can't access Redis..."
```

- 队列名由 **优先级 `priority`** 与 **`suffix`**（`resume` 解析走专用队列，其余 `common`，L425）决定。
- `queue_product` 即向 **Redis Stream `XADD`** 一条消息；消息体只含 task 的 id/页码/digest 等**轻量字段**，重数据（解析配置等）由消费端回查数据库。
- 入队前先 `bulk_insert_into_db(Task, ...)` 把 task 落库、`begin2parse` 标记文档开始解析（L458-459）。

---

## 2. 第二问：do_handle_task 主要逻辑与技术实现

入口：**`do_handle_task(task)`**，位于
[`rag/svr/task_executor.py:1359`](../../ragflow/rag/svr/task_executor.py)。
外层有 `@timeout(3小时)` 装饰，单任务超时保护。

### 2.1 消费侧：collect() 如何从 Redis Stream 取任务

[`collect()` L199](../../ragflow/rag/svr/task_executor.py) 用 **Redis Stream 消费者组**拉消息：
1. 先 `get_unacked_iterator` 处理**未 ACK** 的遗留消息（崩溃恢复）；
2. 再 `queue_consumer(queue, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME)` 拉新消息（即 `XREADGROUP`）；
3. 拿到 `msg["id"]` 后 **回查数据库** `TaskService.get_task(id)` 还原完整任务对象。

> 设计要点：**队列轻、DB 重**——消息只携带 id，避免大消息撑爆队列；消费者组保证多 Task Executor 实例可水平扩展且不重复消费。

### 2.2 按 task_type 分发

`do_handle_task` 先看 `task_type`，分几条路：
- `memory` → 存记忆任务；
- `dataflow` → 走 Pipeline 画布解析 `run_dataflow`；
- `raptor` → RAPTOR 递归聚类摘要建树（L1419）；
- `graphrag` → 知识图谱抽取（L1467）；
- `mindmap` → 占位；
- **else → 标准分块**（最常用，L1535）。

所有分支开始前都会**绑定 embedding 模型并探测向量维度**（L1400-1406，用 `encode(["ok"])` 得到 `vector_size`），再 `init_kb(task, vector_size)` 确保索引存在。

### 2.3 标准分块四步流水线（核心）

| 步骤 | 函数 | 关键技术 | 代码 |
|---|---|---|---|
| ① 解析 + 分块 | `build_chunks` | 从 **MinIO** 取文件二进制 → `FACTORY[parser_id]` 选 chunker → 线程池执行 `chunker.chunk(...)` → 图片切片回传 MinIO；**chunk id = `xxhash64(content_with_weight + doc_id)`**（内容寻址，天然去重） | L272 / L1539 |
| ② 向量化 | `embedding` | **标题向量 + 内容向量加权融合**：`vects = title_w*标题 + (1-title_w)*正文`，`title_w` 默认 0.1（`filename_embd_weight`）；分批 `mdl.encode`（`EMBEDDING_BATCH_SIZE`）；结果写入字段 `q_<dim>_vec` | L680 / L1552 |
| ③ 入库索引 | `insert_chunks` | 写入 **`docStoreConn`（Elasticsearch 或 Infinity，二选一）**；处理 parent-child「母块 mom」；批量 `insert`（`DOC_BULK_SIZE`）；回写 `task.chunk_ids` | L1243 / L1576 |
| ④ 收尾 | — | `increment_chunk_num` 更新文档块数与 token 数 → `progress_callback(prog=1.0)` 标记完成；表格解析还会聚合列元数据 | L1605 / L1662 |

`build_chunks` 细节（L272-381）：
- 超过 `DOC_MAXIMUM_SIZE` 直接失败；
- `chunker = FACTORY[task["parser_id"].lower()]`；
- 表格解析会把 KB 级 `parser_config`（列角色/模式）合并进文档级配置；
- 若 PDF 解析器返回 `__outline__`（书签目录），持久化为文档元数据。

`embedding` 细节（L680-729）：
- 文本会去除 `<table><td>...` 等标签再编码（L689）；
- 全空 chunk 用占位符 `"None"` 防止编码报错。

### 2.4 进度与取消

- `set_progress`（L165）+ `progress_callback` 实时回写进度（0→1.0），前端可见。
- 多处 `has_canceled(task_id)` 检查 Redis 里的取消标志；取消时回滚已插入的 chunk（`finally` 块 L1665-1684）。

---

## 3. 第三问：DeepDoc 深度解析模块与 PDF 解析技术

### 3.1 DeepDoc 内置的解析器

DeepDoc 分两层：`deepdoc/parser/`（文件解析器）与 `deepdoc/vision/`（视觉模型）。

**文件解析器**（[`deepdoc/parser/__init__.py`](../../ragflow/deepdoc/parser/__init__.py) 导出）：

| 解析器 | 文件类型 |
|---|---|
| `RAGFlowPdfParser` (PdfParser) | **PDF（深度解析，重点）** |
| `PlainParser` | PDF（纯文本快速模式） |
| `VisionParser` | PDF（多模态大模型模式） |
| `RAGFlowDocxParser` | Word .docx |
| `RAGFlowExcelParser` | Excel .xls/.xlsx |
| `RAGFlowPptParser` | PowerPoint .ppt/.pptx |
| `RAGFlowHtmlParser` | HTML |
| `RAGFlowMarkdownParser` | Markdown |
| `RAGFlowJsonParser` | JSON |
| `RAGFlowEpubParser` | EPUB |
| `RAGFlowTxtParser` | 纯文本 |
| `resume/` | 简历（结构化抽取） |
| 另有 | `figure_parser`(视觉图片描述)、`docling/mineru/paddleocr/opendataloader/tcadp` 等可选后端 |

**视觉模型**（`deepdoc/vision/`）：
- `ocr.py` — OCR（文字检测 + 识别）
- `layout_recognizer.py` — 版面分析（标题/正文/图/表/页眉页脚分类）
- `table_structure_recognizer.py` — 表格结构识别（TSR）

### 3.2 PDF 三种解析模式（关键！与任务二强相关）

分块器 `rag/app/naive.py` 按 `parser_config["layout_recognize"]` 选择 PDF 解析器
（[L305-318](../../ragflow/rag/app/naive.py) 与 L892）：

| `layout_recognize` 取值 | 解析器 | 能力 | 代价 |
|---|---|---|---|
| `"Plain Text"` / 空 | `PlainParser` | pypdf 纯文本抽取 | 快；**丢失图片、版面、表格结构** |
| `"DeepDOC"`（默认） | `RAGFlowPdfParser` | OCR + 版面分析 + TSR | 慢；保留图文结构 |
| 视觉模型名（IMAGE2TEXT） | `VisionParser` | 整页交给多模态 LLM 理解 | 最慢；对图片型/低质量 PDF 最强 |

此外 `vision_figure_parser_pdf_wrapper`（naive.py L121，调用 `deepdoc/parser/figure_parser.py`）
会对抽取出的**图片**再用视觉 LLM **生成描述文本**，让图片内容可被检索。

> **对任务二的直接含义**：6 个测试问题里第 3~6 题是"第 7 页图中部件位置/尺寸/气流方向"。
> 若用 `PlainParser`（纯文本）则图片信息全丢，必然答错；必须用 **DeepDOC + 视觉图片描述** 或 **VisionParser**，
> 让图中的部件标号、尺寸标注、气流方向被解析成可检索文本。这是后续调优的主攻方向。

### 3.3 RAGFlowPdfParser 深度解析流水线

`RAGFlowPdfParser.__init__`（[pdf_parser.py:57](../../ragflow/deepdoc/parser/pdf_parser.py)）即加载三个深度学习模型：

```python
self.ocr      = OCR()                        # 文字检测+识别
self.layouter = LayoutRecognizer(domain)     # 版面分析
self.tbl_det  = TableStructureRecognizer()   # 表格结构识别
```

主流程 **`__call__`**（[pdf_parser.py:1673](../../ragflow/deepdoc/parser/pdf_parser.py)）是一条 8 步流水线：

```python
self.outlines = extract_pdf_outlines(fnm)     # 1. 提取书签/目录
self.__images__(fnm, zoomin)                  # 2. 按 zoomin=3 渲染高分辨率页面图
self._layouts_rec(zoomin)                     # 3. 版面分析：每个框分类为标题/正文/图/表...
self._table_transformer_job(zoomin, ...)      # 4. 表格结构识别(TSR)：还原行列单元格
self._text_merge()                            # 5. 合并相邻文本框
self._concat_downward()                       # 6. 跨行/跨页向下拼接成段
self._filter_forpages()                       # 7. 过滤页眉页脚等噪声
tbls = self._extract_table_figure(...)        # 8. 抽取表格与图片(可转HTML)
return self.__filterout_scraps(...), tbls
```

技术要点：
- **`zoomin=3`**：把 PDF 页面放大 3 倍渲染成图，提高低分辨率扫描件的 OCR 准确率。
- **OCR**（`deepdoc/vision/ocr.py`）：`detect` 检测文本框 → `recognize_batch` 批量识别，含旋转裁剪校正 `get_rotate_crop_image`。
- **版面分析**（`layout_recognizer.py`）：把页面元素分类（标题/正文/图/表/页眉页脚），决定阅读顺序与噪声过滤。
- **表格结构识别 TSR**（`table_structure_recognizer.py`）：`construct_table` 把表格还原成行列单元格，可输出 HTML，避免表格被打散成无结构文本。
- **文本拼接** `_text_merge` / `_concat_downward`：解决 PDF 文字按坐标散落的问题，按版面把碎片拼成语义完整段落——这正是"修复信息丢失"的关键。
- 还有 `PlainParser`（纯文本快速路径，L2005）与 `VisionParser`（多模态 LLM，L2026）两个轻/重备选。

---

## 4. 小结：一次完整解析的数据流

```
用户点"解析"
  → API Server: queue_tasks() 按 parser_id 切 task → Redis Stream(XADD)
  → Task Executor: collect() 消费者组 XREADGROUP → 回查DB还原task
  → do_handle_task():
       build_chunks(): MinIO取文件 → DeepDoc(OCR/版面/TSR)解析 → 分块
       embedding(): 标题+内容加权向量化 → q_<dim>_vec
       insert_chunks(): 写入 Elasticsearch/Infinity + MinIO存图
  → 进度回写 1.0，知识库可检索
```

**给任务二的关键结论**：图片型工业 PDF 的图文信息能否保留，取决于 §3.2 的 `layout_recognize` 选择。
后续 6 问调优将围绕「解析模式（DeepDOC/Vision）+ 图片描述 + 分块粒度 + 向量/ReRank 权重」展开。
