# mem0 多领域长期记忆系统 — 集成设计文档

**项目**：工单22 · Agent22  
**版本**：v1.0  
**日期**：2026-07-06  
**技术栈**：mem0 2.0 · ChromaDB 1.5 · SiliconFlow API · FastAPI · uvicorn

---

## 1. 技术选型与依据

| 组件 | 选型 | 选择依据 |
|------|------|---------|
| 记忆框架 | mem0 2.0.11 | 开箱即用的 LLM-驱动记忆提取与向量检索；支持多 user_id 隔离 |
| 向量数据库 | ChromaDB 1.5.9（嵌入式） | 本地持久化，无需独立服务；collection `agent22_memory` |
| Embedding | BAAI/bge-large-zh-v1.5（SiliconFlow） | 1024 维中文优化模型，字符上限约 420 字 |
| LLM（记忆提取） | Qwen/Qwen2.5-72B-Instruct（SiliconFlow） | mem0 内部提取摘要；约 20 秒/次，fire-and-forget 异步写入 |
| API 框架 | FastAPI + uvicorn | 原生 async，方便 `asyncio.to_thread` 隔离同步 IO |

---

## 2. 整体架构

```
Browser (index.html)
    │  POST /api/chat/{domain}
    │  GET/DELETE /api/memory/{domain}/{user_id}
    ▼
FastAPI (main.py, port 8022)
    ├── routes_chat.py   ── asyncio.to_thread ──► MedicalAgent
    │                                          ► TravelAgent
    │                                          ► EducationAgent
    └── routes_memory.py ──────────────────────► MemoryClient
                                                      │
                             ┌────────────────────────┴─────────────────────────┐
                             │              MemoryClient (Singleton)             │
                             │  recall()  → mem0.search(filters={user_id})      │
                             │  remember() → Thread(daemon=False) → mem0.add()  │
                             │  list_all() → mem0.get_all(filters={user_id})    │
                             │  clear()   → mem0.delete() × N                  │
                             └────────────────────────┬─────────────────────────┘
                                                      │
                             ┌────────────────────────▼──────┐
                             │        mem0.Memory             │
                             │  LLM:   SiliconFlow/Qwen       │
                             │  Emb:   SiliconFlow/bge        │
                             │  Store: ChromaDB (本地)        │
                             └───────────────────────────────┘

MedicalAgent:
  recall → 注入记忆 → POST http://localhost:8012/chat (工单12 Neo4j)
                       └── 失败/空reply → fallback 本地 Qwen LLM
```

---

## 3. 数据流详解

### 3.1 对话请求流

```
1. POST /api/chat/{domain}
   ChatRequest { user_id, query, session_id? }
       │
       ▼
2. asyncio.to_thread(agent.chat, ...)   [隔离同步 IO，不阻塞事件循环]
       │
       ▼
3. MemoryClient.recall(user_id, query, limit=5)
   → mem0.search(query=query, filters={"user_id": user_id}, limit=5)
   → ChromaDB ANN 检索
   → 返回 [{memory, score}, ...]（失败则降级为 []）

4. 构造 full_query = 历史记忆块 + 本轮问题

5a. [medical] POST http://localhost:8012/chat
    → 若 reply 为空/None → fallback 本地 Qwen
5b. [travel/education] 本地 Qwen LLM.chat(system, user=full_query)

6. MemoryClient.remember(user_id, messages, metadata, blocking=False)
   → _truncate（400字/条）→ Thread(daemon=False).start()
   → Thread: mem0.add(messages, user_id, metadata)
             Qwen 提取摘要（~20s）→ ChromaDB 写入

7. 返回 ChatResponse { reply, recalled, domain, elapsed_ms, source }
```

### 3.2 记忆写入异步机制

- `daemon=False`：进程退出前 Python runtime 等待线程自然结束
- `_pending_threads` 列表追踪所有活跃写入线程
- lifespan shutdown hook 调用 `wait_pending_writes(timeout=60s)` 显式 join
- 双重保障：程序正常退出或 SIGTERM 均不会丢失已提交的写入请求

---

## 4. 用户隔离设计

### 4.1 user_id 命名规则

| domain | 原始 ID | 存储 user_id |
|--------|---------|-------------|
| medical | `user001` | `patient_user001` |
| travel | `user001` | `traveler_user001` |
| education | `user001` | `student_user001` |

同一自然人在不同领域的记忆互相隔离，通过前缀区分。

### 4.2 mem0 过滤器

所有检索和列举均使用 `filters={"user_id": <prefixed_id>}` 过滤，由 ChromaDB 元数据字段保证隔离。

---

## 5. ChromaDB Schema

**Collection name**：`agent22_memory`（mem0 自动管理）  
**向量维度**：1024（BAAI/bge-large-zh-v1.5）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID | mem0 自动生成 |
| `memory` | str | Qwen 提取的摘要文本 |
| `user_id` | str | 如 `patient_user001`（元数据过滤键） |
| `metadata.domain` | str | medical / travel / education |
| `metadata.session_id` | str | 如 `session1_medical` |
| `metadata.created_at` | ISO8601 | mem0 自动添加 |

---

## 6. API 契约

### POST `/api/chat/{domain}`

**请求**
```json
{
  "user_id": "user001",
  "query": "我最近头痛",
  "session_id": "session1_medical"
}
```

**响应**
```json
{
  "reply": "根据您的历史记录...",
  "recalled": [{"memory": "患者对青霉素过敏", "score": 0.8712}],
  "domain": "medical",
  "elapsed_ms": 145,
  "source": "wt12_neo4j"
}
```

`source` 枚举：`wt12_neo4j` | `fallback_llm` | `mock_llm`

### GET `/api/memory/{domain}/{user_id}`

返回该用户在该领域的全部记忆列表（用于前端记忆面板）。

### DELETE `/api/memory/{domain}/{user_id}`

清空该用户在该领域的全部记忆，返回实际删除条数。

---

## 7. 关键配置（`.env`）

```
SILICONFLOW_API_KEY=sk-...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_LLM_MODEL=Qwen/Qwen2.5-72B-Instruct
SILICONFLOW_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
SILICONFLOW_EMBEDDING_DIMS=1024
CHROMA_PATH=./data/chroma
WT12_URL=http://localhost:8012/chat
WT12_TIMEOUT=30
```

**注意**：`.env` 不得纳入版本控制；`.gitignore` 中应添加 `.env`。

---

## 8. 已知限制与注意事项

| 项目 | 说明 |
|------|------|
| bge 字符上限 | 中文约 420 字，超出自动截断至 400 字并附提示 |
| Qwen 提取延迟 | 每次写入约 20 秒（fire-and-forget，不影响响应时间） |
| ChromaDB 并发 | 嵌入式模式不支持多进程写入，生产建议单 worker |
| WT12 超时 | 默认 30 秒，可通过 `.env` 的 `WT12_TIMEOUT` 调整 |
| 记忆截断副作用 | 长医疗描述可能因截断丢失细节，建议用户分多轮输入 |
