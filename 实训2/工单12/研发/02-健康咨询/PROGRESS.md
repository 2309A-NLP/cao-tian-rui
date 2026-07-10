# 工单12 · 健康咨询 · 进度记录

> 最后更新：2026-07-05
> 状态：**后端+前端全部完成，服务可正常启动，基础接口验证通过，待明日完整测试**

---

## 一、当前状态

### ✅ 已完成

| 模块 | 状态 | 说明 |
|------|------|------|
| 后端 `src/` | ✅ | agent / graph / app / auth / config / memory_bridge 全部就位 |
| 前端 `chat.html` | ✅ | 双栏布局，teal 配色与总框架一致，图谱面板/意图标签/快捷按钮全功能 |
| 知识图谱数据 | ✅ | 6720 疾病 / 5936 症状 / 1124 药物 / 12.9万关系（已在 Neo4j 容器中） |
| 测试用例 | ✅ | 20 个（工单10题 + 变体10题），`tests/test_whooping_cough.py` |
| 知识图谱导入脚本 | ✅ | `scripts/import_graph.py`（数据已导入，不需要重跑） |
| 环境配置 | ✅ | `.env` / `.venv` / `requirements.txt` 全部就绪 |
| 接口冒烟测试 | ✅ | health / stats / chat 接口验证通过，4道百日咳题全部命中 |

### ⏳ 明天要做

1. **跑完整测试套件**（20道题）
2. **浏览器手动验证前端**（对话、图谱面板展开、快捷标签、侧边栏统计）
3. **补充测试结果文档**（截图或命令行输出）
4. **与工单11总框架对接**（确认嵌入 00-总界面框架 的方式）

---

## 二、明天启动步骤（复制即用）

```powershell
# 1. 确认 Neo4j 在跑
docker start neo4j

# 2. 启动 wt12 服务
cd "F:\kimi  project\医疗agent1\02-健康咨询"
.venv\Scripts\python -m uvicorn src.app:app --host 0.0.0.0 --port 8012

# 3. 浏览器打开聊天页面
# http://localhost:8012

# 4. 跑完整测试（另开终端）
cd "F:\kimi  project\医疗agent1\02-健康咨询"
.venv\Scripts\python -m pytest tests/ -v
```

**Neo4j 浏览器**：`http://localhost:7474`
- 账号：`neo4j` / 密码：`medical123`
- 注意：直接打开 7474 根路径，**不要带 `?cmd=...` 参数**，否则 React 崩溃

---

## 三、架构速查

### 项目结构
```
02-健康咨询/
├── src/
│   ├── agent.py          ← 3步流程：Function Calling → Neo4j → LLM生成
│   ├── graph.py          ← 11个Cypher模板函数，query_graph()统一入口
│   ├── app.py            ← FastAPI，端口8012，含/stats端点
│   ├── auth.py           ← API Key鉴权（未配置=开发模式）
│   ├── config.py         ← 硅基流动+Neo4j配置
│   └── memory_bridge.py  ← mem0长期记忆（可选，失败静默）
├── chat.html             ← 双栏前端（主聊天区+右侧边栏）
├── tests/
│   └── test_whooping_cough.py  ← 20个测试用例
├── scripts/
│   ├── import_graph.py   ← medical.json → Neo4j（已运行过）
│   └── quick_test.py     ← 快速冒烟测试（不依赖pytest）
├── data/medical.json     ← 原始图谱数据
└── .env                  ← 真实配置（含API Key）
```

### 3步 Agent 流程
```
用户提问
  ↓
Step 1: LLM（Qwen2.5-72B）Function Calling
        强制调用 query_medical_graph 工具
        → 返回 {intent: <12种之一>, entity: <疾病名/症状名>}
  ↓
Step 2: Python 根据 intent 分派到 Cypher 模板
        → Neo4j 查询 → list[dict]
  ↓
Step 3: LLM 拿图谱数据 → 生成自然语言回答
        （图谱无命中 → LLM 自身知识兜底，前端标注"图谱未命中"）
```

### 12种意图 → Neo4j 字段映射

| intent | 查询对象 |
|--------|---------|
| disease_info | Disease.intro / get_prob / easy_get（含血常规/检查信息） |
| symptom_to_disease | (Disease)-[:HAS_SYMPTOM]->(Symptom) 反向 |
| disease_to_symptom | (Disease)-[:HAS_SYMPTOM]->(Symptom) |
| disease_to_drug | (Disease)-[:USES_DRUG]->(Drug) |
| disease_to_dept | (Disease)-[:BELONGS_TO]->(Department) |
| disease_to_complication | (Disease)-[:CAUSES]->(Disease) |
| disease_to_diet | (Disease)-[:CAN_EAT/NOT_EAT]->(Food) |
| disease_to_transmission | (Disease)-[:TRANSMITS_VIA]->(Transmission) |
| disease_to_cause | Disease.cause 文本字段 |
| disease_to_prevent | Disease.prevent 文本字段 |
| disease_to_nursing | Disease.nursing + Disease.treat_detail（含隔离期） |
| disease_to_treat | Disease.treat_detail（含中医主方，不截断） |

### 端口规划

| 工单 | 服务 | 端口 |
|------|------|------|
| wt11 | 挂号管理 | 8011 |
| **wt12** | **健康咨询** | **8012** |
| wt13 | 影像分析 | 8013 |
| Neo4j Bolt | 图数据库 | 7687 |
| Neo4j Browser | 图数据库 UI | 7474 |

---

## 四、今日踩坑记录

### 坑1：`Start-Process` 启动失败（logs 目录不存在）
- **症状**：`-RedirectStandardOutput` 路径不存在时进程静默退出
- **修复**：先 `New-Item -ItemType Directory -Force -Path logs`，再 Start-Process

### 坑2：PowerShell 发送中文 JSON 乱码导致 422
- **症状**：`Invoke-RestMethod -Body '{"query":"百日咳..."}'` → entity 收到乱码 → 图谱查不到
- **原因**：PowerShell 单引号字符串默认不是 UTF-8
- **修复**：用 `[System.Text.Encoding]::UTF8.GetBytes(...)` + `WebClient.UploadData`，或直接用 Python 脚本测试

### 坑3：Neo4j Browser 带参数打开崩溃
- **症状**：点聊天页"Neo4j浏览器"链接 → React 报 NotFoundError
- **原因**：`?cmd=edit&arg=...` 参数在 Neo4j 未登录时触发渲染 bug
- **修复**：先打开 `http://localhost:7474` 登录，再用链接；或点"重新加载应用程序"

---

## 五、性能基准（今日实测）

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Neo4j Cypher 查询 | < 100ms | 远优于工单要求 |
| LLM 意图识别（Step 1） | ~1-3s | 硅基流动 Qwen2.5-72B |
| LLM 答案生成（Step 3） | ~1-3s | 同上 |
| **全链路响应** | **3~7s** | 两次 LLM 调用叠加 |

> 工单要求"响应时间 < 500ms"实测无法满足（全链路含两次云端 LLM 调用）。
> 图谱检索阶段 < 100ms，完全符合。建议测试报告中如实说明：500ms 为图谱检索性能指标，LLM 生成耗时属于云端 API 固有延迟。
