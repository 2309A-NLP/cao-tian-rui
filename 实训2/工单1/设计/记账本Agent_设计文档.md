# 记账本 Agent 智能体 — 设计文档

> 工单编号：人工智能 NLP-Agent 数字人项目-记账本任务
> 版本：v1.0 | 2026-06-22
> 总代码量：1721 行（7 个模块）

---

## 一、项目概述

开发一款家庭记账智能体，用户通过自然语言对话完成账目的**增、删、查、统**。核心技术方案：**OpenAI Function Calling 模式**，让 LLM 自主决定何时调用哪个数据库操作工具。

---

## 二、系统架构

```
┌────────────────────────────────────────────────────┐
│                 浏览器 (index.html)                  │
│             SSE 流式聊天界面 + 快捷测试按钮           │
└──────────────────────┬─────────────────────────────┘
                       │ POST /api/chat  /  /api/chat/stream
                       ▼
┌────────────────────────────────────────────────────┐
│               FastAPI (:8020)                       │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  AgentEngine (agent_engine.py, 276行)        │   │
│  │                                              │   │
│  │  messages = [system_prompt, ...history]       │   │
│  │       │                                      │   │
│  │       ▼                                      │   │
│  │  LLM.chat(messages, tools=TOOLS_SCHEMA)      │   │
│  │       │                                      │   │
│  │       ├── finish="stop" → 返回文本给用户       │   │
│  │       │                                      │   │
│  │       └── finish="tool_calls"                 │   │
│  │              │                               │   │
│  │              ▼                               │   │
│  │       ToolExecutor.execute(name, args)        │   │
│  │              │                               │   │
│  │              ▼                               │   │
│  │       结果回填 messages → 继续循环 (最多5轮)    │   │
│  └─────────────────────────────────────────────┘   │
│                         │                           │
│                    ┌────┴────┐                      │
│                    │  MySQL  │                      │
│                    │money_notes│                    │
│                    └─────────┘                      │
└────────────────────────────────────────────────────┘
```

### 模块清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `backend/agent_engine.py` | 276 | Function Calling 循环引擎 + 会话管理 + 防重复 |
| `backend/tools.py` | 367 | 4 个工具的 OpenAI Schema + ToolExecutor |
| `backend/database.py` | 281 | MySQL 连接 + money_notes 表 CRUD + 汇总 |
| `backend/llm_provider.py` | 155 | OpenAI 兼容接口 + 流式/非流式 + tool_calls 分片拼接 |
| `api/api.py` | 218 | FastAPI 路由：/api/chat, /api/chat/stream, /api/welcome, /api/health |
| `api/static/index.html` | 234 | 聊天界面 + SSE 流式 + 快捷测试 |
| `config.json` | - | LLM (DeepSeek) + MySQL 配置 |

---

## 三、Function Calling 循环设计（核心）

```
用户输入 → add_user → messages = [system, user]
                        │
                        ▼
              LLM.chat(messages, tools=TOOLS_SCHEMA)
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   finish="stop"             finish="tool_calls"
   返回文本给用户               │
   (首条自动拼接开场白)          ▼
                        ToolExecutor.execute()
                          │
                          ├── add_record    → INSERT INTO money_notes
                          ├── query_records → SELECT + 汇总
                          ├── delete_record → (false)搜索/(true)DELETE
                          └── get_summary   → GROUP BY 汇总
                          │
                          ▼
                   结果回填 messages
                   session.add("tool", result)
                          │
                          ▼
                   回到循环顶部，继续 LLM.chat()
                   (LLM 看到 tool 结果后决定是否再调工具)
```

**关键设计决策**：
- 最大 **5 轮**循环，防止死循环
- 循环内用**非流式**获取 tool_calls 决策（避免参数分片拼接），最终回复可选流式
- 同参数 `add_record` 防重复：`called_tools_this_turn` 集合去重

---

## 四、数据库设计

### 表：money_notes

```sql
CREATE TABLE money_notes (
    id          INT AUTO_INCREMENT PRIMARY KEY  COMMENT '记录ID',
    member      VARCHAR(16)  NOT NULL           COMMENT '成员：爸爸/妈妈/女儿',
    amount      DECIMAL(10,2) NOT NULL          COMMENT '金额（正数）',
    type        VARCHAR(8)   NOT NULL           COMMENT '收入/支出',
    category    VARCHAR(32)  NOT NULL DEFAULT '' COMMENT '分类',
    item        VARCHAR(256) NOT NULL DEFAULT '' COMMENT '物品/事项',
    record_date DATE         NOT NULL           COMMENT '记账日期',
    note        VARCHAR(512) DEFAULT ''         COMMENT '备注',
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_member      (member),
    INDEX idx_date        (record_date),
    INDEX idx_type        (type),
    INDEX idx_category    (category),
    INDEX idx_member_date (member, record_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 设计要点

| 决策 | 原因 |
|------|------|
| 金额统一存正数 | 收入/支出由 `type` 字段区分，避免负数计算错误 |
| `record_date` ≠ `created_at` | 记账日期是业务日期（可能是过去），创建时间是系统时间 |
| 5 个索引 | 覆盖所有查询维度（成员+日期是最常用组合） |
| utf8mb4 | 支持 emoji，LLM 回复经常带 😊📊 等符号 |

### 分类体系

| 用户表达 | category | 示例 |
|---------|----------|------|
| 买书/教材/培训 | 学习 | "买三体花了50元" |
| 买衣/鞋/包/化妆品 | 购物 | "登山鞋 499 元" |
| 吃饭/外卖/买菜 | 餐饮 | "请客吃饭花了两百四" |
| 旅游/机票/酒店 | 旅游 | "报旅游团" |
| 工资/奖金 | 工资收入 | "收到工资 8000" |
| 报销/退款 | 报销收入 | "收到报销 1000" |
| 看病/买药 | 医疗 | - |
| 公交/地铁/加油 | 交通 | - |
| 房租/水电/物业 | 住房 | - |

---

## 五、工具定义设计

### 4 个工具

| 工具 | 用途 | 必填参数 | 特殊逻辑 |
|------|------|---------|---------|
| `add_record` | 添加记账 | member, amount, type, category, item, record_date | 字段不完整 → 不调用，追问 |
| `query_records` | 查询记录 | 全部可选 | 支持模糊搜索 keyword |
| `delete_record` | 删除记录 | confirmed (bool) | **两步确认**：false→搜索展示，true→执行 |
| `get_summary` | 汇总统计 | start_date, end_date | 支持 group_by: member/category/type |

### delete_record 的两步确认机制

```
用户: "删除女儿报旅游团的费用"
    │
    ▼
LLM → delete_record(confirmed=false, keyword="旅游团", member="女儿")
    │
    ▼
ToolExecutor → SELECT * FROM money_notes WHERE member='女儿' AND item LIKE '%旅游团%'
    │
    ├── 有结果 → 展示记录列表 → "确认删除吗？"
    │       │
    │       ▼
    │   用户: "确认" → delete_record(confirmed=true, record_id=X)
    │       │
    │       ▼
    │   DELETE → "已删除"
    │
    └── 无结果 → "没有找到相关记录"
```

### add_record 的防重复机制

```python
# 代码层：同参数只执行一次 INSERT
dedup_key = f"add_record|{member}|{item}|{amount}|{date}"
if dedup_key in called_tools_this_turn:
    return "该记录已存在，无需重复添加"
called_tools_this_turn.add(dedup_key)
```

---

## 六、System Prompt 设计

Prompt 是 Agent 行为的核心驱动力。设计分为 5 个层次：

### 层次结构

```
第一层：角色定义     "你是小家专属记账本助手"
第二层：致命规则     3条（强制工具 / 不完整追问 / 删除确认 / 不重复调用）
第三层：指代消解     "我"→根据上下文判断或追问
第四层：领域知识     分类映射表 + 收支判断 + 日期处理
第五层：回复风格     友好 + 回显格式 + 列表展示
```

### 设计原则

1. **具体优于抽象**：不写"正确理解用户意图"，而是写"闺女=女儿，请客吃饭=餐饮，两百四=240"
2. **示例驱动**：删除确认规则包含具体的调用示例
3. **代码兜底**：开场白、防重复等关键行为在代码层直接实现，不依赖 LLM

---

## 七、API 设计

| 端点 | 方法 | 用途 |
|------|------|------|
| `/api/welcome` | GET | 页面加载时返回开场白 + 创建会话 |
| `/api/chat` | POST | 同步对话，返回完整回复 |
| `/api/chat/stream` | POST | 流式对话 (SSE)，逐字推送 |
| `/api/health` | GET | 健康检查 (DB + LLM 状态) |
| `/api/records` | GET | 调试用：直接查看 DB 数据 |

### 流式事件格式

```
data: {"type":"tool_call","name":"add_record","arguments":{...}}
data: {"type":"tool_result","name":"add_record","result":"..."}
data: {"type":"token","content":"记"}
data: {"type":"token","content":"账"}
data: {"type":"token","content":"成"}
data: {"type":"token","content":"功"}
data: {"type":"done","session_id":"ses_xxx"}
```

---

## 八、前端设计

单页 HTML (234行)，核心功能：

| 功能 | 实现 |
|------|------|
| 页面加载即显示开场白 | `fetch('/api/welcome')` 自动调用 |
| SSE 流式接收 | `fetch` + `ReadableStream` 逐行解析 |
| 工具调用可视化 | 显示"📝 正在记账...""🔍 正在查询..."等状态 |
| 快捷测试按钮 | 一键触发验收测试语句 |

---

## 九、设计决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 架构模式 | 纯 Function Calling | 不走 LangChain 等框架，工单要求"通过提示词实现" |
| LLM | DeepSeek (deepseek-chat) | 支持 tools 参数，OpenAI 兼容，中文效果好 |
| 会话存储 | 内存 dict | 单用户场景，无需 Redis |
| 流式策略 | 决策用非流式，回复可流式 | tool_calls 参数分片拼接复杂且易出错 |
| 开场白 | 代码层拼接 | 不依赖 LLM 记得输出，保证 100% 可靠 |
| 防重复 | 代码层去重 + Prompt 规则 | 双重防御 |
