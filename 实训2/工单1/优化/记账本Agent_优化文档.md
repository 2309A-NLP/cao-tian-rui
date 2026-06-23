# 记账本 Agent 智能体 — 优化文档

> 工单编号：人工智能 NLP-Agent 数字人项目-记账本任务
> 版本：v1.0 | 2026-06-22

---

## 一、优化历程

项目经历 **3 轮迭代**，从 0 到 29/29 验收通过。以下记录每轮发现的问题和优化方案。

### 迭代 1 → 2：修核心 Bug

| 问题 | 优化 | 效果 |
|------|------|------|
| `Session.add()` 不支持 dict 传参 | 重写为同时支持 dict 和 **kwargs | 消除 HTTP 500 |
| tool_calls 回填后消息格式错误 | 明确 assistant 消息含 tool_calls 字段时 content 为 None | LLM 正确读取上下文 |
| 热重载不生效 | 每次修改后清 `__pycache__` | 避免幽灵 bug |

### 迭代 2 → 3：完善 Agent 行为

| 问题 | 优化 | 效果 |
|------|------|------|
| 删除操作选了 query_records | 三层防御：Prompt 明确规则 + 工具描述加排除 + 示例 | delete_record 正确触发 |
| 删除时先问身份再搜索 | 修改删除规则为"不管知不知道身份，先搜" | 删除流程不被指代消解打断 |
| 同一笔账重复 INSERT | 代码层 `called_tools_this_turn` 集合去重 | 消除重复记录 |

### 迭代 3 → 最终：体验优化

| 问题 | 优化 | 效果 |
|------|------|------|
| 开场白依赖 LLM，不稳定 | 代码层 `OPENING_MESSAGE` 直接拼接 | 100% 可靠 |
| 前端需输入才显示开场白 | 加 `/api/welcome` 端点 + 页面加载自动调用 | 打开页面即显示 |
| 消息链过长导致 LLM 效率下降 | 保持完整上下文（记账场景消息量不大） | 暂未截断 |

---

## 二、已应用的关键优化

### 2.1 代码层开场白

**优化前**：靠 System Prompt 告诉 LLM 输出开场白。LLM 在 function calling 场景下优先调工具，开场白被跳过。

**优化后**：
```python
# agent_engine.py
OPENING_MESSAGE = "您好，欢迎使用咱们小家专属记账本！..."

# chat() 方法中
if is_first_msg and not session.opening_sent:
    content = OPENING_MESSAGE + "\n\n" + content
session.opening_sent = True
```

**效果**：无论 LLM 回复什么，开场白都会出现在回复最前面。

### 2.2 add_record 防重复调用

**优化前**：LLM 在 Function Calling 循环的第二轮可能再次调用同一个 add_record（参数相同），导致 DB 中重复记录。

**优化后**：
```python
# agent_engine.py
called_tools_this_turn = set()  # 本轮已调用工具的去重键

for tc in tcs:
    dedup_key = f"{name}|{member}|{item}|{amount}|{date}"
    if name == "add_record" and dedup_key in called_tools_this_turn:
        result_text = "该记录已存在，无需重复添加"
        continue  # 跳过执行
    called_tools_this_turn.add(dedup_key)
    result_text = self.executor.execute(name, args)
```

**效果**：同一轮循环中，完全相同的 add_record 只执行一次。

### 2.3 删除确认流程的三层防御

**问题**：用户说"删除X"，LLM 天然倾向调用 `query_records`（只读安全）而非 `delete_record(confirmed=false)`（名字含"delete"）。

**三层修复**：

| 层级 | 修改 | 作用 |
|------|------|------|
| System Prompt | 增加具体示例："直接调 delete_record(confirmed=false, keyword='X')" | 最高优先级指令 |
| delete_record 描述 | "不要用 query_records 代替，因为 confirmed=false 时就是搜索" | 消除 LLM 的安全顾虑 |
| query_records 描述 | "如果用户说'删除'——使用 delete_record，不要用本工具" | 明确排除 |

### 2.4 工具调用的非流式决策

**问题**：流式模式下，tool_calls 的 `arguments` 字段被拆成多个 chunk（如 `{"me` → `mber":` → `"女儿"`），拼接逻辑复杂且容易出错。

**优化**：Agent 循环内部统一用非流式 `chat()` 获取 tool_calls 决策，只在最终回复时提供流式选项。

```python
# agent_engine.py chat() 方法
for rnd in range(self.max_rounds):
    resp = self.llm.chat(session.messages, tools=TOOLS_SCHEMA)  # 非流式
    if finish == "tool_calls":
        # 参数完整，不拆分，直接解析
        args = json.loads(fn["arguments"])
```

### 2.5 指代消解不阻塞操作

**问题**：用户说"删除我的报销记录"，LLM 先追问"您是谁"，而不是先搜索。

**优化**：System Prompt 删除规则中明确：
```
"删除我的报销记录" → delete_record(confirmed=false, keyword="报销")
即使不知道"我"是谁，也不传member参数，搜索全部成员
```

**效果**：先搜到结果（不管是爸爸还是妈妈的报销），再让用户确认——搜索结果本身就说明了是谁的。

---

## 三、代码质量优化

### 3.1 模块职责清晰

| 模块 | 单一职责 |
|------|---------|
| `llm_provider.py` | 只负责 HTTP 调用，不关心业务 |
| `tools.py` | 只定义工具和执行逻辑，不管理对话 |
| `agent_engine.py` | 只管理循环和消息链，不做 DB 操作 |
| `database.py` | 只做 SQL，不知道 Agent 存在 |

### 3.2 错误处理

每个 `chat()` 调用外包了 try/except，LLM API 故障时有友好降级：
```python
except Exception as e:
    err = f"抱歉，服务暂时不可用：{e}"
    session.add_assistant(err)
    return {"reply": err, ...}
```

### 3.3 日志体系

`agent.log` 记录 Agent 每一轮的行为：
```
Agent 第1轮, 消息数=2
执行工具: add_record, 参数: {"member":"女儿",...}
记账成功: id=5, 女儿, 支出, 登山鞋
Agent 第2轮, 消息数=4
回复: 记账成功！...
```

---

## 四、已知限制与改进方向

| 限制 | 影响 | 改进方向 |
|------|------|---------|
| 会话存内存 | 服务重启丢失上下文 | 接入 Redis（参考现有 RAG 项目的 session_manager） |
| 单用户设计 | 多人同时用会话混淆 | 加用户认证，多 session 隔离 |
| 无修改功能 | 记错了只能删掉重记 | 加 `update_record` 工具 |
| 消息不截断 | 长对话影响 LLM 速度 | 超过 20 轮时截断+摘要 |
| 纯文本输出 | 查账结果只有表格文本 | 加 ECharts 柱状图/饼图可视化 |
| 无导出功能 | 无法导出 Excel/CSV | 加 `/api/export` 端点 |
| 分类固定 | 不支持自定义分类 | 加 `categories` 表，用户可增删 |

---

## 五、性能优化

### 当前状态

| 操作 | 耗时 | 瓶颈 |
|------|------|------|
| 简单记账 | 3-5s | LLM API 延迟 (~2s) + DB 写入 (~10ms) |
| 查询 | 2-4s | LLM API 延迟 |
| 删除确认 | 5-8s | 两轮 LLM 调用 |

核心瓶颈是 LLM API 网络延迟，本地代码和 DB 操作均在毫秒级。

### 优化方向

1. **LLM 缓存**：相同意图的 tool_calls 可不调 LLM，直接复用
2. **并行工具调用**：查询+汇总同时进行（当前 DeepSeek 不支持单轮多 tool_call）
3. **本地模型**：部署 Qwen 等本地模型，消除网络延迟
