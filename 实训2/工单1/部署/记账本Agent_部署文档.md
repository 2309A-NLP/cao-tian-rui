# 记账本 Agent 智能体 — 部署文档

> 工单编号：人工智能 NLP-Agent 数字人项目-记账本任务
> 版本：v1.0 | 2026-06-22

---

## 一、环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | ≥3.11 | 必须 |
| MySQL | 8.0.27 | 本地 127.0.0.1:3306 |
| DeepSeek API | - | 需要有效的 API Key，能访问 api.deepseek.com |

### Python 依赖

```
fastapi>=0.104.0      # Web 框架 + SSE 支持
uvicorn>=0.24.0       # ASGI 服务器
pymysql>=1.1.0        # MySQL 连接器
requests>=2.31.0      # HTTP 客户端（LLM API 调用）
```

---

## 二、安装步骤

### 1. 获取代码

```bash
项目目录: E:\实训2\1--工单1\
```

### 2. 安装依赖

```bash
cd E:\实训2\1--工单1
pip install -r requirements.txt
```

### 3. 配置

编辑 `config.json`：

```json
{
  "llm_api_key": "sk-xxx",                    // DeepSeek API Key
  "llm_base_url": "https://api.deepseek.com", // API 地址
  "llm_model": "deepseek-chat",              // 模型名（支持 function calling）
  "llm_temperature": 0.3,                    // 温度（低=更确定）
  "db_host": "127.0.0.1",                    // MySQL 地址
  "db_port": 3306,                           // MySQL 端口
  "db_user": "root",                          // 数据库用户
  "db_password": "root",                      // 数据库密码
  "db_name": "agent_money_book",             // 数据库名（自动创建）
  "log_dir": "logs",                          // 日志目录
  "max_tool_call_rounds": 5                  // Function Calling 最大轮次
}
```

**环境变量覆盖**（可选，适合容器部署）：

| 环境变量 | 对应配置 |
|---------|---------|
| `AGENT_LLM_KEY` | llm_api_key |
| `AGENT_LLM_URL` | llm_base_url |
| `AGENT_LLM_MODEL` | llm_model |
| `AGENT_DB_HOST` | db_host |
| `AGENT_DB_PORT` | db_port |
| `AGENT_DB_USER` | db_user |
| `AGENT_DB_PASS` | db_password |
| `AGENT_DB_NAME` | db_name |

### 4. 确认 MySQL 运行

```bash
# 检查 MySQL 是否在运行
netstat -ano | findstr ":3306"

# 测试连接
python -c "import pymysql; c=pymysql.connect(host='127.0.0.1',port=3306,user='root',password='root'); print('OK'); c.close()"
```

### 5. 启动服务

```bash
python run.py
```

输出示例：
```
========================================
  家庭记账 Agent 智能体
  访问: http://localhost:8020
  API文档: http://localhost:8020/docs
========================================
```

---

## 三、验证部署

### 1. 健康检查

```bash
curl http://localhost:8020/api/health
```

正常返回：
```json
{
  "status": "ok",
  "database": true,
  "llm": "deepseek-chat",
  "agent_ready": true
}
```

### 2. 前端页面

浏览器打开 `http://localhost:8020`，应自动显示开场白：

> 您好，欢迎使用咱们小家专属记账本！请按照"x年x月x日，谁做什么事收入/支出多少钱"的格式来输入。请告诉我你的账目需求吧！

### 3. 功能验证

发送第一条记账消息：
```
今天女儿买了双登山鞋 499 元
```

预期回复：
```
记账成功！已记录：
日期：2026年6月22日 | 成员：女儿 | 分类：购物 | 物品：登山鞋 | 支出 499 元
```

### 4. 数据库验证

```bash
# 浏览器打开
http://localhost:8020/api/records

# 或命令行
python -c "import pymysql; c=pymysql.connect(host='127.0.0.1',port=3306,user='root',password='root',database='agent_money_book'); cur=c.cursor(); cur.execute('SELECT * FROM money_notes'); [print(r) for r in cur.fetchall()]"
```

---

## 四、目录结构

```
E:\实训2\1--工单1\
├── config.json              # 配置文件
├── requirements.txt         # Python 依赖
├── run.py                   # 启动入口
├── test_acceptance.py       # 验收测试脚本
├── api/
│   ├── __init__.py
│   ├── api.py               # FastAPI 路由 (218行)
│   └── static/
│       └── index.html       # 聊天界面 (234行)
├── backend/
│   ├── __init__.py
│   ├── config.py            # 配置加载 + 环境变量覆盖
│   ├── logger.py            # 日志（按天轮转，保留30天）
│   ├── database.py          # MySQL + money_notes CRUD (281行)
│   ├── llm_provider.py      # LLM 调用 + function calling (155行)
│   ├── tools.py             # 工具定义 + ToolExecutor (367行)
│   └── agent_engine.py      # Function Calling 循环引擎 (276行)
├── output/                  # 文档产出
└── logs/                    # 运行日志
    └── agent.log            # 按天轮转的日志文件
```

---

## 五、运维

### 查看日志

```bash
# 实时跟踪
tail -f E:\实训2\1--工单1\logs\agent.log

# 查看最近错误
grep -i "error\|exception" E:\实训2\1--工单1\logs\agent.log
```

### 停止服务

```bash
# Ctrl+C 优雅停止
# 或找到进程
netstat -ano | findstr ":8020"
taskkill /PID <PID> /F
```

### 重启后数据保留

所有记账数据存储在 MySQL `agent_money_book.money_notes` 表中，服务重启不影响数据。

### 清空数据

```sql
-- 登录 MySQL 执行
USE agent_money_book;
TRUNCATE TABLE money_notes;
-- 或只清测试数据
DELETE FROM money_notes WHERE item LIKE '%测试%';
```

---

## 六、常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `database: false` | MySQL 未启动或端口/密码不对 | 检查 MySQL 服务，确认 config.json 中 db_port 为 3306 |
| `agent_ready: false` | LLM API 不通 | 检查 DeepSeek API Key 和网络 |
| 开场白不显示 | 前端未调用 /api/welcome | 刷新页面 |
| 同一笔账重复记录 | LLM 多轮调用了两次 add_record | 代码层已做去重，检查 agent_engine.py 版本 |
| 删除时直接删了没确认 | System Prompt 规则被忽略 | 确认 delete_record 描述中包含两步确认逻辑 |

---

## 七、性能基准

| 指标 | 数值 | 说明 |
|------|------|------|
| 启动时间 | ~3 秒 | 含 MySQL 连接 + 建表 |
| 简单记账响应 | 3-5 秒 | LLM 意图识别 + 1轮 tool_call + INSERT + 回复生成 |
| 查询响应 | 2-4 秒 | 1轮 tool_call + SELECT + 回复生成 |
| 删除确认流程 | 5-8 秒 | 2轮 tool_call（搜索+确认） |
| 内存占用 | ~50 MB | 纯 Python，无重型依赖 |
