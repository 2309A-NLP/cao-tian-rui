现在我有完整的代码视图了。给你一份**结构化学习路线**，按优先级排列：



------



## ***\*一、架构全景（必须能口述）\****

用户输入

  │

  ▼

app.py  ──[鉴权/CORS/输入校验]──► POST /chat

  │

  ▼

agent.py: run_agent()

  │

  ├─ Step1: _pre_validate()  ← 规则拦截，不走LLM（省token）

  │

  ├─ Step2: LLM多轮循环（最多8轮）

  │   │

  │   ├─ call_llm_with_fallback()  ← 主模型Qwen2.5-32B

  │   │     ↓失败

  │   │  fallback Qwen2.5-14B

  │   │     ↓失败

  │   │  规则兜底（纯正则）

  │   │

  │   ├─ LLM返回 tool_calls → _dispatch_tool() → tools_read/write → MySQL

  │   └─ LLM返回纯文字 → 返回给用户

  │

  └─ 挂号守卫：LLM查到号但忘了挂 → 自动补调 book_appointment

**一句话总结架构**：用户发自然语言 → agent驱动LLM多轮调工具 → 工具操作MySQL → LLM整合结果返回自然语言。



------



## ***\*二、必须掌握的核心概念（按重要性）\****

### ***\*1. Function Calling（最核心，必考）\****

**在哪**：[agent_prompts.py:120](F:\kimi project\医疗agent1\01-挂号管理\backend\src\agent_prompts.py#L120)

原理：你把 TOOLS_SCHEMA 传给LLM，LLM看到用户说"挂号"，它不直接回答，而是在 message.tool_calls 字段返回 {"name":"book_appointment", "arguments":{...}}，代码执行这个函数，把结果塞回对话，LLM再继续。

**能说出来的话**：

"我们用OpenAI兼容的Function Calling协议，定义了6个工具的Schema传给LLM，LLM根据用户意图自动决定调哪个工具、填什么参数，代码执行后把结果送回LLM，LLM再决定是否继续调工具还是给用户回复。"

### ***\*2. 多轮工具循环（agent.py的核心）\****

**在哪**：[agent.py:266](F:\kimi project\医疗agent1\01-挂号管理\backend\src\agent.py#L266)

典型挂号流程（3步）：

第1轮 LLM → 并发调 get_family_member + query_schedule

第2轮 LLM → 调 book_appointment(patient_id, sch_id)

第3轮 LLM → 返回自然语言"已为大宝预约成功..."

### ***\*3. 挂号守卫（亮点设计）\****

**在哪**：[agent.py:300-380](F:\kimi project\医疗agent1\01-挂号管理\backend\src\agent.py#L300)

**问题**：LLM偶尔会查到号源后直接回复"已帮您查到号"，忘记实际调 book_appointment。 **解法**：检测到"有挂号意图 + 查了号源 + 但没调book"时，代码**强制注入**一次 book_appointment 调用。

**能说出来的话**：

"这是我们针对LLM幻觉的一个工程保障，确保用户说'挂号'最终真的写入了数据库，而不只是LLM口头说'已挂'。"

### ***\*4. 三级兜底策略\****

**在哪**：[llm_client.py:90](F:\kimi project\医疗agent1\01-挂号管理\backend\src\llm_client.py#L90)

Qwen2.5-32B（主）→ 失败 → Qwen2.5-14B → 失败 → 纯正则规则

每级失败都有日志，规则兜底保证系统不崩溃。

### ***\*5. 前置校验（_pre_validate）\****

**在哪**：[agent.py:169](F:\kimi project\医疗agent1\01-挂号管理\backend\src\agent.py#L169)

三步校验在LLM之前完成：用户是否存在 → 时间表达是否合法 → 科室是否在白名单。这样无效请求直接返回，不消耗LLM token。



------



## ***\*三、适当了解（能回答追问即可）\****

| ***\*文件\****                                               | ***\*了解要点\****                                 |
| ------------------------------------------------------------ | -------------------------------------------------- |
| [config.py](F:\kimi project\医疗agent1\01-挂号管理\backend\src\config.py) | 科室白名单、号源类型、费用在这里统一配置           |
| [auth.py](F:\kimi project\医疗agent1\01-挂号管理\backend\src\auth.py) | API Key鉴权，.env未配置时跳过（开发模式）          |
| [app.py:28](F:\kimi project\医疗agent1\01-挂号管理\backend\src\app.py#L28) | CORS白名单，不含"null"防本地HTML攻击               |
| tools_read/write                                             | 查询4个工具，写入2个工具（挂号/取消），带MySQL事务 |



------



## ***\*四、讲解时的话术框架\****

**开场**（架构）：

"这个模块是医疗Agent的挂号子系统，用FastAPI提供HTTP接口，核心是一个LLM驱动的多轮工具调用Agent。用户发自然语言，Agent通过Function Calling让LLM选工具、填参数，工具操作MySQL数据库，最终LLM把结果整合成自然语言回复。"

**被问"为什么不直接写规则"时**：

"规则覆盖不了'再约上次那个医生'这种上下文依赖的请求，LLM的优势是理解意图和灵活的多步推理。但LLM不可靠，所以我们在关键路径加了守卫和兜底。"

**被问"LLM失败了怎么办"时**：

"三级降级：主模型→备用模型→纯正则兜底。每级都有日志记录，最差情况下系统还能返回一条简易查询结果，不会崩溃。"



------





 