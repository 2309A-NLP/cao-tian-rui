当前时间从哪里来？
完全来自服务器本地系统时钟，没有任何网络请求。

代码里用的全是 Python 标准库：


# utils.py:101
now = datetime.now()    # 本地系统时间（精确到秒）
today = now.date()      # 本地系统日期

# agent_prompts.py:30
today = date.today()    # 本地系统日期

# agent.py:257
today = date.today()    # 同上
没有调 NTP、没有请求任何 API，就是操作系统的系统时钟。所以如果服务器时间错了，挂号日期判断就全错了。

整体逻辑与思路
第一层：HTTP 接口层（app.py）
用户的请求先到这里，做三件事再放行：


请求进来
  ├─ 鉴权（verify_api_key）    → 没带key或key错 → 401直接挡
  ├─ CORS检查                  → 非白名单来源 → 浏览器拒绝
  └─ 输入格式验证              → 日期格式错 / slot_type非法 → 422直接挡
         ↓ 通过
      run_agent(user_id, query)
这一层的设计思路是：脏活在入口解决，干净的请求才进Agent。

第二层：Agent调度层（agent.py）
这是整个系统的大脑，分四个环节：

环节1：前置校验（_pre_validate）


用户ID存在？ → 不存在直接返回"用户不存在"
时间表达合法？→ 解析失败/超出14天范围 → 直接返回错误
科室白名单？ → 说了"XX科"但不在列表里 → 直接返回"没有该科室"
目的是不让垃圾请求消耗LLM token。规则能拦的不给LLM做。

环节2：多轮工具循环（最多8轮）

这是核心，每一轮的逻辑是：


① 把"对话历史+可用工具列表"发给LLM
② LLM看完，做两种选择之一：
   - 选择A：返回 tool_calls（"我要调这个工具，参数是..."）
   - 选择B：返回纯文字（"已为大宝预约成功..."）→ 结束循环

③ 如果是选择A：
   - 代码执行工具 → 拿到结果
   - 把结果塞进对话历史
   - 继续下一轮

④ 如果是选择B：直接返回给用户
一次完整挂号的3轮示意：


第1轮：发给LLM（用户说"帮我挂内科专家号"）
        LLM回：调 get_family_member("我") + query_schedule("内科") ← 并发两个
        执行：patient_id=5, 找到sch_id=23的号源

第2轮：发给LLM（附上上面的结果）
        LLM回：调 book_appointment(patient_id=5, sch_id=23)
        执行：写库成功，返回reg_id=88

第3轮：发给LLM（附上挂号成功结果）
        LLM回：纯文字"已为您预约内科专家号，编号88，..."
        → 返回给用户
环节3：挂号守卫

LLM有时会在第2轮就直接说"已为您找到号源，可以挂号"，然后结束——它查到了但忘了真正去挂。

守卫在LLM返回纯文字时检查：


if (
    有挂号意图("约/挂/预约" in query)
    and 查过号源(query_schedule in tools_called)
    and 但没挂号(book_appointment not in tools_called)
):
    # 从历史消息里翻出 patient_id 和 sch_id
    # 强制注入一次 book_appointment 调用
    # 再给LLM一轮，生成含reg_id的最终回复
这是工程补偿LLM行为不稳定的典型做法。

环节4：快捷路径（短路优化）

挂号成功后，代码不再调LLM润色，直接用模板拼字符串返回：


# agent.py:441
reply = f"已为{alias}预约成功！挂号编号：{reg_id}，{dept} {sch_type}，..."
原因：LLM润色会多花一轮时间（约1-2秒），但挂号成功的回复格式是固定的，模板就够了。

第三层：工具层（tools_read / tools_write）
6个工具，按读写分两类：


查询类（tools_read.py，只有SELECT）：
  get_family_member   → 按昵称查患者ID
  query_schedule      → 查可用号源
  get_user_history    → 查历史挂号记录
  get_doctor_schedule → 查医生排班

写入类（tools_write.py，有INSERT/UPDATE，带事务）：
  book_appointment    → 挂号（号源-1，写挂号记录）
  cancel_appointment  → 取消（号源+1，状态改0）
第四层：兜底层（llm_client.py）

Qwen2.5-32B（主）
  ↓ 超时/429/503，重试3次（1s→2s→4s指数退避）
Qwen2.5-14B（备）
  ↓ 也失败
纯正则（本地）→ 从query里提取科室+意图，查一条号源，告知用户
最差情况系统不崩溃，只是能力降级。

时间处理的完整链路（呼应你的问题）

用户说"明天下午2点"
      ↓
utils.parse_time() 解析（用 date.today() 本地时间）
      ↓
_pre_validate() 校验日期范围是否在[今天, 今天+14天]
      ↓
通过 → agent_prompts.build_system_prompt() 把今天日期注入System Prompt
      ↓
LLM看到"今天=2026-07-06"这个明确信息，才能正确推算"明天=2026-07-07"
System Prompt里的日期是每次请求时动态生成的（[agent_prompts.py:30](F:\kimi  project\医疗agent1\01-挂号管理\backend\src\agent_prompts.py#L30) 不是常量），目的是防止LLM用它训练时的日期知识乱推算。