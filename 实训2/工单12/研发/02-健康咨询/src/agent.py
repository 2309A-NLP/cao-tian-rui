"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-健康咨询

核心 Agent 流程（v2 优化版）：
  Step 0: 缓存查询（完全相同 query → 直接返回）
  Step 1: 规则意图分类（高置信度命中直接用，0ms）
          → 未命中 → LLM Function Calling 兜底
  Step 2: Python 模板化 Cypher → Neo4j 查询
  Step 3: LLM 基于图谱数据生成回答（支持流式）
  兜底：任何一步失败 → 降级回复，不抛异常

质量保证：
  - 规则不确定时一律走 LLM，绝不勉强分类
  - LLM 答案生成阶段完全不动（同模型、同 prompt、同 temperature）
  - 缓存只做完全匹配，不做模糊匹配
"""

import json                    # 标准库：序列化/反序列化 JSON（工具调用参数、图谱数据格式化）
import time                    # 标准库：高精度计时，统计每次请求的 elapsed_ms
from collections import OrderedDict  # 标准库：有序字典，用于实现 LRU 缓存（按访问顺序淘汰）
from typing import Iterator, Optional  # 标准库：类型注解，提高代码可读性

# openai 包：OpenAI Python SDK，同时兼容硅基流动等 OpenAI 兼容 API
# 用于调用 LLM 完成意图分类（Function Calling）和答案生成（普通对话/流式对话）
from openai import OpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL  # 本项目 LLM 连接配置
from src.graph import query_graph, safe_raw_query              # Neo4j 图谱查询统一入口
from src import memory_bridge                                  # 患者历史记忆管理模块
from src.intent_rules import classify as rule_classify         # 规则意图分类器（0ms，高置信度）

# 创建 OpenAI 客户端，base_url 指向硅基流动兼容 API 地址
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# ──────────────────────────────────────────────────────
# LRU 缓存（完全匹配才命中，永远不影响准确率）
# key = f"{patient_id}::{query}"，value = run_agent 返回的完整结果 dict
# ──────────────────────────────────────────────────────

_CACHE_MAX = 256  # 缓存最大条目数，超出时淘汰最久未访问的条目
_cache: "OrderedDict[str, dict]" = OrderedDict()  # 全局 LRU 缓存字典


def _cache_get(key: str) -> Optional[dict]:
    """
    从 LRU 缓存中取值，命中时将该条目移到末尾（表示最近使用）。

    参数:
        key (str): 缓存键（patient_id::query）

    返回:
        dict 或 None（未命中）
    """
    if key in _cache:
        _cache.move_to_end(key)  # 刷新访问顺序（LRU 核心操作）
        return _cache[key]
    return None  # 未命中


def _cache_put(key: str, value: dict) -> None:
    """
    向 LRU 缓存写入一条记录，超出上限时淘汰最久未访问的条目。

    参数:
        key (str): 缓存键
        value (dict): 缓存值（完整 Agent 返回结果）
    """
    _cache[key] = value
    _cache.move_to_end(key)             # 新写入的条目放到末尾（最近使用）
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)      # 弹出最前面的条目（最久未使用）


# ──────────────────────────────────────────────────────
# Function Calling Schema（LLM 兜底用）
# 当规则分类器无法高置信度识别意图时，通过此 tool schema
# 强制 LLM 输出结构化的 intent + entity
# ──────────────────────────────────────────────────────

_EXTRACT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "query_medical_graph",   # 工具名称
            "description": "根据用户问题，识别意图和医学实体，查询医疗知识图谱",
            "parameters": {
                "type": "object",
                "required": ["intent", "entity"],  # 两个字段均为必填
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "用户意图类型",
                        # enum 限定 LLM 只能从这 12 种意图中选择，防止幻觉
                        "enum": [
                            "disease_info",           # 疾病综合介绍/检查/诊断
                            "symptom_to_disease",     # 症状 → 疾病（反查）
                            "disease_to_symptom",     # 疾病 → 症状
                            "disease_to_drug",        # 疾病 → 药物
                            "disease_to_dept",        # 疾病 → 就诊科室
                            "disease_to_complication",# 疾病 → 并发症
                            "disease_to_diet",        # 疾病 → 饮食建议
                            "disease_to_transmission",# 疾病 → 传播途径
                            "disease_to_cause",       # 疾病 → 病因
                            "disease_to_prevent",     # 疾病 → 预防措施
                            "disease_to_nursing",     # 疾病 → 护理/隔离期
                            "disease_to_treat",       # 疾病 → 治疗方案
                        ]
                    },
                    "entity": {
                        "type": "string",
                        "description": "核心医学实体（疾病名或症状名）。症状类意图填症状，其余填疾病名。"
                    }
                }
            }
        }
    }
]

# LLM 系统提示：引导 LLM 正确选择意图并填写 entity
_SYSTEM_PROMPT = """你是一位专业的医疗知识助手，能够回答疾病相关问题。

你必须调用 query_medical_graph 工具来查询知识图谱，获取准确数据后再回答。

意图选择规则：
- 用户提到症状，询问可能是什么病 → symptom_to_disease，entity 填症状
- 问疾病有哪些症状（临床表现、外在症状）→ disease_to_symptom
- 问疾病吃什么药 → disease_to_drug
- 问疾病治疗方案 / 中医治疗 / 怎么治 → disease_to_treat
- 问疾病怎么传播 → disease_to_transmission
- 问疾病的并发症 → disease_to_complication
- 问疾病能吃什么 / 不能吃什么 → disease_to_diet
- 问疾病怎么预防 → disease_to_prevent
- 问疾病护理注意事项 / 隔离期 → disease_to_nursing
- 问疾病病因 / 由什么引起 → disease_to_cause
- 问挂哪个科 → disease_to_dept
- 综合介绍某疾病 / 诊断标准 / 化验检查 / 血常规 / 实验室指标 / 影像学检查 → disease_info

注意：
- entity 填疾病或症状的核心词，不带"的""了"等助词
- 涉及"检查/化验/血常规/影像"等诊断类问题，一律走 disease_info"""


# ──────────────────────────────────────────────────────
# NL2Cypher 兜底层（第三级，仅在模板查询返回空数据时触发）
# ──────────────────────────────────────────────────────

# ── _GRAPH_SCHEMA：完整图谱结构描述，作为 NL2Cypher 的 System Prompt 基础 ──
# 为什么要把 Schema 写得这么详细：
#   LLM 不知道这个图库里有什么节点、什么关系、属性叫什么名字。
#   如果不告诉它，它会凭"常识"猜，极易生成不存在的关系名（如 HAS_DISEASE）或属性名。
#   把 6 种节点、7 种关系、Disease 的全部属性都列出来，LLM 就能准确照着写 Cypher。
# 为什么写"重要约束"那一节：
#   实测 LLM 倾向于用精确匹配 =，但图谱中疾病名可能带括号或空格，= 会查不到。
#   明确要求用 CONTAINS 是关键的兜底提示。
_GRAPH_SCHEMA = """
## Neo4j 医疗知识图谱 Schema

### 节点类型
- Disease    ：疾病，属性：name, intro, cause, prevent, nursing, treat_detail, get_prob, easy_get, insurance, treat_period, treat_cost
- Symptom    ：症状，属性：name
- Drug       ：药物，属性：name
- Department ：科室，属性：name
- Food       ：食物，属性：name
- Transmission：传播途径，属性：name

### 关系类型（全部从 Disease 出发）
- (Disease)-[:HAS_SYMPTOM]->(Symptom)        疾病→症状
- (Disease)-[:USES_DRUG]->(Drug)              疾病→推荐药物
- (Disease)-[:BELONGS_TO]->(Department)      疾病→就诊科室
- (Disease)-[:CAN_EAT]->(Food)               疾病→可吃食物
- (Disease)-[:NOT_EAT]->(Food)               疾病→忌口食物
- (Disease)-[:TRANSMITS_VIA]->(Transmission) 疾病→传播途径
- (Disease)-[:CAUSES]->(Disease)             疾病→并发症（疾病→疾病）

### 重要约束
- 节点名称匹配用 CONTAINS（不要用 =），例：WHERE d.name CONTAINS '糖尿病'
- 只能用 MATCH / WITH / WHERE / RETURN / ORDER BY / LIMIT，禁止写操作
- 结尾必须有 RETURN 和 LIMIT（最多 LIMIT 10）
- 只输出纯 Cypher 代码，不要加 markdown 代码块标记
"""

# ── _NL2CYPHER_PROMPT：在 Schema 基础上追加任务指令，构成完整 System Prompt ──
# Schema 描述"图长什么样"，任务指令描述"要做什么"，二者拼接给 LLM 作为 system 角色消息。
# 把 Schema 放 system、把具体问题放 user，符合 OpenAI 的 role 使用规范：
#   system = 背景知识 + 行为约束
#   user   = 具体输入
_NL2CYPHER_PROMPT = _GRAPH_SCHEMA + """
根据以上 Schema 和用户问题，生成一条能查到答案的 Cypher 查询语句。
要求：只输出 Cypher 语句本身，不加任何解释或代码块标记。
"""


def _nl2cypher(query: str, entity: str) -> list:
    """
    NL2Cypher 兜底查询：让 LLM 根据 Schema 生成 Cypher 并安全执行。

    在整个 Agent 流程中的位置（第三级兜底）：
      模板 Cypher 返回空 → 调用本函数 → 仍空 → LLM 通用知识兜底

    为什么这里是第三级而不是第二级：
      模板 Cypher 速度快（<50ms）且永远语法正确，能覆盖 90% 以上的常见问题。
      NL2Cypher 需要额外一次 LLM 调用（约 2s），只在模板查不到时才值得花这个代价。

    参数:
        query (str): 用户原始问题，完整传入让 LLM 理解语义
        entity (str): 已提取的医学实体，额外传入帮助 LLM 聚焦，减少 Cypher 中实体名猜测错误

    返回:
        list[dict]: Neo4j 查询结果；LLM 调用失败 / Cypher 不合法 / 执行报错均返回 []
    """
    try:
        # 把问题和实体一起放进 user 消息：
        #   - query 提供完整语义（LLM 需要知道用户到底想查什么关系）
        #   - entity 是已经提取好的核心词，直接用，避免 LLM 再从问题里猜实体名
        user_msg = f"用户问题：{query}\n核心实体：{entity}\n\n请生成 Cypher："

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _NL2CYPHER_PROMPT},  # Schema + 任务指令
                {"role": "user",   "content": user_msg},            # 具体问题
            ],
            temperature=0,    # 代码生成必须用 temperature=0：确定性输出，不允许随机变体
            max_tokens=300,   # Cypher 语句通常 50-150 token，300 足够且避免 LLM 输出多余解释
        )

        cypher = resp.choices[0].message.content.strip()

        # ── 清洗 LLM 输出：去掉可能残留的 markdown 代码块标记 ──
        # 即使 Prompt 里要求"只输出 Cypher"，LLM 有时仍会加 ```cypher ... ``` 包裹
        # 用正则把开头的 ```xxx 和结尾的 ``` 都去掉，得到纯 Cypher 字符串
        import re as _re
        cypher = _re.sub(r"^```[a-z]*\n?", "", cypher, flags=_re.IGNORECASE)  # 去开头标记
        cypher = _re.sub(r"\n?```$", "", cypher).strip()                        # 去结尾标记

        # ── 交给安全执行器：做危险关键字过滤 + 合法性校验 + 执行 ──
        # safe_raw_query 负责所有安全检查，本函数不重复做，职责分离
        return safe_raw_query(cypher)

    except Exception:
        # 任何异常（API 超时 / 网络断开 / JSON 解析失败等）都静默返回空列表
        # 上层会接着走 LLM 通用知识兜底，用户不会感知到这里出了问题
        return []


# ──────────────────────────────────────────────────────
# 输入校验
# ──────────────────────────────────────────────────────

def validate_query(query: str) -> Optional[str]:
    """
    校验用户输入的合法性。

    参数:
        query (str): 原始查询字符串

    返回:
        str: 错误原因（校验失败时返回）
        None: 校验通过
    """
    if query is None:
        return "问题不能为空"  # None 类型直接拒绝

    q = query.strip()  # 去除首尾空白

    if not q:
        return "问题不能为空"  # 纯空白字符串拒绝

    if len(q) > 500:
        return "问题过长（最多 500 字）"  # 防止过长输入消耗过多 token

    import re
    # 要求至少包含一个中文字符或英文字母，过滤纯数字/纯符号输入
    if not re.search(r"[一-龥A-Za-z]", q):
        return "请输入有效的医学问题"

    return None  # 校验通过


# ──────────────────────────────────────────────────────
# 意图识别（规则优先 + LLM 兜底）
# ──────────────────────────────────────────────────────

def _classify_intent(query: str) -> tuple[str, str, str]:
    """
    识别用户问题的意图和医学实体。

    策略：
      1. 先调用规则分类器（0ms，高置信度命中才返回）
      2. 规则未命中 → 调用 LLM Function Calling（1-3s）

    参数:
        query (str): 用户问题（已通过 validate_query 校验）

    返回:
        tuple: (intent, entity, source)
          - intent: 意图字符串（如 "disease_to_complication"）
          - entity: 医学实体（如 "百日咳"）
          - source: 来源标识（"rule" 或 "llm"）

    异常:
        Exception: LLM 调用失败或未返回工具调用时抛出
    """
    # ── Step 1: 规则优先 ──
    rule_result = rule_classify(query)
    if rule_result:
        intent, entity = rule_result
        return (intent, entity, "rule")  # 规则命中，直接返回

    # ── Step 2: LLM Function Calling 兜底 ──
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=_EXTRACT_TOOL,
        # tool_choice 强制 LLM 必须调用指定工具，避免直接文本回答
        tool_choice={"type": "function", "function": {"name": "query_medical_graph"}},
        temperature=0,  # temperature=0：确定性输出，意图分类不需要创造性
    )
    msg = resp.choices[0].message
    if not msg.tool_calls:
        raise ValueError("LLM 未返回工具调用")  # 不应发生（tool_choice 强制了），保险起见

    # 解析工具调用参数（JSON 字符串 → dict）
    args = json.loads(msg.tool_calls[0].function.arguments)
    return (args["intent"], args["entity"], "llm")


# ──────────────────────────────────────────────────────
# 主入口（非流式，兼容旧接口）
# ──────────────────────────────────────────────────────

def run_agent(query: str, patient_id: str = "default_patient") -> dict:
    """
    处理用户健康咨询，一次性返回完整答案。

    流程：输入校验 → 缓存查询 → 记忆召回 → 意图分类 → 图谱查询 → LLM 生成 → 缓存写入

    参数:
        query (str): 用户问题
        patient_id (str): 患者唯一标识，用于隔离记忆（默认 "default_patient"）

    返回:
        dict: {
            "reply": 回复文本,
            "intent": 意图分类结果,
            "entity": 提取的医学实体,
            "graph_data": 图谱查询数据列表,
            "elapsed_ms": 总耗时毫秒,
            "recalled_memories": 召回的历史记忆列表,
            "cache_hit": 是否命中缓存,
            "intent_source": 意图来源（"rule"/"llm"/"error"）
        }
    """
    t0 = time.perf_counter()  # 开始计时

    # ── 输入校验 ──
    err = validate_query(query)
    if err:
        return _error_reply(err)  # 校验失败立即返回错误结构

    query = query.strip()  # 统一去除首尾空白

    # ── LRU 缓存查询 ──
    cache_key = f"{patient_id}::{query}"  # 缓存键：患者ID + 问题（完全匹配）
    cached = _cache_get(cache_key)
    if cached:
        # 命中缓存：直接返回，elapsed_ms 更新为本次实际耗时
        return {**cached, "cache_hit": True, "elapsed_ms": round((time.perf_counter() - t0) * 1000)}

    # ── 记忆召回（患者历史对话语义检索）──
    recalled = memory_bridge.recall(patient_id, query, limit=5)      # 检索最相关的 5 条历史
    memory_context = memory_bridge.format_for_prompt(recalled)        # 格式化为 prompt 片段

    # ── 意图分类（规则优先 + LLM 兜底）──
    try:
        intent, entity, source = _classify_intent(query)
    except Exception:
        return _error_reply("抱歉，无法理解您的问题，请换个方式描述。")

    # ── 图谱查询 ──
    graph_result = query_graph(intent, entity)  # 返回 {"ok": bool, "data": [...]}

    # ── 模板查询无数据：进入多级兜底 ──
    # graph_result["ok"]=False 表示 Neo4j 连接异常；data 为空列表表示图谱里确实没有该实体
    # 两种情况都进入兜底逻辑，不区分处理（结果一样：没有可用的图谱数据）
    if not graph_result["ok"] or not graph_result.get("data"):

        # ── 第三级兜底：NL2Cypher ──
        # 模板只覆盖 12 种固定意图，超出范围（如多跳查询、交集查询）或实体名模糊匹配失败时
        # 让 LLM 自由生成 Cypher，有机会从图谱中查到模板覆盖不到的数据
        nl_data = _nl2cypher(query, entity)

        if nl_data:
            # NL2Cypher 查到了数据：复用"有图谱数据"的答案生成函数，保持回答质量一致
            # 对 LLM 来说 nl_data 和模板数据格式相同（都是 list[dict]），无需区分
            reply = _llm_answer_with_graph(query, intent, entity, nl_data, memory_context)
            elapsed = round((time.perf_counter() - t0) * 1000)
            memory_bridge.remember(patient_id, query, reply)  # 异步写入记忆（后台线程）
            result = {
                "reply": reply, "intent": intent, "entity": entity,
                "graph_data": nl_data,          # 把 NL2Cypher 查到的数据也透传给前端
                "elapsed_ms": elapsed,
                "recalled_memories": [m.get("memory", "") for m in recalled],
                "cache_hit": False, "intent_source": source,
                "nl2cypher_used": True,         # 明确标记：本次走了 NL2Cypher 路径，供前端展示
            }
            _cache_put(cache_key, result)       # 写入 LRU 缓存，下次同样问题直接命中
            return result

        # ── 最终兜底：LLM 通用医学知识 ──
        # NL2Cypher 也查不到（实体不在图谱中，或生成的 Cypher 执行失败）
        # 退化到最基础的 LLM 问答：告知"图谱无数据"，让 LLM 用训练知识回答
        # 这是兜底的最后一道，确保用户永远能得到一个回答而不是报错页面
        reply = _llm_answer_without_graph(query, entity, memory_context)
        elapsed = round((time.perf_counter() - t0) * 1000)
        memory_bridge.remember(patient_id, query, reply)
        result = {
            "reply": reply, "intent": intent, "entity": entity,
            "graph_data": [],               # 无图谱数据，前端不展示知识来源
            "elapsed_ms": elapsed,
            "recalled_memories": [m.get("memory", "") for m in recalled],
            "cache_hit": False, "intent_source": source,
            "nl2cypher_used": False,        # 标记：NL2Cypher 也没有成功
        }
        _cache_put(cache_key, result)
        return result

    # ── 图谱有数据时：LLM 基于图谱数据生成精准回答 ──
    reply = _llm_answer_with_graph(query, intent, entity, graph_result["data"], memory_context)
    elapsed = round((time.perf_counter() - t0) * 1000)
    memory_bridge.remember(patient_id, query, reply)  # 异步写入记忆

    result = {
        "reply": reply,
        "intent": intent,
        "entity": entity,
        "graph_data": graph_result["data"],
        "elapsed_ms": elapsed,
        "recalled_memories": [m.get("memory", "") for m in recalled],
        "cache_hit": False,
        "intent_source": source,
        "nl2cypher_used": False,
    }
    _cache_put(cache_key, result)  # 写入缓存
    return result


# ──────────────────────────────────────────────────────
# 流式接口（SSE 用）
# ──────────────────────────────────────────────────────

def run_agent_stream(query: str, patient_id: str = "default_patient") -> Iterator[dict]:
    """
    流式返回 Agent 处理结果（用于 SSE 接口）。

    yield 的事件类型：
      {"type": "meta",  "intent": ..., "entity": ..., "intent_source": ...,
       "graph_data": ..., "recalled_memories": ...}  ← 首条事件，意图识别完成后立即推送
      {"type": "token", "content": "..."}            ← 多次，每次一个文字片段
      {"type": "done",  "elapsed_ms": ..., "reply": "...", "cache_hit": ...}  ← 最后一条
      {"type": "error", "message": "..."}            ← 出错时代替上述所有

    参数:
        query (str): 用户问题
        patient_id (str): 患者唯一标识

    生成:
        Iterator[dict]: 事件字典序列
    """
    t0 = time.perf_counter()

    # ── 输入校验 ──
    err = validate_query(query)
    if err:
        yield {"type": "error", "message": err}  # 校验失败，推送 error 事件后终止
        return

    query = query.strip()
    cache_key = f"{patient_id}::{query}"

    # ── 缓存命中：复用上次结果，模拟流式推送 ──
    cached = _cache_get(cache_key)
    if cached:
        # 先推 meta 事件（意图信息）
        yield {
            "type": "meta",
            "intent": cached["intent"], "entity": cached["entity"],
            "intent_source": cached.get("intent_source", "cache"),
            "graph_data": cached.get("graph_data", []),
            "recalled_memories": cached.get("recalled_memories", []),
        }
        yield {"type": "token", "content": cached["reply"]}  # 全文一次性推送（缓存场景）
        yield {
            "type": "done",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            "reply": cached["reply"],
            "cache_hit": True,
        }
        return

    # ── 记忆召回 ──
    recalled = memory_bridge.recall(patient_id, query, limit=5)
    memory_context = memory_bridge.format_for_prompt(recalled)

    # ── 意图分类 ──
    try:
        intent, entity, source = _classify_intent(query)
    except Exception:
        yield {"type": "error", "message": "抱歉，无法理解您的问题，请换个方式描述。"}
        return

    # ── 图谱查询 ──
    graph_result = query_graph(intent, entity)
    graph_data = graph_result.get("data") if graph_result["ok"] else []  # 查询失败时降为空列表

    # ── 推送 meta 事件（意图识别结果，客户端可据此显示"正在查询图谱..."）──
    yield {
        "type": "meta",
        "intent": intent, "entity": entity, "intent_source": source,
        "graph_data": graph_data or [],
        "recalled_memories": [m.get("memory", "") for m in recalled],
    }

    # ── 流式生成回答（含 NL2Cypher 三级兜底）──
    nl2cypher_used = False  # 记录本次是否触发了 NL2Cypher，最终写入缓存和 done 事件

    if not graph_data:
        # 模板查询为空，进入 NL2Cypher 兜底
        # 注意：_nl2cypher 是阻塞调用（需要一次 LLM 往返），在这里会等待约 2s
        # 流式接口能容忍这个等待，因为首个 meta 事件已经在前面推送出去了，
        # 用户看到"正在查询..."的状态，不会感觉卡顿
        nl_data = _nl2cypher(query, entity)

        if nl_data:
            # NL2Cypher 查到数据：覆盖 graph_data，后续走有图谱路径生成答案
            graph_data = nl_data
            nl2cypher_used = True

            # 推送第二条 meta 事件，把 NL2Cypher 查到的图谱数据更新给前端
            # 前端会用这份数据更新"执行过程"面板里展示的图谱来源
            # nl2cypher_used=True 是额外标志，前端可据此显示"深度查询"标签
            yield {
                "type": "meta",
                "intent": intent, "entity": entity, "intent_source": source,
                "graph_data": graph_data,
                "recalled_memories": [m.get("memory", "") for m in recalled],
                "nl2cypher_used": True,
            }
        # nl_data 为空时不做任何处理，graph_data 保持空列表，继续往下走最终兜底

    if graph_data:
        # 有图谱数据（来源：模板 Cypher 或 NL2Cypher，两者对 LLM 完全透明）
        # 基于图谱数据流式生成回答，每个 token 立即推送给客户端
        buf = []
        for tok in _llm_answer_with_graph_stream(query, intent, entity, graph_data, memory_context):
            buf.append(tok)
            yield {"type": "token", "content": tok}  # 逐 token 推送，用户看到打字机效果
        reply = "".join(buf)  # 收集完整回复文本，后续写缓存和记忆用
    else:
        # 最终降级：模板和 NL2Cypher 都没有数据，退化为 LLM 通用医学知识回答
        # 同样流式推送，体验一致
        buf = []
        for tok in _llm_answer_without_graph_stream(query, entity, memory_context):
            buf.append(tok)
            yield {"type": "token", "content": tok}
        reply = "".join(buf)

    elapsed = round((time.perf_counter() - t0) * 1000)

    # ── 异步写入记忆 ──
    memory_bridge.remember(patient_id, query, reply)

    # ── 写入缓存 ──
    result = {
        "reply": reply, "intent": intent, "entity": entity,
        "graph_data": graph_data or [], "elapsed_ms": elapsed,
        "recalled_memories": [m.get("memory", "") for m in recalled],
        "cache_hit": False, "intent_source": source,
        "nl2cypher_used": nl2cypher_used,
    }
    _cache_put(cache_key, result)

    # ── 推送 done 事件（通知客户端流结束）──
    yield {"type": "done", "elapsed_ms": elapsed, "reply": reply, "cache_hit": False, "nl2cypher_used": nl2cypher_used}


# ──────────────────────────────────────────────────────
# LLM 答案生成（保持与 v1 完全一致的 prompt / temperature）
# ──────────────────────────────────────────────────────

def _build_prompt_with_graph(query: str, intent: str, entity: str, data: list, memory_context: str) -> str:
    """
    构建"基于图谱数据回答"的 prompt。

    参数:
        query (str): 用户原始问题
        intent (str): 意图分类结果
        entity (str): 医学实体
        data (list): 图谱查询返回的数据列表
        memory_context (str): 患者历史记忆文本块（空时不插入）

    返回:
        str: 完整 prompt 字符串
    """
    data_str = json.dumps(data, ensure_ascii=False, indent=2)  # 图谱数据格式化为 JSON 字符串
    # 仅在有历史记忆时才插入记忆块
    memory_block = f"[患者历史记忆]\n{memory_context}\n[/患者历史记忆]\n\n" if memory_context else ""
    return f"""{memory_block}用户问题：{query}

知识图谱查询结果（意图={intent}，实体={entity}）：
{data_str}

请严格基于以上图谱数据回答用户问题，遵循以下规则：

1. **精准定位**：识别用户问题中的关键限定词，在数据中找到对应段落再回答，不要跨限定词混答。
   - 若问题限定了治疗体系（如"中医"或"西医"），只从该体系段落取答案，两者不混用。
   - 若问题限定了病程阶段（如"初期"/"恢复期"等），只取该阶段对应内容。
   - 若问题问"主方是什么"，直接给出方剂/药物名称，再简要说明用途。
   - 若问题限定了人群（如"儿童"/"老人"），优先取对应人群的描述。

2. **数据里没有**：明确说"知识图谱中未收录该项具体信息"，再给通用建议，不用其他段落硬凑。

3. **表达要求**：
   - 用自然语言，不要复述 JSON 结构
   - 若涉及药物/方剂，把关键名称写清楚
   - 长度：120~250 字，按问题复杂度伸缩
   - 结尾加"以上信息仅供参考，具体用药请遵医嘱"（问题不涉及用药/治疗则跳过）"""


def _llm_answer_with_graph(query: str, intent: str, entity: str, data: list, memory_context: str = "") -> str:
    """
    非流式 LLM 答案生成（有图谱数据版本）。

    参数:
        query, intent, entity, data, memory_context: 同 _build_prompt_with_graph

    返回:
        str: LLM 生成的回答文本；出错时返回图谱原始数据的 JSON 字符串（降级兜底）
    """
    prompt = _build_prompt_with_graph(query, intent, entity, data, memory_context)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,   # 略有随机性，回答更自然，但不影响准确性
            max_tokens=512,    # 限制输出长度，控制 API 成本
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # LLM 调用失败时降级：直接返回原始图谱数据，确保有内容可用
        return f"根据医疗知识图谱查询结果：{json.dumps(data, ensure_ascii=False)}"


def _llm_answer_with_graph_stream(query: str, intent: str, entity: str, data: list, memory_context: str = "") -> Iterator[str]:
    """
    流式 LLM 答案生成（有图谱数据版本）。

    参数:
        query, intent, entity, data, memory_context: 同 _build_prompt_with_graph

    生成:
        Iterator[str]: 逐个 token 字符串
    """
    prompt = _build_prompt_with_graph(query, intent, entity, data, memory_context)
    try:
        # stream=True：启用流式输出，LLM 边生成边返回
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=512, stream=True,
        )
        for chunk in stream:
            # 检查 chunk 是否有内容（最后一个 chunk 可能 content 为 None）
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content  # yield 单个 token
    except Exception:
        yield f"根据医疗知识图谱查询结果：{json.dumps(data, ensure_ascii=False)}"


def _llm_answer_without_graph(query: str, entity: str, memory_context: str = "") -> str:
    """
    非流式 LLM 答案生成（无图谱数据版本，使用通用医学知识兜底）。

    参数:
        query (str): 用户问题
        entity (str): 提取的医学实体（用于提示 LLM 图谱无数据）
        memory_context (str): 患者历史记忆文本

    返回:
        str: LLM 生成的通用医学建议
    """
    memory_block = f"[患者历史记忆]\n{memory_context}\n[/患者历史记忆]\n\n" if memory_context else ""
    # 在 user 消息中明确告知 LLM 图谱无数据，让其用通用知识回答
    user_msg = f"{memory_block}{query}（注意：知识图谱中未找到关于'{entity}'的记录，请用通用医学知识回答）"
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是专业医疗知识助手，用简洁专业的语言回答问题，结尾建议咨询医生。"},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3, max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"抱歉，知识图谱中未找到关于'{entity}'的信息，建议咨询专业医生。"


def _llm_answer_without_graph_stream(query: str, entity: str, memory_context: str = "") -> Iterator[str]:
    """
    流式 LLM 答案生成（无图谱数据版本）。

    参数:
        query, entity, memory_context: 同 _llm_answer_without_graph

    生成:
        Iterator[str]: 逐个 token 字符串
    """
    memory_block = f"[患者历史记忆]\n{memory_context}\n[/患者历史记忆]\n\n" if memory_context else ""
    user_msg = f"{memory_block}{query}（注意：知识图谱中未找到关于'{entity}'的记录，请用通用医学知识回答）"
    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是专业医疗知识助手，用简洁专业的语言回答问题，结尾建议咨询医生。"},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3, max_tokens=400, stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception:
        yield f"抱歉，知识图谱中未找到关于'{entity}'的信息，建议咨询专业医生。"


def _error_reply(msg: str) -> dict:
    """
    构造统一的错误返回结构（输入校验失败或意图识别失败时使用）。

    参数:
        msg (str): 错误原因描述

    返回:
        dict: 与 run_agent 返回格式一致的错误结构
    """
    return {
        "reply": msg,
        "intent": "error",   # 特殊意图标记，供上层判断
        "entity": "",
        "graph_data": [],
        "elapsed_ms": 0,
        "recalled_memories": [],
        "cache_hit": False,
        "intent_source": "error",
    }


# ──────────────────────────────────────────────────────
# 缓存管理接口（便于测试/运维）
# ──────────────────────────────────────────────────────

def clear_cache() -> int:
    """
    清空全局 LRU 缓存。

    返回:
        int: 清空前的缓存条目数（用于日志记录）
    """
    n = len(_cache)
    _cache.clear()  # OrderedDict.clear() 清空所有条目
    return n


def cache_stats() -> dict:
    """
    返回缓存当前状态（供 /cache/stats 接口调用）。

    返回:
        dict: {"size": 当前条目数, "max": 最大容量}
    """
    return {"size": len(_cache), "max": _CACHE_MAX}
