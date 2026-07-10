"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-健康咨询

Neo4j 连接管理 + 所有 Cypher 查询封装。
外部只需调用 query_graph(intent, entity) 即可。

图谱 Schema（基于 medical.json 实际数据）：
  节点：Disease / Symptom / Drug / Department / Food / Transmission
  关系：
    (Disease)-[:HAS_SYMPTOM]->(Symptom)
    (Disease)-[:USES_DRUG]->(Drug)
    (Disease)-[:BELONGS_TO]->(Department)
    (Disease)-[:CAN_EAT]->(Food)
    (Disease)-[:NOT_EAT]->(Food)
    (Disease)-[:TRANSMITS_VIA]->(Transmission)
    (Disease)-[:CAUSES]->(Disease)          # 并发症关系
  文本字段（存于 Disease 节点属性）：
    cause / prevent / nursing / treat_detail / intro
"""
# neo4j 包：Neo4j 官方 Python 驱动
# GraphDatabase 是驱动入口类，用于创建连接池（driver）
from neo4j import GraphDatabase

# 从 config 模块导入 Neo4j 连接参数（URI / 用户名 / 密码）
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# 全局驱动单例（延迟初始化，避免模块导入时就建立连接）
_driver = None


def get_driver():
    """
    获取 Neo4j 驱动单例（懒加载）。

    第一次调用时创建驱动（连接池），后续调用复用同一实例。
    驱动对象是线程安全的，可在多个请求之间共享。

    返回:
        neo4j.Driver: Neo4j 驱动对象
    """
    global _driver
    if _driver is None:
        # auth 参数为 (用户名, 密码) 元组，GraphDatabase.driver() 建立连接池
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def _run(cypher: str, **params) -> list:
    """
    执行 Cypher 查询并将结果转换为 Python 字典列表。

    参数:
        cypher (str): Cypher 查询语句（使用 $param 参数占位符防注入）
        **params: 查询参数，对应 Cypher 中的 $xxx 占位符

    返回:
        list[dict]: 查询结果，每行记录转为一个 dict
    """
    # with get_driver().session() 自动管理 Session 生命周期（用完即释放回连接池）
    with get_driver().session() as s:
        # dict(r) 将 neo4j.Record 对象转换为普通 Python 字典
        return [dict(r) for r in s.run(cypher, **params)]


# ──────────────────────────────────────────────────────
# Cypher 模板库（每个函数返回 list[dict]）
# 设计原则：不让 LLM 直接写 Cypher，全部使用 Python 模板，防注入 + 防错
# ──────────────────────────────────────────────────────

def q_disease_info(disease: str) -> list:
    """
    查询疾病综合介绍信息（intro 字段含临床特征、实验室检查、诊断要点）。

    参数:
        disease (str): 疾病名称（模糊匹配，使用 CONTAINS）

    返回:
        list[dict]: 最多 1 条，包含 name/intro/get_prob/easy_get/insurance/treat_period/treat_cost
    """
    return _run("""
        MATCH (d:Disease)
        WHERE d.name CONTAINS $disease
        RETURN d.name AS name,
               d.intro AS intro,
               d.get_prob AS get_prob,
               d.easy_get AS easy_get,
               d.insurance AS insurance,
               d.treat_period AS treat_period,
               d.treat_cost AS treat_cost
        LIMIT 1
    """, disease=disease)


def q_symptom_to_disease(symptom: str) -> list:
    """
    通过症状反查可能患有的疾病（症状 → 疾病）。

    参数:
        symptom (str): 症状名称（模糊匹配）

    返回:
        list[dict]: 最多 5 条疾病，每条含 disease 名称和 symptoms 列表
    """
    return _run("""
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE s.name CONTAINS $symptom
        RETURN d.name AS disease, collect(s.name) AS symptoms
        LIMIT 5
    """, symptom=symptom)


def q_disease_to_symptom(disease: str) -> list:
    """
    查询疾病的所有症状（疾病 → 症状列表）。

    参数:
        disease (str): 疾病名称

    返回:
        list[dict]: 最多 1 条，包含 disease 名称和 symptoms 列表
    """
    return _run("""
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE d.name CONTAINS $disease
        RETURN d.name AS disease, collect(s.name) AS symptoms
        LIMIT 1
    """, disease=disease)


def q_disease_to_drug(disease: str) -> list:
    """
    查询疾病的推荐药物（疾病 → 药物）。

    参数:
        disease (str): 疾病名称

    返回:
        list[dict]: 最多 1 条，包含 disease 名称和 drugs 列表
    """
    return _run("""
        MATCH (d:Disease)-[:USES_DRUG]->(m:Drug)
        WHERE d.name CONTAINS $disease
        RETURN d.name AS disease, collect(m.name) AS drugs
        LIMIT 1
    """, disease=disease)


def q_disease_to_dept(disease: str) -> list:
    """
    查询疾病的就诊科室（疾病 → 科室）。

    参数:
        disease (str): 疾病名称

    返回:
        list[dict]: 最多 1 条，包含 disease 名称和 departments 列表
    """
    return _run("""
        MATCH (d:Disease)-[:BELONGS_TO]->(p:Department)
        WHERE d.name CONTAINS $disease
        RETURN d.name AS disease, collect(p.name) AS departments
        LIMIT 1
    """, disease=disease)


def q_disease_to_complication(disease: str) -> list:
    """
    查询疾病的并发症（CAUSES 关系：疾病 → 疾病）。

    参数:
        disease (str): 疾病名称

    返回:
        list[dict]: 最多 1 条，包含 disease 名称和 complications 列表
    """
    return _run("""
        MATCH (d:Disease)-[:CAUSES]->(c:Disease)
        WHERE d.name CONTAINS $disease
        RETURN d.name AS disease, collect(c.name) AS complications
        LIMIT 1
    """, disease=disease)


def q_disease_to_diet(disease: str) -> list:
    """
    查询疾病的饮食建议（可吃食物 / 忌口食物）。

    使用 OPTIONAL MATCH 确保两类食物都为 null 时也能返回疾病记录。

    参数:
        disease (str): 疾病名称

    返回:
        list[dict]: 最多 1 条，包含 disease/can_eat/not_eat 三个字段
    """
    return _run("""
        MATCH (d:Disease)
        WHERE d.name CONTAINS $disease
        OPTIONAL MATCH (d)-[:CAN_EAT]->(can:Food)
        OPTIONAL MATCH (d)-[:NOT_EAT]->(not:Food)
        RETURN d.name AS disease,
               collect(DISTINCT can.name) AS can_eat,
               collect(DISTINCT not.name) AS not_eat
        LIMIT 1
    """, disease=disease)


def q_disease_to_transmission(disease: str) -> list:
    """
    查询疾病的传播途径（疾病 → 传播方式节点）。

    参数:
        disease (str): 疾病名称

    返回:
        list[dict]: 最多 1 条，包含 disease 名称和 transmission 列表
    """
    return _run("""
        MATCH (d:Disease)-[:TRANSMITS_VIA]->(t:Transmission)
        WHERE d.name CONTAINS $disease
        RETURN d.name AS disease, collect(t.name) AS transmission
        LIMIT 1
    """, disease=disease)


def q_disease_to_cause(disease: str) -> list:
    """
    查询疾病的病因（Disease.cause 文本属性，截取前 800 字符避免过长）。

    参数:
        disease (str): 疾病名称

    返回:
        list[dict]: 最多 1 条，包含 disease 名称和 cause 文本
    """
    return _run("""
        MATCH (d:Disease)
        WHERE d.name CONTAINS $disease
        RETURN d.name AS disease, substring(d.cause, 0, 800) AS cause
        LIMIT 1
    """, disease=disease)


def q_disease_to_prevent(disease: str) -> list:
    """
    查询疾病的预防措施（Disease.prevent 文本属性，截取前 800 字符）。

    参数:
        disease (str): 疾病名称

    返回:
        list[dict]: 最多 1 条，包含 disease 名称和 prevent 文本
    """
    return _run("""
        MATCH (d:Disease)
        WHERE d.name CONTAINS $disease
        RETURN d.name AS disease, substring(d.prevent, 0, 800) AS prevent
        LIMIT 1
    """, disease=disease)


def q_disease_to_nursing(disease: str) -> list:
    """
    查询疾病的护理要点（nursing 字段 + treat_detail，不截断）。
    护理细节和隔离期信息常在 treat_detail 字段中，因此同时返回两字段。

    参数:
        disease (str): 疾病名称

    返回:
        list[dict]: 最多 1 条，包含 disease/nursing/treat_detail 三个字段
    """
    return _run("""
        MATCH (d:Disease)
        WHERE d.name CONTAINS $disease
        RETURN d.name AS disease,
               d.nursing AS nursing,
               d.treat_detail AS treat_detail
        LIMIT 1
    """, disease=disease)


def q_disease_treat_detail(disease: str) -> list:
    """
    查询疾病的详细治疗方案（含中西医，不截断）。
    中医主方/首选抗生素等信息均在此字段中。

    参数:
        disease (str): 疾病名称

    返回:
        list[dict]: 最多 1 条，包含 disease 名称和完整 treat_detail 文本
    """
    return _run("""
        MATCH (d:Disease)
        WHERE d.name CONTAINS $disease
        RETURN d.name AS disease, d.treat_detail AS treat_detail
        LIMIT 1
    """, disease=disease)


# ──────────────────────────────────────────────────────
# 统一查询入口：意图 → 对应处理函数的映射表
# ──────────────────────────────────────────────────────

INTENT_HANDLERS = {
    "disease_info":             q_disease_info,            # 综合介绍/检查/诊断
    "symptom_to_disease":       q_symptom_to_disease,      # 症状反查疾病
    "disease_to_symptom":       q_disease_to_symptom,      # 疾病→症状
    "disease_to_drug":          q_disease_to_drug,         # 疾病→药物
    "disease_to_dept":          q_disease_to_dept,         # 疾病→科室
    "disease_to_complication":  q_disease_to_complication, # 疾病→并发症
    "disease_to_diet":          q_disease_to_diet,         # 疾病→饮食建议
    "disease_to_transmission":  q_disease_to_transmission, # 疾病→传播途径
    "disease_to_cause":         q_disease_to_cause,        # 疾病→病因
    "disease_to_prevent":       q_disease_to_prevent,      # 疾病→预防
    "disease_to_nursing":       q_disease_to_nursing,      # 疾病→护理
    "disease_to_treat":         q_disease_treat_detail,    # 疾病→治疗方案
}


def safe_raw_query(cypher: str, limit: int = 10) -> list:
    """
    执行 LLM 生成的 Cypher（NL2Cypher 兜底层专用）。

    为什么需要这个函数而不直接调 _run()：
      _run() 是内部工具函数，对输入不做任何检查，直接执行。
      LLM 生成的 Cypher 是不可信的外部输入，必须先过安全检查：
        1. 危险关键字过滤（防止写操作破坏图数据库）
        2. 合法性校验（必须含 MATCH + RETURN 才是查询语句）
        3. 强制覆盖 LIMIT（防止全表扫描打爆内存）
      三道关卡都通过后才允许执行。

    参数:
        cypher (str): LLM 生成的 Cypher 查询语句（未经检验的原始字符串）
        limit (int): 强制追加的最大返回行数，默认 10，防止大结果集

    返回:
        list[dict]: 查询结果列表；任何检查失败或执行异常均返回空列表 []
    """
    import re as _re

    # ── 第一道关卡：危险关键字黑名单 ──
    # 将 Cypher 转大写统一比对，避免大小写绕过
    # "SET " 带空格是为了避免误判 "RESET" 这类词（虽然 Neo4j 不常见，保险起见）
    upper = cypher.upper()
    dangerous = ["CREATE", "DELETE", "SET ", "MERGE", "DROP", "REMOVE", "CALL"]
    if any(kw in upper for kw in dangerous):
        # 含写操作关键字，直接拒绝，返回空列表（不抛异常，让上层静默降级）
        return []

    # ── 第二道关卡：必须是合法的查询语句 ──
    # 合法 Cypher 查询语句必须同时含 MATCH（数据源）和 RETURN（输出字段）
    # 缺少任一说明 LLM 生成了不完整的片段，执行会报错，直接拦截
    if "MATCH" not in upper or "RETURN" not in upper:
        return []

    # ── 第三道关卡：强制覆盖 LIMIT，防止全表扫描 ──
    # 用正则去掉 LLM 已写的 LIMIT N（无论大小写），再统一追加我们设定的上限
    # 这样即使 LLM 写了 LIMIT 100000 也会被替换为 LIMIT 10
    cypher_clean = _re.sub(r"\bLIMIT\s+\d+", "", cypher, flags=_re.IGNORECASE).strip()
    final = f"{cypher_clean} LIMIT {limit}"  # 拼接最终执行语句

    # ── 执行并容错 ──
    # 即使通过了前三关，Cypher 语法依然可能有细微错误（如关系名拼错）
    # try/except 确保任何执行异常都静默返回空列表，不影响主流程
    try:
        return _run(final)
    except Exception:
        return []


def query_graph(intent: str, entity: str) -> dict:
    """
    统一图谱查询对外接口：根据意图选择对应的 Cypher 处理函数并执行。

    参数:
        intent (str): 意图字符串（来自 agent._classify_intent()）
        entity (str): 医学实体名称（疾病名或症状名）

    返回:
        dict: 成功时 {"ok": True, "intent": intent, "entity": entity, "data": list}
              失败时 {"ok": False, "error": "错误描述"}
    """
    handler = INTENT_HANDLERS.get(intent)  # 从映射表中查找对应处理函数

    if not handler:
        return {"ok": False, "error": f"未知意图: {intent}"}  # 未知意图（不应发生）

    try:
        data = handler(entity)  # 调用对应查询函数，传入实体名
        return {"ok": True, "intent": intent, "entity": entity, "data": data}
    except Exception as e:
        # Neo4j 连接失败或 Cypher 执行异常时捕获，返回错误字典而非抛出
        return {"ok": False, "error": str(e)}
