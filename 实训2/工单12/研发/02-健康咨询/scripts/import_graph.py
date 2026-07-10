"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-健康咨询

将 medical.json 导入 Neo4j 医疗知识图谱（幂等，可重复运行）。

图谱 Schema：
  节点：Disease / Symptom / Drug / Department / Food / Transmission
  关系：
    (Disease)-[:HAS_SYMPTOM]->(Symptom)
    (Disease)-[:USES_DRUG]->(Drug)
    (Disease)-[:BELONGS_TO]->(Department)
    (Disease)-[:CAN_EAT]->(Food)
    (Disease)-[:NOT_EAT]->(Food)
    (Disease)-[:TRANSMITS_VIA]->(Transmission)
    (Disease)-[:CAUSES]->(Disease)          # 并发症

用法：
    cd F:\\kimi  project\\医疗agent1\\02-健康咨询
    .venv\\Scripts\\python scripts/import_graph.py
    .venv\\Scripts\\python scripts/import_graph.py --limit 100   # 只导前100条（快速测试）
"""

import json      # 标准库：解析 medical.json 中每行的 JSON 对象
import os        # 标准库：读取环境变量（NEO4J_URI 等）
import sys       # 标准库：修改 stdout 编码、退出程序
import argparse  # 标准库：解析命令行参数（--limit / --data）

# ── Windows GBK/CP936 终端编码修复 ──
# 某些 Windows 终端默认 GBK 编码，中文输出会乱码，强制切换为 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('gbk', 'cp936', 'gb2312'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path       # 标准库：跨平台路径操作，比字符串拼接更安全
from dotenv import load_dotenv  # python-dotenv 包：从 .env 文件加载环境变量到 os.environ
from neo4j import GraphDatabase  # neo4j 包：Neo4j 官方 Python 驱动，用于连接图数据库并执行 Cypher 查询

# 加载项目根目录下的 .env 文件，使 NEO4J_PASSWORD 等变量生效
load_dotenv()

# 从环境变量读取 Neo4j 连接参数（优先 .env，其次默认值）
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")  # Bolt 协议地址
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")                  # 用户名
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")                            # 密码（必须配置）

# 密码未配置时立即退出，避免后续连接失败时产生误导性错误
if not NEO4J_PASSWORD:
    raise SystemExit("错误：NEO4J_PASSWORD 未配置，请在 .env 中设置")

# 默认数据路径：相对于本脚本所在目录往上两级再进入 data/medical.json
_DEFAULT_DATA = str(Path(__file__).parent.parent / "data" / "medical.json")


def load_records(path: str, limit: int = None) -> list:
    """
    读取 medical.json 文件（每行一个 JSON 对象的 JSONL 格式）。

    参数:
        path (str): medical.json 文件路径
        limit (int): 仅读取前 N 条记录（None 表示全部读取，用于快速测试）

    返回:
        list: 解析成功的 dict 记录列表
    """
    records = []
    # errors='ignore' 跳过编码错误字节，防止非法字符导致整批失败
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行（文件末尾常有空行）
            try:
                records.append(json.loads(line))  # 解析单行 JSON 对象
            except json.JSONDecodeError:
                continue  # 解析失败则跳过该行（容错）
            if limit and len(records) >= limit:
                break  # 达到限制条数后停止读取
    return records


def clean_list(val) -> list:
    """
    将字段值规范化为干净的字符串列表，过滤 None / 空字符串。

    参数:
        val: 原始字段值（可能是 None / str / list）

    返回:
        list: 去除首尾空格后的非空字符串列表
    """
    if not val:
        return []  # None 或空值直接返回空列表
    if isinstance(val, str):
        val = [val]  # 单字符串包装为列表，统一处理
    # strip() 去除首尾空格，过滤空字符串
    return [v.strip() for v in val if v and v.strip()]


def create_constraints(session):
    """
    为所有节点标签创建唯一约束（幂等：IF NOT EXISTS 保证重复执行安全）。

    参数:
        session: Neo4j Session 对象（由 driver.session() 返回）
    """
    constraints = [
        # 每种节点类型以 name 属性为唯一键
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease)      REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom)      REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Drug)         REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Department)   REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Food)         REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transmission) REQUIRE t.name IS UNIQUE",
    ]
    for c in constraints:
        session.run(c)  # 逐条执行约束创建语句
    print("✓ 约束创建完毕")


def import_batch(session, records: list):
    """
    逐条将疾病记录 MERGE 导入 Neo4j（幂等：重复运行不产生重复节点/关系）。

    MERGE 语义：节点/关系不存在则创建，已存在则更新属性（SET 语句）。

    参数:
        session: Neo4j Session 对象
        records (list): load_records() 返回的疾病记录列表
    """
    for i, r in enumerate(records):
        name = r.get("name", "").strip()  # 疾病名称是唯一标识，必须非空
        if not name:
            continue  # 没有名称的记录跳过

        # ── 1. MERGE Disease 节点并写入所有文本属性 ──
        session.run("""
            MERGE (d:Disease {name: $name})
            SET d.intro        = $intro,
                d.insurance    = $insurance,
                d.get_prob     = $get_prob,
                d.easy_get     = $easy_get,
                d.treat_prob   = $treat_prob,
                d.treat_period = $treat_period,
                d.treat_cost   = $treat_cost,
                d.cause        = $cause,
                d.prevent      = $prevent,
                d.nursing      = $nursing,
                d.treat_detail = $treat_detail
        """, name=name,
             intro=r.get("intro", ""),          # 疾病简介（含诊断/化验描述）
             insurance=r.get("insurance", ""),  # 医保类型
             get_prob=r.get("get_prob", ""),    # 患病概率描述
             easy_get=r.get("easy_get", ""),    # 易感人群
             treat_prob=r.get("treat_prob", ""),    # 治愈率
             treat_period=r.get("treat_period", ""), # 治疗周期
             treat_cost=r.get("treat_cost", ""),     # 治疗费用
             cause=r.get("cause", ""),          # 病因文本
             prevent=r.get("prevent", ""),      # 预防措施文本
             nursing=r.get("nursing", ""),      # 护理要点文本
             treat_detail=r.get("treat_detail", ""))  # 详细治疗方案（含中西医）

        # ── 2. 症状关系：(Disease)-[:HAS_SYMPTOM]->(Symptom) ──
        for sym in clean_list(r.get("symptom")):
            session.run("""
                MERGE (s:Symptom {name: $sym})
                MERGE (d:Disease {name: $name})
                MERGE (d)-[:HAS_SYMPTOM]->(s)
            """, sym=sym, name=name)  # MERGE 关系也是幂等的

        # ── 3. 药物关系：(Disease)-[:USES_DRUG]->(Drug) ──
        for drug in clean_list(r.get("drug")):
            session.run("""
                MERGE (m:Drug {name: $drug})
                MERGE (d:Disease {name: $name})
                MERGE (d)-[:USES_DRUG]->(m)
            """, drug=drug, name=name)

        # ── 4. 科室关系：(Disease)-[:BELONGS_TO]->(Department) ──
        for dept in clean_list(r.get("cure_dept")):
            session.run("""
                MERGE (p:Department {name: $dept})
                MERGE (d:Disease {name: $name})
                MERGE (d)-[:BELONGS_TO]->(p)
            """, dept=dept, name=name)

        # ── 5. 可吃食物：(Disease)-[:CAN_EAT]->(Food) ──
        for food in clean_list(r.get("can_eat")):
            session.run("""
                MERGE (f:Food {name: $food})
                MERGE (d:Disease {name: $name})
                MERGE (d)-[:CAN_EAT]->(f)
            """, food=food, name=name)

        # ── 6. 忌口食物：(Disease)-[:NOT_EAT]->(Food) ──
        for food in clean_list(r.get("not_eat")):
            session.run("""
                MERGE (f:Food {name: $food})
                MERGE (d:Disease {name: $name})
                MERGE (d)-[:NOT_EAT]->(f)
            """, food=food, name=name)

        # ── 7. 传播途径：(Disease)-[:TRANSMITS_VIA]->(Transmission) ──
        way = r.get("get_way", "").strip()  # get_way 是单字符串字段
        if way:
            session.run("""
                MERGE (t:Transmission {name: $way})
                MERGE (d:Disease {name: $name})
                MERGE (d)-[:TRANSMITS_VIA]->(t)
            """, way=way, name=name)

        # ── 8. 并发症（疾病→疾病）：(Disease)-[:CAUSES]->(Disease) ──
        for comp in clean_list(r.get("neopathy")):
            session.run("""
                MERGE (c:Disease {name: $comp})
                MERGE (d:Disease {name: $name})
                MERGE (d)-[:CAUSES]->(c)
            """, comp=comp, name=name)

        # 每 500 条打印一次进度，避免长时间无输出
        if (i + 1) % 500 == 0:
            print(f"  已导入 {i+1}/{len(records)} 条...")


def main():
    """
    主函数：解析参数 → 连接 Neo4j → 读数据 → 建约束 → 导入 → 打印统计。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int,   default=None,          help="只导前 N 条（测试用）")
    parser.add_argument("--data",  type=str,   default=_DEFAULT_DATA, help="medical.json 路径")
    args = parser.parse_args()

    print(f"连接 Neo4j: {NEO4J_URI}")
    # GraphDatabase.driver() 创建连接池；auth 传入 (用户名, 密码) 元组
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()  # 发送 PING 验证网络和认证是否正常
        print("✓ Neo4j 连接成功")
    except Exception as e:
        # 连接失败时打印提示并退出，避免后续报错更难理解
        print(f"✗ Neo4j 连接失败: {e}")
        print("  请先启动：docker start neo4j")
        sys.exit(1)

    print(f"读取数据: {args.data}")
    records = load_records(args.data, limit=args.limit)  # 读取 JSONL 文件
    print(f"✓ 解析完成，共 {len(records)} 条疾病记录")

    # 使用 with driver.session() 确保 Session 使用后自动关闭
    with driver.session() as session:
        create_constraints(session)               # 先建约束（幂等）
        print("开始导入图谱节点和关系...")
        import_batch(session, records)            # 逐条 MERGE 导入

    # ── 打印节点/关系统计数字 ──
    with driver.session() as session:
        stats = {}
        for label in ["Disease", "Symptom", "Drug", "Department", "Food", "Transmission"]:
            # 用 f-string 动态构造 Cypher，查询各标签节点总数
            c = session.run(f"MATCH (n:{label}) RETURN count(n) as c").single()["c"]
            stats[label] = c
        # 查询全图关系总数
        rel_c = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]

    print("\n=== 导入完成 ===")
    for k, v in stats.items():
        print(f"  {k:15}: {v:,} 个节点")   # 千分位格式化方便阅读
    print(f"  {'关系总数':15}: {rel_c:,} 条")
    print("\nNeo4j 浏览器：http://localhost:7474")

    driver.close()  # 关闭驱动连接池，释放资源


if __name__ == "__main__":
    main()
