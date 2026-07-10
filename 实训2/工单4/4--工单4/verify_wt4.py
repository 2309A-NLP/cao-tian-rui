# -*- coding: utf-8 -*-
"""工单4完整性验证脚本"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 50)
print("工单4 完整性验证")
print("=" * 50)

# 1. answers.jsonl 覆盖情况
print("\n[1] answers.jsonl 检查")
with open('outputs/answers.jsonl', encoding='utf-8') as f:
    answers = [json.loads(l) for l in f if l.strip()]
ids = {a['id'] for a in answers}
missing = set(range(1000)) - ids
print(f"  总条数: {len(answers)}")
print(f"  ID范围: 0 ~ {max(ids)}")
print(f"  缺失ID: {missing if missing else '无，全部覆盖'}")

# 答案分类
normal, prospectus, nodata, error = [], [], [], []
for a in answers:
    ans = a.get('answer', '')
    if any(k in ans for k in ['无法理解', 'Connection error', '429', 'Error code']):
        error.append(a['id'])
    elif ans == '__PROSPECTUS__':
        prospectus.append(a['id'])
    elif any(k in ans for k in ['未查询到相关数据', '招股书文本内容']):
        nodata.append(a['id'])
    else:
        normal.append(a['id'])
print(f"  正常DB答案: {len(normal)}")
print(f"  招股书跳过: {len(prospectus)}")
print(f"  无数据:     {len(nodata)}")
print(f"  API错误:    {len(error)}")

# 2. 工单6 Tool 接口
print("\n[2] 工单6 Tool 接口检查")
from fund_agent import fund_db_tool, get_agent, classify_question
if fund_db_tool:
    print(f"  fund_db_tool: OK")
    print(f"  Tool.name: {fund_db_tool.name}")
    print(f"  Tool.func: {fund_db_tool.func.__name__}")
else:
    print("  fund_db_tool: 未初始化（langchain未安装？）")

# 3. 预分类器
print("\n[3] 预分类器测试")
cases = [
    ('20210105综合金融涨跌幅最大股票', 'db'),
    ('云南沃森生物竞争优势是什么', 'prospectus'),
    ('景顺长城中短债前三大债券持仓', 'db'),
    ('正式在册员工人数是多少', 'prospectus'),
    ('基金规模变动情况20210331', 'db'),
]
all_pass = True
for q, expect in cases:
    got = classify_question(q)
    ok = got == expect
    all_pass = all_pass and ok
    print(f"  {'OK' if ok else 'FAIL'} expect={expect} got={got} | {q}")
print(f"  预分类器: {'全部通过' if all_pass else '有失败项'}")

# 4. 工单6调用示例说明
print("\n[4] 工单6调用方式")
print("  方式A（LangChain Tool）：")
print("    from src.fund_agent import fund_db_tool")
print("    answer = fund_db_tool.run('你的问题')")
print("  方式B（直接调用）：")
print("    from src.fund_agent import get_agent")
print("    answer = get_agent().query('你的问题')")
print("  注意：返回 '__PROSPECTUS__' 时说明是招股书问题，转给工单5处理")

print("\n" + "=" * 50)
print("验证完成")
print("=" * 50)
