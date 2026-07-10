# -*- coding: utf-8 -*-
"""
工单4 功能演示脚本 - 用于截图提交
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
logging.basicConfig(level=logging.WARNING)  # 只显示警告，不显示INFO日志，截图更干净

from fund_agent import get_agent, classify_question

agent = get_agent()

print("=" * 60)
print("  工单4：基金数据问答智能体 - 功能演示")
print("=" * 60)

demo_questions = [
    {
        "desc": "【A股行情查询】",
        "q": "在20210105，中信行业分类划分的一级行业为综合金融行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？百分数保留两位小数。"
    },
    {
        "desc": "【基金持仓查询】",
        "q": "景顺长城中短债债券C基金在20210331的季报里，前三大持仓占比的债券名称是什么？"
    },
    {
        "desc": "【行业统计查询】",
        "q": "请帮我查询出20210415日，建筑材料一级行业涨幅超过5%（不包含）的股票数量。"
    },
    {
        "desc": "【基金规模查询】",
        "q": "2019年中期报告里，华夏基金管理有限公司管理的基金中，机构投资者持有份额比个人投资者多的基金有多少只？"
    },
    {
        "desc": "【招股书问题识别（预分类跳过）】",
        "q": "兰州海默科技股份有限公司的正式在册员工人数是多少？"
    },
]

for i, item in enumerate(demo_questions, 1):
    print(f"\n[问题 {i}] {item['desc']}")
    print(f"Q: {item['q']}")

    qtype = classify_question(item['q'])
    if qtype == 'prospectus':
        print(f"A: [预分类识别为招股书问题，无需查询数据库，转交工单5处理]")
    else:
        t0 = time.time()
        answer = agent.query(item['q'])
        elapsed = time.time() - t0
        print(f"A: {answer}")
        print(f"   （耗时 {elapsed:.1f}s）")

print("\n" + "=" * 60)
print("  演示完成")
print("  输出文件：outputs/answers.jsonl（1000题完整答案）")
print("=" * 60)
