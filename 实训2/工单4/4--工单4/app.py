# -*- coding: utf-8 -*-
"""
工单4：基金数据问答智能体 - Gradio 前端
"""
import sys, time, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
logging.basicConfig(level=logging.WARNING)

import gradio as gr
from fund_agent import get_agent
from db_utils import get_full_schema

# 预加载（触发 schema lru_cache 和单例初始化，避免第一次提问时卡顿）
get_full_schema()
get_agent()

EXAMPLES = [
    "在20210105，中信行业分类划分的一级行业为综合金融行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？百分数保留两位小数。",
    "景顺长城中短债债券C基金在20210331的季报里，前三大持仓占比的债券名称是什么？",
    "请帮我查询出20210415日，建筑材料一级行业涨幅超过5%（不包含）的股票数量。",
    "2019年中期报告里，华夏基金管理有限公司管理的基金中，机构投资者持有份额比个人投资者多的基金有多少只？",
    "请查询在2021年度，688338股票涨停天数？（收盘价/昨日收盘价-1）>=9.8% 视作涨停。",
    "在20201022，按照中信行业分类的行业划分标准，哪个一级行业的A股公司数量最多？",
]


def answer_question(question: str):
    """流式生成器：每步 yield (answer_box, sql_box, type_box)"""
    if not question.strip():
        yield "请输入问题", "", "未知"
        return

    t0 = time.time()
    answer_box = ""
    sql_box = ""

    for evt, content in get_agent().query_stream(question):
        if evt == "status":
            yield answer_box, sql_box, content
        elif evt == "sql":
            sql_box = content
            yield answer_box, sql_box, "正在执行查询…"
        elif evt == "answer":
            total = time.time() - t0
            if content == "__PROSPECTUS__":
                yield (
                    "该问题涉及招股说明书内容，不在基金结构化数据库范围内。\n请使用招股书问答工具（工单5）查询。",
                    "无需生成SQL（预分类识别为招股书问题）",
                    f"招股书类（共 {total:.1f}s）",
                )
            else:
                yield content, sql_box, f"数据库类（共 {total:.1f}s）"
        elif evt == "error":
            yield f"查询失败：{content}", sql_box, f"失败（{time.time()-t0:.1f}s）"


with gr.Blocks(title="基金数据问答智能体") as demo:
    gr.Markdown("""
    # 基金数据问答智能体
    **工单4** · 基于 NL2SQL 技术，查询基金/股票/债券结构化数据（2019-2021年）

    数据来源：博金杯比赛数据库，包含10张数据表：基金基本信息、股票持仓明细、债券持仓明细、日行情等
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="输入问题",
                placeholder="例如：在20210105，综合金融行业涨跌幅最大的股票代码是？",
                lines=3,
            )
            submit_btn = gr.Button("查询", variant="primary")

        with gr.Column(scale=1):
            type_output = gr.Textbox(label="问题类型", interactive=False)

    answer_output = gr.Textbox(label="回答", lines=4, interactive=False)
    sql_output = gr.Textbox(label="生成的SQL语句", lines=5, interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=question_input,
        label="示例问题（点击填入）",
    )

    submit_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, sql_output, type_output],
    )
    question_input.submit(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, sql_output, type_output],
    )

    gr.Markdown("""
    ---
    **技术说明**：用户问题 → LLM生成SQL → SQLite执行 → LLM生成自然语言回答

    招股书类问题（如公司竞争优势、员工人数等）由预分类器识别后转交工单5处理
    """)

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
