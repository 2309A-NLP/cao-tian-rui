# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
RAGAS 评测器 - 简化版

负责：
- 对 RAG 系统进行离线评估
- 使用预设问题测试回答质量和来源
- 生成评估报告（JSON）

⚠️ 常改动的地方：
1. 评估方法（evaluate）中的默认角色（role="lawyer"）
2. 生成报告时的输出路径（output_path）
3. 可扩展更多评估指标（如答案相关性、忠实度等）

⚠️ 注意事项：
1. 当前实现为简化版，仅记录问题、答案和来源，未计算 RAGAS 标准指标（faithfulness, answer_relevancy 等）
2. rag_instance.chat 方法调用时传入 user_id=1, username="test"（硬编码），可能需要根据实际调整
3. 异常捕获仅记录错误字符串，未向上抛出，便于批量执行不受个别问题中断
"""

import json
from typing import List, Dict
from datetime import datetime


class RAGEvaluator:
    """RAG 评估器"""

    def __init__(self):
        self.results = []

    def evaluate(self, rag_instance, questions: List[str], role: str = "lawyer") -> Dict:
        """
        评估 RAG 系统
        参数:
            rag_instance: RAG 引擎实例（必须具备 chat 方法）
            questions: 测试问题列表
            role: 使用的角色类型，默认 "lawyer"
        返回:
            包含总数和每条结果（含问题、答案、来源或错误）的字典
        """
        results = []
        for q in questions:
            try:
                # ⚠️ 常改动：user_id 和 username 当前硬编码，若需要测试不同用户可改为参数
                resp = rag_instance.chat(1, "test", q, role)
                results.append({
                    "question": q,
                    "answer": resp.get("answer", ""),
                    "sources": resp.get("sources", [])
                })
            except Exception as e:
                results.append({"question": q, "error": str(e)})

        self.results = results
        return {"total": len(results), "results": results}

    def generate_report(self, output_path: str = "rag_evaluation_report.json"):
        """将评估结果保存为 JSON 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)