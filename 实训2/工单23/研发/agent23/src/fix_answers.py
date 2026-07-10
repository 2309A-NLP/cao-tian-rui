"""
答案质量修复脚本（离线后处理工具）。

功能：
  扫描 answer.jsonl，识别低质量答案（原始搜索片段、过长、含降级语等），
  对每个低质量答案调用 LLM 重新提取精确答案，
  最终将修复后的结果写入 answer_fixed.jsonl。

使用场景：
  批量评测（run_eval.py）完成后，运行此脚���对结果做二次清洗，
  提升提交给评测平台的答案质量。

用法：
  python fix_answers.py
"""
import json   # 标准库：JSON 读写，用于解析 JSONL 文件
import sys    # 标准库：系统接口，用于修改模块搜索路径和写 stderr
import os     # 标准库：路径操作
import re     # 标准库：正则表达式（本文件实际未直接用，导入备用）

# 将 src/ 目录加入搜索路径，使得下方的本地模块可以直接 import
sys.path.insert(0, os.path.dirname(__file__))

from config import Config      # 全局配置（API Key 等，通过 llm_client 间接使用）
from llm_client import chat    # LLM 调用函数：发送消息给 Qwen，返回文本

# ── 文件路径配置 ──────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")  # 项目根目录（src/ 的上级）
QUESTION_FILE = os.path.join(ROOT, "question.jsonl")  # 原始题目文件（含 id 和 question）
ANSWER_FILE   = os.path.join(ROOT, "answer.jsonl")    # 批量评测生成的答案文件（输入）
OUTPUT_FILE   = os.path.join(ROOT, "answer_fixed.jsonl")  # 修复后输出的答案文件


def load_jsonl(path):
    """
    读取 JSONL 文件（每行一个 JSON 对象）并返回列表。

    :param path: JSONL 文件路径
    :return:     解析后的字典列表
    """
    with open(path, encoding="utf-8") as f:
        # 过滤空行（strip() 后为空时跳过），对非空行做 JSON 解析
        return [json.loads(l) for l in f if l.strip()]


def is_bad(answer: str) -> bool:
    """
    判断答案是否为低质量（需要 LLM 修复）。

    判断条件（满足任一即为低质量）：
    - 答案为空
    - 以 "[1]" 开头（搜索结果的原始索引格式，说明答案是未处理的搜索片段）
    - 超过 200 字（过长，说明 LLM 给出的是解释而非精确答案）
    - 包含降级兜底短语（"无法确定"、"没有找到"等）
    - 以句号结尾且含"是"/"为"等谓语词（解释性句子，非精确答案）

    :param answer: 待检查的答案字符串
    :return:       True 表示质量差，需要 LLM 修复
    """
    if not answer or len(answer) < 1:  # 空答案
        return True
    if answer.startswith("[1]"):        # 以 "[1]" 开头说明是搜索结果原文片段
        return True
    if len(answer) > 200:               # 明显过长，说明答案包含解释
        return True

    # 常见降级/兜底短语列表
    bad_phrases = [
        "无法确定", "没有找到", "需要更多", "并非任何", "根据现有",
        "Unable to", "cannot determine",
    ]
    if any(p in answer for p in bad_phrases):  # 含降级兜底语
        return True

    # 以中文句号结尾且含谓语词（说明是解释性句子而非精确答案）
    if answer.endswith("。") and any(k in answer for k in ["是", "为", "名称", "官方"]):
        return True

    return False  # 通过所有检查，质量可接受


def fix_answer(question: str, raw_answer: str) -> str:
    """
    调用 LLM 从低质量原始答案中提取精确答案。

    :param question:   原始问题（用于 LLM 理解答案应该是什么类型）
    :param raw_answer: 低质量的原始答案（可能含搜索片段或冗长解释）
    :return:           LLM 提取的精确答案；LLM 失败时返回原始答案
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an answer extractor. Given a question and a messy raw answer "
                "(which may contain search snippets or verbose explanations), "
                "extract ONLY the precise, minimal answer. "
                "Output ONLY the answer — no explanation, no punctuation unless part of the answer. "
                "Match the language of the question."  # 语言要与问题一致
            )
        },
        {
            "role": "user",
            # raw_answer[:800] 截断防止 token 超限
            "content": f"Question: {question}\n\nRaw answer: {raw_answer[:800]}\n\nPrecise answer:"
        }
    ]
    try:
        # max_tokens=100：精确答案通常很短，100 token 足够
        return chat(messages, max_tokens=100).strip()
    except Exception as e:
        # LLM 调用失败时打印错误并返回原始答案（不修复，保留原样）
        print(f"  LLM error: {e}")
        return raw_answer  # 降级返回未修复的答案


def main():
    """
    主函数：加载题目和答案，修复低质量答案，写出结果文件。
    """
    # 加载题目列表，构建 id → question 映射字典，方便按 id 查找题目
    questions = {q["id"]: q["question"] for q in load_jsonl(QUESTION_FILE)}
    # 加载答案列表（每项含 id 和 answer）
    answers   = load_jsonl(ANSWER_FILE)

    fixed = 0     # 记录修复的答案数量
    results = []  # 存放最终结果（原样保留的 + 修复后的）

    for item in answers:
        qid = item["id"]      # 题目 ID
        ans = item["answer"]  # 当前答案

        if is_bad(ans):  # 判断是否需要修复
            q = questions.get(qid, "")          # 获取对应题目（找不到时用空字符串）
            new_ans = fix_answer(q, ans)        # 调 LLM 修复
            # 将修复信息写入 stderr（避免污染 stdout 的结果输出）
            sys.stderr.buffer.write(f"[{qid:>3}] FIXED\n".encode("utf-8"))
            results.append({"id": qid, "answer": new_ans})  # 保存修复后的答案
            fixed += 1  # 修复计数 +1
        else:
            results.append(item)  # 质量OK，保留原样

    # 按题号升序排序（确保输出顺序与题目顺序一致）
    results.sort(key=lambda x: x["id"])

    # 写出 JSONL 格式结果文件（每行一个 JSON 对象）
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            # ensure_ascii=False 保留中文字符；每行末尾加换行
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印汇总信息
    print(f"\n完成：修复 {fixed} 条，输出 → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()  # 仅作为脚本直接运行时执行 main()
