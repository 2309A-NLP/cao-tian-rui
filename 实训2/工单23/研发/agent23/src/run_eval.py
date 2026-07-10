"""
本地批量评测脚本（不依赖 PAI-EAS，可本地直接运行）。

功能：
  读取 question.jsonl → 多线程并发调用 process_question → 写 answer.jsonl

特点：
  - 使用 ThreadPoolExecutor 并发处理，EVAL_CONCURRENCY 控制并发数
  - 实时打印进度（当前/总数、失败数、已用时、预计剩余时间）
  - 支持 --limit 参数只跑前 N 道题，方便调试
  - 按题号排序写出，确保输出顺序与输入顺序一致

用法：
  python run_eval.py                    # 跑全部100题
  python run_eval.py --limit 5          # 只跑前5题（调试用）
  python run_eval.py --concurrency 5    # 5并发
"""
import json    # 标准库：JSON 读写，解析 JSONL 文件
import logging # 标准库：日志记录
import os      # 标准库：路径操作
import sys     # 标准库：修改模块搜索路径
import time    # 标准库：时间函数，计算总耗时和 ETA

# concurrent.futures：标准库，提供高级并发原语
# ThreadPoolExecutor - 线程池，用于多线程并发处理题目
# as_completed       - 迭代器，按完成顺序返回 Future 结果（而非提交顺序）
from concurrent.futures import ThreadPoolExecutor, as_completed

# 将 src/ 目录加入搜索路径，保证本地模块可以直接 import
sys.path.insert(0, os.path.dirname(__file__))

from config import Config, validate_config  # 全局配置 + 配置校验
from app import process_question            # 核心处理函数（预处理→ReAct→答案生成）

# 配置日志格式（简单格式，无 JSON，方便命令行查看）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"  # 时间 + 级别 + 消息
)
logger = logging.getLogger(__name__)

# ── 文件路径配置 ──────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")          # 项目根目录
QUESTION_FILE = os.path.join(_ROOT, "question.jsonl")          # 题目文件（输入）
ANSWER_FILE = os.path.join(_ROOT, "answer.jsonl")              # 答案文件（输出）


def load_questions(path: str) -> list[dict]:
    """
    从 JSONL 文件加载题目列表。
    每行一个 JSON 对象，格式：{"id": N, "question": "..."}

    :param path: JSONL 文件路径
    :return:     解析后的字典列表
    """
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()   # 去首尾空格和换行
            if line:              # 跳过空行
                questions.append(json.loads(line))  # 解析 JSON
    return questions


def run_eval(
    question_file: str = QUESTION_FILE,
    answer_file: str = ANSWER_FILE,
    concurrency: int = Config.EVAL_CONCURRENCY,  # 并发线程数，默认3
    limit: int | None = None,                    # 调试用：只跑前 N 道题；None 表示全部
):
    """
    批量评测主函数：并发处理所有题目，写出答案文件。

    :param question_file: 题目 JSONL 文件路径
    :param answer_file:   输出答案 JSONL 文件路径
    :param concurrency:   并发线程数
    :param limit:         最多处理的题目数（None 表示不限制）
    """
    # 启动前检查配置（API Key 等）
    missing = validate_config()
    if missing:
        logger.error("Config missing: %s", missing)
        sys.exit(1)  # 配置不完整时退出，避免大量失败请求

    questions = load_questions(question_file)  # 加载所有题目
    if limit:
        questions = questions[:limit]  # 调试时截取前 N 道题

    total = len(questions)  # 总题数
    logger.info("Loaded %d questions, concurrency=%d", total, concurrency)

    results: dict[int, str] = {}  # {题目ID: 答案}，按 ID 存储结果
    failed = 0                    # 失败（返回 "Unknown"）的题目数
    t_start = time.time()         # 评测开始时间

    def _task(item: dict) -> tuple[int, str]:
        """
        单道题目的处理任务（在子线程中执行）。

        :param item: 题目字典，含 "id" 和 "question" 字段
        :return:     (题目ID, 答案字符串) 元组
        """
        qid = item["id"]
        q = item["question"]
        try:
            # process_question 返回 dict，取 "answer" 字段
            answer = process_question(q)
            # process_question 返回 dict（含 answer/trace 等），只取 answer
            if isinstance(answer, dict):
                answer = answer.get("answer", "Unknown")
            return qid, answer
        except Exception as e:
            logger.error("Q%d failed: %s", qid, e)
            return qid, "Unknown"  # 任何异常都返回 "Unknown"，保证程序继续

    # 使用线程池并发处理所有题目
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # 提交所有任务，futures 是 {Future: 题目ID} 映射
        futures = {pool.submit(_task, item): item["id"] for item in questions}

        done = 0  # 已完成的题目数

        # as_completed 按完成顺序返回 Future（先完成的先返回，不按提交顺序）
        for future in as_completed(futures):
            qid, answer = future.result()  # 获取任务结果（阻塞等待直到完成）
            results[qid] = answer          # 存入结果字典
            done += 1
            if answer == "Unknown":
                failed += 1  # 统计失败数

            # 实时打印进度：当前/总数、失败数、已用时、预计剩余时间（ETA）
            elapsed = time.time() - t_start
            # ETA 估算：(已用时/已完成数) × 剩余数
            eta = (elapsed / done) * (total - done) if done > 0 else 0
            # \r 回到行首覆盖打印，end="" 不换行，实现动态进度条效果
            print(f"\r[{done}/{total}] failed={failed} elapsed={elapsed:.0f}s ETA={eta:.0f}s  ",
                  end="", flush=True)  # flush=True 立即刷新缓冲区

    print()  # 评测完成后换行（避免进度行被覆盖）

    # 按题号升序排序写出（evaluator 可能依赖 ID 顺序）
    with open(answer_file, "w", encoding="utf-8") as f:
        for qid in sorted(results.keys()):
            # ensure_ascii=False 保留中文字符
            f.write(json.dumps({"id": qid, "answer": results[qid]}, ensure_ascii=False) + "\n")

    total_time = time.time() - t_start
    logger.info("Done! %d answers written to %s (failed=%d, time=%.1fs)",
                total, answer_file, failed, total_time)


if __name__ == "__main__":
    # 命令行入口：支持参数覆盖默认配置
    import argparse
    parser = argparse.ArgumentParser(description="Batch evaluation for Research Agent")
    # --limit：只跑前 N 道题（调试用，不传则跑全部）
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run first N questions (debug)")
    # --concurrency：并发线程数（默认从 Config 读取）
    parser.add_argument("--concurrency", type=int, default=Config.EVAL_CONCURRENCY)
    # --question-file：自定义题目文件路径
    parser.add_argument("--question-file", default=QUESTION_FILE)
    # --answer-file：自定义答案输出路径
    parser.add_argument("--answer-file", default=ANSWER_FILE)
    args = parser.parse_args()

    # 用命令行参数调用 run_eval
    run_eval(
        question_file=args.question_file,
        answer_file=args.answer_file,
        concurrency=args.concurrency,
        limit=args.limit,
    )
