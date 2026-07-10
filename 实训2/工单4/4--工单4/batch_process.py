"""
batch_process.py - 批量处理 question.jsonl，生成 answers.jsonl
支持断点续跑：已有答案的 id 自动跳过
"""
import json
import logging
import sys
import time
import os
from pathlib import Path

# 将 src 加入路径（使 src 内模块可直接 import，不带 src. 前缀）
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import QUESTION_FILE, OUTPUT_FILE, REQUEST_INTERVAL
from fund_agent import get_agent

_ROOT = Path(__file__).parent
_LOG_FILE = _ROOT / "outputs" / "batch.log"
_ROOT.joinpath("outputs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_questions(path: Path) -> list[dict]:
    """读取 question.jsonl"""
    if not path.exists():
        raise FileNotFoundError(f"问题文件不存在：{path}")
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    logger.info(f"共加载 {len(questions)} 道题目")
    return questions


def load_existing_answers(path: Path) -> dict[int, str]:
    """读取已有答案（断点续跑用）"""
    if not path.exists():
        return {}
    answered = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if "answer" in item:
                    answered[item["id"]] = item["answer"]
    logger.info(f"已有 {len(answered)} 道题的答案，将跳过")
    return answered


def process_all(
    question_file: Path = QUESTION_FILE,
    output_file: Path = OUTPUT_FILE,
):
    """批量处理所有题目"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    questions = load_questions(question_file)
    existing = load_existing_answers(output_file)
    agent = get_agent()

    total = len(questions)
    skipped = 0
    success = 0
    failed = 0
    consecutive_api_errors = 0  # 连续API错误计数（熔断用）
    MAX_CONSECUTIVE_ERRORS = 10  # 连续10次API错误就熔断停止

    def is_api_error(ans: str) -> bool:
        """判断答案是否为API错误（非SQL逻辑错误）"""
        return any(k in ans for k in ["无法理解", "Connection error", "429", "Error code"])

    def is_prospectus(ans: str) -> bool:
        """招股书类问题，规则直接跳过无需重跑"""
        return ans == "__PROSPECTUS__"

    # 追加模式写入（断点续跑）
    with open(output_file, "a", encoding="utf-8") as out_f:
        for i, item in enumerate(questions):
            qid = item["id"]
            question = item["question"]

            # 断点续跑：跳过已有答案的题目
            if qid in existing:
                skipped += 1
                continue

            logger.info(f"[{i+1}/{total}] ID={qid} 处理中...")

            try:
                answer = agent.query(question)
                result = {"id": qid, "question": question, "answer": answer}
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                success += 1
                logger.info(f"[{i+1}/{total}] ID={qid} ✓ 答案：{answer[:60]}...")

                # 熔断检测：检查答案是否为API错误
                if is_api_error(answer):
                    consecutive_api_errors += 1
                    logger.warning(f"⚠ 连续API错误计数: {consecutive_api_errors}/{MAX_CONSECUTIVE_ERRORS}")
                else:
                    consecutive_api_errors = 0  # 有成功就重置

            except Exception as e:
                answer = f"查询失败：{e}"
                result = {"id": qid, "question": question, "answer": answer}
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                failed += 1
                consecutive_api_errors += 1
                logger.error(f"[{i+1}/{total}] ID={qid} ✗ 错误：{e}")

            # 熔断：连续N次API错误时停止
            if consecutive_api_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error(
                    f"\n🛑 熔断触发！连续 {MAX_CONSECUTIVE_ERRORS} 次API错误，已停止运行。\n"
                    f"可能原因：API限流持续、网络中断、API余额用尽\n"
                    f"请检查后重新运行 python batch_process.py（会自动跳过已完成）"
                )
                break

            # 限流间隔（本题实际处理了才等，跳过的不等）
            if i < total - 1:
                time.sleep(REQUEST_INTERVAL)

    logger.info(
        f"\n=== 批量处理完成 ===\n"
        f"总题数: {total}\n"
        f"成功:   {success}\n"
        f"失败:   {failed}\n"
        f"跳过:   {skipped}\n"
        f"结果文件: {output_file}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量处理基金问答题目")
    parser.add_argument("--questions", type=Path, default=QUESTION_FILE, help="问题文件路径")
    parser.add_argument("--output",    type=Path, default=OUTPUT_FILE,   help="输出文件路径")
    parser.add_argument("--limit",     type=int,  default=0,             help="只处理前N题（测试用，0=全部）")
    args = parser.parse_args()

    if args.limit > 0:
        logger.info(f"测试模式：只处理前 {args.limit} 道题")
        qs = load_questions(args.questions)[: args.limit]
        # 写入临时文件
        tmp = args.questions.parent / "question_test.jsonl"
        with open(tmp, "w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        process_all(question_file=tmp, output_file=args.output)
    else:
        process_all(question_file=args.questions, output_file=args.output)
