# -*- coding: utf-8 -*-
"""
清除 answers.jsonl 中的错误答案，让 batch_process.py 重新处理这些题
"""
import json, shutil
from pathlib import Path

answers_file = Path(__file__).parent / "outputs" / "answers.jsonl"
backup_file  = Path(__file__).parent / "outputs" / "answers_backup.jsonl"

# 备份原文件
shutil.copy(answers_file, backup_file)
print(f"已备份原文件到: {backup_file}")

kept, removed = [], []

with open(answers_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        a = item.get("answer", "")
        if any(k in a for k in ["无法理解", "Connection error", "429", "Error code"]):
            removed.append(item["id"])
        else:
            kept.append(item)

# 写回只保留正常答案
with open(answers_file, "w", encoding="utf-8") as f:
    for item in kept:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"保留答案: {len(kept)} 条")
print(f"清除错误: {len(removed)} 条（将被重新处理）")
print(f"\n现在运行 python batch_process.py 即可重新处理这 {len(removed)} 道题")
