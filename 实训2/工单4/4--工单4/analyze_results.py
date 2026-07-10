# -*- coding: utf-8 -*-
import json
from pathlib import Path

answers_file = Path(__file__).parent / "outputs" / "answers.jsonl"

answers, errors, no_data, prospectus_skip, normal = [], [], [], [], []

with open(answers_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        answers.append(item)
        a = item.get("answer", "")
        if any(k in a for k in ["无法理解", "Connection error", "429", "Error code"]):
            errors.append(item["id"])
        elif a == "__PROSPECTUS__":
            prospectus_skip.append(item["id"])
        elif any(k in a for k in ["未查询到相关数据", "招股书文本内容"]):
            no_data.append(item["id"])
        else:
            normal.append(item["id"])

print(f"总答案数:           {len(answers)}")
print(f"正常DB答案:         {len(normal)} 个")
print(f"招股书跳过(规则):   {len(prospectus_skip)} 个  ← 零API成本")
print(f"无数据(API处理后):  {len(no_data)} 个")
print(f"API错误需重跑:      {len(errors)} 个")
print()
print(f"API错误的ID（前30个）: {errors[:30]}")
print()

# 写出需要重跑的ID列表
retry_file = Path(__file__).parent / "outputs" / "retry_ids.txt"
with open(retry_file, "w", encoding="utf-8") as f:
    for eid in errors:
        f.write(str(eid) + "\n")
print(f"需重跑ID已写入: {retry_file}")
