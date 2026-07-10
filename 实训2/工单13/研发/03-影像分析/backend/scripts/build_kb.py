"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
知识库构建：从 SLAKE train.json 抽取中文 QA 对，写入 ChromaDB。

SLAKE 数据集是一个医疗视觉问答数据集，包含中英文问答对和对应的医学影像。
本脚本从中提取中文 QA 数据，将每条 QA 组合成自然语言文档后写入
ChromaDB（一种向量数据库），用于后续 RAG（检索增强生成）任务。

策略：
- 只用中文 QA（q_lang=zh），无需翻译
- 每条 QA 拼成一条自然语言"影像医学问答知识"文档
- 元数据保留 modality/location/answer_type/content_type，便于过滤
- 去重：按 (question + answer) 去重
"""

# json：Python 内置模块，用于读取和解析 JSON 格式文件
import json

# sys：Python 内置模块，用于操作 Python 解释器的运行时环境（此处用于修改模块搜索路径和退出程序）
import sys

# Counter：Python 内置 collections 模块中的计数器类，用于统计可哈希对象出现次数
from collections import Counter

# pathlib.Path：Python 内置模块，提供面向对象的文件系统路径操作（比 os.path 更现代）
from pathlib import Path

# 获取当前脚本的绝对路径，然后向上两级得到项目根目录（backend/）
# __file__ 是当前脚本的路径，.resolve() 转为绝对路径，.parent.parent 向上两级
ROOT = Path(__file__).resolve().parent.parent

# 将项目根目录插入 Python 模块搜索路径的首位，使 src 包可以被正确导入
sys.path.insert(0, str(ROOT))

# 从本项目的 rag_store 模块导入向量库操作函数：
# add_documents：批量写入文档到向量库
# count_docs：查询向量库中的文档总数
from src.rag_store import add_documents, count_docs  # noqa: E402

# SLAKE 数据集 JSON 文件的路径（需要用户提前下载并放置在 data/ 目录下）
SLAKE_JSON = ROOT / "data" / "slake_train.json"


def load_slake_zh() -> list[dict]:
    """
    读取 SLAKE 数据集 JSON 文件，筛选出中文问答条目。

    返回值：
        list[dict]：中文 QA 条目列表，每个元素是一个包含 question/answer/modality 等字段的字典
    """
    # 读取整个 JSON 文件内容并解析为 Python 对象（列表）
    data = json.loads(SLAKE_JSON.read_text(encoding="utf-8"))

    # 过滤：只保留 q_lang 字段为 "zh"（中文）的条目
    zh = [x for x in data if x.get("q_lang") == "zh"]
    return zh


def make_doc(item: dict, idx: int) -> dict:
    """
    将一条 SLAKE QA 条目转换为向量库所需的文档格式。

    参数：
        item (dict)：SLAKE 数据集中的一条原始 QA 记录
        idx (int)：当前条目在列表中的顺序索引（用于生成唯一 ID）

    返回值：
        dict：包含 id、text、metadata 三个字段的文档字典
    """
    # 提取问题文本，去除首尾空白字符
    q = item["question"].strip()

    # 提取答案文本，转为字符串（因为答案可能是数字类型），去除首尾空白
    a = str(item["answer"]).strip()

    # 提取影像模态（如 X-Ray、CT、MRI），若不存在则默认"未知"
    modality = item.get("modality", "未知")

    # 提取解剖部位（如 Chest、Abdomen），若不存在则默认"未知"
    location = item.get("location", "未知")

    # 提取内容类型（如 abnormality、organ），若不存在则默认空字符串
    content_type = item.get("content_type", "")

    # 将 QA 信息拼接成自然语言格式的文本，供向量嵌入使用
    # 使用【】标记各字段，便于人工阅读和模型理解
    text = (
        f"【模态】{modality}\n"      # 成像方式（X线/CT/MRI等）
        f"【部位】{location}\n"      # 检查部位（胸部/腹部等）
        f"【问题】{q}\n"             # 原始问题
        f"【答案】{a}"               # 对应答案
    )

    # 返回标准文档格式
    return {
        # 文档唯一 ID：使用 qid 字段（若存在）+ 序号，确保唯一性
        "id": f"slake-{item.get('qid', idx)}-{idx}",
        "text": text,               # 用于向量嵌入的文本内容
        "metadata": {               # 附加元数据，可用于向量库过滤检索
            "source": "SLAKE",          # 数据来源标识
            "modality": modality,        # 影像模态
            "location": location,        # 解剖部位
            "answer_type": item.get("answer_type", ""),   # 答案类型（open/closed）
            "content_type": content_type,                  # 内容类型
            "title": f"{modality} {location} - {content_type}",  # 展示用标题
        },
    }


def main():
    """
    主函数：加载 SLAKE 中文数据 → 去重 → 分批写入 ChromaDB 向量库。
    """
    # 检查数据文件是否存在，不存在则打印错误信息并退出
    if not SLAKE_JSON.exists():
        print(f"[error] 缺文件: {SLAKE_JSON}")
        sys.exit(1)  # 以退出码 1（异常）退出程序

    # 加载中文 QA 数据
    zh_items = load_slake_zh()
    print(f"[load] SLAKE 中文 QA: {len(zh_items)} 条")

    # ── 去重处理 ──
    # seen：用集合存储已处理的 (问题, 答案) 元组，集合的查找是 O(1) 时间复杂度
    seen: set[tuple] = set()
    docs = []  # 存储去重后的文档列表
    for i, x in enumerate(zh_items):
        # 以 (问题文本, 答案文本) 作为去重键
        key = (x["question"].strip(), str(x["answer"]).strip())
        if key in seen:
            continue  # 已存在则跳过此条
        seen.add(key)           # 将新键加入集合
        docs.append(make_doc(x, i))  # 转换为文档格式并添加

    print(f"[dedup] 去重后: {len(docs)} 条")

    # ── 统计分布（便于了解数据组成）──
    # Counter 统计每种 modality 出现次数
    mods = Counter(d["metadata"]["modality"] for d in docs)
    # Counter 统计每种 location 出现次数
    locs = Counter(d["metadata"]["location"] for d in docs)
    print(f"[dist] modality={dict(mods)}")   # 打印模态分布
    print(f"[dist] location={dict(locs)}")   # 打印部位分布

    # ── 分批写入向量库 ──
    BATCH = 200   # 每批写入 200 条，避免内存占用过大或单次请求超时
    total = 0     # 记录已成功写入的总条数
    for i in range(0, len(docs), BATCH):
        batch = docs[i : i + BATCH]   # 切片取当前批次的文档
        n = add_documents(batch)       # 调用向量库写入函数，返回实际写入条数
        total += n
        print(f"[upsert] {i + n}/{len(docs)}")  # 打印进度

    # 打印最终向量库中的文档总数
    print(f"\n[done] 知识库总条数: {count_docs()}")


# 仅当直接运行此脚本时执行 main()（被 import 时不执行）
if __name__ == "__main__":
    main()
