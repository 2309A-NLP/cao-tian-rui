"""
========================================
本地数据准备脚本（用于生成微调数据集）
功能：
1. 解析PDF并分块
2. 调用大模型API生成问答对（anchor-positive）
3. 使用本地bge模型挖掘难负例，生成三元组
4. 输出 train_triplets.jsonl 和 eval_queries.jsonl
========================================
"""

import os
import json
import re
import torch
import random
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util


# ============================================================
# 第一步：PDF解析 + 文本分块
# ============================================================
def parse_and_chunk_pdfs(folder_path: str, chunk_size: int = 512, overlap: int = 64) -> List[Dict]:
    """
    解析文件夹中的所有PDF，进行固定长度分块（带重叠）
    参数:
        folder_path: 存放PDF的文件夹路径
        chunk_size: 每块字符数（建议512）
        overlap: 重叠字符数（建议64）
    返回:
        chunks列表，每个元素格式: {"id": int, "text": str}
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("请安装 pypdf: pip install pypdf")

    print(f"开始解析文件夹: {folder_path}")
    all_chunks = []
    chunk_id = 0

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(folder_path, filename)
        print(f"  正在解析: {filename}")

        try:
            reader = PdfReader(pdf_path)
            full_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # 基础清洗：合并空白、移除多余换行
                    text = re.sub(r'\s+', ' ', text).strip()
                    full_text.append(text)

            document_text = " ".join(full_text)

            # 滑动窗口分块
            stride = chunk_size - overlap
            for i in range(0, len(document_text), stride):
                chunk = document_text[i:i + chunk_size]
                if len(chunk.strip()) > 50:  # 过滤过短的块
                    all_chunks.append({"id": chunk_id, "text": chunk.strip()})
                    chunk_id += 1

        except Exception as e:
            print(f"  解析失败: {filename}, 错误: {e}")
            continue

    print(f"✅ 分块完成，共生成 {len(all_chunks)} 个文本块")
    return all_chunks


# ============================================================
# 第二步：调用大模型API生成 (anchor, positive) 正例对
# ============================================================
def generate_qa_pairs(
        chunks: List[Dict],
        api_key: str,
        api_url: str = "https://api.deepseek.com/v1/chat/completions",
        model_name: str = "deepseek-chat",
        questions_per_chunk: int = 2
) -> List[Dict]:
    """
    调用大模型API为每个文本块生成问题
    参数:
        chunks: 文本块列表
        api_key: DeepSeek API密钥
        api_url: API地址
        model_name: 模型名称
        questions_per_chunk: 每个块生成几个问题
    返回:
        qa_pairs列表，每项: {"anchor": "问题", "positive": "原文本"}
    """
    print(f"开始生成问答对，共 {len(chunks)} 个文本块，每块生成 {questions_per_chunk} 个问题")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt_template = """你是一个招股说明书分析师。以下是一个段落，请生成{num}个投资者可能提出的问题。

段落：{chunk_text}

要求：
1. 问题必须基于段落内容，使用专业金融术语
2. 问题应体现投资者实际关心角度
3. 每行一个问题，不要编号，不要前缀

只输出问题，每行一个，不要输出其他内容。"""

    qa_pairs = []

    for idx, chunk in enumerate(chunks):
        if idx % 50 == 0:
            print(f"  进度: {idx}/{len(chunks)}")

        prompt = prompt_template.format(num=questions_per_chunk, chunk_text=chunk["text"])

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 500
                },
                timeout=30
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                questions = [q.strip() for q in content.strip().split("\n") if q.strip()]
                for q in questions:
                    qa_pairs.append({
                        "anchor": q,
                        "positive": chunk["text"]
                    })
            else:
                print(f"  API调用失败 (chunk {idx}): {response.status_code}")

        except Exception as e:
            print(f"  chunk {idx} 生成失败: {e}")
            continue

    print(f"✅ 问答对生成完成，共 {len(qa_pairs)} 条")
    return qa_pairs


# ============================================================
# 第三步：挖掘难负例（使用本地bge模型）
# ============================================================
def mine_hard_negatives(
        qa_pairs: List[Dict],
        all_chunks: List[Dict],
        model_path: str,
        top_k: int = 10,
        negatives_per_query: int = 2
) -> List[Dict]:
    """
    使用本地bge模型挖掘难负例
    参数:
        qa_pairs: 正例对列表 [{"anchor": "...", "positive": "..."}]
        all_chunks: 所有文本块列表（用于检索负例）
        model_path: 本地 bge-base-zh-v1.5 模型路径
        top_k: 检索相似文档的数量
        negatives_per_query: 每个 anchor 保留多少个 negative
    返回:
        三元组列表 [{"anchor": "...", "positive": "...", "negative": "..."}]
    """
    print("开始难负例挖掘...")
    print(f"  模型路径: {model_path}")

    # 1. 加载本地模型
    try:
        model = SentenceTransformer(model_path)
        if torch.cuda.is_available():
            model = model.cuda()
            print("  使用 GPU 进行编码")
        else:
            print("  使用 CPU 进行编码（较慢）")
    except Exception as e:
        print(f"  模型加载失败: {e}")
        return []

    # 2. 构建文档库映射
    chunk_texts = [c["text"] for c in all_chunks]
    chunk_ids = [str(c["id"]) for c in all_chunks]
    # 建立 text -> id 的快速查找表
    text_to_id = {c["text"]: str(c["id"]) for c in all_chunks}

    # 3. 编码所有文档块
    print(f"  正在编码 {len(chunk_texts)} 个文档块...")
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=True)

    # 4. 为每个 query 挖掘难负例
    triplets = []
    total = len(qa_pairs)

    for idx, qa in enumerate(qa_pairs):
        if idx % 100 == 0:
            print(f"  进度: {idx}/{total}")

        anchor = qa["anchor"]
        positive_text = qa["positive"]

        # 查找 positive 对应的 chunk_id
        positive_id = text_to_id.get(positive_text)
        if positive_id is None:
            continue  # 理论上应该存在，若不存在则跳过

        # 编码 query
        query_emb = model.encode([anchor], convert_to_tensor=True)

        # 计算相似度
        scores = util.cos_sim(query_emb, chunk_embeddings)[0]
        top_indices = torch.argsort(scores, descending=True)[:top_k].tolist()

        # 筛选负例（排除 positive 自身）
        neg_count = 0
        for chunk_idx in top_indices:
            chunk_id = chunk_ids[chunk_idx]
            if chunk_id == positive_id:
                continue
            triplets.append({
                "anchor": anchor,
                "positive": positive_text,
                "negative": chunk_texts[chunk_idx]
            })
            neg_count += 1
            if neg_count >= negatives_per_query:
                break

    print(f"✅ 难负例挖掘完成，生成 {len(triplets)} 条三元组")
    return triplets


# ============================================================
# 主函数（本地数据准备）
# ============================================================
def main():
    # ========== 配置参数（请根据实际情况修改）==========
    PDF_FOLDER = "./data"  # PDF文件夹路径
    OUTPUT_DIR = "./output"  # 输出目录
    DEEPSEEK_API_KEY = "xxx"  # DeepSeek API密钥（需要申请）

    # 本地 bge-base-zh-v1.5 模型路径（必须正确）
    LOCAL_BGE_PATH = r"F:\4--专业所有安装的软件及改动设置\2-3--专高3\2：bge-base-zh-v1.5\bge-base-zh-v1.5"

    # 分块参数
    CHUNK_SIZE = 512
    OVERLAP = 64

    # 问答对生成参数
    QUESTIONS_PER_CHUNK = 2  # 每个文本块生成几个问题

    # 难负例挖掘参数
    TOP_K = 10  # 检索相似文档数量
    NEGATIVES_PER_QUERY = 2  # 每个 query 保留几个负例

    # 训练/测试集分割比例
    TRAIN_RATIO = 0.8

    # ========== 执行流程 ==========
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. PDF解析与分块
    chunks = parse_and_chunk_pdfs(PDF_FOLDER, CHUNK_SIZE, OVERLAP)
    chunks_file = os.path.join(OUTPUT_DIR, "chunks.jsonl")
    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"💾 分块结果已保存: {chunks_file}")

    # 2. 生成问答对（需要API密钥）
    if DEEPSEEK_API_KEY == "your-deepseek-api-key":
        print("⚠️ 请先配置 DEEPSEEK_API_KEY，跳过问答对生成")
        # 尝试加载已有数据
        qa_file = os.path.join(OUTPUT_DIR, "qa_pairs.jsonl")
        if os.path.exists(qa_file):
            with open(qa_file, "r", encoding="utf-8") as f:
                qa_pairs = [json.loads(line) for line in f]
            print(f"✅ 从已有文件加载问答对: {len(qa_pairs)} 条")
        else:
            raise ValueError("请配置 DEEPSEEK_API_KEY 或提供已有的 qa_pairs.jsonl")
    else:
        qa_pairs = generate_qa_pairs(
            chunks, DEEPSEEK_API_KEY,
            questions_per_chunk=QUESTIONS_PER_CHUNK
        )
        qa_file = os.path.join(OUTPUT_DIR, "qa_pairs.jsonl")
        with open(qa_file, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        print(f"💾 问答对已保存: {qa_file}")

    # 3. 分割训练集和测试集
    random.seed(42)
    random.shuffle(qa_pairs)
    split_idx = int(len(qa_pairs) * TRAIN_RATIO)
    train_qa = qa_pairs[:split_idx]
    eval_qa = qa_pairs[split_idx:]

    print(f"📊 训练集问答对: {len(train_qa)} 条")
    print(f"📊 测试集问答对: {len(eval_qa)} 条")

    # 保存测试集（用于后续评估）
    eval_file = os.path.join(OUTPUT_DIR, "eval_queries.jsonl")
    with open(eval_file, "w", encoding="utf-8") as f:
        for qa in eval_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"💾 测试集已保存: {eval_file}")

    # 4. 难负例挖掘（仅在训练集上进行）
    print("\n" + "=" * 50)
    triplets = mine_hard_negatives(
        train_qa, chunks, LOCAL_BGE_PATH,
        top_k=TOP_K,
        negatives_per_query=NEGATIVES_PER_QUERY
    )

    # 保存三元组训练数据
    triplets_file = os.path.join(OUTPUT_DIR, "train_triplets.jsonl")
    with open(triplets_file, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"💾 三元组训练数据已保存: {triplets_file}")

    # 5. 输出统计信息
    print("\n" + "=" * 50)
    print("✅ 本地数据准备完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📄 生成的文件:")
    print(f"   - chunks.jsonl         : {len(chunks)} 个文本块")
    print(f"   - qa_pairs.jsonl       : {len(qa_pairs)} 条问答对")
    print(f"   - eval_queries.jsonl   : {len(eval_qa)} 条测试集")
    print(f"   - train_triplets.jsonl : {len(triplets)} 条三元组")
    print("\n下一步:")
    print("1. 将 output 文件夹整体打包上传到恒源云")
    print("2. 在恒源云上使用 train_triplets.jsonl 进行模型微调")
    print("3. 使用 eval_queries.jsonl 评估微调效果")


if __name__ == "__main__":
    main()