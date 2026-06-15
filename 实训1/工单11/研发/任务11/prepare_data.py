"""
本地数据准备脚本（用于生成微调三元组数据）
输入：两个分块 JSON 文件（数组格式）
输出：train_triplets.jsonl, eval_queries.jsonl, qa_pairs.jsonl
"""

import os
import json
import random
import torch
import requests
from sentence_transformers import SentenceTransformer, util

# ============================================================
# 配置参数（请根据实际情况修改）
# ============================================================

# 输入文件（你的两个分块产物）
CHUNK_FILE_1 = r"E:\10--agent--任务\工单产物汇总\任务11\data\chunks_招股说明书1-无水印.json"
CHUNK_FILE_2 = r"E:\10--agent--任务\工单产物汇总\任务11\data\chunks_招股说明书2--无水印.json"

# 输出目录
OUTPUT_DIR = "./output"

# DeepSeek API 配置（需要替换为真实密钥）
DEEPSEEK_API_KEY = "xxx"   # 请填写你的 API Key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 本地 bge-base-zh-v1.5 模型路径（请修改为你的实际路径）
BGE_MODEL_PATH = r"F:\4--专业所有安装的软件及改动设置\2-3--专高3\2：bge-base-zh-v1.5\bge-base-zh-v1.5"

# 数据处理参数
SAMPLE_RATIO = 0.3            # 采样比例（30%）
QUESTIONS_PER_CHUNK = 2       # 每个文本块生成的问题数
TOP_K = 10                    # 难负例检索数量
NEGATIVES_PER_QUERY = 2       # 每个 query 保留的负例数
TRAIN_RATIO = 0.8             # 训练集比例

# 缓存文件（避免重复编码文档）
CACHE_FILE = os.path.join(OUTPUT_DIR, "chunk_embeddings.pt")

# ============================================================
# 1. 加载分块文件（JSON 数组格式）
# ============================================================
def load_chunks_from_json(file_path: str):
    """加载 JSON 数组格式的分块文件，返回 chunks 列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)   # 整个数组
    chunks = []
    for item in data:
        # 提取 chunk_id 和 text（支持不同字段名）
        chunk_id = item.get("chunk_id") or item.get("id")
        text = item.get("text") or item.get("content")
        if chunk_id is not None and text:
            chunks.append({"id": str(chunk_id), "text": text.strip()})
    print(f"从 {os.path.basename(file_path)} 加载 {len(chunks)} 个分块")
    return chunks

def load_all_chunks(file_paths):
    """加载多个分块文件，合并并重新分配全局 id"""
    all_chunks = []
    global_id = 0
    for file_path in file_paths:
        chunks = load_chunks_from_json(file_path)
        for c in chunks:
            # 重新分配 id，避免重复
            all_chunks.append({"id": str(global_id), "text": c["text"]})
            global_id += 1
    print(f"合并后共 {len(all_chunks)} 个分块")
    return all_chunks

# ============================================================
# 2. 采样并生成问答对（调用 DeepSeek API）
# ============================================================
def generate_qa_pairs_sampled(chunks, sample_ratio, api_key, api_url, questions_per_chunk=2):
    """随机采样 chunks，调用 API 生成 (question, chunk_text) 对"""
    sample_size = max(1, int(len(chunks) * sample_ratio))
    sampled = random.sample(chunks, sample_size)
    print(f"随机采样 {len(sampled)} 个分块用于生成问答对 (比例 {sample_ratio*100:.0f}%)")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt_template = """你是一个招股说明书分析师。以下是一个段落，请生成{num}个投资者可能提出的问题。

段落：{chunk_text}

要求：
1. 问题必须基于段落内容，使用专业金融术语
2. 问题应体现投资者实际关心角度
3. 每行一个问题，不要编号，不要前缀

只输出问题，每行一个。"""

    qa_pairs = []
    for idx, chunk in enumerate(sampled):
        if idx % 20 == 0:
            print(f"API 生成进度: {idx}/{len(sampled)}")
        prompt = prompt_template.format(num=questions_per_chunk, chunk_text=chunk["text"])
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 500
                },
                timeout=30
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                questions = [q.strip() for q in content.split("\n") if q.strip()]
                for q in questions[:questions_per_chunk]:
                    qa_pairs.append({
                        "anchor": q,
                        "positive": chunk["text"]
                    })
            else:
                print(f"  请求失败 (chunk {idx}): HTTP {response.status_code}")
        except Exception as e:
            print(f"  请求异常 (chunk {idx}): {e}")
            continue
    print(f"共生成 {len(qa_pairs)} 条问答对")
    return qa_pairs

# ============================================================
# 3. 难负例挖掘（使用本地 bge 模型）
# ============================================================
def mine_hard_negatives(qa_pairs, all_chunks, model_path, cache_file, top_k=10, neg_per_query=2):
    """为每个 (anchor, positive) 挖掘难负例，生成三元组"""
    print("开始难负例挖掘...")
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    model = SentenceTransformer(model_path, device=device)

    # 准备文档库
    chunk_texts = [c["text"] for c in all_chunks]
    chunk_ids = [c["id"] for c in all_chunks]
    text_to_id = {c["text"]: c["id"] for c in all_chunks}

    # 编码文档块（使用缓存）
    if os.path.exists(cache_file):
        print(f"从缓存加载文档 embeddings: {cache_file}")
        chunk_embeddings = torch.load(cache_file, map_location=device)
    else:
        print(f"正在编码 {len(chunk_texts)} 个文档块...")
        chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=True)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(chunk_embeddings, cache_file)
        print(f"已保存 embeddings 到 {cache_file}")

    triplets = []
    total = len(qa_pairs)
    for idx, qa in enumerate(qa_pairs):
        if idx % 100 == 0:
            print(f"难负例挖掘进度: {idx}/{total}")
        anchor = qa["anchor"]
        positive_text = qa["positive"]
        positive_id = text_to_id.get(positive_text)
        if positive_id is None:
            continue

        # 编码 query
        query_emb = model.encode([anchor], convert_to_tensor=True)
        scores = util.cos_sim(query_emb, chunk_embeddings)[0]
        top_indices = torch.argsort(scores, descending=True)[:top_k].tolist()

        neg_count = 0
        for chunk_idx in top_indices:
            if chunk_ids[chunk_idx] == positive_id:
                continue
            triplets.append({
                "anchor": anchor,
                "positive": positive_text,
                "negative": chunk_texts[chunk_idx]
            })
            neg_count += 1
            if neg_count >= neg_per_query:
                break
    print(f"共生成 {len(triplets)} 条三元组")
    return triplets

# ============================================================
# 主函数
# ============================================================
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载并合并分块
    chunk_files = [CHUNK_FILE_1, CHUNK_FILE_2]
    chunks = load_all_chunks(chunk_files)

    # 2. 生成问答对（采样）
    if DEEPSEEK_API_KEY == "your-deepseek-api-key":
        print("⚠️ 未配置 DeepSeek API Key，将尝试加载已有 qa_pairs.jsonl")
        qa_file = os.path.join(OUTPUT_DIR, "qa_pairs.jsonl")
        if os.path.exists(qa_file):
            with open(qa_file, "r", encoding="utf-8") as f:
                qa_pairs = [json.loads(line) for line in f]
            print(f"从已有文件加载 {len(qa_pairs)} 条问答对")
        else:
            raise ValueError("请配置有效的 DEEPSEEK_API_KEY 或提供已有的 qa_pairs.jsonl")
    else:
        qa_pairs = generate_qa_pairs_sampled(
            chunks, SAMPLE_RATIO, DEEPSEEK_API_KEY, DEEPSEEK_API_URL,
            questions_per_chunk=QUESTIONS_PER_CHUNK
        )
        # 保存问答对
        qa_file = os.path.join(OUTPUT_DIR, "qa_pairs.jsonl")
        with open(qa_file, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        print(f"问答对已保存至 {qa_file}")

    # 3. 划分训练集和测试集
    random.seed(42)
    random.shuffle(qa_pairs)
    split_idx = int(len(qa_pairs) * TRAIN_RATIO)
    train_qa = qa_pairs[:split_idx]
    eval_qa = qa_pairs[split_idx:]
    print(f"训练集问答对: {len(train_qa)}，测试集问答对: {len(eval_qa)}")

    # 保存测试集（用于后续评估）
    eval_file = os.path.join(OUTPUT_DIR, "eval_queries.jsonl")
    with open(eval_file, "w", encoding="utf-8") as f:
        for qa in eval_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"测试集已保存至 {eval_file}")

    # 4. 难负例挖掘（仅训练集）
    triplets = mine_hard_negatives(
        train_qa, chunks, BGE_MODEL_PATH, CACHE_FILE,
        top_k=TOP_K, neg_per_query=NEGATIVES_PER_QUERY
    )

    # 保存三元组
    triplets_file = os.path.join(OUTPUT_DIR, "train_triplets.jsonl")
    with open(triplets_file, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"✅ 三元组训练数据已保存至 {triplets_file}")

    # 5. 输出统计信息
    print("\n" + "="*50)
    print("数据准备完成！统计信息：")
    print(f"  - 分块总数: {len(chunks)}")
    print(f"  - 问答对总数: {len(qa_pairs)}")
    print(f"  - 训练集三元组数: {len(triplets)}")
    print(f"  - 缓存文件: {CACHE_FILE}")
    print("\n下一步：将 output 文件夹打包上传至恒源云，使用 train_triplets.jsonl 进行模型微调。")

if __name__ == "__main__":
    main()