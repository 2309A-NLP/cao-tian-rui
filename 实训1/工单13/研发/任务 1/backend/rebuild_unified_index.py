"""
rebuild_unified_index.py — 统一索引重建

目的（修复 hybrid 失效的根因）：
  当前 向量检索(Milvus) 与 BM25(all_chunks.json) 跑在两套不同的 chunk_id 上，
  导致 RRF 永远无法融合、hybrid 模式退化成噪声。
  本脚本把 Milvus 直接用 all_chunks.json 重建，使
    向量(Milvus).chunk_id  ==  BM25(all_chunks.json).chunk_id  ==  全局唯一 id
  之后 RRF 才能真正融合，hybrid 才能生效，且 9 家年报也进入向量检索。

为什么低风险：
  - all_chunks.json 是当前 fulltext(94%) 用的同一份内容，质量已验证。
  - 只重建向量库，源文件 all_chunks.json 不动；要回滚就重跑 run_hybrid_reindex.py。
  - 不调用任何外部 API（不耗 Qwen/VL 配额）。
  - 代价：2 个招股书在 Milvus 里的 VL markdown 会被替换成 all_chunks 的纯文本
    （纯文本在 fulltext 已拿到 94%）。VL 内容仍保留在 output/hybrid_vl_output/，可后续再合并。

用法（gpu_env，项目根目录）：
  python backend/rebuild_unified_index.py            # 交互确认
  python backend/rebuild_unified_index.py --yes      # 跳过确认

完成后用 eval 验证（关键：对比 hybrid 是否已能融合）：
  python backend/eval_quick.py --retrieval-mode fulltext hybrid
  python backend/eval_compare.py
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig
from logger import get_logger
from vector_store import VectorStore

logger = get_logger("rebuild")


def _find_all_chunks(config: AppConfig) -> Path:
    """定位 all_chunks.json（output 目录已被 config 解析为绝对路径）"""
    candidates = [
        Path(config.output_dir) / "chunks" / "all_chunks.json",
        Path(config.project_root) / "output" / "chunks" / "all_chunks.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"找不到 all_chunks.json，已试: {[str(c) for c in candidates]}")


def main():
    parser = argparse.ArgumentParser(description="统一索引重建：Milvus 从 all_chunks.json 重建")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认")
    parser.add_argument("--config", default="", help="config.json 路径（默认自动检测）")
    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = AppConfig.load(args.config)
    else:
        root_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
        config = AppConfig.load(root_cfg) if os.path.exists(root_cfg) else AppConfig.load()

    # 读取 all_chunks.json
    chunks_path = _find_all_chunks(config)
    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = json.load(f)

    n = len(all_chunks)
    ids = [c.get("chunk_id") for c in all_chunks]
    n_unique = len(set(ids))
    pages = [c.get("page", 0) for c in all_chunks]

    print("=" * 64)
    print("  统一索引重建预检")
    print("=" * 64)
    print(f"  源文件: {chunks_path}")
    print(f"  chunk 总数: {n}")
    print(f"  chunk_id 唯一数: {n_unique}  (应等于总数才能保证全局唯一)")
    print(f"  page 范围: {min(pages)} ~ {max(pages)}")
    print(f"  Milvus 集合: {config.milvus_collection} @ {config.milvus_host}:{config.milvus_port}")
    print(f"  嵌入模型: {config.embedding_model_type} @ {config.embedding_model_path}")
    print("=" * 64)

    if n_unique != n:
        print(f"⚠️ 警告：chunk_id 不是全局唯一（{n_unique}/{n}）。重建后 RRF 仍会冲突！")
        print("   请先确保 all_chunks.json 的 chunk_id 全局唯一，再重建。")
        if not args.yes:
            return 1

    if not args.yes:
        ans = input("将 DROP 当前 Milvus 集合并用上述 chunk 全量重嵌入，确认？(yes/no): ").strip().lower()
        if ans not in ("y", "yes"):
            print("已取消。")
            return 0

    # 初始化向量库
    vs = VectorStore(
        model_path=config.embedding_model_path,
        milvus_host=config.milvus_host,
        milvus_port=config.milvus_port,
        collection_name=config.milvus_collection,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
        model_type=config.embedding_model_type,
    )
    if not vs.load_model():
        print("❌ 嵌入模型加载失败")
        return 1
    if not vs.connect_milvus():
        print("❌ Milvus 连接失败")
        return 1

    # DROP + 重建集合
    print("\n🗑️  删除旧集合...")
    vs.drop_collection()
    if not vs.connect_milvus():
        print("❌ 重新连接 Milvus 失败")
        return 1

    # 全量嵌入（保留 all_chunks 的全局 chunk_id；filename=None 不额外加公司前缀，
    # 因为 all_chunks 文本里多数已含【文件】标记）
    print(f"\n🔢 开始嵌入 {n} 个 chunk（保留全局 chunk_id）...")
    ok = vs.index_documents(all_chunks, filename=None, config=config)
    if not ok:
        print("❌ 索引失败")
        return 1

    final = vs.count()
    print("\n" + "=" * 64)
    print(f"✅ 统一索引重建完成：Milvus 现有 {final} 个向量")
    print("   现在 向量 与 BM25 共用 all_chunks.json 的全局 chunk_id，RRF 可正常融合。")
    print("=" * 64)
    print("\n下一步验证（务必对比 hybrid 是否已能超过 fulltext）：")
    print("  python backend/eval_quick.py --retrieval-mode fulltext hybrid")
    print("  python backend/eval_compare.py")
    print("\n若 hybrid 的 Hit@10 / Recall@10 明显高于 fulltext，再把 config.json 的")
    print("  retrieval_mode 改成 hybrid；否则保持 fulltext。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
