"""
RAG 对话系统 - 主入口 + 命令行界面

用法：
  python main.py                           # 交互式对话
  python main.py --ask "问题"              # 单次问答
  python main.py --index "docs/xxx.pdf"    # 索引 PDF
  python main.py --show-config             # 显示配置

快捷键：
  /rag          切换到 RAG 模式
  /direct       切换到直接问答模式
  /clear        清除对话历史
  /history      显示历史记录
  /mode         显示当前模式
  /index <path> 索引一个 PDF 文件
  /save         保存索引到磁盘
  /quit         退出
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Optional

# 确保项目根路径在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig
from llm_provider import LLMFactory
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine, ChatSession
from database import DatabaseManager


# ─── 索引 PDF ──────────────────────────────────────────

def index_pdf(pdf_path: str, config: AppConfig, vector_store: VectorStore,
              drop_existing: bool = False, force: bool = False,
              ocr_images: bool = False,
              tesseract_cmd: str = "tesseract") -> int:
    """索引一个 PDF 文件（支持增量追加），返回 chunk 数量

    流程：
      1. 连接 Milvus / 加载模型
      2. 检查知识库中是否已存在同名文件
      3. 如果已存在且未强制索引：跳过，不重复索引
      4. 如果不存在或强制索引：解析 PDF → 分块 → 向量化 → 追加写入
    
    ocr_images: 是否对提取的图片进行 OCR 识别（需安装 tesseract-ocr 及中文语言包）
    tesseract_cmd: tesseract 命令路径（Windows 下可能需要完整路径）
    """
    if not os.path.exists(pdf_path):
        print(f"错误：文件不存在 - {pdf_path}")
        return 0

    filename = os.path.basename(pdf_path)
    print(f"\n正在处理 PDF：{filename}")

    # 先连接 Milvus 和加载模型
    if not vector_store.load_model():
        print("错误：嵌入模型加载失败")
        return 0
    if not vector_store.connect_milvus():
        print("错误：Milvus 连接失败")
        return 0

    # 检查知识库中是否已有这个文件
    if not force and vector_store.document_exists(filename):
        existing = vector_store.list_documents()
        print(f"知识库中已存在该文件: {filename}")
        print(f"当前知识库 ({len(existing)} 个文件): {', '.join(existing)}")
        print("跳过索引（如需重新索引请使用 --force 或 --reindex）")
        return 0

    # 如果 drop_existing 为 True，清空整个集合
    if drop_existing:
        vector_store.drop_collection()
        if not vector_store.connect_milvus():
            print("错误：重新连接 Milvus 失败")
            return 0

    # 解析 PDF
    with PDFProcessor(pdf_path, output_dir=config.output_dir) as processor:
        if not processor.process(ocr_images=ocr_images, tesseract_cmd=tesseract_cmd):
            print("错误：PDF 处理失败")
            return 0

        chunks = processor.get_chunks(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        print(f"生成了 {len(chunks)} 个文档块")

        # 向量化 + 存入 Milvus（追加模式）
        print("正在向量化并存入 Milvus...")
        if not vector_store.index_documents(chunks, filename=filename):
            print("错误：文档索引失败")
            return 0

        count = vector_store.count()
        print(f"索引完成，Milvus 中现有 {count} 个向量")

        stats = processor.processing_stats
        print(f"统计：{stats['pages']} 页, {stats['tables']} 表格, "
              f"{stats['images']} 图片, 耗时 {stats['time_seconds']:.1f}s")

    return len(chunks)


# ─── 命令行界面 ──────────────────────────────────────────

def run_interactive(config: AppConfig):
    """交互式对话"""
    print("\n" + "=" * 60)
    print("  RAG 对话系统 v2.0")
    print("=" * 60)
    print("  /help 查看可用命令")
    print(f"  嵌入模型：BGE-M3 ({config.embedding_model_path})")
    print(f"  LLM 提供商：{config.llm_provider} / {config.llm_model}")
    print("=" * 60)

    # 初始化 LLM
    llm = LLMFactory.create(
        provider=config.llm_provider,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
    )

    # 初始化向量存储
    vec = VectorStore(
        model_path=config.embedding_model_path,
        milvus_host=config.milvus_host,
        milvus_port=config.milvus_port,
        device=config.embedding_device,
    )
    vec.load_model()
    vec.connect_milvus()

    # 初始化 RAG 引擎
    rag = RAGEngine(
        vector_store=vec,
        llm_provider=llm,
        top_k=config.top_k,
        similarity_threshold=config.similarity_threshold,
    )

    # 初始化会话
    session = ChatSession(user_id=0, session_id="cli_main")

    while True:
        try:
            print("\n你 > ", end="", flush=True)
            lines = []
            consecutive_blank = 0
            while True:
                line = input()
                if not line:
                    consecutive_blank += 1
                    if consecutive_blank >= 2:
                        break  # 连续两个空行结束多行输入
                    continue
                consecutive_blank = 0
                # 检查是否是命令（只检查第一行）
                if not lines and line.startswith("/"):
                    lines = [line]
                    break
                lines.append(line)
            user_input = " ".join(lines).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # ── 命令处理 ──────────────────────────────
        if user_input.startswith("/"):
            cmd_parts = user_input[1:].strip().split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                print("再见！")
                break
            elif cmd == "rag":
                session.set_mode("rag")
                print("[系统] 已切换到 RAG 模式（检索增强生成）")
            elif cmd == "direct":
                session.set_mode("direct")
                print("[系统] 已切换到 直接问答模式（纯 LLM）")
            elif cmd == "clear":
                session.clear()
                print("[系统] 对话历史已清除")
            elif cmd == "history":
                if not session.history:
                    print("[系统] 暂无对话历史")
                else:
                    print("\n--- 对话历史 ---")
                    for i, msg in enumerate(session.history[-10:]):
                        label = "你" if msg["role"] == "user" else "AI"
                        preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                        print(f"  [{i+1}] {label}: {preview}")
                    print("-----------------\n")
            elif cmd == "mode":
                print(f"[系统] 当前模式：{session.mode}")
            elif cmd == "index":
                if not arg:
                    print("用法：/index <pdf_path>")
                    continue
                index_pdf(arg, config, vec)
            elif cmd == "save":
                vec.disconnect()
                print("[系统] Milvus 索引已持久化")
            elif cmd == "help":
                print("""
可用命令：
  /rag           切换到 RAG 模式（检索增强）
  /direct        切换到直接问答模式
  /clear         清除对话历史
  /history       显示最近 10 轮对话
  /mode          显示当前模式
  /index <path>  索引一个 PDF 文件
  /save          断开 Milvus 连接
  /help          显示帮助
  /quit          退出
                """)
            else:
                print(f"未知命令：/{cmd}，输入 /help 查看可用命令")
            continue

        # ── 对话 ──────────────────────────────────
        start = datetime.now()
        print("\nAI > ", end="", flush=True)

        result = rag.answer(user_input, session)
        answer_text = result["answer"]
        print(answer_text)

        elapsed = (datetime.now() - start).total_seconds()
        parts = [f"\n\n[完成 | {elapsed:.1f}s"]
        if result["retrieval_time_ms"] > 0:
            parts.append(f" 检索:{result['retrieval_time_ms']}ms")
        if result["llm_time_ms"] > 0:
            parts.append(f" LLM:{result['llm_time_ms']}ms")
        parts.append(f" 模式:{result['mode']}")
        if result["sources"]:
            pages = sorted(set(s["page"] for s in result["sources"]))
            parts.append(f" 来源:第{pages}页")
        parts.append("]")
        print("".join(parts))


# ─── 工具函数 ──────────────────────────────────────────

def single_ask(question: str, config: AppConfig):
    """单次问答"""
    llm = LLMFactory.create(
        provider=config.llm_provider,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
    )
    vec = VectorStore(
        model_path=config.embedding_model_path,
        milvus_host=config.milvus_host,
        milvus_port=config.milvus_port,
        device=config.embedding_device,
    )
    vec.load_model()
    vec.connect_milvus()

    rag = RAGEngine(
        vector_store=vec,
        llm_provider=llm,
        top_k=config.top_k,
        similarity_threshold=config.similarity_threshold,
    )
    session = ChatSession(user_id=0, session_id="single_ask")
    # 默认用 rag 模式
    session.set_mode("rag")

    result = rag.answer(question, session)
    print(f"\n问题：{question}")
    print(f"答案：{result['answer']}")
    if result["sources"]:
        pages = sorted(set(s["page"] for s in result["sources"]))
        print(f"来源：第{pages}页")
    print(f"耗时：{result['total_time_ms']}ms")


# ─── 入口 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG 对话系统 - 基于文档的智能问答"
    )
    parser.add_argument("--ask", type=str, help="单次提问并退出")
    parser.add_argument("--index", type=str, nargs="+", help="索引一个或多个 PDF 文件（增量追加，自动跳过已索引文件）")
    parser.add_argument("--reindex", action="store_true", help="清空知识库并重新索引所有 PDF（等效于 --force 完全重建）")
    parser.add_argument("--force", action="store_true", help="强制重新索引（即使文件已存在也重新处理）")
    parser.add_argument("--scan", nargs="?", const="", default=None,
                        help="扫描 knowledge_base 目录下的所有 PDF，增量索引未入库的文件。可选参数：指定目录路径，默认 knowledge_base/")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config.json"),
                        help="指定配置文件路径")
    parser.add_argument("--show-config", action="store_true", help="显示配置并退出")
    args = parser.parse_args()

    # 加载配置
    config = AppConfig.load(args.config)

    # 确保目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "chunks"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "images"), exist_ok=True)
    os.makedirs(config.knowledge_base_dir, exist_ok=True)

    if args.show_config:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return

    # ── --scan 知识库扫描 ──────────────────────────────
    if args.scan is not None:
        scan_dir = args.scan if args.scan else config.knowledge_base_dir
        if not os.path.isdir(scan_dir):
            print(f"错误：目录不存在 - {scan_dir}")
            return
        pdf_files = sorted([f for f in os.listdir(scan_dir) if f.lower().endswith(".pdf")])
        if not pdf_files:
            print(f"知识库目录 '{scan_dir}' 中没有找到 PDF 文件")
            return
        print(f"发现 {len(pdf_files)} 个 PDF 文件")
        vec = VectorStore(
            model_path=config.embedding_model_path,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            device=config.embedding_device,
        )
        total_chunks = 0
        skipped = 0
        for pdf_file in pdf_files:
            pdf_path = os.path.join(scan_dir, pdf_file)
            chunks = index_pdf(pdf_path, config, vec, drop_existing=False, force=args.force)
            if chunks == 0:
                skipped += 1
            total_chunks += chunks
        print(f"\n扫描完成！共索引 {total_chunks} 个文档块，跳过 {skipped} 个已存在的文件")
        return

    # ── --reindex 全部重新索引 ────────────────────────────
    if args.reindex and not args.index:
        scan_dir = config.knowledge_base_dir
        if not os.path.isdir(scan_dir):
            print(f"错误：知识库目录不存在 - {scan_dir}")
            return
        pdf_files = sorted([f for f in os.listdir(scan_dir) if f.lower().endswith(".pdf")])
        if not pdf_files:
            print(f"知识库目录 '{scan_dir}' 中没有找到 PDF 文件")
            return
        print(f"发现 {len(pdf_files)} 个 PDF 文件，正在重新索引（清空旧数据）...")
        vec = VectorStore(
            model_path=config.embedding_model_path,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            device=config.embedding_device,
        )
        total_chunks = 0
        # 先清空 Milvus（只清一次）
        if vec.load_model() and vec.connect_milvus():
            vec.drop_collection()
            vec.connect_milvus()
        else:
            print("错误：向量存储初始化失败")
            return
        for pdf_file in pdf_files:
            pdf_path = os.path.join(scan_dir, pdf_file)
            print(f"\n{'='*60}")
            print(f"正在处理: {pdf_file}")
            print(f"{'='*60}")
            # 每个文件单独处理，drop_existing=False 因为已经清空了
            chunks = index_pdf(pdf_path, config, vec,
                              drop_existing=False, force=True)
            total_chunks += chunks
        print(f"\n全部完成！共索引 {total_chunks} 个文档块")
        print(f"当前 Milvus 向量总数: {vec.count() if hasattr(vec, 'count') else '?'}")
        return

    if args.index:
        # 初始化向量存储用于索引
        vec = VectorStore(
            model_path=config.embedding_model_path,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            device=config.embedding_device,
        )
        total_chunks = 0
        for pdf_path in args.index:
            chunks = index_pdf(
                pdf_path, config, vec,
                drop_existing=(total_chunks == 0 and args.reindex),
                force=args.force,
            )
            total_chunks += chunks
        if total_chunks > 0:
            print(f"\n全部完成！共索引 {total_chunks} 个文档块")
        return

    if args.ask:
        single_ask(args.ask, config)
        return

    # 默认：交互模式
    run_interactive(config)


if __name__ == "__main__":
    main()
