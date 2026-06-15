"""
补索引脚本：将现有表格和图片数据作为独立 chunk 写入 Milvus

用法（Windows gpu_env）:
  cd E:\10--agent--任务\任务 1\backend
  python supplement_index.py

说明：
- 表格：读取 output/tables/*.txt，生成 {field_type: table} chunk
- 图片：读取 output/images/*，用 PaddleOCR 提取文字，生成 {field_type: image} chunk
- 不会重复写入已存在的 chunk（按文件名/page 去重）
"""
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig
from logger import get_logger, log_exception

logger = get_logger("supplement")


def load_config() -> AppConfig:
    """加载配置（支持 RAG_CONFIG_PATH 环境变量）"""
    config_path = os.environ.get("RAG_CONFIG_PATH", "")
    if config_path:
        return AppConfig.load(config_path)
    return AppConfig.load()


def get_existing_table_pages(store) -> set:
    """查询 Milvus 中已有 table type 的 page 集合，避免重复写入"""
    if not store._has_field_type:
        return set()
    try:
        store.collection.load()
        results = store.collection.query(
            expr='field_type == "table"',
            output_fields=["page"],
            limit=10000,
        )
        return set(r["page"] for r in results)
    except Exception as e:
        logger.debug(f"查询已有 table 失败: {e}")
        return set()


def get_existing_image_pages(store) -> set:
    """查询 Milvus 中已有 image type 的 page 集合"""
    if not store._has_field_type:
        return set()
    try:
        store.collection.load()
        results = store.collection.query(
            expr='field_type == "image"',
            output_fields=["page"],
            limit=10000,
        )
        return set(r["page"] for r in results)
    except Exception as e:
        logger.debug(f"查询已有 image 失败: {e}")
        return set()


def read_table_txt(file_path: str) -> str:
    """读取表格 .txt 文件，返回纯文本内容"""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def parse_table_filename(filename: str) -> dict:
    """从文件名解析 page 和 table_id"""
    # 格式: page_0001_table_1.txt
    import re
    m = re.match(r'page_(\d+)_table_(\d+)', filename)
    if m:
        return {"page": int(m.group(1)), "table_id": int(m.group(2))}
    return {"page": 0, "table_id": 0}


def build_table_chunks(output_dir: str, project_root: str,
                       existing_pages: set) -> list[dict]:
    """从 tables/ 目录读取 .txt 文件，生成 table chunk"""
    table_dir = Path(output_dir) / "tables"
    if not table_dir.exists():
        logger.warning(f"表格目录不存在: {table_dir}")
        return []

    chunk_id_offset = 100000  # 用大偏移量，避免与正文 chunk_id 冲突
    chunks = []

    txt_files = sorted(table_dir.glob("*.txt"))
    logger.info(f"发现 {len(txt_files)} 个表格文件")

    for txt_file in txt_files:
        info = parse_table_filename(txt_file.stem)
        page = info["page"]

        # 如果该页的 table 已经存在，跳过
        if page in existing_pages:
            logger.debug(f"  跳过已存在的表格 (页 {page}): {txt_file.name}")
            continue

        text = read_table_txt(txt_file)
        if len(text.strip()) < 10:
            continue

        # 构建表格 chunk
        # 第一行通常是标题行（=== 第 X 页 表格 N ===），提取作为前缀
        lines = text.strip().split("\n", 1)
        title_line = lines[0] if lines else ""
        body = lines[1] if len(lines) > 1 else text

        chunk_text = (
            f"【表格】第{page}页 | {title_line}\n"
            f"【表格内容】\n{body}"
        )

        chunk = {
            "chunk_id": chunk_id_offset + info["table_id"],
            "page": page,
            "field_type": "table",
            "text": chunk_text,
            "source": str(txt_file.name),
        }
        chunks.append(chunk)

    logger.info(f"生成 {len(chunks)} 个表格 chunk")
    return chunks


def build_image_chunks(output_dir: str, project_root: str,
                       existing_pages: set) -> list[dict]:
    """读取 images/ 目录下的图片，用 PaddleOCR 提取文字，生成 image chunk"""
    img_dir = Path(output_dir) / "images"
    if not img_dir.exists():
        logger.warning(f"图片目录不存在: {img_dir}")
        return []

    # 读取图片元数据
    report_path = Path(output_dir) / "images_report.json"
    if not report_path.exists():
        logger.warning(f"图片报告不存在: {report_path}")
        return []

    with open(report_path, encoding="utf-8") as f:
        images_meta = json.load(f)

    # 按 page 分组
    from collections import defaultdict
    page_images = defaultdict(list)
    for img in images_meta:
        page_images[img["page"]].append(img)

    logger.info(f"发现 {len(images_meta)} 张图片，分布在 {len(page_images)} 个页面")

    # 用 Tesseract OCR（系统已安装）
    import subprocess
    import tempfile

    chunks = []
    chunk_id_offset = 200000

    for page_num, imgs in sorted(page_images.items()):
        if page_num in existing_pages:
            logger.debug(f"  跳过已有的图片 chunk (页 {page_num})")
            continue

        ocr_results = []
        for img_meta in imgs:
            img_path = img_dir / f"page_{page_num:04d}_{img_meta['format'].upper()}_{img_meta['image_id']}.{img_meta['format']}"
            if not img_path.exists():
                continue
            try:
                result = subprocess.run(
                    ["tesseract", str(img_path), "stdout", "-l", "chi_sim+eng", "--psm", "6"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    text = result.stdout.strip()
                    if text and len(text) > 10:
                        ocr_results.append(text)
                        logger.debug(f"  Tesseract OCR (页 {page_num}): {len(text)}字符")
            except Exception as e:
                logger.debug(f"  OCR 失败 (页 {page_num}, {img_path.name}): {e}")

        if not ocr_results:
            logger.debug(f"  页 {page_num}: OCR 未提取到文字，跳过")
            continue

        ocr_text = "\n".join(ocr_results)
        chunk_text = (
            f"【图片文字】第{page_num}页\n"
            f"【OCR 识别内容】\n{ocr_text}"
        )

        chunks.append({
            "chunk_id": chunk_id_offset + page_num,
            "page": page_num,
            "field_type": "image",
            "text": chunk_text,
            "source": f"page_{page_num}",
        })
        logger.info(f"  页 {page_num}: OCR 提取 {len(ocr_text)} 字符, {len(ocr_results)} 张图")

    logger.info(f"生成 {len(chunks)} 个图片 chunk")
    return chunks


def main():
    config = load_config()

    # 解析 output_dir 为绝对路径
    output_dir = config.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(config.project_root, output_dir)

    logger.info("=" * 60)
    logger.info("补索引：表格 + 图片")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Milvus 集合: {config.milvus_collection}")
    logger.info("=" * 60)

    # 初始化 VectorStore
    from vector_store import VectorStore

    store = VectorStore(
        model_path=config.embedding_model_path,
        milvus_host=config.milvus_host,
        milvus_port=config.milvus_port,
        collection_name=config.milvus_collection,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
    )
    if not store.load_model():
        logger.error("模型加载失败")
        sys.exit(1)
    if not store.connect_milvus():
        logger.error("Milvus 连接失败")
        sys.exit(1)

    # 查询已有数据
    existing_tables = get_existing_table_pages(store)
    existing_images = get_existing_image_pages(store)
    logger.info(f"已有表格页面: {len(existing_tables)} 个")
    logger.info(f"已有图片页面: {len(existing_images)} 个")

    # 1. 表格 chunk
    table_chunks = build_table_chunks(output_dir, config.project_root, existing_tables)
    if table_chunks:
        logger.info(f"索引 {len(table_chunks)} 个表格 chunk...")
        store.index_documents(table_chunks, filename=None, config=config)
    else:
        logger.info("没有新的表格需要索引")

    # 2. 图片 chunk（OCR）
    image_chunks = build_image_chunks(output_dir, config.project_root, existing_images)
    if image_chunks:
        logger.info(f"索引 {len(image_chunks)} 个图片 chunk...")
        store.index_documents(image_chunks, filename=None, config=config)
    else:
        logger.info("没有新的图片需要索引")

    # 3. 保存补充后的 chunks_metadata 到文件
    all_chunks_path = os.path.join(output_dir, "chunks", "all_chunks.json")
    if os.path.exists(all_chunks_path):
        with open(all_chunks_path, encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        all_chunks = []
    all_chunks.extend(table_chunks)
    all_chunks.extend(image_chunks)
    os.makedirs(os.path.dirname(all_chunks_path), exist_ok=True)
    with open(all_chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"all_chunks.json 已更新: {len(all_chunks)} 条")

    logger.info("补索引完成！")


if __name__ == "__main__":
    main()
