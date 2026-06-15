#!/usr/bin/env python3
"""
PaddleOCR-VL 混合策略测试脚本 v2

优化后的混合策略逻辑：
1. 纯文字页面：pymupdf直接提取（高效，无需API）
2. 表格/图表页面：PaddleOCR-VL处理（更准确）
3. 混合页面：分离处理（文字+表格）

用途：
1. 测试页面分类功能
2. 验证不同类型页面的处理策略
3. 对比标准策略和混合策略的处理效果

使用方法：
python test_hybrid.py --pdf <PDF路径> [--config <配置文件>] [--api-key <API密钥>] [--api-secret <API密钥>]
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_processor import HybridPDFProcessor
from config import AppConfig
from pdf_processor import PDFProcessor


def test_page_classification(pdf_path: str):
    """测试页面分类功能"""
    print("📄 测试页面分类功能...")
    
    processor = HybridPDFProcessor(pdf_path=pdf_path)
    if not processor.load():
        print("❌ PDF 加载失败")
        return False
    
    # 测试前几页的分类
    test_pages = min(5, len(processor.pdf))
    classifications = {}
    
    for i in range(test_pages):
        page = processor.pdf[i]
        page_type = processor.classifier.classify_page(page)
        classifications[i+1] = page_type
        print(f"  第{i+1}页: {page_type}")
    
    print(f"📊 分类统计: {classifications}")
    return True


def test_hybrid_processing(pdf_path: str, config: AppConfig):
    """测试混合策略处理"""
    print("🔄 测试混合策略处理...")
    
    processor = HybridPDFProcessor(
        pdf_path=pdf_path,
        output_dir=config.output_dir,
        api_key=config.paddleocr_api_key,
        api_secret=config.paddleocr_api_secret
    )
    
    if not processor.load():
        print("❌ PDF 加载失败")
        return False
    
    try:
        chunks = processor.process()
        processor.save_output(chunks)
        
        print(f"✅ 处理完成，生成 {len(chunks)} 个文档块")
        print(f"📊 统计信息:")
        for key, value in processor.stats.items():
            print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ 混合策略处理失败: {e}")
        return False


def compare_strategies(pdf_path: str, config: AppConfig):
    """对比标准策略和混合策略"""
    print("🔍 对比标准策略 vs 混合策略...")
    
    # 标准策略
    print("\n--- 标准策略 ---")
    with PDFProcessor(pdf_path, output_dir=config.output_dir) as processor:
        if processor.process():
            chunks = processor.get_chunks(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            print(f"生成 {len(chunks)} 个文档块")
            stats = processor.processing_stats
            print(f"统计: {stats}")
    
    # 混合策略
    print("\n--- 混合策略 ---")
    if config.paddleocr_api_key:
        processor = HybridPDFProcessor(
            pdf_path=pdf_path,
            output_dir=config.output_dir,
            api_key=config.paddleocr_api_key,
            api_secret=config.paddleocr_api_secret
        )
        if processor.load():
            chunks = processor.process()
            print(f"生成 {len(chunks)} 个文档块")
            stats = processor.stats
            print(f"统计: {stats}")
        else:
            print("❌ 混合策略加载失败")
    else:
        print("⚠️ 未配置API密钥，跳过混合策略测试")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL 混合策略测试")
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--api-key", help="PaddleOCR API Key")
    parser.add_argument("--api-secret", help="PaddleOCR API Secret")
    parser.add_argument("--test-classification", action="store_true", 
                       help="仅测试页面分类")
    parser.add_argument("--test-hybrid", action="store_true", 
                       help="测试混合策略处理")
    parser.add_argument("--compare", action="store_true", 
                       help="对比两种策略")
    
    args = parser.parse_args()
    
    # 加载配置
    config = AppConfig.load(args.config)
    
    # 覆盖API密钥
    if args.api_key:
        config.paddleocr_api_key = args.api_key
    if args.api_secret:
        config.paddleocr_api_secret = args.api_secret
    
    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    
    if not os.path.exists(args.pdf):
        print(f"❌ PDF文件不存在: {args.pdf}")
        return
    
    print(f"📁 测试文件: {args.pdf}")
    print(f"⚙️ 配置文件: {args.config}")
    
    if args.test_classification:
        test_page_classification(args.pdf)
    elif args.test_hybrid:
        test_hybrid_processing(args.pdf, config)
    elif args.compare:
        compare_strategies(args.pdf, config)
    else:
        # 默认执行所有测试
        print("🧪 执行所有测试...")
        test_page_classification(args.pdf)
        if config.paddleocr_api_key:
            test_hybrid_processing(args.pdf, config)
            compare_strategies(args.pdf, config)
        else:
            print("⚠️ 未配置API密钥，跳过混合策略测试")


if __name__ == "__main__":
    main()