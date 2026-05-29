# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
清理 Milvus 集合工具
功能：删除所有 Milvus 集合，清空向量数据
使用场景：切换 embedding 模型、修复维度不匹配问题
运行方式：python scripts/clear_milvus.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, utility


def clear_all_collections():
    """删除所有 Milvus 集合"""
    print("正在连接 Milvus...")
    connections.connect(host="localhost", port="19530")

    collections = utility.list_collections()
    print(f"当前集合: {collections}")

    if not collections:
        print("没有找到任何集合")
        return

    for col in collections:
        utility.drop_collection(col)
        print(f"✓ 已删除: {col}")

    print("\n所有集合已删除！")
    print("请重启 main.py 重新创建向量库")


def clear_specific_collections():
    """仅删除指定的集合"""
    print("正在连接 Milvus...")
    connections.connect(host="localhost", port="19530")

    # 要删除的集合列表
    target_collections = ["rag_chatbot_v2", "long_term_memory", "bm25_collection"]

    for col in target_collections:
        if utility.has_collection(col):
            utility.drop_collection(col)
            print(f"✓ 已删除: {col}")
        else:
            print(f"集合不存在: {col}")

    print("\n指定集合已删除！")


if __name__ == "__main__":
    print("=" * 50)
    print("Milvus 集合清理工具")
    print("=" * 50)
    print("1. 删除所有集合")
    print("2. 仅删除 rag_chatbot_v2、long_term_memory")
    choice = input("请选择 (1/2): ").strip()

    if choice == "1":
        clear_all_collections()
    elif choice == "2":
        clear_specific_collections()
    else:
        print("无效选择")