# -*- coding: utf-8 -*-
import os

# 定义项目根目录
PROJECT_ROOT = "rag-roleplay-system"

# 定义需要创建的文件夹和文件结构（严格匹配需求架构）
structure = {
    # 根目录文件
    "files": [
        "main.py",
        "config.py",
        "requirements.txt"
    ],
    # 子目录（键：目录路径，值：该目录下的空文件/子文件夹）
    "dirs": {
        "core": ["__init__.py", "rag_engine.py", "memory.py", "retriever.py", "prompt_manager.py"],
        "processor": ["__init__.py", "document_loader.py", "text_cleaner.py", "chunker.py", "advanced_parser.py"],
        "storage": ["__init__.py", "milvus_client.py", "mysql_client.py", "redis_client.py"],
        "api": ["__init__.py", "routes.py", "models.py"],
        "utils": ["__init__.py", "logger.py", "helpers.py"],
        "evaluation": ["__init__.py", "ragas_evaluator.py"],
        "scripts": ["clear_milvus.py", "init_db.py"],
        "static": ["index.html"],
        "knowledge_base": ["医学知识/", "法律知识/"]  # 知识库子文件夹（不创建文件，仅创建文件夹）
    }
}


def create_project():
    # 创建项目根目录（不存在则创建）
    if not os.path.exists(PROJECT_ROOT):
        os.makedirs(PROJECT_ROOT)
        print(f"创建项目根目录：{PROJECT_ROOT}")

    # 创建根目录下的空文件，每个文件开头添加编码注释
    for file in structure["files"]:
        file_path = os.path.join(PROJECT_ROOT, file)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# -*- coding: utf-8 -*-\n")  # 固定编码注释，无多余内容
        print(f"创建空文件：{file_path}")

    # 遍历所有子目录，创建目录及对应空文件
    for dir_path, files in structure["dirs"].items():
        full_dir_path = os.path.join(PROJECT_ROOT, dir_path)
        # 递归创建子目录（确保多级目录可正常生成）
        if not os.path.exists(full_dir_path):
            os.makedirs(full_dir_path)
            print(f"创建子目录：{full_dir_path}")
        # 区分创建文件和子文件夹
        for item in files:
            # 结尾带"/"的是子文件夹，仅创建文件夹，不创建文件
            if item.endswith("/"):
                sub_dir_path = os.path.join(full_dir_path, item.rstrip("/"))
                if not os.path.exists(sub_dir_path):
                    os.makedirs(sub_dir_path)
                    print(f"创建子文件夹：{sub_dir_path}")
            # 否则是文件，创建空文件并添加编码注释
            else:
                file_path = os.path.join(full_dir_path, item)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("# -*- coding: utf-8 -*-\n")
                print(f"创建空文件：{file_path}")


if __name__ == "__main__":
    create_project()
    print("\n✅ 项目空文件夹和空文件全部生成完成！")