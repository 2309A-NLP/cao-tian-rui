"""
一键混合策略重新索引
使用方法：
  1. 在 PyCharm 中：右键 → Run 'run_hybrid_reindex'
  2. 或在终端：python run_hybrid_reindex.py

等效于命令行：python backend/main.py --reindex --hybrid --config config.json
"""
import subprocess
import sys
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

cmd = [
    sys.executable,                           # 当前 Python 解释器
    os.path.join(PROJECT_ROOT, "backend", "main.py"),  # 入口
    "--reindex",
    "--hybrid",
    "--config",
    os.path.join(PROJECT_ROOT, "config.json"),
]

print("=" * 60)
print("  RAG 混合策略重新索引")
print("=" * 60)
print(f"  项目目录: {PROJECT_ROOT}")
print(f"  运行命令: {' '.join(cmd)}")
print("=" * 60)

result = subprocess.run(cmd, cwd=PROJECT_ROOT)
sys.exit(result.returncode)
