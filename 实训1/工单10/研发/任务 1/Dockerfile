# RAG 金融问答系统 — 后端镜像（CPU 版）
# 设计要点：
#   1) 模型(BGE-M3 4.3G)/数据/输出 全部走卷挂载，绝不打进镜像 → 镜像保持精简、可移植
#   2) 先单独装 CPU 版 torch，再装 requirements，避免 pip 拉进 CUDA 大包（~4G）
#   3) device=cpu 由 config.docker.json 指定；embedding_provider 的 auto 在无 GPU 时也会回落 CPU
FROM python:3.10-slim

# 系统依赖：curl 用于 healthcheck；build 基础库给少数需要编译的轮子兜底
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_CONFIG_PATH=/app/config.docker.json

WORKDIR /app

# ── 依赖层（与代码分离，便于缓存）──
# CPU 版 torch 单独装（关键：省 ~4G CUDA 体积）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# requirements 里的 torch>=2.0.0 已被上一步满足，pip 会跳过，不会改装 GPU 版
RUN pip install --no-cache-dir -r requirements.txt

# transformers 4.57 加载 BGE-M3 的 pytorch_model.bin 走 torch.load，要求 torch>=2.6
# （宿主 gpu_env 用 torch 2.11 无此问题）。单独升级，保留上面的 requirements 缓存层。
# 精确钉 2.6.0 并放宽超时/重试，适配慢网。
RUN pip install --no-cache-dir --timeout 180 --retries 10 \
        "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu

# ── 代码层 ──
COPY backend/ ./backend/
COPY api/ ./api/
COPY run_api.py ./
COPY config.docker.json ./

# 运行时挂载点（compose 提供卷）：
#   /models/bge-m3   模型（只读）
#   /app/output      chunks/索引产物（含 all_chunks.json，BM25 全文检索依赖）
#   /app/logs        日志
#   /app/knowledge_base  PDF 知识库
EXPOSE 8010

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS http://localhost:8010/api/health || exit 1

CMD ["python", "run_api.py"]
