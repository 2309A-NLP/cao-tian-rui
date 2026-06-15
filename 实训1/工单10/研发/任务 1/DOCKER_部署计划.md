# RAG 金融问答系统 — Docker 部署计划

## 目标（对齐工单）
把 RAG 金融问答系统以 Docker 容器形式部署到本机（Win11 + Docker Desktop），`docker compose up` 一条命令全起，验收通过。

## 已探明的现状（决定方案的关键事实）
- **后端**：FastAPI/uvicorn，端口 8010，前端单页 `api/static/index.html` 由后端同域托管（`const API=''` 相对路径，零改动）。
- **DB 自动建表**：`database.py:_init_tables()` 自动建库建表 → MySQL 无需手工初始化脚本。
- **空库可跑**：`api.py:129-144` 只要 Milvus 连上（哪怕空），就从挂载的 `output/chunks/all_chunks.json` 构建 BM25；生产默认 `retrieval_mode=fulltext`（纯 BM25），**Milvus 空也能跑通问答**。
- **device=auto**：`embedding_provider.py:91` 用 `torch.cuda.is_available()` 自动判断 → CPU 版 torch 自动回落 CPU，**代码零改动**。
- **config 环境变量**：已支持 `APP_DB_*`/`APP_MILVUS_*`，**缺 `APP_REDIS_*`/`APP_LLM_*_2`** → 需补几行映射。
- **现有依赖容器**：本机已有 `milvus-standalone`(+etcd+minio)、`mariadb`(3307→3306)，Redis 在宿主机。compose 将新建一套独立编排，不动现有容器（用不同项目名/卷，避免冲突）。
- **模型**：BGE-M3 4.3GB 在 `F:\...\bge-m3` → 只读卷挂载，不进镜像。

## 交付物（工单要求）
1. `Dockerfile`（rag-app，CPU 版）
2. `docker-compose.yml`（rag-app + milvus + etcd + minio + mysql + redis）
3. `.dockerignore`
4. `config.docker.json`（host 改服务名）
5. `DEPLOY.md`（部署文档 + 验收对照）
6. `部署问题记录.md`（过程问题记录）

## 实施步骤

### 第 1 步：补 config.py 环境变量映射（最小代码改动）
给 `_apply_env_overrides` 的 `env_map` 增加：
- `APP_REDIS_HOST→redis_host`、`APP_REDIS_PORT→redis_port`、`APP_REDIS_PASSWORD→redis_password`
- `APP_EMBEDDING_DEVICE→embedding_device`、`APP_EMBEDDING_MODEL_PATH→embedding_model_path`
- 把 `redis_port` 加入 `_INT_FIELDS`
（其余配置走 `config.docker.json` + `RAG_CONFIG_PATH`，密钥不写死）

### 第 2 步：Dockerfile（rag-app，CPU）
- `python:3.10-slim` 基础镜像
- 先单独装 **CPU 版 torch**（`--index-url https://download.pytorch.org/whl/cpu`），再装 requirements.txt（镜像从 ~6GB→~2GB）
- `COPY backend/ api/ run_api.py config.docker.json`
- 模型/数据/输出全走卷，不 COPY
- `CMD ["python","run_api.py"]`，`EXPOSE 8010`
- 健康检查：`curl -f http://localhost:8010/api/health`

### 第 3 步：docker-compose.yml（5+服务一张网 rag-net）
| 服务 | 镜像 | 卷 | 对外端口 |
|------|------|----|---------|
| rag-app | 自建 | 模型(只读)、output、logs、knowledge_base | **8010** |
| milvus | milvusdb/milvus | milvus-data | 19530（可选暴露） |
| etcd | coreos-etcd | etcd-data | 内部 |
| minio | minio | minio-data | 内部 |
| mysql | mysql:8.0 | mysql-data | 3307→3306（可选） |
| redis | redis:7-alpine | redis-data | 内部 |
- 服务间用**服务名**互通（rag-app 连 `milvus`/`mysql`/`redis`）→ 满足"容器间通信/网络配置"
- `depends_on` + healthcheck 控制启动顺序，rag-app 等依赖 healthy 再起
- 命名卷做持久化 → 满足"卷持久化/数据不丢"
- `restart: unless-stopped`

### 第 4 步：config.docker.json
复制 config.json，改：`milvus_host=milvus`、`db_host=mysql`、`db_port=3306`、`redis_host=redis`、`embedding_model_path=/models/bge-m3`、`embedding_device=cpu`、保留 `retrieval_mode=fulltext`（先保证可跑通）。密钥通过 compose 的 `environment` 注入而非写死。

### 第 5 步：数据初始化
- **MySQL**：首启自动建表，无需操作。
- **Milvus**：方案A（默认，最稳）——先靠 fulltext 跑通验收（挂载 all_chunks.json 即可）；方案B（可选增强）——提供 `make-init` 容器把现有索引灌入新 Milvus（复用 rebuild_unified_index.py）。
- **挂载** `output/`（含 all_chunks.json 18443 chunk）到容器，BM25 全文检索立即可用。

### 第 6 步：构建 + 启动 + 验收
```
docker compose build
docker compose up -d
docker compose ps        # 全 healthy
curl http://localhost:8010/api/health
# 浏览器开 http://localhost:8010 问一个金融问题
```
验收对照（写进 DEPLOY.md）：
- ✅ docker run/compose 成功启动、指定端口提供服务
- ✅ 无异常日志（`docker compose logs rag-app`）
- ✅ 卷持久化（`docker volume ls` + 重启容器数据仍在）
- ✅ 容器间通信（rag-app 用服务名连 milvus/mysql/redis）
- ✅ 网络配置正确（rag-net bridge）

### 第 7 步：写文档
`DEPLOY.md`（部署步骤、卷说明、验收清单、常见问题、回滚）+ `部署问题记录.md`。

## 风险与对策
- **镜像大/torch**：CPU 版 + 分层装，控制在 ~2GB。
- **模型路径 Windows→容器**：Docker Desktop 支持挂载 `F:\` 盘；compose 里写 `F:/.../bge-m3:/models/bge-m3:ro`。
- **与现有容器冲突**：用独立 compose 项目名 `rag-deploy` + 独立命名卷，端口若占用则换宿主端口（如 8010→保持，milvus 内部不暴露）。
- **首次嵌入慢**：CPU 下建索引慢，但日常问答只嵌一句话，无感；建索引可仍用宿主 gpu_env 预生成。

## 不做的事
- 不动现有 milvus-standalone/mariadb 容器和数据
- 不改检索/生成逻辑（这是部署任务，非算法任务）
- 不打模型进镜像
