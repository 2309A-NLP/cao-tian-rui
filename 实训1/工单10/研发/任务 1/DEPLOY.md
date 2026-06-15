# RAG 金融问答系统 — Docker 部署文档

将 RAG 金融问答系统以 Docker 容器形式部署，一条命令拉起全部服务。

## 一、架构

```
                    ┌─────────────────────────────────────────┐
                    │            docker network: rag-net        │
  浏览器/接口  ──8010──►  rag-app (FastAPI + 前端)               │
                    │        │      │        │                  │
                    │     milvus  mysql    redis                │
                    │     │    │                                │
                    │   etcd  minio                             │
                    └─────────────────────────────────────────┘
   宿主机挂载：BGE-M3 模型(只读) / output / logs / knowledge_base
   远程：DeepSeek / SiliconFlow LLM API（走外网）
```

| 服务 | 镜像 | 作用 | 对外端口 |
|------|------|------|---------|
| rag-app | 自建 `rag-app:cpu`（含 torch 2.6.0+cpu / transformers 4.57 / pymilvus 2.4.15） | FastAPI 后端 + 前端页面 | **8010** |
| milvus | milvusdb/milvus:v3.0-beta | 向量库 | 内部 |
| etcd / minio | 官方 | Milvus 元数据/对象存储 | 内部 |
| mysql | mariadb:10.11（兼容 MySQL，pymysql 直连） | 对话历史/反馈 | 内部 |
| redis | redis:7-alpine | 会话缓存 | 内部 |

> 镜像版本选用本机已有、且与宿主 gpu_env 实证可用的组合（pymilvus 2.4.x ↔ milvus v3.0-beta），避免慢网重下。
> 仅 rag-app 的 8010 对外暴露，其余服务只在 rag-net 内部用服务名互通——既安全，也**不与宿主机已有的 milvus/mariadb/redis 端口冲突**。

## 二、前置条件

1. Docker Desktop（已含 docker compose v2）。
2. 本地 BGE-M3 模型目录（4.3G）。
3. 能访问 DeepSeek/SiliconFlow 的网络（LLM 生成）。

## 三、配置

编辑根目录 `.env`（已提供，按需改）：

```ini
MODEL_PATH=F:/.../bge-m3          # 宿主机模型目录，正斜杠
APP_PORT=8010                     # 对外端口
APP_RETRIEVAL_MODE=fulltext       # fulltext(默认,最稳) / vector / hybrid
# LLM（DeepSeek 官方欠费时切 SiliconFlow，OpenAI 兼容端点）
APP_LLM_PROVIDER=openai
APP_LLM_API_KEY=sk-xxxx
APP_LLM_BASE_URL=https://api.siliconflow.cn/v1
APP_LLM_MODEL=deepseek-ai/DeepSeek-V3
```

容器内运行参数见 `config.docker.json`（host 已指向服务名 milvus/mysql/redis，`embedding_device=cpu`）。

## 四、启动

```bash
docker compose build          # 构建 rag-app 镜像（首次 ~5-10 分钟）
docker compose up -d          # 拉起全部服务
docker compose ps             # 等待全部 healthy（首次 milvus/mysql 约 1-2 分钟）
```

启动顺序由 `depends_on + healthcheck` 控制：etcd/minio → milvus、mysql、redis 就绪后，rag-app 才启动。

## 五、验证（对照工单验收标准）

### 1. 服务可用
```bash
curl http://localhost:8010/api/health
# 期望：{"status":"ok","vector_count":...,"database":true,"redis":true,...}
```
浏览器打开 `http://localhost:8010`，问一个金融问题（如"平安银行2019年盈利增长的关键因素"），能返回带引用的答案即通过。

### 2. 容器启动与运行 ✅
- `docker compose ps` 全部 `Up (healthy)`。
- `docker compose logs rag-app` 无异常堆栈（首次会打印模型加载、BM25 构建日志）。

### 3. 容器数据管理（卷持久化）✅
```bash
docker volume ls | grep rag-deploy   # milvus/etcd/minio/mysql/redis 数据卷
# 验证持久化：重启后历史对话仍在
docker compose restart
```
- 关键数据：MySQL（对话历史/反馈）、Milvus（向量）、Redis（会话）走命名卷；
- 模型/知识库/output 走宿主机绑定挂载，**容器间共享**（rag-app 读 output 里的 all_chunks.json 构建 BM25）。

### 4. 网络配置 ✅
- 所有服务在 `rag-net` bridge 网络，用服务名通信。
- 验证容器间连通：
```bash
docker compose exec rag-app curl -fsS http://milvus:9091/healthz   # app→milvus
docker compose exec rag-app sh -c "curl -s redis:6379 || echo reachable"
```
- 与其他服务（如 RTMP 服务器）通信：把对方服务加入 `rag-net`，或用 `docker network connect rag-net <容器>` 即可用服务名互访。

## 六、检索模式说明

- **fulltext（默认）**：纯 BM25，仅依赖挂载的 `output/chunks/all_chunks.json`，**Milvus 空也能跑**，最稳，覆盖全部文档。
- **vector / hybrid**：需 Milvus 有向量数据。灌数据见下。

### 给容器内 Milvus 灌向量（可选，启用 vector/hybrid）
```bash
# 在 rag-app 容器内用挂载的 all_chunks.json 重建（CPU 较慢，一次性）
docker compose exec rag-app python backend/rebuild_unified_index.py --yes
# 完成后把 .env 的 APP_RETRIEVAL_MODE 改 hybrid，重启 rag-app
docker compose up -d rag-app
```

## 七、运维

```bash
docker compose logs -f rag-app     # 跟踪日志
docker compose down                # 停止（保留卷/数据）
docker compose down -v             # 停止并删除卷（清空数据，慎用）
docker compose up -d --build       # 改代码后重建并重启
```

## 八、回滚

- 仅停服务：`docker compose down`（数据卷保留，再 `up -d` 即恢复）。
- 镜像有问题：`docker compose build --no-cache rag-app` 重建。
- 本部署独立于你现有的开发用 milvus-standalone/mariadb 容器，互不影响。
