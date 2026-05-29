# -*- coding: utf-8 -*-
"""
RAG 角色扮演系统 - 主入口

⚠️ 常改动的地方：
1. 端口号（8001）如需更改，请修改 uvicorn.run 中的 port 参数
2. 知识库路径（KNOWLEDGE_BASE_PATH）已在 config.py 中配置，若需切换请修改 config
3. 向量库集合名（MILVUS_COLLECTION_NAME）在 config.py 中
4. 分块参数（chunk_size, chunk_overlap）在 init_knowledge_base 中调用 DataProcessor 时设置
"""

import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# 确保项目根目录在 Python 路径中，以便直接导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
# 注意：日志级别可改为 DEBUG 以便调试，生产环境建议 INFO 或 WARNING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入配置和模块
from config import KNOWLEDGE_BASE_PATH, EMBEDDING_MODEL_PATH
from storage.mysql_client import mysql_client
from storage.redis_client import redis_client
from storage.milvus_client import milvus_client
from api.routes import router, init_rag_components
from processor.data_processor import DataProcessor


# ==================== Embedding 包装类 ====================

class EmbeddingWrapper:
    """
    包装 embedding 模型，提供 embed_query 和 embed_documents 方法
    用于与 LangChain 的 Milvus 向量存储兼容

    ⚠️ 注意事项：
    1. 确保模型路径（EMBEDDING_MODEL_PATH）正确，且模型维度与 Milvus 集合一致
    2. 当前使用 sentence-transformers 框架，若更换模型需确保 encode 方法返回 numpy 数组或列表
    3. 批量处理时注意内存占用，超大文档集建议分批
    """

    def __init__(self, model_path: str):
        from sentence_transformers import SentenceTransformer
        logger.info(f"加载 Embedding 模型: {model_path}")
        self.model = SentenceTransformer(model_path)

    def embed_query(self, text: str):
        """单个查询文本嵌入，返回列表格式"""
        # 注意：.tolist() 将 numpy 数组转为 Python 列表，LangChain 需要
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list):
        """批量文档嵌入，返回列表的列表"""
        # 注意：批量编码可提速，但需确保 texts 不为空
        return self.model.encode(texts).tolist()


# ==================== 初始化知识库 ====================

def init_knowledge_base():
    """
    初始化知识库：扫描 KNOWLEDGE_BASE_PATH 目录，解析文档并生成分块
    返回 Document 列表（用于 LangChain）

    ⚠️ 常改动：
    1. chunk_size（分块大小）和 chunk_overlap（重叠长度）：根据文档类型调整
       - 医疗指南：700/100
       - 法律条文：900/150
       此处默认使用 800/100，实际处理时 DataProcessor 内部会根据文件类型选择策略
    2. 如需强制重新处理所有文档，可先清空 data/processed_docs/ 目录并删除 Milvus 集合

    ⚠️ 注意事项：
    1. 如果知识库目录不存在，则跳过初始化并返回 None
    2. 该方法会生成 knowledge_base_preview.json 用于前端预览
    3. 返回的 documents 可直接用于 Milvus.from_documents()
    """
    logger.info("正在加载知识库...")

    if not KNOWLEDGE_BASE_PATH.exists():
        logger.warning(f"知识库目录不存在: {KNOWLEDGE_BASE_PATH}")
        return None

    # 实例化数据处理器，默认分块参数（实际处理时会被策略覆盖）
    processor = DataProcessor(chunk_size=800, chunk_overlap=100)
    documents = processor.process_directory(str(KNOWLEDGE_BASE_PATH))

    if not documents:
        logger.warning("没有找到知识库文档")
        return None

    # 保存预览文件（用于前端展示文档统计信息）
    processor.save_preview(documents, "knowledge_base_preview.json")
    processor.print_summary()
    return documents


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器（FastAPI 启动时和关闭时的钩子）

    ⚠️ 注意事项：
    1. 启动时会依次连接 MySQL、Redis、Milvus，任何一个失败都会记录错误但继续启动
    2. 向量库（Milvus）如果不存在，则尝试根据知识库文档创建；若创建失败则 RAG 不可用
    3. 关闭时会断开数据库连接，避免资源泄露
    """
    logger.info("=" * 50)
    logger.info("启动 RAG 角色扮演系统")
    logger.info("=" * 50)

    # 1. 连接数据库（MySQL, Redis, Milvus）
    try:
        mysql_client.connect()
        redis_client.connect()
        milvus_client.connect()
        logger.info("所有服务连接成功")
    except Exception as e:
        logger.error(f"服务连接失败: {e}")

    # 2. 初始化向量存储（Milvus + Embedding）
    from langchain_community.vectorstores import Milvus
    from config import MILVUS_COLLECTION_NAME, MILVUS_INDEX_PARAMS

    embedding = EmbeddingWrapper(EMBEDDING_MODEL_PATH)

    # 检查 Milvus 中是否已存在集合
    if milvus_client.has_collection(MILVUS_COLLECTION_NAME):
        # 已有集合：直接加载，不再重复处理文档（避免覆盖已存在的向量）
        vector_store = Milvus(
            embedding_function=embedding,
            collection_name=MILVUS_COLLECTION_NAME,
            connection_args={"host": "localhost", "port": "19530"}
        )
        logger.info(f"加载已有向量库: {MILVUS_COLLECTION_NAME}")
    else:
        # 首次运行，需要创建向量库：处理知识库文档 → 生成向量 → 存入 Milvus
        documents = init_knowledge_base()
        if documents:
            vector_store = Milvus.from_documents(
                documents=documents,
                embedding=embedding,
                collection_name=MILVUS_COLLECTION_NAME,
                connection_args={"host": "localhost", "port": "19530"},
                index_params=MILVUS_INDEX_PARAMS,
                drop_old=True          # 若同名集合存在则删除重建（但上面已判断不存在，此处安全）
            )
            logger.info(f"创建向量库成功，共 {len(documents)} 个向量")
        else:
            vector_store = None
            logger.warning("无法创建向量库")

    # 3. 初始化 RAG 组件（检索器、LLM 等），该函数在 api/routes.py 中定义
    if vector_store:
        init_rag_components(vector_store, embedding)
        logger.info("RAG 组件初始化成功")

    yield  # 此处应用开始处理请求

    # 关闭时执行清理
    logger.info("正在关闭服务...")
    milvus_client.disconnect()
    mysql_client.disconnect()


# ==================== 创建应用 ====================

app = FastAPI(
    title="AI智能助手",
    description="基于RAG的角色扮演对话系统",
    version="1.0.0",
    lifespan=lifespan
)

# 跨域中间件：允许所有来源（开发环境实用，生产环境建议限制具体域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由（定义在 api/routes.py）
app.include_router(router)

# 静态文件目录（存放前端页面）
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """根路径返回前端聊天页面"""
    return FileResponse("static/index.html")


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("启动 Web 服务...")
    print("访问地址: http://127.0.0.1:8001")
    print("API 文档: http://127.0.0.1:8001/docs")
    print("=" * 60 + "\n")

    # ⚠️ 常改动：如需更换端口，修改 port 参数；若需生产部署（禁用 reload），设置 reload=False
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=True          # 开发模式下自动重启，生产环境请设为 False
    )