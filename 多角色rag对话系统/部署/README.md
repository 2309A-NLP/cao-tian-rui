## 安装依赖

```bash
pip install -r requirements.txt

# 或手动安装
pip install fastapi uvicorn pydantic
pip install langchain langchain-community langchain-huggingface langchain-milvus
pip install pymilvus redis pymysql
pip install sentence-transformers PyMuPDF PyPDF2 python-docx
pip install ollama
pip install python-multipart   # 文件上传功能必需
pip install sseclient-py       # 流式输出测试（可选）