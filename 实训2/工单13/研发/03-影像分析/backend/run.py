"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
启动入口：python run.py

本文件是整个后端服务的启动脚本，使用 uvicorn 这一高性能 ASGI 服务器
来运行 FastAPI 应用。直接运行此文件即可启动 HTTP 服务。
"""

# uvicorn：高性能异步 Web 服务器，专为 ASGI 框架（如 FastAPI）设计
# 用来将 FastAPI 应用暴露为 HTTP 服务
import uvicorn

# Python 约定：当此脚本被直接执行（而非被其他模块 import）时，才运行下面的代码
if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",   # 指定 ASGI 应用的位置，格式为"模块路径:对象名"
        host="0.0.0.0",  # 监听所有网络接口，允许外部访问（非仅 localhost）
        port=8013,        # 服务监听的 TCP 端口号
        reload=False,     # 关闭热重载（生产环境不开，开发时可改为 True 以自动重启）
        log_level="info", # 日志级别：info 表示输出一般信息（不含 debug 细节）
    )
