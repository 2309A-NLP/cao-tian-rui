import uvicorn

if __name__ == "__main__":
    port = 8010
    print("启动 FastAPI 后端服务器...")
    print(f"访问地址: http://localhost:{port}")
    print(f"API 文档: http://localhost:{port}/docs")
    print("按 Ctrl+C 停止服务器")
    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=8010,
        reload=False
    )
