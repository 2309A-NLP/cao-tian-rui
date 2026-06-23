"""
启动入口
启动命令：python run.py
"""
import uvicorn

if __name__ == "__main__":
    port = 8020
    print(f"========================================")
    print(f"  家庭记账 Agent 智能体")
    print(f"  访问: http://localhost:{port}")
    print(f"  API文档: http://localhost:{port}/docs")
    print(f"========================================")
    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
