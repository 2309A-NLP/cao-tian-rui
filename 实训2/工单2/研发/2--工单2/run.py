"""
日程提醒智能体 — 启动入口
工单编号：人工智能 NLP-Agent 数字人项目-日程提醒智能体任务

用法：python run.py
然后浏览器打开 http://localhost:8001
"""
import uvicorn

if __name__ == "__main__":
    print("日程提醒智能体「小暖」启动中...")
    print("浏览器打开: http://localhost:8001")
    print("=" * 50)
    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )
