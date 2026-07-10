@echo off
chcp 65001 > nul
echo.
echo  Agent22 - mem0 多领域长期记忆系统
echo  =====================================
echo  后端 API: http://localhost:8022
echo  前端页面: http://localhost:8022
echo  API 文档: http://localhost:8022/docs
echo.

cd /d "F:\kimi  project\agent22\backend"
call venv\Scripts\activate.bat
uvicorn main:app --host 0.0.0.0 --port 8022 --reload
