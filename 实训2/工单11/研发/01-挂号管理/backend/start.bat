@echo off
chcp 65001 > nul
echo ========================================
echo  工单11 医疗挂号Agent  端口: 8011
echo ========================================

cd /d "%~dp0"

REM 检查 venv
if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境，请先执行:
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM 首次运行或数据库为空时初始化
venv\Scripts\python.exe scripts/init_db.py
if errorlevel 1 (
    echo [警告] 数据库初始化出现错误，但服务仍会尝试启动
)

echo.
echo [启动 FastAPI 服务...]
venv\Scripts\python.exe -m uvicorn src.app:app --host 0.0.0.0 --port 8011 --reload

pause
