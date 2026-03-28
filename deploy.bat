@echo off
chcp 65001 >nul 2>&1
:: ============================================
:: Attendance AI - Windows 部署脚本
:: 部署到 80 端口
:: ============================================

setlocal enabledelayedexpansion

set APP_DIR=%~dp0
set VENV_DIR=%APP_DIR%venv
set PORT=80
set SERVICE_NAME=AttendanceAI

echo ==========================================
echo  Attendance AI 部署脚本 (Windows)
echo ==========================================
echo.

:: 1. 检查 Python
echo [1/4] 检查 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo   %%i

:: 2. 检查 Tesseract
echo [2/4] 检查 Tesseract OCR...
tesseract --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   警告: 未找到 Tesseract OCR
    echo   请从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装
    echo   安装后将其添加到系统 PATH 环境变量
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('tesseract --version 2^>^&1') do (
    echo   Tesseract: %%i
    goto :tesseract_done
)
:tesseract_done

:: 3. 创建虚拟环境并安装依赖
echo [3/4] 创建虚拟环境并安装依赖...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    python -m venv "%VENV_DIR%"
)
call "%VENV_DIR%\Scripts\activate.bat"
pip install --upgrade pip -q
pip install -r "%APP_DIR%requirements.txt" -q
echo   依赖安装完成

:: 4. 启动服务
echo [4/4] 启动服务 (端口 %PORT%)...
echo.
echo ==========================================
echo  服务启动中...
echo  访问地址: http://localhost:%PORT%
echo  按 Ctrl+C 停止服务
echo ==========================================
echo.

cd /d "%APP_DIR%"
uvicorn server:app --host 0.0.0.0 --port %PORT%

pause
