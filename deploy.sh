#!/bin/bash
# ============================================
# Attendance AI - Linux 部署脚本
# 部署到 80 端口
# ============================================

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="attendance-ai"
PORT=80

echo "=========================================="
echo " Attendance AI 部署脚本 (Linux)"
echo "=========================================="

# 1. 检查 Python
echo "[1/5] 检查 Python 环境..."
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "错误: 未找到 Python，请先安装 Python 3.8+"
    exit 1
fi
echo "  Python: $($PYTHON --version)"

# 2. 检查 Tesseract
echo "[2/5] 检查 Tesseract OCR..."
if ! command -v tesseract &>/dev/null; then
    echo "  Tesseract 未安装，尝试自动安装..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim
    elif command -v yum &>/dev/null; then
        sudo yum install -y tesseract tesseract-langpack-chi_sim
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y tesseract tesseract-langpack-chi_sim
    else
        echo "错误: 无法自动安装 Tesseract，请手动安装"
        exit 1
    fi
fi
echo "  Tesseract: $(tesseract --version 2>&1 | head -1)"

# 3. 创建虚拟环境并安装依赖
echo "[3/5] 创建虚拟环境并安装依赖..."
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$APP_DIR/requirements.txt" -q
echo "  依赖安装完成"

# 4. 创建 systemd 服务
echo "[4/5] 配置 systemd 服务..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Attendance AI Service
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/uvicorn server:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# 5. 启动服务
echo "[5/5] 启动服务..."
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}

sleep 2
if sudo systemctl is-active --quiet ${SERVICE_NAME}; then
    echo ""
    echo "=========================================="
    echo " 部署成功!"
    echo " 访问地址: http://$(hostname -I | awk '{print $1}'):${PORT}"
    echo " 服务状态: sudo systemctl status ${SERVICE_NAME}"
    echo " 查看日志: sudo journalctl -u ${SERVICE_NAME} -f"
    echo " 停止服务: sudo systemctl stop ${SERVICE_NAME}"
    echo "=========================================="
else
    echo ""
    echo "错误: 服务启动失败，请检查日志:"
    echo "  sudo journalctl -u ${SERVICE_NAME} -n 50"
    exit 1
fi
