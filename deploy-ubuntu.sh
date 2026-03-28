#!/bin/bash
# ============================================
# Attendance AI - Ubuntu Server 24.04 LTS 部署脚本
# 部署到 80 端口
# ============================================

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="attendance-ai"
PORT=80

echo "=========================================="
echo " Attendance AI 部署脚本"
echo " Ubuntu Server 24.04 LTS 64bit"
echo "=========================================="
echo ""

# 1. 安装系统依赖
echo "[1/6] 安装系统依赖..."
sudo apt-get update -y
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng

echo "  Python3: $(python3 --version)"
echo "  Tesseract: $(tesseract --version 2>&1 | head -1)"

# 2. 创建虚拟环境
echo "[2/6] 创建 Python 虚拟环境..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  虚拟环境已创建: $VENV_DIR"
else
    echo "  虚拟环境已存在，跳过创建"
fi

# 3. 安装 Python 依赖
echo "[3/6] 安装 Python 依赖..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q
pip install \
    "opencv-python-headless>=4.8.0" \
    "numpy>=1.24.0" \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "python-multipart>=0.0.6" \
    "pytesseract>=0.3.10" \
    "scipy>=1.10.0" \
    -q
echo "  Python 依赖安装完成"

# 4. 创建必要目录
echo "[4/6] 创建必要目录..."
mkdir -p "$APP_DIR/uploads"
mkdir -p "$APP_DIR/debug_output"
echo "  目录已就绪"

# 5. 创建 systemd 服务
echo "[5/6] 配置 systemd 服务..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Attendance AI Service
After=network.target

[Service]
Type=simple
User=$(whoami)
Group=$(id -gn)
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/uvicorn server:app --host 0.0.0.0 --port $PORT --workers 2
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=LANG=en_US.UTF-8

# 安全加固
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$APP_DIR/uploads $APP_DIR/debug_output

[Install]
WantedBy=multi-user.target
EOF

# 6. 启动服务
echo "[6/6] 启动服务..."
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}

sleep 3
if sudo systemctl is-active --quiet ${SERVICE_NAME}; then
    IP=$(hostname -I | awk '{print $1}')
    echo ""
    echo "=========================================="
    echo " 部署成功!"
    echo "=========================================="
    echo " 访问地址:  http://${IP}"
    echo ""
    echo " 常用命令:"
    echo "   查看状态: sudo systemctl status ${SERVICE_NAME}"
    echo "   查看日志: sudo journalctl -u ${SERVICE_NAME} -f"
    echo "   重启服务: sudo systemctl restart ${SERVICE_NAME}"
    echo "   停止服务: sudo systemctl stop ${SERVICE_NAME}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo " 错误: 服务启动失败!"
    echo "=========================================="
    echo " 请检查日志:"
    echo "   sudo journalctl -u ${SERVICE_NAME} -n 50 --no-pager"
    echo ""
    echo " 也可以手动测试运行:"
    echo "   cd $APP_DIR"
    echo "   source venv/bin/activate"
    echo "   uvicorn server:app --host 0.0.0.0 --port 80"
    echo "=========================================="
    exit 1
fi
