#!/bin/bash

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git

mkdir -p /app
cd /app

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install flwr==1.5.0
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets pandas numpy scikit-learn fastapi uvicorn

cat << 'EOF' > server.py
{{SERVER_CODE}}
EOF

cat << 'EOF' > inference_api.py
{{INFERENCE_CODE}}
EOF

# Servidor de FL en puerto 8080
nohup python server.py > server.log 2>&1 &

# API de Inferencia en puerto 8000
nohup python inference_api.py > api.log 2>&1 &