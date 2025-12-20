#!/bin/bash
set -e # Detener si hay error

echo "--- Inicio de Provisionamiento (Startup Script) ---"

# 1. Actualizar sistema
apt-get update
apt-get install -y python3-pip python3-venv git

# 2. Preparar directorio de trabajo
mkdir -p /app
cd /app

# 3. Crear entorno virtual
echo "Creando entorno virtual..."
python3 -m venv venv
source venv/bin/activate

# 4. Instalar librerías
echo "Instalando dependencias Python..."
pip install --upgrade pip
pip install flwr torch transformers google-cloud-storage

echo "--- Dependencias instaladas correctamente ---"
# El control vuelve al metadata script inyectado por Pulumi