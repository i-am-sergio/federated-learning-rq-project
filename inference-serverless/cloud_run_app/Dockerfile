FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Instalar PyTorch CPU (Ligero)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Instalar dependencias python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiar código fuente (app.py)
COPY app.py .

# Nota: Ya NO copiamos ni descargamos modelos aquí. 
# La imagen queda "agnóstica" al modelo.

ENV PORT=8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app