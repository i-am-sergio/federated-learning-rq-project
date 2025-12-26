FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Instalar PyTorch CPU
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Instalar resto de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- NUEVO PASO: "Hornear" (Bake) el modelo base en la imagen ---
# Esto descarga config.json, tokenizer.json y vocab.txt y los guarda en /app/model_base
# Así NO necesitamos pedirlos a HuggingFace cuando arranque el servidor.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    save_path = './model_base'; \
    print('⬇️ Descargando modelo base durante el build...'); \
    AutoTokenizer.from_pretrained('microsoft/mpnet-base').save_pretrained(save_path); \
    AutoModelForSequenceClassification.from_pretrained('microsoft/mpnet-base').save_pretrained(save_path)"

# 3. Copiar código
COPY app.py .

ENV PORT=8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app