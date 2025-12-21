FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
# Streamlit corre en 8501, pero Cloud Run inyecta el puerto en $PORT
CMD sh -c "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"