from fastapi import FastAPI
import os
import traceback
import sys

app = FastAPI()

# Variables globales para estado
model = None
tokenizer = None
startup_error = None  # Aquí guardaremos el error si explota

# Intentamos cargar todo dentro de un bloque seguro
try:
    print("--- INICIANDO CARGA DE MODELO ---")
    from google.cloud import storage
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import shutil

    # Configuración
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    LOCAL_MODEL_PATH = "/tmp/prod_model"
    # Usamos un modelo más ligero por si acaso (TinyBERT) o el mismo base
    MODEL_BASE = "microsoft/mpnet-base" 

    def download_model():
        if not BUCKET_NAME:
            print("Warning: BUCKET_NAME no definido.")
            return False
            
        print(f"Buscando modelo en bucket: {BUCKET_NAME}...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        # Verificamos si podemos listar (prueba de permisos)
        blobs = list(bucket.list_blobs(prefix="prod_model/"))
        
        if not blobs:
            print("No se encontró modelo entrenado. Usando base.")
            return False

        if os.path.exists(LOCAL_MODEL_PATH):
            shutil.rmtree(LOCAL_MODEL_PATH)
        os.makedirs(LOCAL_MODEL_PATH)

        for blob in blobs:
            filename = os.path.basename(blob.name)
            if filename:
                blob.download_to_filename(f"{LOCAL_MODEL_PATH}/{filename}")
        return True

    # Ejecutar lógica
    has_trained_model = download_model()
    
    print("Cargando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    
    print("Cargando Modelo...")
    model_path = LOCAL_MODEL_PATH if has_trained_model else MODEL_BASE
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print("--- CARGA EXITOSA ---")

except Exception as e:
    # SI ALGO FALLA, NO MATAMOS EL CONTENEDOR. GUARDAMOS EL ERROR.
    startup_error = traceback.format_exc()
    print(f"CRITICAL STARTUP ERROR: {e}")
    # No hacemos 'raise', dejamos que uvicorn inicie

from pydantic import BaseModel
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: TextRequest):
    if startup_error:
        return {"error": "Service failed to start correctly", "details": startup_error}
    if not model:
        return {"error": "Model not loaded properly"}
        
    inputs = tokenizer(payload.text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_idx = torch.argmax(logits, dim=-1).item()
    label = "F" if pred_idx == 0 else "NF"
    confidence = torch.softmax(logits, dim=-1).max().item()
    return {"label": label, "confidence": confidence}

@app.get("/health")
def health():
    # Este endpoint nos dirá la verdad
    if startup_error:
        print(f"Health check failed with startup error: {startup_error}")
        return {"status": "error", "logs": startup_error}
    return {"status": "ok", "model_loaded": model is not None}