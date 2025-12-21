from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from contextlib import asynccontextmanager

# Configuración
MODEL_PATH = "mpnet_fed_requirements.pth"
BASE_MODEL = "microsoft/mpnet-base"
device = torch.device("cpu")

# Variables globales
tokenizer = None
model = None

def load_model_logic():
    """Lógica central para cargar/recargar el modelo"""
    global tokenizer, model
    print("--- (RE)CARGANDO MODELO ---")
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    temp_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    
    if not os.path.exists(MODEL_PATH):
        bucket_name = os.environ.get("MODEL_BUCKET_NAME")
        if bucket_name:
            print(f"Modelo local no encontrado. Buscando en Bucket: {bucket_name}...")
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(MODEL_PATH)
                if blob.exists():
                    print("Descargando de GCS...")
                    blob.download_to_filename(MODEL_PATH)
                    print("Descarga completada.")
            except Exception as e:
                print(f"No se pudo descargar del bucket: {e}")
                
    if os.path.exists(MODEL_PATH):
        print(f"Cargando pesos federados desde: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        temp_model.load_state_dict(state_dict)
        status = "Federated Model Loaded"
    else:
        print("! Modelo federado no encontrado. Usando base (Esperando entrenamiento...)")
        status = "Base Model (Untrained)"
    
    temp_model.to(device)
    temp_model.eval()
    
    model = temp_model
    return status

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_logic()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequirementRequest(BaseModel):
    text: str

@app.post("/reload_model")
def reload_model_endpoint():
    """Endpoint para forzar la recarga del modelo sin reiniciar el servidor"""
    status = load_model_logic()
    return {"message": "Modelo recargado exitosamente", "current_status": status}

@app.post("/predict")
def predict(req: RequirementRequest):
    if not tokenizer or not model:
        return {"error": "El modelo aún no está listo."}

    inputs = tokenizer(
        req.text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    pred_idx = torch.argmax(logits, dim=-1).item()
    confidence = torch.softmax(logits, dim=-1).max().item()
    
    label_str = "Funcional (F)" if pred_idx == 0 else "No Funcional (NF)"
    
    return {
        "label": label_str,
        "confidence": confidence,
        "raw_pred": pred_idx
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_exists": os.path.exists(MODEL_PATH)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)