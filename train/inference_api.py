from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuración CORS para que Cloud Run pueda hablar con la VM
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración del Modelo
MODEL_PATH = "mpnet_fed_requirements.pth" # El archivo que genera tu FL
BASE_MODEL = "microsoft/mpnet-base"
device = torch.device("cpu")

class RequirementRequest(BaseModel):
    text: str

# Variables globales
tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    print("--- INICIANDO API DE INFERENCIA ---")
    
    # 1. Cargar Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 2. Cargar Estructura
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    # 3. Intentar cargar pesos federados
    if os.path.exists(MODEL_PATH):
        print(f"✓ Cargando modelo federado desde: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("! Advertencia: Modelo federado no encontrado. Usando modelo base sin entrenar.")
    
    model.to(device)
    model.eval()

@app.post("/predict")
def predict(req: RequirementRequest):
    if not tokenizer or not model:
        return {"error": "El modelo aún no está listo."}

    # Recargar el modelo si acaba de terminar una ronda de FL y el archivo cambió
    # (Opcional: lógica simplificada para no complicar el código)
    
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
    
    # Mapeo de etiquetas
    label_str = "Funcional (F)" if pred_idx == 0 else "No Funcional (NF)"
    
    return {
        "label": label_str,
        "confidence": confidence,
        "raw_pred": pred_idx
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)