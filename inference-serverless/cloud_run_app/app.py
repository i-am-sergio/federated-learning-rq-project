import os
import torch
from flask import Flask, request, jsonify
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Configuración
BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME")
MODEL_FILE = "mpnet_fed_requirements.pth"
# Ruta donde guardamos los pesos entrenados (GCS)
LOCAL_WEIGHTS_PATH = f"/tmp/{MODEL_FILE}"
# Ruta donde guardamos la arquitectura base (Docker) <--- NUEVO
LOCAL_BASE_PATH = "./model_base" 

print("🚀 CLOUD RUN: Iniciando servicio...")

# 1. Descargar pesos entrenados (si no existen)
if not os.path.exists(LOCAL_WEIGHTS_PATH):
    print(f"⬇️ Descargando pesos entrenados de GCS...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE)
    blob.download_to_filename(LOCAL_WEIGHTS_PATH)
    print("✅ Descarga de pesos completada.")

# 2. Cargar Modelo usando la BASE LOCAL
print("🧠 Cargando PyTorch Model desde disco (Offline)...")

# AQUÍ ESTÁ EL CAMBIO: Usamos LOCAL_BASE_PATH en lugar de "microsoft/mpnet-base"
tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE_PATH)
model = AutoModelForSequenceClassification.from_pretrained(LOCAL_BASE_PATH, num_labels=2)

# Cargar tus pesos entrenados encima de la base
model.load_state_dict(torch.load(LOCAL_WEIGHTS_PATH))
model.eval()
print("✅ Servicio listo (Sin conexión a HuggingFace).")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
    
    label = "NF" if pred.item() == 1 else "F"
    
    return jsonify({
        "prediction": label,
        "confidence": float(conf.item()),
        "source": "CLOUD (DeepModel - MPNet)",
        "offloaded": True
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))