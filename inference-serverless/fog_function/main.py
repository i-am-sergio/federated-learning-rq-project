import functions_framework
import pickle
import os
import json
import requests
from google.cloud import storage

# Configuración
BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME")
CLOUD_RUN_URL = os.environ.get("CLOUD_RUN_URL") # URL del Cloud Run
MODEL_FILE = "fast_model.pkl"
LOCAL_PATH = f"/tmp/{MODEL_FILE}"

# --- WARMUP RÁPIDO ---
print("⚡ FOG: Cargando FastModel...")
if not os.path.exists(LOCAL_PATH):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE)
    blob.download_to_filename(LOCAL_PATH)

with open(LOCAL_PATH, 'rb') as f:
    fast_model = pickle.load(f)

@functions_framework.http
def fog_predict(request):
    # CORS
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    request_json = request.get_json(silent=True)
    text = request_json['text']
    
    # 1. Inferencia Local
    probs = fast_model.predict_proba([text])[0]
    confidence = max(probs)
    pred_idx = probs.argmax()
    label = "NF" if pred_idx == 1 else "F"
    
    # 2. Decisión de Offloading (Umbral 0.75)
    if confidence >= 0.75:
        return (json.dumps({
            "prediction": label,
            "confidence": float(confidence),
            "source": "FOG (FastModel)",
            "offloaded": False
        }), 200, {'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json'})
    
    # 3. Offloading a Cloud Run
    else:
        print(f"📉 Offloading a Cloud Run: {CLOUD_RUN_URL}")
        try:
            # Llamada al contenedor pesado
            resp = requests.post(f"{CLOUD_RUN_URL}/predict", json={"text": text})
            return (resp.content, resp.status_code, {'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json'})
        except Exception as e:
            return (json.dumps({"error": str(e)}), 500, {'Access-Control-Allow-Origin': '*'})