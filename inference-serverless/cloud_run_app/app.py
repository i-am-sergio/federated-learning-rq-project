import os
import torch
from flask import Flask, request, jsonify
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Configuración
BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME")

TEMPLATE_PREFIX = "templates/mpnet-base"
WEIGHTS_FILE = "mpnet_fed_requirements.pth"

# Rutas Locales (En Cloud Run /tmp es un disco en memoria RAM)
LOCAL_BASE_PATH = "/tmp/model_base"
LOCAL_WEIGHTS_PATH = f"/tmp/{WEIGHTS_FILE}"


def download_folder_from_gcs(bucket_name, prefix, local_path):
    """Descarga todos los archivos de una 'carpeta' en GCS"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        
    print(f"⬇️ Descargando plantilla desde gs://{bucket_name}/{prefix}...")
    count = 0
    for blob in blobs:
        if blob.name.endswith("/"): continue # Ignorar directorios virtuales
        
        # Obtener nombre de archivo limpio
        filename = os.path.basename(blob.name)
        destination = os.path.join(local_path, filename)
        
        blob.download_to_filename(destination)
        count += 1
    print(f"✅ {count} archivos de plantilla descargados.")

print("🚀 CLOUD RUN: Iniciando servicio de inferencia...")

# 1. Descargar Plantilla Base (Config, Tokenizer) si no existe
if not os.path.exists(LOCAL_BASE_PATH):
    download_folder_from_gcs(BUCKET_NAME, TEMPLATE_PREFIX, LOCAL_BASE_PATH)

# 2. Descargar Pesos Entrenados si no existen
if not os.path.exists(LOCAL_WEIGHTS_PATH):
    print(f"⬇️ Descargando pesos entrenados gs://{BUCKET_NAME}/{WEIGHTS_FILE}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(WEIGHTS_FILE)
    blob.download_to_filename(LOCAL_WEIGHTS_PATH)

# 3. Cargar Modelo OFFLINE
print("🧠 Cargando modelo PyTorch desde /tmp...")
try:
    # Cargar tokenizer y config desde la carpeta local descargada
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE_PATH, local_files_only=True)
    
    # Cargar arquitectura base
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_BASE_PATH, num_labels=2, local_files_only=True)
    
    # Cargar tus pesos entrenados (map_location='cpu' es vital en Cloud Run básico)
    state_dict = torch.load(LOCAL_WEIGHTS_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    model.eval()
    print("✅ Modelo cargado exitosamente. Listo para recibir peticiones.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO cargando modelo: {e}")
    # No matamos la app aquí para que Cloud Run logee el error, pero fallará la request

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
    except Exception as e:
        print(f"Error en inferencia: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))