import os
import torch
from flask import Flask, request, jsonify
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Configuración
BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME")
# Variable clave: Define si este contenedor servirá Binario o Multiclase
SERVING_MODE = os.environ.get("SERVING_MODE", "binary") 

TEMPLATE_PREFIX = "templates/mpnet-base"

# Nombres de archivos
WEIGHTS_BINARY = "mpnet_fed_requirements.pth"
WEIGHTS_MULTI = "mpnet_fed_multiclass.pth"

# Rutas Locales
LOCAL_BASE_PATH = "/tmp/model_base"
LOCAL_W_BIN = f"/tmp/{WEIGHTS_BINARY}"
LOCAL_W_MULTI = f"/tmp/{WEIGHTS_MULTI}"

# Mapeo de Clases
LABELS_MULTI = ['A', 'F', 'FT', 'L', 'LF', 'MN', 'O', 'PE', 'PO', 'SC', 'SE', 'US']
ID2LABEL_MULTI = {i: label for i, label in enumerate(LABELS_MULTI)}

def download_folder_from_gcs(bucket_name, prefix, local_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    for blob in blobs:
        if blob.name.endswith("/"): continue
        filename = os.path.basename(blob.name)
        blob.download_to_filename(os.path.join(local_path, filename))

def download_file(bucket_name, blob_name, local_path):
    if not os.path.exists(local_path):
        print(f"⬇️ Descargando {blob_name}...")
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
        except Exception as e:
            print(f"⚠️ Error descargando {blob_name}: {e}")

print(f"🚀 CLOUD RUN ({SERVING_MODE.upper()}): Iniciando servicio...")

# 1. Descargar Plantilla Base (Común)
if not os.path.exists(LOCAL_BASE_PATH):
    download_folder_from_gcs(BUCKET_NAME, TEMPLATE_PREFIX, LOCAL_BASE_PATH)

# 2. Descargar SOLO el peso necesario según el modo
if SERVING_MODE == 'multiclass':
    download_file(BUCKET_NAME, WEIGHTS_MULTI, LOCAL_W_MULTI)
else:
    download_file(BUCKET_NAME, WEIGHTS_BINARY, LOCAL_W_BIN)

# 3. Carga de Modelo en Memoria
try:
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE_PATH, local_files_only=True)
except Exception as e:
    print(f"❌ Error fatal cargando Tokenizer: {e}")
    raise e

model = None

def load_model(weights_path, num_labels):
    try:
        # ignore_mismatched_sizes=True permite adaptar la arquitectura base
        m = AutoModelForSequenceClassification.from_pretrained(
            LOCAL_BASE_PATH, 
            num_labels=num_labels, 
            local_files_only=True,
            ignore_mismatched_sizes=True 
        )
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            m.load_state_dict(state_dict)
            m.eval()
            print(f"✅ Modelo {SERVING_MODE} cargado exitosamente.")
            return m
        else:
            print(f"❌ Archivo no encontrado: {weights_path}")
            return None
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None

# Cargar UN solo modelo
if SERVING_MODE == 'multiclass':
    model = load_model(LOCAL_W_MULTI, num_labels=12)
else:
    model = load_model(LOCAL_W_BIN, num_labels=2)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load"}), 500

    try:
        data = request.get_json()
        text = data.get('text', '')
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            conf, pred_id = torch.max(probs, dim=-1)
            
        pred_idx = pred_id.item()
        confidence = float(conf.item())
        
        # Decodificar etiqueta según el modo actual
        if SERVING_MODE == 'multiclass':
            label = ID2LABEL_MULTI.get(pred_idx, "UNKNOWN")
        else:
            label = "F" if pred_idx == 0 else "NF"
        
        return jsonify({
            "prediction": label,
            "confidence": confidence,
            "source": f"CLOUD (DeepModel - {SERVING_MODE})",
            "offloaded": True
        })

    except Exception as e:
        print(f"Error inferencia: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))