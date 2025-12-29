import functions_framework
import pickle
import os
import json
import requests
import traceback
from google.cloud import storage

# Configuración
BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME")
CLOUD_RUN_BINARY_URL = os.environ.get("CLOUD_RUN_BINARY_URL")
CLOUD_RUN_MULTI_URL = os.environ.get("CLOUD_RUN_MULTI_URL")

MODEL_BINARY = "fast_model.pkl"
MODEL_MULTI = "multiclass.pkl"
PATH_BINARY = f"/tmp/{MODEL_BINARY}"
PATH_MULTI = f"/tmp/{MODEL_MULTI}"

def download_if_not_exists(bucket_name, blob_name, local_path):
    if not os.path.exists(local_path):
        print(f"⬇️ Descargando {blob_name}...")
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
        except Exception as e:
            print(f"⚠️ No se pudo descargar {blob_name}: {e}")

# --- WARMUP SEGURO ---
print("⚡ FOG: Inicializando modelos...")
try:
    download_if_not_exists(BUCKET_NAME, MODEL_BINARY, PATH_BINARY)
    download_if_not_exists(BUCKET_NAME, MODEL_MULTI, PATH_MULTI)
except Exception as e:
    print(f"⚠️ Error descargando modelos: {e}")

model_bin = None
model_multi = None
load_error = None

# Intentamos cargar, pero si falla, NO ROMPEMOS LA APP
try:
    if os.path.exists(PATH_BINARY):
        with open(PATH_BINARY, 'rb') as f:
            model_bin = pickle.load(f)
    
    if os.path.exists(PATH_MULTI):
        with open(PATH_MULTI, 'rb') as f:
            model_multi = pickle.load(f)
    print("✅ Modelos cargados en memoria")
except Exception as e:
    load_error = str(e)
    print(f"❌ ERROR CRÍTICO CARGANDO MODELOS LOCALES: {load_error}")
    print(traceback.format_exc())

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
    text = request_json.get('text', '')
    mode = request_json.get('mode', 'binary') 
    
    # URL Destino
    target_url = CLOUD_RUN_MULTI_URL if mode == 'multiclass' else CLOUD_RUN_BINARY_URL

    # --- LÓGICA DE FALLBACK ---
    # Si los modelos locales no existen, OFF-LOAD directo
    if model_bin is None or model_multi is None:
        print(f"⚠️ Modelos locales no disponibles. Forzando Cloud Run. Razón: {load_error}")
        return call_cloud_run(target_url, text, mode)

    # Selección del modelo local
    active_model = model_multi if mode == 'multiclass' else model_bin
    
    # 1. Inferencia Local Intentada
    try:
        # Obtener probabilidades
        probs = active_model.predict_proba([text])[0]
        
        # Casting explícito a tipos nativos de Python para evitar error de JSON
        confidence = float(max(probs))
        pred_idx = int(probs.argmax())
        
        # Obtener etiqueta cruda
        raw_label = active_model.classes_[pred_idx]
        
        # --- LÓGICA ESPECÍFICA POR MODO ---
        threshold_conf = 0.0
        threshold_len = 0
        pred_label = ""

        if mode == 'multiclass':
            # MODO MULTICLASE
            # Labels son strings ('SE', 'US', etc.) según tu entrenamiento
            pred_label = str(raw_label) 
            
            # Requisitos: Confianza > 0.50 Y Largo < 120
            threshold_conf = 0.3
            threshold_len = 120
            
        else:
            # MODO BINARIO
            # Labels son 0 (F) o 1 (NF) según tu entrenamiento
            # 0 -> F, 1 -> NF
            if str(raw_label) == '0':
                pred_label = "F"
            else:
                pred_label = "NF"
                
            # Requisitos: Confianza >= 0.70 Y Largo < 100
            threshold_conf = 0.65
            threshold_len = 120

        # 2. Decisión de Offloading
        # Si la confianza supera el umbral Y el texto es corto => RESPONDER DESDE FOG
        if confidence > threshold_conf and len(text) < threshold_len:
            return (json.dumps({
                "prediction": pred_label,
                "confidence": confidence,
                "source": f"FOG (FastModel - {mode})",
                "offloaded": False
            }), 200, {'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json'})
        
        else:
            return call_cloud_run(target_url, text, mode)

    except Exception as e:
        print(f"❌ Error durante inferencia local: {e}")
        print(traceback.format_exc())
        return call_cloud_run(target_url, text, mode)

def call_cloud_run(url, text, mode):
    print(f"📉 Offloading a Cloud Run ({mode}): {url}")
    try:
        resp = requests.post(f"{url}/predict", json={"text": text, "mode": mode})
        return (resp.content, resp.status_code, {'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json'})
    except Exception as e:
        return (json.dumps({"error": f"Fallo total (Fog y Cloud): {str(e)}"}), 500, {'Access-Control-Allow-Origin': '*'})