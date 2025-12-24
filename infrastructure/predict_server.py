from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = Flask(__name__)
CORS(app)  # Permite peticiones desde otros dominios (como Colab o una web)

# ====================================================
# CONFIGURACIÓN Y CARGA DEL MODELO
# ====================================================
MODEL_NAME = "microsoft/mpnet-base"
MODEL_PATH = "mpnet_fed_requirements.pth"
ID2LABEL = {0: "Funcional (F)", 1: "No Funcional (NF)"}
DEVICE = torch.device("cpu")

print("Cargando modelo y tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✓ Pesos federados cargados desde {MODEL_PATH}")
else:
    print(f"⚠ Alerta: No se encontró {MODEL_PATH}. Usando modelo base sin entrenar.")

model.to(DEVICE)
model.eval()

# ====================================================
# RUTA DE PREDICCIÓN
# ====================================================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No se proporcionó el campo 'text'"}), 400

    text = data['text']
    
    # Preprocesamiento
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=32
    ).to(DEVICE)

    # Inferencia
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][prediction].item()

    return jsonify({
        "requirement": text,
        "prediction": ID2LABEL[prediction],
        "class_id": prediction,
        "confidence": round(confidence, 4)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ready", "model": MODEL_NAME}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)