import functions_framework
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

# Configuración de rutas
MODEL_PATH = "mpnet_fed_requirements.pth"
MODEL_NAME = "microsoft/mpnet-base"

# Carga global para Warm Start
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

@functions_framework.http
def predict_requirement(request):
    request_json = request.get_json(silent=True)
    if not request_json or 'text' not in request_json:
        return {"error": "JSON con campo 'text' requerido"}, 400

    text = request_json['text']
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    label = "F" if prediction == 0 else "NF"
    return (json.dumps({
        "text": text,
        "prediction": label,
        "confidence": float(probs[0][prediction])
    }), 200, {'Content-Type': 'application/json'})