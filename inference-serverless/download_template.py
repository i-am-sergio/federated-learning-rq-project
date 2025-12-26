# download_template.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Carpeta donde guardaremos la plantilla para que Pulumi la suba
save_path = "./model_base_template" 
os.makedirs(save_path, exist_ok=True)

print(f"⬇️ Descargando plantilla mpnet-base a {save_path}...")
model_name = "microsoft/mpnet-base"

# Descargamos solo la config y tokenizer. Los pesos base vienen también pero
# son pequeños comparados con el historial completo de git de HF.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("✅ Plantilla lista para Pulumi.")
