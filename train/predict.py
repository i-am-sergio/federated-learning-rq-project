import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_requirement(text):
    # 1. Configuración de nombres y etiquetas
    model_name = "microsoft/mpnet-base"
    id2label = {0: "Funcional (F)", 1: "No Funcional (NF)"}
    device = torch.device("cpu") # Usamos CPU en el server

    print(f"\nAnalizando requisito: '{text}'")

    # 2. Cargar el Tokenizer original
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Cargar la estructura del modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )

    # 4. Cargar tus pesos entrenados (.pth)
    try:
        state_dict = torch.load("mpnet_fed_requirements.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Modo evaluación (importante)
        print("✓ Modelo federado cargado exitosamente.")
    except FileNotFoundError:
        print("X Error: No se encontró el archivo 'mpnet_fed_requirements.pth'")
        return

    # 5. Preprocesar el texto
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=32
    ).to(device)

    # 6. Inferencia (Predicción)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        
        # Opcional: Calcular probabilidad (Softmax)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][prediction].item()

    print(f"\nRESULTADO:")
    print(f"Clase predicha: {id2label[prediction]}")
    print(f"Confianza: {confidence:.2%}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Permite pasar la frase por argumento de terminal
        user_text = " ".join(sys.argv[1:])
        predict_requirement(user_text)
    else:
        # Frases de prueba por defecto
        test_phrases = [
            "The system shall allow users to login with their email.",
            "The database should respond to queries in less than 500ms.",
            "All personal data must be encrypted using AES-256."
        ]
        for phrase in test_phrases:
            predict_requirement(phrase)