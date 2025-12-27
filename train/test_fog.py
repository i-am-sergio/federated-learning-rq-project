import requests

API_URL = "https://fog-node-fn-c159cd6-y5dphoazqq-uc.a.run.app"

def run_tests():
    test_cases = [
        # ==============================================================================
        # BLOQUE FOG (Cortos < 100 chars)
        # ==============================================================================
        
        # --- 4 Fog NF (Non-Functional - Existentes) ---
        "The application shall be compatible with iOS and Android platforms.",
        "The product shall be compatible with web browsers.",
        "The app must load in under 2 seconds.",
        "The system shall run seamlessly on Unix and Windows operating systems.",
        
        # --- 4 Fog F (Functional) -> Pred: F (Corregidos con frases exactas del CSV) ---
        "The system shall calculate tax for the order.",
        "The system shall display all the available promotions to the user.",
        "The system shall allow user to select the financing option.",
        "The secondary database shall store historical statistics.",
        # ==============================================================================
        # BLOQUE CLOUD (Largos > 100 chars)
        # ==============================================================================

        # --- 4 Cloud NF (Non-Functional - Existentes) ---
        "The application must ensure data integrity using AES-256 encryption according to GDPR compliance mandates for all european users.",
        "Ideally, the overarching microservices architecture should seamlessly integrate disjointed legacy protocols while ensuring latency stays low.",
        "The Kubernetes cluster must automatically scale the pods horizontally based on custom metrics derived from the Prometheus adapter.",
        "It is absolutely critical that the system provides a comprehensive audit trail of all transaction logs including timestamps and user IDs.",

        # --- 4 Cloud F (Functional - Nuevos/Expandidos) ---
        "The system shall allow the user to enter the order information for tracking and display the current tracking information about the order.",
        "The system shall allow the administrator to manually input new inventory levels including SKU numbers and warehouse locations.",
        "The system shall calculate the total cost of the shopping cart including local taxes and shipping fees based on the user's address.",
        "The system shall enable the user to enter their reviews and ratings for the items they purchased in the transaction.",
    ]

    print(f"{'TEXTO (Recortado)':<45} | {'CHARS':<5} | {'DESTINO':<7} | {'PRED':<4} | {'CONF %'}")
    print("-" * 90)

    for text in test_cases:
        text = text.strip()
        if not text: continue

        try:
            response = requests.post(API_URL, json={"text": text})
            
            if response.status_code == 200:
                res = response.json()
                
                is_offloaded = res.get("offloaded", False)
                source = "CLOUD" if is_offloaded else "FOG"
                
                prediction = res.get("prediction", "N/A")
                confidence = res.get("confidence", 0) * 100
                chars_count = len(text)
                
                text_preview = (text[:40] + '..') if len(text) > 40 else text
                
                # Colores: Verde (FOG), Amarillo (CLOUD)
                color = "\033[92m" if source == "FOG" else "\033[93m"
                reset = "\033[0m"
                
                print(f"{text_preview:<45} | {chars_count:<5} | {color}{source:<7}{reset} | {prediction:<4} | {confidence:.2f}%")
            
            else:
                print(f"Error {response.status_code} para: {text[:20]}...")
                
        except Exception as e:
            print(f"Error de conexion: {e}")

if __name__ == "__main__":
    run_tests()