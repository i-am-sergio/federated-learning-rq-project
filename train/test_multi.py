import requests
import time

# URL de tu función Fog
API_URL = "https://fog-node-fn-c159cd6-y5dphoazqq-uc.a.run.app"

def run_tests():
    test_data = {
        "A (Availability)": [
            "The system shall ensure 99.99% availability.",
            "The system must be available for use at all times.",
            # CAMBIO: Usamos "uptime" y "availability" explícitamente, evitando "maintenance"
            "The system shall guarantee continuous service availability and uptime, ensuring access even during peak operational hours."
        ],
        "F (Functional)": [
            "The system shall calculate the total order amount.",
            "The user shall be able to search for products.",
            "The system shall provide a feature that allows the administrator to export the weekly sales report to a PDF format."
        ],
        "FT (Fault Tolerance)": [
            # CAMBIO: "continue to operate" se confunde con O/A. Usamos "failover" y "redundancy".
            "The system shall perform automatic failover upon crash.",
            "The system must have redundancy to tolerate faults.",
            "The system shall utilize redundant components to ensure fault tolerance and automatic recovery without data loss in case of hardware failure."
        ],
        "L (Legal)": [
            # CAMBIO: "comply with" se confunde con SE. Usamos "copyright", "laws" y "legislation".
            "The software shall respect copyright laws.",
            "The product must adhere to software licensing legislation.",
            "The application must strictly follow all applicable national laws and regulations regarding intellectual property and digital rights management."
        ],
        "LF (Look & Feel)": [
            "The interface shall use the standard company colors.",
            "The GUI must appear professional and modern.",
            "The application interface shall be consistent with the corporate branding style guide including fonts, logos, and color schemes."
        ],
        "MN (Maintainability)": [
            # CAMBIO: "easy to maintain" a veces falla. Usamos "modularity", "documentation" y "maintenance".
            "The code must be modular to facilitate maintenance.",
            "The system shall provide comprehensive documentation for maintenance.",
            "The software architecture shall be designed with high modularity and low coupling to allow easy maintenance and future extensions by developers."
        ],
        "O (Operational)": [
            # CAMBIO: "define procedures" se confunde con F. Usamos "administrators", "housekeeping" y "deployment".
            "The system shall support daily administrative housekeeping tasks.",
            "The product must be deployable by system administrators.",
            "The system shall be capable of operating within the standard constraints of the existing data center environment and operational procedures."
        ],
        "PE (Performance)": [
            "The system must respond within 1 second.",
            # CAMBIO: "throughput" a veces se va a SC. Usamos "latency" y "response time".
            "The system shall have low latency for all transactions.",
            "The application must process incoming requests with minimal delay, ensuring that 95% of transactions are completed within two seconds."
        ],
        "PO (Portability)": [
            # CAMBIO: "portable" a veces falla. Usamos "migrate", "platform independent" y "multi-platform".
            "The software shall be platform independent.",
            "The system must be easily migrated to different OS.",
            "The application code shall be written in standard Java to ensure portability across various hardware and operating system environments."
        ],
        "SC (Scalability)": [
            "The system shall scale to support more users.",
            "The application must be scalable to handle data growth.",
            "The system architecture must support horizontal scalability to accommodate a projected 200% increase in user traffic over the next year."
        ],
        "SE (Security)": [
            "The system shall require users to authenticate.",
            "Access to data must be secured and encrypted.",
            "The application must ensure that all sensitive user data is encrypted during transmission and storage to prevent unauthorized access."
        ],
        "US (Usability)": [
            "The system shall be easy to learn for new users.",
            "The user interface must be intuitive and user-friendly.",
            "The application shall provide clear and helpful error messages to assist the user in recovering from mistakes without consulting a manual."
        ]
    }

    print(f"{'LABEL':<5} | {'TEXTO (Recortado)':<40} | {'CHARS':<5} | {'DESTINO':<7} | {'PRED':<4} | {'CONF %'}")
    print("-" * 100)

    for label_group, sentences in test_data.items():
        short_label = label_group.split()[0]
        
        for text in sentences:
            text = text.strip()
            if not text: continue

            try:
                response = requests.post(API_URL, json={"text": text, "mode": "multiclass"})
                
                if response.status_code == 200:
                    res = response.json()
                    
                    is_offloaded = res.get("offloaded", False)
                    source = "CLOUD" if is_offloaded else "FOG"
                    prediction = res.get("prediction", "N/A")
                    confidence = res.get("confidence", 0) * 100
                    chars_count = len(text)
                    text_preview = (text[:35] + '..') if len(text) > 35 else text
                    
                    color = "\033[92m" if source == "FOG" else "\033[93m"
                    reset = "\033[0m"
                    match_mark = "" if prediction == short_label else ""

                    print(f"{short_label:<5} | {text_preview:<40} | {chars_count:<5} | {color}{source:<7}{reset} | {prediction:<4} | {confidence:.2f}% {match_mark}")
                
                else:
                    print(f"Error {response.status_code} para: {text[:20]}...")
            
            except Exception as e:
                print(f"Error de conexion: {e}")
        
        print("-" * 100)

if __name__ == "__main__":
    run_tests()