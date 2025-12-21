import streamlit as st
import requests
import os
import time

# Configuración de página
st.set_page_config(page_title="Requerimientos AI", page_icon="🤖")

st.title("🤖 Clasificador de Requisitos")
st.markdown("Escribe un requisito de software y el modelo federado determinará si es **Funcional** o **No Funcional**.")

# Obtener URL de la VM desde variables de entorno (Inyectada por Pulumi)
VM_API_URL = os.environ.get("VM_API_URL", "http://LOCALHOST:8000")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario (Estilo Chat Gemini)
if prompt := st.chat_input("Escribe tu requisito aquí..."):
    # 1. Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Consultar a la VM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Pensando...")
        
        try:
            # Petición a la API en la VM
            response = requests.post(
                f"{VM_API_URL}/predict",
                json={"text": prompt},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                label = data['label']
                conf = data['confidence']
                
                # Formatear respuesta bonita
                if "No Funcional" in label:
                    color = "red"
                    icon = "🛡️"
                else:
                    color = "green"
                    icon = "⚙️"
                
                full_response = f"### Resultado: :{color}[{label}]\n\n**Confianza:** {conf:.2%}\n\n{icon} *Procesado por el Modelo Federado en VM*"
            else:
                full_response = f"⚠️ Error en el servidor: {response.status_code}"

        except Exception as e:
            full_response = f"❌ **Error de conexión:** No se pudo contactar a la VM.\n\nDetalle: `{e}`\n\nAsegúrate de que la VM esté encendida y el puerto 8000 abierto."

        # Mostrar respuesta final
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})