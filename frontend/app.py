import streamlit as st
import requests
import os

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA (Estilo Formal)
# ==========================================
st.set_page_config(
    page_title="Sistema de Clasificación de Requisitos",
    page_icon="💠",
    layout="centered"
)

st.markdown("""
<style>
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    /* Estilo sutil para diferenciar mensajes */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #f9f9f9; 
    }
    /* Ajuste para modo oscuro automática */
    @media (prefers-color-scheme: dark) {
        [data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #262730; 
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BARRA LATERAL (Sidebar)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/ios/100/4a90e2/artificial-intelligence.png", width=80)
    st.header("Panel de Control")
    st.markdown("""
    Este sistema utiliza un modelo de **Aprendizaje Federado (MPNet)** para clasificar requisitos de software.
    
    **Guía de Uso:**
    1. Ingrese el requisito en el chat.
    2. El sistema inferirá si es **Funcional** o **No Funcional**.
    """)
    st.divider()
    st.caption("Estado del Sistema: 🟢 En Línea")
    st.caption("Versión: 2.0.1 (Production)")

# ==========================================
# 3. ENCABEZADO PRINCIPAL
# ==========================================
st.title("💠 Clasificador Inteligente")
st.markdown("##### Asistente de Ingeniería de Requisitos")
st.info("💡 Consejo: Sea específico con el requisito para obtener una mayor confianza en la predicción.")

# Obtener URL de la VM
VM_API_URL = os.environ.get("VM_API_URL", "http://LOCALHOST:8000")

# ==========================================
# 4. GESTIÓN DEL CHAT
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar_icon = "👤" if message["role"] == "user" else "💠"
    
    with st.chat_message(message["role"], avatar=avatar_icon):
        if isinstance(message["content"], dict):
            cols = st.columns([1, 4])
            with cols[0]:
                st.metric("Confianza", f"{message['content']['conf']:.1%}")
            with cols[1]:
                if "No Funcional" in message['content']['label']:
                    st.error(f"🛡️ **No Funcional (NF)**\n\n{message['content']['text']}")
                else:
                    st.success(f"⚙️ **Funcional (F)**\n\n{message['content']['text']}")
        else:
            st.markdown(message["content"])

# ==========================================
# 5. LÓGICA DE INTERACCIÓN
# ==========================================
if prompt := st.chat_input("Ingrese el requisito del sistema..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="💠"):
        with st.spinner("Analizando semántica del requisito..."):
            try:
                response = requests.post(
                    f"{VM_API_URL}/predict",
                    json={"text": prompt},
                    timeout=8
                )
                if response.status_code == 200:
                    data = response.json()
                    label = data['label']
                    conf = data['confidence']
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Certeza", f"{conf:.1%}")
                    with col2:
                        if "No Funcional" in label:
                            st.error(f"**Clasificación: {label}**")
                            explanation = "Este requisito describe una restricción del sistema (seguridad, rendimiento, etc.)."
                        else:
                            st.success(f"**Clasificación: {label}**")
                            explanation = "Este requisito describe un comportamiento o función específica del sistema."
                        st.caption(explanation)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": {"label": label, "conf": conf, "text": explanation}
                    })
                else:
                    st.error(f"⚠️ Error del servidor: {response.status_code}")
            except Exception as e:
                st.warning("No se pudo conectar con el motor de inferencia.")
                st.caption(f"Error técnico: {e}")