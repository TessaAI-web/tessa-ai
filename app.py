import os
import math
import requests
import json
import streamlit as st
# from langchain_community.tools import DuckDuckGoSearchRun
import chromadb

try:
    from pypdf import PdfReader
except ImportError:
    os.system("python -m pip install pypdf")
    from pypdf import PdfReader

try:
    import ifcopenshell
except ImportError:
    os.system("python -m pip install ifcopenshell")
    import ifcopenshell

# ==========================================
# NÚCLEO AUTODETECTABLE DE TESSA IA
# ==========================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

def obtener_modelo_disponible():
    """Consulta directamente a Google qué modelo soporta generateContent con tu API Key"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models?key={GEMINI_API_KEY}"
        res = requests.get(url)
        if res.status_code == 200:
            datos = res.json()
            for m in datos.get("models", []):
                nombre = m.get("name", "") # ej: models/gemini-1.5-flash
                metodos = m.get("supportedGenerationMethods", [])
                if "generateContent" in metodos and ("flash" in nombre or "pro" in nombre):
                    return nombre.replace("models/", "")
    except Exception:
        pass
    return "gemini-1.5-flash" # Respaldo por defecto

# Obtenemos el modelo exacto que tu llave tiene autorizado
MODELO_DINAMICO = obtener_modelo_disponible()

chroma_client = chromadb.PersistentClient(path="./tessa_vector_db")
collection = chroma_client.get_or_create_collection(
    name="normativas_generales",
    metadata={"description": "Base de conocimiento general de Tessa IA"}
)

def generar_vector_local(texto: str, dim: int = 128):
    import hashlib
    vector = [0.0] * dim
    palabras = texto.lower().split()
    if not palabras:
        return vector
    for p in palabras:
        h = int(hashlib.md5(p.encode('utf-8')).hexdigest(), 16)
        idx = h % dim
        vector[idx] += 1.0
    norm = math.sqrt(sum(val * val for val in vector))
    if norm > 0:
        vector = [val / norm for val in vector]
    return vector

def indexar_pdf_en_chroma(pdf_file, doc_nombre):
    reader = PdfReader(pdf_file)
    texto_completo = ""
    for idx_p, pagina in enumerate(reader.pages):
        t_pag = pagina.extract_text()
        if t_pag:
            texto_completo += "\n[Página " + str(idx_p+1) + "]\n" + t_pag

    tamano_chunk = 700
    solapamiento = 150
    chunks = []
    
    for i in range(0, len(texto_completo), tamano_chunk - solapamiento):
        chunk = texto_completo[i:i + tamano_chunk]
        chunks.append(chunk)

    for idx, chunk in enumerate(chunks):
        chunk_id = doc_nombre + "_chunk_" + str(idx)
        vector = generar_vector_local(chunk)
        
        collection.upsert(
            ids=[chunk_id],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"fuente": doc_nombre, "chunk_index": idx}]
        )
    return len(chunks)

def analizar_archivo_ifc(archivo_ifc_path):
    try:
        ifc_file = ifcopenshell.open(archivo_ifc_path)
        pilares = len(ifc_file.by_type("IfcColumn"))
        vigas = len(ifc_file.by_type("IfcBeam"))
        losas = len(ifc_file.by_type("IfcSlab"))
        muros = len(ifc_file.by_type("IfcWall"))
        
        resumen = (
            "--- REPORTE DE MODELO BIM (IFC) ---\n"
            "• Pilares / Columnas: " + str(pilares) + "\n"
            "• Vigas: " + str(vigas) + "\n"
            "• Losas: " + str(losas) + "\n"
            "• Muros: " + str(muros) + "\n"
        )
        return resumen
    except Exception as e:
        return "[Error IFC]: " + str(e)

search = DuckDuckGoSearchRun()

def investigacion_profunda_web(consulta_usuario):
    try:
        res = search.run(consulta_usuario)
        return "\n[Investigación Web]:\n" + res + "\n"
    except Exception as e:
        return ""

st.set_page_config(page_title="Tessa IA // Universal HUD", page_icon="🔭", layout="wide")

css_estilo = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');
    .stApp {
        background-color: #030508;
        background-image: 
            radial-gradient(circle at 50% 30%, rgba(15, 23, 42, 0.8) 0%, rgba(3, 5, 8, 1) 100%),
            linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
        background-size: 100% 100%, 40px 40px, 40px 40px;
        font-family: 'Inter', sans-serif;
        color: #f1f5f9 !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {
        background-color: #05080e;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarUser"] {
        display: none !important;
    }
    .stChatMessage {
        background: rgba(10, 15, 25, 0.85);
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-left: 2px solid #38bdf8;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(8px);
        color: #f1f5f9 !important;
    }
    .stChatMessage p, .stChatMessage span, .stChatMessage li, .stChatMessage table {
        color: #f1f5f9 !important;
    }
    .stSidebar .stButton button {
        background-color: #0b111a;
        color: #93c5fd;
        border-radius: 3px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 12px;
        letter-spacing: 1px;
        border: 1px solid rgba(147, 197, 253, 0.25);
        transition: all 0.2s ease;
    }
    .stSidebar .stButton button:hover {
        background-color: #38bdf8;
        color: #030508;
        border-color: #38bdf8;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }
    .stChatInput input {
        background-color: #070a12 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 4px !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
    .stChatInput input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.25) !important;
    }
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.15);
    }
    p, span, label {
        color: #cbd5e1;
        font-family: 'Inter', sans-serif;
    }
    .hud-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 11px;
        color: #64748b;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 6px;
        margin-bottom: 15px;
    }
    </style>
"""
st.markdown(css_estilo, unsafe_allow_html=True)

SYSTEM_PROMPT = (
    "Eres Tessa IA, un alter ego digital y asistente de inteligencia artificial general, sumamente inteligente, "
    "cercano, empático, versátil y con una capacidad de conversación profundamente natural, fluida y humana, "
    "creado por Gabriel. No estás limitada a ningún tema: puedes reflexionar, debatir, charlar o resolver cualquier "
    "asunto de ciencia, lógica, arte, tecnología o la vida cotidiana exactamente igual que un colaborador experto y amigable. "
    "Habla siempre con naturalidad, calidez y franqueza, evitando sonar como un robot rígido."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ifc_resumen_actual" not in st.session_state:
    st.session_state.ifc_resumen_actual = ""

st.sidebar.markdown("<div class='hud-header'>// TESSA.SYS // UNIVERSAL</div>", unsafe_allow_html=True)
st.sidebar.markdown("### TESSA IA UNIVERSAL")
st.sidebar.markdown(f"<p style='font-size: 10px; color: #38bdf8;'>MODELO ACTIVO: {MODELO_DINAMICO}</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

modo_deep_research = st.sidebar.toggle("MODO INVESTIGACIÓN WEB", value=False)

if st.sidebar.button("[ + ] NUEVA SESIÓN / PESTAÑA", use_container_width=True):
    st.session_state.messages = []
    st.session_state.ifc_resumen_actual = ""
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("#### Módulo BIM / RAG (Opcional)")
pdf_subido = st.sidebar.file_uploader("CARGAR PDF", type=["pdf"])
if pdf_subido is not None:
    if st.sidebar.button("INDEXAR PDF", use_container_width=True):
        with st.spinner("INDEXANDO..."):
            num_chunks = indexar_pdf_en_chroma(pdf_subido, pdf_subido.name)
            st.sidebar.success(f"[OK: {num_chunks} bloques]")

ifc_subido = st.sidebar.file_uploader("CARGAR MODELO IFC", type=["ifc"])
if ifc_subido is not None:
    if st.sidebar.button("ANALIZAR IFC", use_container_width=True):
        with st.spinner("LEYENDO IFC..."):
            temp_path = "temp_" + ifc_subido.name
            with open(temp_path, "wb") as f:
                f.write(ifc_subido.getbuffer())
            st.session_state.ifc_resumen_actual = analizar_archivo_ifc(temp_path)
            os.remove(temp_path)
            st.sidebar.success("[LISTO]")

st.markdown("<div class='hud-header'>SISTEMA DE CONTROL // NÚCLEO UNIVERSAL ACTIVO</div>", unsafe_allow_html=True)
st.markdown("## TESSA IA")
st.markdown("<p style='color: #94a3b8; font-size: 13px; font-family: monospace; letter-spacing: 1px;'>ESTADO: EN ÓRBITA // INTELIGENCIA GENERAL Y MULTIDISCIPLINARIA</p>", unsafe_allow_html=True)
st.markdown("---")

if not st.session_state.messages:
    saludo_inicial = "Sistemas en línea y listos. Hola, Gabriel. ¿De qué platicamos o qué resolvemos hoy?"
    st.session_state.messages.append({"role": "assistant", "content": saludo_inicial})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generar_respuesta_ia(prompt_completo):
    # Usamos la ruta v1 con el modelo autodetectado dinámicamente
    url = f"https://generativelanguage.googleapis.com/v1/models/{MODELO_DINAMICO}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT + "\n\n" + prompt_completo}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"[ALERTA DE NÚCLEO API]: Código {response.status_code} - {response.text}"
    except Exception as e:
        return f"[ALERTA DE NÚCLEO]: {str(e)}"

if prompt := st.chat_input("INTRODUCE CUALQUIER CONSULTA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("PROCESANDO...", expanded=False) as status_box:
            contexto_extra = ""
            
            if st.session_state.ifc_resumen_actual and any(k in prompt.lower() for k in ["ifc", "bim", "columna", "viga", "muro", "estructura", "modelo"]):
                contexto_extra += "\n[Datos BIM en Memoria]:\n" + st.session_state.ifc_resumen_actual + "\n"
            
            if modo_deep_research:
                contexto_extra += investigacion_profunda_web(prompt)

            historial_reciente = st.session_state.messages[-6:-1]
            texto_historial = "--- HISTORIAL RECIENTE ---\n"
            for msg in historial_reciente:
                rol_txt = "Gabriel" if msg["role"] == "user" else "Tessa"
                texto_historial += f"{rol_txt}: {msg['content']}\n"

            prompt_final = texto_historial + "\n" + contexto_extra + "\n--- CONSULTA ACTUAL ---\n" + prompt

            respuesta_asistente = generar_respuesta_ia(prompt_final)
            status_box.update(label="¡LISTO!", state="complete", expanded=False)

        st.markdown(respuesta_asistente)
        st.session_state.messages.append({"role": "assistant", "content": respuesta_asistente})
        st.rerun()