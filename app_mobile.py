import json
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Simulador Financiero PyME AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Simulador Financiero PyME + Asistente AI")

# --- BARRA LATERAL: CONFIGURACIÓN DE GEMINI ---
with st.sidebar:
  st.header("⚙️ Configuración AI")
  api_key = st.text_input("Gemini API Key:", type="password")
  st.info("Ingresa tu API Key para habilitar la entrada por voz e inteligencia.")

# --- ENTRADA DE COMANDO DE VOZ EN EL NAVEGADOR ---
st.subheader("🎙️ Dictado por Voz / Texto")
audio_input = st.audio_input("Presiona para grabar tu comando de voz:")

# Manejo del estado del formulario
if "ci" not in st.session_state:
  st.session_state.update({
      "ci": 200.0,
      "cn": 50.0,
      "mo": 90.0,
      "cif": 90.0,
      "tc_base": 11.20,
      "tc_mercado": 11.20,
      "margen": 15.0,
  })

# Procesamiento de voz si se graba un audio en el navegador
if audio_input and api_key:
  with st.spinner("Procesando comando con Gemini AI..."):
    try:
      client = genai.Client(api_key=api_key)
      bytes_data = audio_input.getvalue()

      prompt = """
            Analiza el audio o texto del usuario y extrae los datos para una hoja de costos PyME.
            Responde ÚNICAMENTE con un objeto JSON válido con los valores numéricos detectados:
            {"ci": float, "cn": float, "mo": float, "cif": float, "tc_base": float, "tc_mercado": float, "margen": float}
            Si algún dato no es mencionado, asigna null.
            """

      response = client.models.generate_content(
          model="gemini-2.5-flash",
          contents=[
              types.Part.from_bytes(data=bytes_data, mime_type="audio/wav"),
              prompt,
          ],
          config=types.GenerateContentConfig(
              response_mime_type="application/json"
          ),
      )

      datos_extraidos = json.loads(response.text)
      for k, v in datos_extraidos.items():
        if v is not None:
          st.session_state[k] = float(v)
      st.success("¡Datos actualizados desde la entrada de voz!")
    except Exception as e:
      st.error(f"Error procesando audio: {e}")

# --- FORMULARIO DE COSTOS EN EL MÓVIL ---
st.subheader("📋 Hoja de Costos")
col1, col2 = st.columns(2)

with col1:
  ci = st.number_input(
      "Insumos Dolarizados (Bs):", value=st.session_state["ci"]
  )
  cn = st.number_input("Insumos Nacionales (Bs):", value=st.session_state["cn"])
  mo = st.number_input(
      "Mano de Obra Directa (Bs):", value=st.session_state["mo"]
  )
  cif = st.number_input(
      "Costos Indirectos CIF (Bs):", value=st.session_state["cif"]
  )

with col2:
  tc_base = st.number_input(
      "TC Base / Oficial (Bs):", value=st.session_state["tc_base"]
  )
  tc_mercado = st.number_input(
      "TC Mercado / Paralelo (Bs):", value=st.session_state["tc_mercado"]
  )
  margen = st.number_input(
      "Margen Deseado (%):", value=st.session_state["margen"]
  )

# --- CÁLCULOS FINANCIEROS ---
factor_ajuste = tc_mercado / tc_base if tc_base > 0 else 1.0
ci_revalorizado = ci * factor_ajuste
cr_real = ci_revalorizado + cn + mo + cif
margen_pct = margen / 100.0
pv_sugerido = cr_real / (1.0 - margen_pct) if margen_pct < 1.0 else 0.0

st.markdown("---")
st.metric(
    label="COSTO DE REPOSICIÓN REAL (CR)", value=f"{cr_real:.2f} Bs."
)
st.metric(
    label="PRECIO DE VENTA SUGERIDO (PV)", value=f"{pv_sugerido:.2f} Bs."
)
