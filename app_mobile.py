import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Simulador Financiero PyME AI", page_icon="📊", layout="wide"
)

# Estilo de paleta Corporate Teal
PRIMARY_COLOR = "#008080"

st.title("📊 Simulador Financiero PyME + Asistente AI")

# --- BARRA LATERAL: CONFIGURACIÓN ---
with st.sidebar:
  st.header("⚙️ Configuración AI")
  api_key = st.text_input("Clave API de Gemini:", type="password")
  st.info("Ingresa tu clave API para habilitar la entrada por voz e inteligencia.")

# --- ENTRADA DE VOZ ---
st.subheader("🎙️ Dictado por Voz / Texto")
audio_input = st.audio_input("Presiona para grabar tu comando de voz:")

# Estado de la sesión
if "ci" not in st.session_state:
  st.session_state.update({
      "ci": 200.0,
      "cn": 50.0,
      "mo": 90.0,
      "cif": 90.0,
      "tc_base": 11.20,
      "tc_mercado": 11.20,
      "margen": 15.0,
      "beta_tc": -0.45,
      "tc_simulado": 11.20,
  })

if audio_input and api_key:
  with st.spinner("Procesando comando con Gemini AI..."):
    try:
      client = genai.Client(api_key=api_key)
      bytes_data = audio_input.getvalue()
      prompt = """
            Analiza el audio y extrae datos financieros. Responde ÚNICAMENTE en formato JSON:
            {"ci": float, "cn": float, "mo": float, "cif": float, "tc_base": float, "tc_mercado": float, "margen": float}
            Si un dato no se menciona, devuelve null.
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
      datos = json.loads(response.text)
      for k, v in datos.items():
        if v is not None:
          st.session_state[k] = float(v)
      st.success("¡Datos actualizados por voz!")
    except Exception as e:
      st.error(f"Error al procesar audio: {e}")

# --- PESTAÑAS CON TODOS LOS MÓDULOS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Hoja de Costos",
    "📈 Motor OLS (Demanda)",
    "⚠️ Riesgo y Sensibilidad",
    "📊 Módulo EPD",
])

# MODULO 1: HOJA DE COSTOS
with tab1:
  st.header("Hoja de Costos de Reposición")
  col1, col2 = st.columns(2)
  with col1:
    ci = st.number_input(
        "Insumos Dolarizados (Bs):",
        value=st.session_state["ci"],
        key="input_ci",
    )
    cn = st.number_input(
        "Insumos Nacionales (Bs):",
        value=st.session_state["cn"],
        key="input_cn",
    )
    mo = st.number_input(
        "Mano de Obra Directa (Bs):",
        value=st.session_state["mo"],
        key="input_mo",
    )
    cif = st.number_input(
        "Costos Indirectos CIF (Bs):",
        value=st.session_state["cif"],
        key="input_cif",
    )
  with col2:
    tc_base = st.number_input(
        "TC Base / Oficial (Bs):",
        value=st.session_state["tc_base"],
        key="input_tc_base",
    )
    tc_mercado = st.number_input(
        "TC Mercado / Paralelo (Bs):",
        value=st.session_state["tc_mercado"],
        key="input_tc_mercado",
    )
    margen = st.number_input(
        "Margen Deseado (%):",
        value=st.session_state["margen"],
        key="input_margen",
    )

  factor = tc_mercado / tc_base if tc_base > 0 else 1.0
  ci_rev = ci * factor
  cr_real = ci_rev + cn + mo + cif
  pv_sugerido = cr_real / (1.0 - (margen / 100.0)) if margen < 100 else 0.0

  st.divider()
  m1, m2 = st.columns(2)
  m1.metric("COSTO DE REPOSICIÓN REAL (CR)", f"{cr_real:.2f} Bs.")
  m2.metric("PRECIO DE VENTA SUGERIDO (PV)", f"{pv_sugerido:.2f} Bs.")

# MODULO 2: MOTOR OLS
with tab2:
  st.header("Estimación de Demanda (Regresión OLS)")
  beta_tc = st.slider(
      "Sensibilidad de la Demanda ante el Tipo de Cambio (Beta TC):",
      -2.0,
      0.0,
      st.session_state["beta_tc"],
  )

  tc_range = np.linspace(tc_base, tc_base * 1.8, 20)
  demanda_base = 1000
  demanda_estimada = demanda_base * (1 + beta_tc * ((tc_range - tc_base) / tc_base))

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(tc_range, demanda_estimada, color=PRIMARY_COLOR, linewidth=2.5, marker="o")
  ax.set_title("Curva de Proyección de Demanda vs Tipo de Cambio")
  ax.set_xlabel("Tipo de Cambio (Bs)")
  ax.set_ylabel("Unidades Demandadas")
  ax.grid(True, linestyle="--", alpha=0.6)
  st.pyplot(fig)

# MODULO 3: RIESGO Y SENSIBILIDAD
with tab3:
  st.header("Análisis de Sensibilidad de Margen")
  tc_sim = st.slider(
      "Simular TC de Mercado:", tc_base, tc_base * 2.0, tc_mercado
  )
  factor_sim = tc_sim / tc_base if tc_base > 0 else 1.0
  cr_sim = (ci * factor_sim) + cn + mo + cif
  utilidad_sim = pv_sugerido - cr_sim
  margen_real_sim = (utilidad_sim / pv_sugerido) * 100 if pv_sugerido > 0 else 0

  st.metric("Costo Estimado con TC Simulado", f"{cr_sim:.2f} Bs.")
  st.metric(
      "Margen de Ganancia Resultante",
      f"{margen_real_sim:.2f} %",
      delta=f"{margen_real_sim - margen:.2f}%",
  )

# MODULO 4: EPD
with tab4:
  st.header("Estructura Porcentual de Decisión (EPD)")
  datos_pie = pd.DataFrame({
      "Componente": [
          "Insumos Dolarizados",
          "Insumos Nacionales",
          "Mano de Obra",
          "CIF",
      ],
      "Monto": [ci_rev, cn, mo, cif],
  })

  fig2, ax2 = plt.subplots(figsize=(6, 6))
  ax2.pie(
      datos_pie["Monto"],
      labels=datos_pie["Componente"],
      autopct="%1.1f%%",
      colors=["#008080", "#20B2AA", "#48D1CC", "#E0F2F1"],
      startangle=90,
  )
  ax2.set_title("Composición del Costo Real")
  st.pyplot(fig2)
