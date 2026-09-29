import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# Configuración de página
st.set_page_config(
    page_title="Simulador Financiero PyME AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta Corporativa Teal
TEAL_PRIMARY = "#008080"
TEAL_SECONDARY = "#20B2AA"
TEAL_LIGHT = "#E0F2F1"
ACCENT_DARK = "#004D4D"

st.title("📊 Simulador Financiero PyME + Asistente AI")

# --- BARRA LATERAL: CONFIGURACIÓN AI ---
with st.sidebar:
  st.header("⚙️ Configuración AI")
  api_key = st.text_input("Clave API de Gemini:", type="password")
  st.info("Ingresa tu API Key para habilitar el dictado por voz y análisis.")

# --- ENTRADA DE VOZ GLOBAL ---
st.subheader("🎙️ Dictado por Voz / Comando AI")
audio_input = st.audio_input("Presiona el micrófono para dictar datos o costos:")

# Estado de variables de sesión
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
  })

# Procesar dictado con Gemini API
if audio_input and api_key:
  with st.spinner("Procesando comando de voz con Gemini AI..."):
    try:
      client = genai.Client(api_key=api_key)
      bytes_data = audio_input.getvalue()
      prompt = """
            Analiza el audio y extrae parámetros numéricos para la hoja de costos PyME.
            Devuelve UNICAMENTE un JSON válido con esta estructura:
            {"ci": float, "cn": float, "mo": float, "cif": float, "tc_base": float, "tc_mercado": float, "margen": float}
            Si un dato no se menciona, asígnalo como null.
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
      st.success("¡Datos actualizados exitosamente vía Inteligencia Artificial!")
    except Exception as e:
      st.error(f"Error procesando el audio: {e}")

# --- PESTAÑAS PRINCIPALES (4 MÓDULOS DEL ESCRITORIO) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Hoja de Costos",
    "📈 Motor OLS (Demanda)",
    "⚠️ Riesgo y Sensibilidad",
    "📊 Módulo EPD",
])

# ==========================================
# MÓDULO 1: HOJA DE COSTOS
# ==========================================
with tab1:
  st.markdown("### 📋 Definición de Costos de Reposición")
  col1, col2 = st.columns(2)

  with col1:
    ci = st.number_input(
        "Insumos Dolarizados (CI - Bs):",
        value=st.session_state["ci"],
        key="ci_in",
    )
    cn = st.number_input(
        "Insumos Nacionales (CN - Bs):",
        value=st.session_state["cn"],
        key="cn_in",
    )
    mo = st.number_input(
        "Mano de Obra Directa (MO - Bs):",
        value=st.session_state["mo"],
        key="mo_in",
    )
    cif = st.number_input(
        "Costos Indirectos (CIF - Bs):",
        value=st.session_state["cif"],
        key="cif_in",
    )

  with col2:
    tc_base = st.number_input(
        "Tipo de Cambio Base / Oficial (Bs):",
        value=st.session_state["tc_base"],
        key="tcb_in",
    )
    tc_mercado = st.number_input(
        "Tipo de Cambio Mercado / Paralelo (Bs):",
        value=st.session_state["tc_mercado"],
        key="tcm_in",
    )
    margen = st.number_input(
        "Margen Deseado (%):",
        value=st.session_state["margen"],
        key="mg_in",
    )

  # Cálculos
  factor_ajuste = tc_mercado / tc_base if tc_base > 0 else 1.0
  ci_revalorizado = ci * factor_ajuste
  cr_real = ci_revalorizado + cn + mo + cif
  margen_pct = margen / 100.0
  pv_sugerido = cr_real / (1.0 - margen_pct) if margen_pct < 1.0 else 0.0

  st.divider()
  m1, m2 = st.columns(2)
  m1.metric(
      label="COSTO DE REPOSICIÓN REAL (CR)",
      value=f"{cr_real:.2f} Bs.",
      delta=f"Factor TC: {factor_ajuste:.2f}x",
  )
  m2.metric(
      label="PRECIO DE VENTA SUGERIDO (PV)",
      value=f"{pv_sugerido:.2f} Bs.",
      delta=f"Margen: {margen:.1f}%",
  )

# ==========================================
# MÓDULO 2: MOTOR OLS (CURVA DE DEMANDA)
# ==========================================
with tab2:
  st.markdown("### 📈 Estimación de Demanda mediante Regresión OLS")
  col_ols1, col_ols2 = st.columns([1, 2])

  with col_ols1:
    st.write("**Parámetros de Elasticidad**")
    beta_tc = st.slider(
        "Sensibilidad de Demanda (Beta TC):",
        min_value=-2.0,
        max_value=0.0,
        value=st.session_state["beta_tc"],
        step=0.05,
    )
    st.info(
        "Un Beta negativo indica que a mayor Tipo de Cambio / Precio, la"
        " demanda disminuye."
    )

  with col_ols2:
    # Gráfico Matplotlib estilo OLS
    tc_rango = np.linspace(tc_base, tc_base * 1.8, 25)
    demanda_base = 1000
    demanda_est = demanda_base * (
        1 + beta_tc * ((tc_rango - tc_base) / tc_base)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        tc_rango,
        demanda_est,
        color=TEAL_PRIMARY,
        linewidth=2.5,
        marker="o",
        label="Demanda Estimada",
    )
    ax.axvline(
        x=tc_mercado,
        color="red",
        linestyle="--",
        label=f"TC Mercado Actual ({tc_mercado})",
    )
    ax.set_title("Curva OLS: Proyección de Demanda vs. Volatilidad Cambiaria")
    ax.set_xlabel("Tipo de Cambio (Bs)")
    ax.set_ylabel("Unidades Demandadas")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)

# ==========================================
# MÓDULO 3: RIESGO Y SENSIBILIDAD
# ==========================================
with tab3:
  st.markdown("### ⚠️ Escenarios de Riesgo y Sensibilidad")
  tc_sim = st.slider(
      "Simular Escenario de TC de Mercado:",
      float(tc_base),
      float(tc_base * 2.2),
      float(tc_mercado),
      step=0.10,
  )

  factor_sim = tc_sim / tc_base if tc_base > 0 else 1.0
  cr_sim = (ci * factor_sim) + cn + mo + cif
  utilidad_sim = pv_sugerido - cr_sim
  margen_real_sim = (utilidad_sim / pv_sugerido) * 100 if pv_sugerido > 0 else 0

  col_r1, col_r2, col_r3 = st.columns(3)
  col_r1.metric("Costo Real Simulado", f"{cr_sim:.2f} Bs.")
  col_r2.metric("Utilidad Unitaria Simulado", f"{utilidad_sim:.2f} Bs.")
  col_r3.metric(
      "Margen Resultante",
      f"{margen_real_sim:.2f} %",
      delta=f"{margen_real_sim - margen:.2f}%",
  )

  # Gráfico de barras de utilidad
  fig_bar, ax_bar = plt.subplots(figsize=(7, 3))
  categorias = ["Costo Reposición", "Utilidad Esperada"]
  valores = [cr_sim, max(0, utilidad_sim)]
  colores = [TEAL_PRIMARY, TEAL_SECONDARY if utilidad_sim > 0 else "salmon"]
  ax_bar.bar(categorias, valores, color=colores, width=0.4)
  ax_bar.set_ylabel("Monto en Bs.")
  ax_bar.set_title("Desglose del Precio Sugerido en Escenario Simulado")
  st.pyplot(fig_bar)

# ==========================================
# MÓDULO 4: MÓDULO EPD
# ==========================================
with tab4:
  st.markdown("### 📊 Estructura Porcentual de Decisión (EPD)")

  df_epd = pd.DataFrame({
      "Componente": [
          "Insumos Dolarizados (Revalorizados)",
          "Insumos Nacionales",
          "Mano de Obra",
          "CIF",
      ],
      "Monto (Bs)": [ci_revalorizado, cn, mo, cif],
  })

  col_e1, col_e2 = st.columns([1, 1])

  with col_e1:
    st.dataframe(df_epd, use_container_width=True)

  with col_e2:
    fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
    colors = [TEAL_PRIMARY, TEAL_SECONDARY, "#48D1CC", TEAL_LIGHT]
    ax_pie.pie(
        df_epd["Monto (Bs)"],
        labels=df_epd["Componente"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
    )
    ax_pie.set_title("Distribución Porcentual del Costo Real")
    st.pyplot(fig_pie)
