import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Simulador de Mareas - Pichilemu", layout="centered")

st.title("🌊 Simulador de Mareas – Pichilemu")

# ============================
# FUNCION PRINCIPAL DE MAREAS
# ============================

def generar_marea(dias, amplitud=1.8, ruido=0.10):
    """
    Genera una marea simulada usando senos combinados.
    - dias: cuántos días simular
    - amplitud: altura promedio de las mareas
    - ruido: variabilidad random
    """
    minutos = dias * 24 * 60
    t = np.linspace(0, dias * 24, minutos)

    # Combinación de funciones seno que simulan las mareas reales
    marea = (
        amplitud * np.sin(2 * np.pi * t / 12.42) +  # Componente principal lunar
        0.4 * np.sin(2 * np.pi * t / 24.0) +        # Desfase diario solar
        0.25 * np.sin(2 * np.pi * t / 8.3)          # Componente menor
    )

    # Ruido para hacerlo más realista
    marea += np.random.normal(0, ruido, minutos)

    return t, marea

# ============================
# CONFIGURACIÓN DE USUARIO
# ============================

dias = st.slider("Días a simular", min_value=1, max_value=14, value=5)
amplitud = st.slider("Amplitud aproximada de marea", min_value=0.5, max_value=3.5, value=1.8, step=0.1)
ruido = st.slider("Nivel de variabilidad", min_value=0.00, max_value=0.40, value=0.10, step=0.01)

# ============================
# EJECUCIÓN DEL MODELO
# ============================

t, marea = generar_marea(dias, amplitud, ruido)

# Convertimos a un dataframe descargable
inicio = datetime(2025, 1, 1)
fechas = [inicio + timedelta(minutes=i) for i in range(len(t))]

df = pd.DataFrame({
    "fecha": fechas,
    "altura": marea
})

# ============================
# GRAFICAR CON PLOTLY
# ============================

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["fecha"], y=df["altura"], mode='lines', name="Marea"))

fig.update_layout(
    title="Simulación de Marea en Pichilemu",
    xaxis_title="Tiempo",
    yaxis_title="Altura (m)",
    template="plotly"
)

st.plotly_chart(fig, use_container_width=True)

# ============================
# DESCARGA DEL CSV
# ============================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Descargar CSV de mareas",
    data=csv,
    file_name="mareas_pichilemu.csv",
    mime="text/csv"
)

st.success("Simulación completada.")

