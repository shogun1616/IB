# simulador_mareas_exploracion_ib.py
# Único archivo: carga/ajusta/modela/exporta + plantilla de informe IB
# Requisitos: streamlit, numpy, pandas
# Ejecutar: streamlit run simulador_mareas_exploracion_ib.py

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Simulador Mareas — Exploración IB", layout="wide")

# -------------------------
# Texto introductorio / plantilla
# -------------------------
st.title("Optimización del Tiempo de Surf en Pichilemu: Simulador y Plantilla para la Exploración IB")
st.markdown(
    "Esta herramienta te ayuda a desarrollar tu Exploración Matemática del IB: "
    "ajusta un modelo sinusoidal a datos de marea, identifica pleamares y ventanas óptimas, "
    "genera gráficos y exporta resultados y una plantilla de informe."
)

# Sidebar: configuración y datos
st.sidebar.header("Datos y configuración")

modo = st.sidebar.radio("Modo de datos", ("Generar datos de ejemplo", "Subir CSV (SHOA u otro)"))

# Parámetros de ejemplo
fecha_ejemplo = st.sidebar.date_input("Fecha de ejemplo (si generas datos)", value=datetime(2024,6,15).date())
ruido = st.sidebar.slider("Ruido (si generas datos)", 0.0, 0.4, 0.06, 0.01)
dias_ejemplo = st.sidebar.slider("Días a simular (ejemplo)", 1, 3, 1)

# Subir CSV
uploaded = None
if modo == "Subir CSV (SHOA u otro)":
    uploaded = st.sidebar.file_uploader("Sube CSV con columnas 'datetime' o 'hora' y 'altura' (m)", type=["csv"])

# Opciones de ajuste
st.sidebar.markdown("---")
st.sidebar.subheader("Opciones de modelado")
usar_periodo_fijo = st.sidebar.checkbox("Fijar periodo semidiurno = 12.42 h (recomendado)", value=True)
periodo_input = st.sidebar.number_input("Periodo (horas, si NO lo fijas)", value=12.42, step=0.01, format="%.2f")
estimar_periodo_fft = st.sidebar.checkbox("Estimar periodo por FFT (si no fijas)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Rango óptimo (ejemplo)")
rango_low = st.sidebar.number_input("Rango mínimo (m)", value=1.30, step=0.01, format="%.2f")
rango_high = st.sidebar.number_input("Rango máximo (m)", value=1.70, step=0.01, format="%.2f")
res_min = st.sidebar.slider("Resolución para ventanas (minutos)", 1, 60, 5)
st.sidebar.markdown("---")
export_csv_flag = st.sidebar.checkbox("Permitir exportar predicciones (CSV)", value=True)
export_template_flag = st.sidebar.checkbox("Permitir descargar plantilla de informe (TXT)", value=True)

# -------------------------
# Funciones matemáticas
# -------------------------
def h_model(t, a, b, c, d):
    return a * np.sin(b * (t - c)) + d

def estimate_period_fft(t_hours, h_values):
    """
    Estima periodo dominante mediante FFT.
    t_hours: array de horas (posiblemente no uniformes) -> convierte a serie igualmente espaciada
    h_values: alturas correspondientes
    """
    # Re-muestrear uniformemente a resolución en minutos mínima
    n = len(t_hours)
    # convert to minutes from first time
    try:
        # if t_hours are floats in hours
        t0 = t_hours[0]
        mins = np.round((t_hours - t0) * 60).astype(int)
        total_minutes = mins[-1] - mins[0] + 1
        # build uniform grid
        grid = np.linspace(t_hours[0], t_hours[-1], total_minutes)
        # interpolate
        h_uniform = np.interp(grid, t_hours, h_values)
    except Exception:
        # fallback: use given arrays
        grid = t_hours
        h_uniform = h_values

    # detrend
    y = h_uniform - np.mean(h_uniform)
    N = len(y)
    yf = np.fft.rfft(y)
    xf = np.fft.rfftfreq(N, d=(grid[1]-grid[0])*3600)  # freq in Hz (samples are hours -> convert to seconds)
    # convert freq to period in hours: period_h = 1 / (freq in Hz) / 3600
    # find peak ignoring zero freq
    mag = np.abs(yf)
    mag[0] = 0
    idx = np.argmax(mag)
    freq_hz = xf[idx]
    if freq_hz <= 0:
        return None
    period_hours = 1.0 / freq_hz / 3600.0
    return period_hours

def analytic_pleamares(a, b, c, d, t_start=0.0, t_end=24.0):
    """Calcula pleamares analíticos en [t_start,t_end] (horas)"""
    if a == 0 or b == 0:
        return []
    ks = []
    k_min = int(np.floor((b*(t_start - c) - np.pi/2) / np.pi)) - 1
    k_max = int(np.ceil((b*(t_end - c) - np.pi/2) / np.pi)) + 1
    res = []
    for k in range(k_min, k_max+1):
        t = c + (np.pi/2 + k*np.pi) / b
        if t_start - 1e-9 <= t <= t_end + 1e-9:
            res.append(t % 24.0)
    return sorted(res)

def ventanas_por_rango(a, b, c, d, low, high, resolution_min=5, t_start=0.0, t_end=24.0):
    step = resolution_min / 60.0
    t_grid = np.arange(t_start, t_end + step, step)
    h_grid = h_model(t_grid, a, b, c, d)
    inside = (h_grid >= low - 1e-9) & (h_grid <= high + 1e-9)
    intervals = []
    if not np.any(inside):
        return intervals
    idx = np.where(inside)[0]
    start_idx = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        else:
            t0 = t_grid[start_idx]
            t1 = t_grid[prev]
            intervals.append((t0, t1))
            start_idx = i
            prev = i
    t0 = t_grid[start_idx]; t1 = t_grid[prev]
    intervals.append((t0, t1))
    return intervals

def hora_str_from_float(t_hours):
    total_seconds = int(round(t_hours * 3600)) % (24*3600)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    return f"{hh:02d}:{mm:02d}"

def calcular_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return float('nan')
    return 1 - ss_res / ss_tot

# -------------------------
# Preparar datos (subida o ejemplo)
# -------------------------
df = None
if modo == "Generar datos de ejemplo":
    # generar datos de ejemplo para un día: muestreo cada 30 min
    fecha0 = pd.to_datetime(fecha_ejemplo)
    t_hours = np.arange(0, 24, 0.5)
    # parámetros reales simulados (puedes variar)
    d_true = 1.5
    a_true = 0.9
    b_true = 2*np.pi/12.42
    c_true = 4.5
    alturas = h_model(t_hours, a_true, b_true, c_true, d_true) + np.random.normal(0, ruido, len(t_hours))
    datetimes = [fecha0 + pd.to_timedelta(h, 'h') for h in t_hours]
    df = pd.DataFrame({"datetime": datetimes, "hora": t_hours, "altura": alturas})
else:
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # Normalize columns
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['hora'] = df['datetime'].dt.hour + df['datetime'].dt.minute/60.0 + df['datetime'].dt.second/3600.0
            if 'hora' not in df.columns:
                st.error("CSV debe incluir columna 'datetime' o 'hora'.")
                st.stop()
            if 'altura' not in df.columns:
                # try common alt col names
                if 'height' in df.columns:
                    df.rename(columns={'height':'altura'}, inplace=True)
                elif 'nivel' in df.columns:
                    df.rename(columns={'nivel':'altura'}, inplace=True)
                else:
                    st.error("CSV debe contener la columna 'altura' (en metros).")
                    st.stop()
            df = df.sort_values('hora').reset_index(drop=True)
        except Exception as e:
            st.error(f"Error leyendo CSV: {e}")
            st.stop()
    else:
        # user selected upload but didn't upload; leave df None
        pass

# Mostrar datos cargados / generados
st.subheader("Datos (muestra)")
if df is None:
    st.info("No hay datos: genera datos de ejemplo o sube un CSV con datos de marea.")
else:
    st.dataframe(df.head(20))

# -------------------------
# Botón ejecutar análisis
# -------------------------
if st.button("Ejecutar análisis y ajustar modelo"):
    if df is None:
        st.error("No hay datos para ajustar.")
    else:
        t_obs = df['hora'].values
        h_obs = df['altura'].values

        # Estimar periodo (opcional)
        if usar_periodo_fijo:
            periodo = 12.42
            st.write("Usando periodo fijo:", periodo, "horas (mareas semidiurnas).")
        else:
            if estimar_periodo_fft:
                p_est = estimate_period_fft(t_obs, h_obs)
                if p_est is None:
                    periodo = periodo_input
                    st.warning("No se pudo estimar periodo por FFT. Usando valor manual.")
                else:
                    periodo = p_est
                    st.success(f"Periodo estimado por FFT: {periodo:.3f} h")
            else:
                periodo = periodo_input
                st.write("Usando periodo manual:", periodo, "horas.")

        b = 2*np.pi / periodo

        # d and a from data
        altura_max = np.max(h_obs)
        altura_min = np.min(h_obs)
        d = (altura_max + altura_min) / 2.0
        a = (altura_max - altura_min) / 2.0

        # estimate c (phase) by grid search (0..period)
        # minimize sum squared error between model and observations over c
        cs = np.linspace(0, 24, 2401)  # paso 0.01 h
        best_c = None
        best_err = np.inf
        for c_try in cs:
            pred = h_model(t_obs, a, b, c_try, d)
            err = np.sum((h_obs - pred)**2)
            if err < best_err:
                best_err = err
                best_c = c_try
        c = best_c

        # compute model on fine grid
        t_sim = np.linspace(0, 24, int(24*60/res_min)+1)  # resolución según res_min
        h_sim = h_model(t_sim, a, b, c, d)

        # R^2 computed on observation times
        h_pred_obs = h_model(t_obs, a, b, c, d)
        r2 = calcular_r2(h_obs, h_pred_obs)

        # show parameters
        st.subheader("Parámetros del modelo ajustado (estimados desde datos)")
        st.write(f"- a (amplitud) = {a:.4f} m")
        st.write(f"- b (frecuencia) = {b:.6f} rad/h  → periodo = {periodo:.4f} h")
        st.write(f"- c (fase) = {c:.4f} h desde medianoche (estimado por búsqueda)")
        st.write(f"- d (línea media) = {d:.4f} m")
        st.write(f"- R² del ajuste a puntos observados = {r2:.4f}")

        # Tabla comparativa (observado vs predicho en tiempos observados)
        df_comp = pd.DataFrame({
            "hora": t_obs,
            "altura_observada": h_obs,
            "altura_modelo": h_pred_obs
        })
        st.subheader("Comparación observados vs modelo (muestra)")
        st.dataframe(df_comp.head(20))

        # Graficar: combinamos series (observado y modelo en grid)
        df_plot = pd.DataFrame({
            "hora": t_sim,
            "altura_modelo": h_sim
        })
        # To include observed points, create a series at observed times (interpolated to grid)
        df_plot_obs = pd.DataFrame({"hora": t_obs, "observado": h_obs})
        # Merge into wide dataframe for plotting
        plot_df = pd.merge_asof(
    df_plot.sort_values("hora"),
    df_plot_obs.sort_values("hora"),
    on="hora",
    direction="nearest"
)

plot_df["altura_modelo"] = pd.to_numeric(plot_df["altura_modelo"], errors="coerce")
plot_df["observado"] = pd.to_numeric(plot_df["observado"], errors="coerce")
plot_df = plot_df.dropna(subset=["altura_modelo", "observado"])

plot_df = plot_df.set_index(pd.to_timedelta(plot_df["hora"], unit="h"))

st.line_chart(plot_df[["altura_modelo", "observado"]])

        # Pleamares analíticos
        pleamas = analytic_pleamares(a, b, c, d, 0.0, 24.0)
        st.subheader("Pleamares (horas estimadas por el modelo)")
        if pleamas:
            for t_p in pleamas:
                st.write(f"- {hora_str_from_float(t_p)}  → altura ≈ {h_model(t_p,a,b,c,d):.3f} m")
        else:
            st.write("No se encontraron pleamares en 0-24 h.")

        # Ventanas óptimas por rango
        ventanas = ventanas_por_rango(a, b, c, d, rango_low, rango_high, resolution_min=res_min)
        st.subheader(f"Ventanas en que la marea está entre {rango_low} y {rango_high} m")
        if ventanas:
            for (t0,t1) in ventanas:
                st.write(f"- Desde {hora_str_from_float(t0)} hasta {hora_str_from_float(t1)}  (duración {(t1-t0):.2f} h)")
        else:
            st.write("No hay ventanas en el día en que la marea esté dentro del rango indicado.")

        # Exportar predicciones (CSV)
        if export_csv_flag:
            # construir predicciones con datetimes usando fecha de primer dato o ejemplo
            if 'datetime' in df.columns:
                fecha0 = pd.to_datetime(df['datetime'].iloc[0]).normalize()
            else:
                fecha0 = pd.to_datetime(fecha_ejemplo)
            datetimes = [ (fecha0 + pd.to_timedelta(h, 'h')) for h in t_sim ]
            df_pred = pd.DataFrame({"datetime": datetimes, "hora": t_sim, "altura_modelo": h_sim})
            csv_bytes = df_pred.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar predicciones (CSV)", data=csv_bytes, file_name="predicciones_marea_modelo.csv", mime="text/csv")

        # Descargar plantilla de informe (texto) con guía IB
        if export_template_flag:
            template = f"""Título de la Exploración
"Optimización del Tiempo de Surf en Pichilemu: Un Modelo Sinusoidal de Mareas"

(Plantilla generada automáticamente - completa con tus textos y resultados)

1. Introducción
- Contexto sobre Pichilemu...
- Pregunta de investigación: ¿Cuál es la hora óptima...?

2. Recopilación de Datos y Modelado Sinusoidal
- Fuente de datos: ...
- Tabla de datos: (adjunta CSV)
- Modelo propuesto: h(t) = a*sin(b*(t-c)) + d
- Parámetros estimados:
  a = {a:.4f}
  b = {b:.6f}  (periodo ≈ {periodo:.4f} h)
  c = {c:.4f} h desde medianoche
  d = {d:.4f} m

3. Validación y Ajuste
- R² = {r2:.4f}
- Comparación gráfica entre datos y modelo (inserta figura)

4. Determinación de la \"Hora Óptima\"
- Criterio 1 (pleamares): horas estimadas → {', '.join([hora_str_from_float(t) for t in pleamas]) if pleamas else 'N/A'}
- Criterio 2 (rango {rango_low}–{rango_high} m): ventanas → {', '.join([f'{hora_str_from_float(t0)}-{hora_str_from_float(t1)}' for t0,t1 in ventanas]) if ventanas else 'N/A'}

5. Reflexión y Conclusión
- Limitaciones: fases lunares, viento, presión, topografía...
- Extensiones sugeridas: suma de sinusoides, incluir swell y viento, comparar múltiples días.

(Agrega capturas de tus gráficos y describe los pasos con detalle para cumplir los criterios A-E del IB.)
"""
            st.download_button("📥 Descargar plantilla de informe (TXT)", data=template.encode('utf-8'), file_name="plantilla_exploracion_ib.txt", mime="text/plain")

        st.success("Análisis completado. Usa los resultados en tu informe y consulta las sugerencias de extensión incluidas en la plantilla.")

# -------------------------
# Pie / ayuda rápida
# -------------------------
st.markdown("---")
st.caption("Consejo: para usar datos reales del SHOA, descarga su CSV para la estación más cercana a Punta de Lobos y sube el archivo en 'Subir CSV'. Si necesitas, puedo adaptar el script para leer el formato exacto del SHOA si pegas 6-10 filas aquí.")


