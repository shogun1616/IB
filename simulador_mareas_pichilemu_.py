# simulador_mareas_pichilemu_final.py
# Simulador de mareas para Exploración Matemática IB
# Usa solo: streamlit, pandas, numpy, math, altair
# Ejecutar: streamlit run simulador_mareas_pichilemu_final.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math

st.set_page_config(page_title="Simulador Mareas — Exploración IB (Pichilemu)", layout="wide")

# -------------------------
# Interfaz y texto intro
# -------------------------
st.title("Optimización del Tiempo de Surf en Pichilemu — Simulador de Mareas")
st.markdown(
    "Herramienta para la Exploración Matemática del IB:\n\n"
    "**“Optimización del Tiempo de Surf en Pichilemu: Un Modelo Sinusoidal de Mareas”**\n\n"
    "Sube un CSV con las columnas obligatorias **`fecha_hora`** (formato `YYYY-MM-DD HH:MM`) y **`altura_m`** (metros)."
)

st.sidebar.header("Entradas y configuración")

# Modo de datos: ejemplo o subir
modo = st.sidebar.radio("Modo de datos", ("Generar datos de ejemplo", "Subir CSV (formato requerido)"))

# Parámetros ejemplo
fecha_ejemplo = st.sidebar.date_input("Fecha de ejemplo (si generas datos)", value=pd.to_datetime("2025-01-01").date())
ruido = st.sidebar.slider("Ruido (si generas datos)", 0.0, 0.6, 0.06, 0.01)
dias_ejemplo = st.sidebar.slider("Días a simular (ejemplo)", 1, 5, 1)

# CSV uploader (solo acepta CSV)
uploaded = None
if modo == "Subir CSV (formato requerido)":
    uploaded = st.sidebar.file_uploader(
        "Sube CSV con columnas EXACTAS: 'fecha_hora' (YYYY-MM-DD HH:MM) y 'altura_m' (metros).",
        type=["csv"]
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Opciones de modelado")
usar_periodo_fijo = st.sidebar.checkbox("Fijar periodo semidiurno = 12.42 h (recomendado)", value=True)
periodo_input = st.sidebar.number_input("Periodo (horas) si NO lo fijas", value=12.42, step=0.01, format="%.2f")
estimar_periodo_fft = st.sidebar.checkbox("Estimar periodo por FFT (si no fijas)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Rango óptimo (para ventanas de surf)")
rango_low = st.sidebar.number_input("Rango mínimo (m)", value=1.30, step=0.01, format="%.2f")
rango_high = st.sidebar.number_input("Rango máximo (m)", value=1.70, step=0.01, format="%.2f")
res_min = st.sidebar.slider("Resolución para ventanas (minutos)", 1, 60, 5)

st.sidebar.markdown("---")
export_csv_flag = st.sidebar.checkbox("Permitir exportar predicciones y ventanas (CSV)", value=True)
export_template_flag = st.sidebar.checkbox("Permitir descargar plantilla de informe (TXT)", value=True)

# -------------------------
# Funciones auxiliares
# -------------------------
def h_model(t_hours, a, b, c, d):
    """t_hours: float or array en horas (puede ir >24 si hay varios días)"""
    return a * np.sin(b * (t_hours - c)) + d

def calcular_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1 - ss_res / ss_tot

def detectar_pleamares_analiticos(a, b, c, d, t_start, t_end):
    """Devuelve lista de instantes t (en horas) entre t_start y t_end donde hay pleamares (máx locales del modelo)."""
    if a == 0 or b == 0:
        return []
    # Condición para máximos: sin(b(t-c)) = 1 -> b(t-c) = pi/2 + 2k*pi
    ks = []
    # calcular rango de k que cubre [t_start, t_end]
    k_min = math.floor((b*(t_start - c) - math.pi/2) / (2*math.pi)) - 1
    k_max = math.ceil((b*(t_end - c) - math.pi/2) / (2*math.pi)) + 1
    pleamas = []
    for k in range(k_min, k_max+1):
        t = c + (math.pi/2 + 2*k*math.pi) / b
        if t_start - 1e-9 <= t <= t_end + 1e-9:
            pleamas.append(t)
    return sorted(pleamas)

def detectar_bajamares_analiticos(a, b, c, d, t_start, t_end):
    """mínimos: sin(...) = -1"""
    if a == 0 or b == 0:
        return []
    k_min = math.floor((b*(t_start - c) + math.pi/2) / (2*math.pi)) - 1
    k_max = math.ceil((b*(t_end - c) + math.pi/2) / (2*math.pi)) + 1
    bajamas = []
    for k in range(k_min, k_max+1):
        t = c + (-math.pi/2 + 2*k*math.pi) / b
        if t_start - 1e-9 <= t <= t_end + 1e-9:
            bajamas.append(t)
    return sorted(bajamas)

def encontrar_intervalos_en_rango(t_grid, h_grid, low, high):
    """Devuelve lista de (t0,t1) contiguos donde h_grid está entre low y high. t en horas."""
    inside = (h_grid >= (low - 1e-9)) & (h_grid <= (high + 1e-9))
    if not np.any(inside):
        return []
    idx = np.where(inside)[0]
    intervals = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            intervals.append((t_grid[start], t_grid[prev]))
            start = i
            prev = i
    intervals.append((t_grid[start], t_grid[prev]))
    return intervals

def hora_str_from_hours(t_hours):
    """Convierte horas (puede ser float) a HH:MM en formato 24h, sin fecha."""
    # normalizar al día (mod 24)
    t = float(t_hours) % 24.0
    hh = int(t)
    mm = int(round((t - hh) * 60))
    if mm == 60:
        hh = (hh + 1) % 24
        mm = 0
    return f"{hh:02d}:{mm:02d}"

def intentar_mapear_columnas(df):
    """
    Debe terminar con columnas: 'fecha_hora' (datetime) y 'altura_m' (float)
    Se aceptan columnas comunes y se renombra si es necesario.
    """
    cols = [c.lower() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    # posibles nombres fecha
    posibles_fecha = ['fecha_hora', 'fecha hora', 'datetime', 'date_time', 'timestamp', 'time', 'fecha']
    posibles_altura = ['altura_m', 'altura', 'height', 'nivel', 'nivel_m', 'nivel_metros']
    fecha_col = None
    altura_col = None
    for p in posibles_fecha:
        if p in colmap:
            fecha_col = colmap[p]
            break
    for p in posibles_altura:
        if p in colmap:
            altura_col = colmap[p]
            break
    return fecha_col, altura_col

# -------------------------
# Preparar / limpiar datos
# -------------------------
df = None
errores = []

if modo == "Generar datos de ejemplo":
    # Generar datos sintéticos para 1..dias_ejemplo días con muestreo cada 30 min
    base = pd.to_datetime(fecha_ejemplo)
    minutos = np.arange(0, 24 * 60 * dias_ejemplo, 30)  # cada 30 min
    datetimes = base + pd.to_timedelta(minutos, unit='m')
    # parámetros reales simulados
    d_true = 1.5
    a_true = 0.9
    b_true = 2*math.pi / 12.42
    c_true = 4.5  # hora de fase
    t_hours = (minutos / 60.0)
    alturas = h_model(t_hours, a_true, b_true, c_true, d_true) + np.random.normal(0, ruido, len(t_hours))
    df = pd.DataFrame({"fecha_hora": datetimes, "altura_m": alturas})
else:
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # intentar mapear columnas comunes a 'fecha_hora' y 'altura_m'
            fecha_col, altura_col = intentar_mapear_columnas(df)
            if fecha_col is None or altura_col is None:
                st.error(
                    "CSV inválido: debe contener columnas con fecha y altura. "
                    "Nombres esperados: 'fecha_hora' y 'altura_m'. "
                    "Puedo intentar mapear nombres comunes como 'datetime'->fecha_hora o 'altura'->altura_m."
                )
                st.stop()
            # renombrar
            df = df.rename(columns={fecha_col: 'fecha_hora', altura_col: 'altura_m'})
            # parseo fecha
            df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], errors='coerce', dayfirst=False)
            # parseo altura
            df['altura_m'] = pd.to_numeric(df['altura_m'], errors='coerce')
            # eliminar filas malas
            n_before = len(df)
            df = df.dropna(subset=['fecha_hora', 'altura_m']).reset_index(drop=True)
            n_after = len(df)
            if n_after < n_before:
                errores.append(f"Se eliminaron {n_before - n_after} filas con fecha/altura inválidas.")
            # ordenar cronológicamente
            df = df.sort_values('fecha_hora').reset_index(drop=True)
        except Exception as e:
            st.error(f"Error leyendo CSV: {e}")
            st.stop()
    else:
        # no subido y no generar -> mostrar instrucción
        pass

st.subheader("Datos (vista previa y limpieza)")
if df is None:
    st.info("No hay datos cargados. Elige 'Generar datos de ejemplo' o sube un CSV con columnas 'fecha_hora' y 'altura_m'.")
else:
    # Mostrar errores si existen
    if errores:
        for er in errores:
            st.warning(er)
    st.dataframe(df.head(30))

# -------------------------
# Botón para ejecutar ajuste y análisis
# -------------------------
if st.button("Ejecutar análisis y ajustar modelo"):
    if df is None or df.empty:
        st.error("No hay datos válidos para ajustar.")
    else:
        # crear variable t_hours relativa al inicio (horas desde la primera medianoche del primer día)
        primer_dia = df['fecha_hora'].dt.normalize().iloc[0]  # midnight del primer día
        # t_rel puede superar 24 si hay múltiples días (eso está bien)
        df['t_horas'] = (df['fecha_hora'] - primer_dia).dt.total_seconds() / 3600.0

        t_obs = df['t_horas'].values  # horas desde primer_dia
        h_obs = df['altura_m'].values

        # Periodo
        if usar_periodo_fijo:
            periodo = 12.42
            st.write(f"Usando periodo fijo semidiurno = {periodo:.2f} h.")
        else:
            if estimar_periodo_fft:
                # estimación básica por FFT: re-muestrear uniformemente a la resolución mínima observada
                try:
                    # re-muestrear
                    times_seconds = (df['fecha_hora'] - df['fecha_hora'].iloc[0]).dt.total_seconds().values
                    # uniform grid based on median step
                    median_step = np.median(np.diff(times_seconds))
                    N = int(round((times_seconds[-1] - times_seconds[0]) / median_step)) + 1
                    grid_seconds = np.linspace(times_seconds[0], times_seconds[-1], N)
                    h_uniform = np.interp(grid_seconds, times_seconds, h_obs)
                    y = h_uniform - np.mean(h_uniform)
                    yf = np.fft.rfft(y)
                    xf = np.fft.rfftfreq(len(y), d=median_step)  # freq in Hz
                    mag = np.abs(yf)
                    mag[0] = 0
                    idx = np.argmax(mag)
                    freq_hz = xf[idx]
                    if freq_hz > 0:
                        # periodo en horas = 1/freq_hz / 3600
                        periodo = 1.0 / freq_hz / 3600.0
                        st.success(f"Periodo estimado por FFT = {periodo:.3f} h")
                    else:
                        periodo = periodo_input
                        st.warning("No se pudo estimar periodo por FFT; usando valor manual.")
                except Exception:
                    periodo = periodo_input
                    st.warning("Estimación por FFT falló; usando valor manual.")
            else:
                periodo = periodo_input
                st.write(f"Usando periodo manual = {periodo:.3f} h.")

        b = 2 * math.pi / periodo

        # a y d desde observaciones (max/min)
        altura_max = np.max(h_obs)
        altura_min = np.min(h_obs)
        d = (altura_max + altura_min) / 2.0
        a = (altura_max - altura_min) / 2.0

        # Detectar c: instante de primera pleamar del primer día
        # Buscamos máximos locales en el segmento correspondiente al primer día (0..24h)
        mask_primer_dia = (df['t_horas'] >= 0.0) & (df['t_horas'] < 24.0)
        c_detectado = None
        if mask_primer_dia.sum() >= 3:
            df_pd = df[mask_primer_dia].reset_index(drop=True)
            h_pd = df_pd['altura_m'].values
            t_pd = df_pd['t_horas'].values
            # detectar máximos locales simple (punto mayor que vecinos inmediatos)
            for i in range(1, len(h_pd)-1):
                if (h_pd[i] >= h_pd[i-1]) and (h_pd[i] >= h_pd[i+1]) and (h_pd[i] > (np.mean(h_pd))):
                    c_detectado = t_pd[i]
                    break
        # Si no se detectó, usar búsqueda por minimización del SSE variando c en rango [0, periodo)
        if c_detectado is None:
            cs = np.linspace(0, periodo, 801)
            best_c = None
            best_err = np.inf
            for c_try in cs:
                pred_try = h_model(t_obs, a, b, c_try, d)
                err = np.sum((h_obs - pred_try) ** 2)
                if err < best_err:
                    best_err = err
                    best_c = c_try
            c = float(best_c)
            st.info(f"No se detectó pleamar local clara en el primer día; fase estimada por búsqueda: c = {c:.3f} h desde {primer_dia.date()}.")
        else:
            c = float(c_detectado)
            st.success(f"Primera pleamar del día detectada automáticamente: c = {c:.3f} h desde {primer_dia.date()} ({hora_str_from_hours(c)})")

        # Generar simulación fina para gráficos y ventanas (resolución según res_min)
        t_sim = np.arange(np.min(t_obs), np.max(t_obs) + (res_min/60.0), res_min/60.0)
        h_sim = h_model(t_sim, a, b, c, d)

        # R^2 (comparación en tiempos observados)
        h_pred_obs = h_model(t_obs, a, b, c, d)
        r2 = calcular_r2(h_obs, h_pred_obs)

        # Mostrar parámetros
        st.subheader("Parámetros del modelo sinusoidal ajustado")
        st.write(f"- a (amplitud) = **{a:.4f} m** (=(max-min)/2)")
        st.write(f"- d (línea media) = **{d:.4f} m** (=(max+min)/2)")
        st.write(f"- b (frecuencia) = **{b:.6f} rad/h**  → periodo = **{periodo:.4f} h**")
        st.write(f"- c (fase) = **{c:.4f} h** desde medianoche del primer día ({primer_dia.date()})")
        st.write(f"- R² entre modelo y datos observados = **{r2:.4f}**")

        # Tabla comparativa
        df_comp = pd.DataFrame({
            "fecha_hora": df['fecha_hora'],
            "t_horas": df['t_horas'],
            "altura_observada_m": df['altura_m'],
            "altura_modelo_m": h_pred_obs,
            "diferencia_m": df['altura_m'] - h_pred_obs
        })
        st.subheader("Comparación: observada vs modelo (muestra)")
        st.dataframe(df_comp.head(40))

        # -------------------------
        # GRAFICOS con Altair
        # -------------------------
        st.subheader("Gráficos")

        # Dataset para gráficas (convertir t_sim a 'fecha_hora' reales para ejes legibles)
        fechas_sim = pd.to_datetime(primer_dia) + pd.to_timedelta(t_sim, unit='h')
        df_sim = pd.DataFrame({
            "fecha_hora": fechas_sim,
            "t_horas": t_sim,
            "altura_modelo": h_sim
        })
        df_obs_plot = pd.DataFrame({
            "fecha_hora": df['fecha_hora'],
            "t_horas": df['t_horas'],
            "altura_observada": df['altura_m']
        })

        # Chart 1: puntos observados (scatter)
        chart_pts = alt.Chart(df_obs_plot).mark_circle(size=40).encode(
            x=alt.X('fecha_hora:T', title='Fecha y hora'),
            y=alt.Y('altura_observada:Q', title='Altura (m)')
        ).properties(width=800, height=250, title="Puntos observados")

        # Chart 2: curva del modelo sinusoidal (línea)
        chart_model = alt.Chart(df_sim).mark_line().encode(
            x=alt.X('fecha_hora:T', title='Fecha y hora'),
            y=alt.Y('altura_modelo:Q', title='Altura (m)')
        ).properties(width=800, height=250, title="Curva del modelo sinusoidal")

        # Chart 3: comparación (línea modelo + puntos observados)
        chart_comp = alt.layer(
            chart_model.encode(color=alt.value("#1f77b4")),
            chart_pts.encode(color=alt.value("#d62728"))
        ).resolve_scale(y='shared').properties(title="Comparación: modelo vs datos")

        st.altair_chart(chart_comp, use_container_width=True)

        # Chart 4: sombreado de rango óptimo
        # Marcar in_range para df_sim
        df_sim['in_range'] = ((df_sim['altura_modelo'] >= (rango_low - 1e-9)) & (df_sim['altura_modelo'] <= (rango_high + 1e-9)))
        area_inrange = alt.Chart(df_sim[df_sim['in_range']]).mark_area(opacity=0.20).encode(
            x='fecha_hora:T',
            y='altura_modelo:Q'
        )

        line_model = alt.Chart(df_sim).mark_line().encode(x='fecha_hora:T', y='altura_modelo:Q')
        pts_obs = alt.Chart(df_obs_plot).mark_circle(size=35).encode(x='fecha_hora:T', y='altura_observada:Q')

        chart_range = (area_inrange + line_model + pts_obs).properties(
            width=900, height=300, title=f"Intervalos donde la marea modelo está en [{rango_low:.2f} , {rango_high:.2f}] m (sombreado)"
        )
        st.altair_chart(chart_range, use_container_width=True)

        # -------------------------
        # Pleamares / Bajamares analíticos
        # -------------------------
        t_min = np.min(t_sim)
        t_max = np.max(t_sim)
        pleamas = detectar_pleamares_analiticos(a, b, c, d, t_min, t_max)
        bajamas = detectar_bajamares_analiticos(a, b, c, d, t_min, t_max)

        st.subheader("Pleamares y bajamares estimados por el modelo (horas)")
        if pleamas:
            for t in pleamas:
                fecha_t = pd.to_datetime(primer_dia) + pd.to_timedelta(t, unit='h')
                st.write(f"- Pleamar: {fecha_t}  (hora {hora_str_from_hours(t)}), altura ≈ {h_model(t,a,b,c,d):.3f} m")
        else:
            st.write("- No se detectaron pleamares en el rango modelado.")

        if bajamas:
            for t in bajamas:
                fecha_t = pd.to_datetime(primer_dia) + pd.to_timedelta(t, unit='h')
                st.write(f"- Bajamar: {fecha_t}  (hora {hora_str_from_hours(t)}), altura ≈ {h_model(t,a,b,c,d):.3f} m")
        else:
            st.write("- No se detectaron bajamares en el rango modelado.")

        # -------------------------
        # Ventanas óptimas (intervalos horarios)
        # -------------------------
        ventanas = encontrar_intervalos_en_rango(df_sim['t_horas'].values, df_sim['altura_modelo'].values, rango_low, rango_high)
        st.subheader(f"Ventanas horarias donde h(t) ∈ [{rango_low:.2f}, {rango_high:.2f}] m")
        if ventanas:
            tabla_ventanas = []
            for (t0, t1) in ventanas:
                start_dt = pd.to_datetime(primer_dia) + pd.to_timedelta(t0, unit='h')
                end_dt = pd.to_datetime(primer_dia) + pd.to_timedelta(t1, unit='h')
                dur = (t1 - t0)
                tabla_ventanas.append({
                    "inicio": start_dt,
                    "fin": end_dt,
                    "hora_inicio": hora_str_from_hours(t0),
                    "hora_fin": hora_str_from_hours(t1),
                    "duracion_h": round(dur, 3)
                })
                st.write(f"- Desde {start_dt} ({hora_str_from_hours(t0)}) hasta {end_dt} ({hora_str_from_hours(t1)})  — duración {dur:.2f} h")
            df_ventanas = pd.DataFrame(tabla_ventanas)
        else:
            st.write("No hay ventanas en el rango indicado en el periodo analizado.")
            df_ventanas = pd.DataFrame(columns=["inicio","fin","hora_inicio","hora_fin","duracion_h"])

        # -------------------------
        # Exportar resultados
        # -------------------------
        if export_csv_flag:
            # CSV 1: observados + modelo + diferencia
            df_export = df_comp.copy()
            csv1 = df_export.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV: observados_vs_modelo.csv", data=csv1, file_name="observados_vs_modelo.csv", mime="text/csv")

            # CSV 2: ventanas óptimas
            csv2 = df_ventanas.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV: ventanas_optimas.csv", data=csv2, file_name="ventanas_optimas.csv", mime="text/csv")

        if export_template_flag:
            # Generar texto explicativo automático para IA (plantilla)
            pleamas_str = ", ".join([hora_str_from_hours(t) for t in pleamas]) if pleamas else "N/A"
            ventanas_str = ", ".join([f"{row['hora_inicio']}-{row['hora_fin']}" for _, row in df_ventanas.iterrows()]) if not df_ventanas.empty else "N/A"

            texto = f"""Título:
Optimización del Tiempo de Surf en Pichilemu: Un Modelo Sinusoidal de Mareas

1) Introducción - contexto
Se modelan las mareas de Pichilemu mediante un modelo sinusoidal simple para identificar ventanas óptimas de surf. 
Datos: columnas obligatorias 'fecha_hora' (YYYY-MM-DD HH:MM) y 'altura_m' (m).

2) Modelo propuesto
h(t) = a * sin(b (t - c)) + d

Parámetros estimados a partir de los datos:
- a (amplitud) = {a:.4f} m  (calculado como (max-min)/2)
- d (línea media) = {d:.4f} m  (calculado como (max+min)/2)
- b (frecuencia) = {b:.6f} rad/h  → periodo = {periodo:.4f} h
- c (fase) = {c:.4f} h desde la medianoche del primer día ({primer_dia.date()})
- R² entre modelo y datos = {r2:.4f}

3) Interpretación de pleamares y bajamares
Pleamares estimadas (horas): {pleamas_str}
Bajamares estimadas (horas): {', '.join([hora_str_from_hours(t) for t in bajamas]) if bajamas else 'N/A'}

4) Rango óptimo elegido por el usuario
Rango: {rango_low:.2f} m ≤ h(t) ≤ {rango_high:.2f} m
Ventanas encontradas: {ventanas_str}

5) Limitaciones
- Modelo de una sola sinusoide: no captura compuestos semi-diurnos y diurnos simultáneos,
  efectos meteorológicos (viento, presión), ni variaciones locales por topografía.
- El ajuste depende de la calidad y cobertura temporal de los datos.
- Se recomienda usar múltiples sinusoides o análisis espectral para modelos más precisos.

6) Uso en la Exploración IB
- Mostrar gráficos (observados vs modelo), tabla de ventanas y cálculo analítico de máximos/minimos.
- Discutir fuentes de error y justificar elección del periodo (12.42 h por mareas semidiurnas).

(Fin de plantilla — completa con tus observaciones y conclusiones.)
"""
            st.download_button("📥 Descargar explicación / plantilla (TXT)", data=texto.encode('utf-8'), file_name="plantilla_exploracion_ib.txt", mime="text/plain")

        st.success("Análisis completado — puedes descargar los resultados y usar la plantilla en tu Exploración IB.")
        st.markdown("---")
        st.caption("Nota: este es un modelo simple (una sinusoide). Para mejoras: combinar varias sinusoides, incluir términos de tendencia, o usar datos de varias estaciones.")


