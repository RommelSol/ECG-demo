import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ecg_utils import compute_r_peaks_and_hr, build_ecg_figure_with_grid, load_npz as _load_npz, choose_best_lead

st.set_page_config(page_title="ECG App – Index loader", layout="wide")
st.title("ECG – Visualización y Análisis (Index-aware)")

@st.cache_data
def load_index():
    # Prefer data/index.csv, fallback to samples/index.csv
    for path in ("data/index.csv", "samples/index.csv"):
        if os.path.exists(path):
            return pd.read_csv(path), path
    return None, None

@st.cache_data
def load_from_index_row(row):
    # Load NPZ
    sig, fs, leads = _load_npz(row["npz_path"])
    # Virtual slicing if start_s/end_s exist
    if "start_s" in row and "end_s" in row and not (pd.isna(row["start_s"]) or pd.isna(row["end_s"])):
        start_s = float(row["start_s"]); end_s = float(row["end_s"])
        i0 = int(max(0, round(start_s * fs))); i1 = int(min(sig.shape[0], round(end_s * fs)))
        sig = sig[i0:i1, :]
    return sig, fs, leads

idx, idx_path = load_index()
if idx is None:
    st.error("No se encontró data/index.csv ni samples/index.csv")
    st.stop()

st.caption(f"Índice: {idx_path} ({len(idx)} filas)")

# Choose row
row_i = st.sidebar.selectbox("Segmento / Registro", idx.index, format_func=lambda i: str(idx.loc[i, "segment_id"] if "segment_id" in idx.columns else idx.loc[i, "record_id"]))
row = idx.loc[row_i]

# Load
sig, fs, leads = load_from_index_row(row)

# --- Sidebar controls ---
st.sidebar.write(f"fs detectada: {fs:.0f} Hz")
force_fs = st.sidebar.selectbox(
    "Forzar fs", options=[None, 500, 360, 250], index=0,
    format_func=lambda x: "No forzar" if x is None else f"{x} Hz"
)
if force_fs:
    fs = float(force_fs)

auto = st.sidebar.checkbox("Elegir lead automáticamente (II > V2 > V5)", value=True)

if auto:
    lead_idx = choose_best_lead(sig, fs, leads)
    lead = leads[lead_idx]
else:
    lead = st.sidebar.selectbox("Lead", leads.tolist())
    lead_idx = list(leads).index(lead)

# Señal del lead elegido
sig_lead = sig[:, lead_idx]

# Opcional: invertir polaridad
invert = st.sidebar.checkbox("Invertir señal (R negativos)", value=False)
if invert:
    sig_lead = -sig_lead
st.sidebar.caption(f"Lead actual: **{lead}**")

# Ventana
window_sec = st.sidebar.slider("Ventana (s)", 8, 12, 10)
window_n = int(window_sec * fs)
sig_view = sig_lead[-window_n:] if len(sig_lead) > window_n else sig_lead
t = np.arange(len(sig_view)) / fs

# FC máxima esperada -> RR mínimo (refractario)
max_hr = st.sidebar.slider("FC máxima esperada (lpm)", 100, 150, 200, step=10)
rr_min_sec = 60.0 / max_hr

tabs = st.tabs(["Visualización", "Análisis", "Meta"])

with tabs[0]:
    st.subheader("Visualización tipo papel ECG")
    fig = build_ecg_figure_with_grid(t, sig_view)
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Frecuencia Cardíaca (picos R + HR)")
    r_idx, hr_inst, hr_value = compute_r_peaks_and_hr(sig_view, fs, rr_min_sec=rr_min_sec, rr_max_sec=2.0)

    # Señal con picos
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=sig_view, mode="lines", name="ECG"))
    if len(r_idx) > 0:
        fig2.add_trace(go.Scatter(x=t[r_idx], y=sig_view[r_idx], mode="markers", name="R-peaks"))
    fig2.update_layout(xaxis_title="Tiempo (s)", yaxis_title="Amplitud (mV)", height=420)
    st.plotly_chart(fig2, use_container_width=True)

    # Etiquetas de interpretación (no diagnóstico)
    if np.isfinite(hr_value):
        if hr_value < 60:
            st.info(f"FC: {hr_value:.0f} lpm (bradicardia) – Solo informativo, no diagnóstico.")
        elif hr_value <= 100:
            st.success(f"FC: {hr_value:.0f} lpm (rango de reposo) – Solo informativo, no diagnóstico.")
        elif hr_value <= 150:
            st.warning(f"FC: {hr_value:.0f} lpm (taquicardia) – Solo informativo, no diagnóstico.")
        else:
            st.warning("⚠️ FC alta (≥150 lpm). Esto NO es un diagnóstico. Verifica fs/lead/artefactos; si es real, corresponde a taquicardia.")
    else:
        st.warning("No se pudo calcular FC")

    # Sanity check (conteo vs RR)
    beats = int(len(r_idx))
    dur = len(sig_view) / fs
    hr_count = 60.0 * beats / max(dur, 1e-6)
    hr_text = f"{hr_value:.0f}" if np.isfinite(hr_value) else "NaN"
    st.caption(f"HR por conteo en ventana ≈ {hr_count:.0f} lpm  |  HR (mediana RR) = {hr_text} lpm")
    if np.isfinite(hr_value) and hr_count > 0 and abs(hr_value - hr_count) > 0.2 * hr_count:
        st.warning("Posible artefacto / fs incorrecta / doble conteo. Prueba otro lead, invierte la señal o ajusta 'FC máxima esperada'.")

with tabs[2]:
    st.write("Row:", dict(row))
    st.write("fs:", fs, "| leads:", leads.tolist(), "| shape:", sig.shape)

st.info("Esta app es educativa e informativa; no provee diagnóstico médico.")