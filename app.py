
import time
import numpy as np
import streamlit as st
import pandas as pd
from collections import deque
from src.data_io import load_manifest,load_slice
from src.hr_analysis import ecg_hr
from src.visualization import fig_ecg_plotly

st.sidebar.subheader("Modo")
mode = st.sidebar.radio("Visualización", ["Explorador", "Monitor"])

manifest = pd.read_csv("data/manifest.csv")

# Selector en la barra lateral (elige por segment_id)
segment_id = st.sidebar.selectbox("Selecciona segmento", manifest["segment_id"].tolist())

# Recuperar la fila correspondiente
row = manifest[manifest["segment_id"] == segment_id].iloc[0]

# Cargar un segmento largo (o concatenate varios)
sig, fs, lead = load_slice(f"data/slices/{row['segment_id']}.npz")

if mode == "Monitor":
    win_s   = st.sidebar.slider("Ventana (s)", 5.0, min(15.0, len(sig)/fs), 8.0, 0.5)
    fps     = st.sidebar.slider("FPS", 10, 40, 25, 1)
    speed_x = st.sidebar.selectbox("Velocidad", ["1x","2x","0.5x"], index=0)
    sp_mult = {"0.5x":0.5,"1x":1.0,"2x":2.0}[speed_x]

    N_win = int(win_s*fs)
    buf   = deque(maxlen=N_win)
    # arranque con ceros
    for _ in range(N_win): buf.append(0.0)

    # índices de picos (si ya los tienes computados)
    res = ecg_hr(sig, fs)
    r_idx_all = res["r_idx"] if res["r_idx"].size else np.array([], dtype=int)

    placeholder = st.empty()
    run = st.toggle("▶ Play / ⏸ Pause", value=False, help="Activa/desactiva el barrido")
    i = 0
    while run and i < len(sig):
        # “streaming” de bloques pequeños para lograr FPS estable
        hop = max(1, int((fs / fps) * sp_mult // 2))  # la mitad del tamaño
        blk = sig[i:i+hop]
        for v in blk: buf.append(float(v))

        # r-peaks en la ventana actual
        start = i - (len(buf)-hop)
        start = max(0, start)
        end   = i
        r_in  = r_idx_all[(r_idx_all >= start) & (r_idx_all < end)] - (end - len(buf))
        r_in  = r_in[(r_in >= 0) & (r_in < len(buf))].astype(int)

        # figura plotly con cuadrícula ECG
        fig = fig_ecg_plotly(np.array(buf), fs, r_idx=r_in)
        placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        i += hop
        time.sleep(1.0 / fps)

    # si se sale del while (pause o fin) dibuja el frame actual
    fig = fig_ecg_plotly(np.array(buf), fs, r_idx=None if not res["r_idx"].size else r_in)
    if r_idx_all is not None and len(r_idx_all) > 0:
        r_idx = np.asarray(r_idx, dtype=int)
        rr = np.diff(r_idx) / fs
        hr = np.where(rr>0, 60.0/rr, np.nan)

        # extender hr a mismo largo que r_idx (último sin valor)
        hr_full = np.full(len(r_idx), np.nan); hr_full[:len(rr)] = hr

        # colores por BPM
        colors = []
        for h in hr_full:
            if np.isnan(h): colors.append("rgba(0,0,0,0)")   # último sin HR
            elif h < 60:    colors.append("blue")
            elif h <= 100:  colors.append("green")
            else:           colors.append("red")

        scatter_r = go.Scatter(
            x=t[r_idx], y=sig[r_idx], mode="markers", name="R-peaks",
            marker=dict(size=7, color=colors, line=dict(width=2)),
            customdata=np.stack([np.r_[rr, np.nan], hr_full], axis=1),
            hovertemplate="R @ t=%{x:.3f}s<br>mV=%{y:.3f}"
                        "<br>RR→=%{customdata[0]:.3f}s"
                        "<br>HR→=%{customdata[1]:.1f} bpm"
            )
    placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:

    st.set_page_config(page_title="ECG App",layout="wide")
    mf=load_manifest()
    row=st.sidebar.selectbox("Segment",mf.to_dict("records"),
        format_func=lambda r:f"{r['record_id']}|{r['lead']}|{r['start_s']}-{r['end_s']}s")
    show_r=st.sidebar.checkbox("Show R-peaks",True)
    sig,fs,lead=load_slice(row["path"])
    res=ecg_hr(sig,fs)
    fig = fig_ecg_plotly(sig, fs, r_idx=(res["r_idx"] if show_r else None))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    if res["hr_series"].size:
        st.metric("Median HR",f"{res['hr_avg']:.1f} bpm")
