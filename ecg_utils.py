import numpy as np
import neurokit2 as nk
import plotly.graph_objects as go
from scipy import signal as sp_signal

# ------------------ I/O ------------------
def load_npz(path):
    """Carga NPZ robustamente y devuelve (signal_2d, fs, lead_names)."""
    with np.load(path, allow_pickle=True) as d:
        sig = d["signal"]
        fs = float(d["fs"])
        if sig.ndim == 1:
            sig = sig[:, None]  # (N,) -> (N,1)
        # leads
        if "leads" in d.files:
            leads = d["leads"]
            try:
                leads = np.array([str(x) for x in np.array(leads).ravel().tolist()])
            except Exception:
                leads = np.array([f"L{i+1}" for i in range(sig.shape[1])])
            if leads.shape[0] != sig.shape[1]:
                leads = np.array([f"L{i+1}" for i in range(sig.shape[1])])
        else:
            leads = np.array([f"L{i+1}" for i in range(sig.shape[1])])
        return sig.astype(float), fs, leads

# ------------------ Lead choice ------------------
def choose_best_lead(sig_2d, fs, lead_names):
    """Escoge el lead con mayor cantidad de RR validos. Prioriza II, V2, V5 si existen."""
    prefer = ["II", "V2", "V5"]
    try_order = [list(lead_names).index(x) for x in prefer if x in lead_names]
    if not try_order:
        try_order = range(sig_2d.shape[1])

    best_i, best_score = 0, -1
    for i in try_order:
        r_idx, hr_inst, hr_val = compute_r_peaks_and_hr(sig_2d[:, i], fs)
        score = len(hr_inst)  # numero de RR validos
        if score > best_score:
            best_i, best_score = i, score
    return best_i

# ------------------ R-peaks & HR (robusto) ------------------
def compute_r_peaks_and_hr(sig, fs, rr_min_sec=0.30, rr_max_sec=2.0, use_neurokit=True):
    """
    Detecta picos R y calcula HR (mediana de 60/RR).
    - Filtro band-pass 5-25 Hz para resaltar QRS.
    - Prueba ambas polaridades (por si R es negativo).
    - Opcion NeuroKit2 (limpieza + ecg_peaks) y fallback SciPy (find_peaks).
    - Refractario minimo = rr_min_sec (evita doble conteo).
    Devuelve: r_idx, hr_inst, hr_value
    """
    x = np.asarray(sig, dtype=float)
    x = np.nan_to_num(x) - np.nanmedian(x)

    # Band-pass para QRS (5-25 Hz)
    low, high = 5.0/(fs/2.0), 25.0/(fs/2.0)
    low = max(low, 1e-6); high = min(high, 0.99)
    b, a = sp_signal.butter(2, [low, high], btype="band")
    xf = sp_signal.filtfilt(b, a, x)

    min_dist = max(1, int(rr_min_sec * fs))  # muestras
    rr_min = rr_min_sec
    rr_max = rr_max_sec

    def enforce_refractory(idx):
        if idx is None or len(idx) == 0:
            return np.array([], dtype=int)
        keep = [int(idx[0])]
        for k in idx[1:]:
            if int(k) - keep[-1] >= min_dist:
                keep.append(int(k))
        return np.array(keep, dtype=int)

    def score_indices(idx):
        idx = enforce_refractory(idx)
        if idx.size < 2:
            return -1, np.array([]), np.nan, idx
        rr = np.diff(idx) / fs
        rr = rr[(rr > rr_min) & (rr < rr_max)]
        if rr.size == 0:
            return 0, np.array([]), np.nan, idx
        hr_inst = 60.0 / rr
        hr_value = float(np.nanmedian(hr_inst))
        return int(rr.size), hr_inst, hr_value, idx

    candidates = []
    for pol in (1, -1):  # señal e invertida
        y = pol * xf

        if use_neurokit:
            clean = nk.ecg_clean(y, sampling_rate=fs, method="neurokit")
            peaks, _ = nk.ecg_peaks(clean, sampling_rate=fs)
            idx_nk = peaks.get("ECG_R_Peaks", np.array([], dtype=int))
            s_nk = score_indices(idx_nk)
            candidates.append(("neurokit", pol, *s_nk))

        # Fallback con SciPy
        prom = max(0.1 * np.std(y), 0.02)
        idx_sp, _ = sp_signal.find_peaks(y, distance=min_dist, prominence=prom)
        s_sp = score_indices(idx_sp)
        candidates.append(("scipy", pol, *s_sp))

    # Elige el que tenga mas RR validos
    best = max(candidates, key=lambda c: c[2]) if candidates else None
    if best is None or best[2] <= 0:
        return np.array([], dtype=int), np.array([]), float("nan")
    _, pol, valid_rr, hr_inst, hr_value, r_idx = best
    return r_idx, hr_inst, hr_value

# ------------------ ECG paper-like plot ------------------
def build_ecg_figure_with_grid(t, sig, v_range=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sig, mode="lines", name="ECG"))
    t0, t1 = float(np.min(t)), float(np.max(t))
    if v_range is None:
        vmax = max(1.0, float(np.nanmax(np.abs(sig))))
        v_range = (-vmax, vmax)
    y0, y1 = v_range

    shapes = []
    # Horizontal (mV): minor 0.1, major 0.5
    minor_y = np.arange(np.floor(y0/0.1)*0.1, np.ceil(y1/0.1)*0.1 + 1e-9, 0.1)
    major_y = np.arange(np.floor(y0/0.5)*0.5, np.ceil(y1/0.5)*0.5 + 1e-9, 0.5)
    for y in minor_y:
        shapes.append(dict(type="line", x0=t0, x1=t1, y0=y, y1=y, line=dict(width=0.5), layer="below"))
    for y in major_y:
        shapes.append(dict(type="line", x0=t0, x1=t1, y0=y, y1=y, line=dict(width=1.2), layer="below"))
    # Vertical (s): minor 0.04, major 0.20
    minor_x = np.arange(np.floor(t0/0.04)*0.04, np.ceil(t1/0.04)*0.04 + 1e-9, 0.04)
    major_x = np.arange(np.floor(t0/0.20)*0.20, np.ceil(t1/0.20)*0.20 + 1e-9, 0.20)
    for x in minor_x:
        shapes.append(dict(type="line", x0=x, x1=x, y0=y0, y1=y1, line=dict(width=0.5), layer="below"))
    for x in major_x:
        shapes.append(dict(type="line", x0=x, x1=x, y0=y0, y1=y1, line=dict(width=1.2), layer="below"))

    fig.update_layout(
        shapes=shapes,
        xaxis_title="Tiempo (s)",
        yaxis_title="Amplitud (mV)",
        height=480,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
    )
    return fig