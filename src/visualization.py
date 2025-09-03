# visualization.py
import numpy as np
import plotly.graph_objects as go

def fig_ecg_plotly(sig, fs, r_idx=None):
    t = np.arange(len(sig)) / fs

    # Señal
    trace = go.Scatter(x=t, y=sig, mode="lines", name="ECG", hovertemplate="t=%{x:.3f}s<br>mV=%{y:.3f}")

    # Cuadrícula tipo papel: grandes (0.2 s / 0.5 mV) y pequeñas (0.04 s / 0.1 mV)
    major_t, minor_t = 0.2, 0.04
    major_v, minor_v = 0.5, 0.1
    ypad = 0.2 + 0.1 * (np.max(np.abs(sig)) + 1e-6)
    ymin, ymax = float(np.min(sig)-ypad), float(np.max(sig)+ypad)

    shapes = []
    # verticales
    x0 = 0.0
    while x0 <= t[-1]+1e-9:
        shapes.append(dict(type="line", x0=x0, x1=x0, y0=ymin, y1=ymax,
                           line=dict(width=1 if abs((x0/major_t)-round(x0/major_t))<1e-9 else 0.5,
                                     color="rgba(255,0,0,0.6)" if abs((x0/major_t)-round(x0/major_t))<1e-9 else "rgba(255,0,0,0.3)")))
        x0 += minor_t
    # horizontales
    y0 = np.floor(ymin/minor_v)*minor_v
    while y0 <= ymax+1e-9:
        major = abs((y0/major_v)-round(y0/major_v))<1e-9
        shapes.append(dict(type="line", x0=t[0], x1=t[-1], y0=y0, y1=y0,
                           line=dict(width=1 if major else 0.5,
                                     color="rgba(255,0,0,0.6)" if major else "rgba(255,0,0,0.3)")))
        y0 += minor_v

    # R-peaks con hover de RR y HR (respecto al siguiente pico)
    scatter_r = None
    if r_idx is not None and len(r_idx) > 0:
        r_idx = np.asarray(r_idx, dtype=int)
        rr = np.diff(r_idx) / fs
        hr = np.where(rr > 0, 60.0/rr, np.nan)

        # extender HR a mismo largo que r_idx (último sin valor)
        hr_full = np.full(len(r_idx), np.nan)
        hr_full[:len(hr)] = hr

        # colores por BPM
        colors = []
        for h in hr_full:
            if np.isnan(h):
                colors.append("rgba(0,0,0,0)")   # último pico sin HR→
            elif h < 60:
                colors.append("blue")
            elif h <= 100:
                colors.append("green")
            else:
                colors.append("red")

        # customdata: [RR, HR] para hover
        cd = np.stack([np.r_[rr, np.nan], hr_full], axis=1)

        scatter_r = go.Scatter(
            x=t[r_idx], y=sig[r_idx], mode="markers", name="R-peaks",
            marker=dict(size=7, color=colors, line=dict(width=2)),
            customdata=cd,
            hovertemplate=(
                "R @ t=%{x:.3f}s<br>mV=%{y:.3f}"
                "<br>RR→=%{customdata[0]:.3f}s"
                "<br>HR→=%{customdata[1]:.1f} bpm"
            )
        )

    fig = go.Figure(data=[trace] + ([scatter_r] if scatter_r else []))
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title="Tiempo (s)", zeroline=False, range=[t[0], t[-1]]),
        yaxis=dict(title="mV", zeroline=False, range=[ymin, ymax]),
        shapes=shapes,
        margin=dict(l=40, r=20, t=10, b=40),
        dragmode="pan"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
