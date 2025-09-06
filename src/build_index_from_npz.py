import os, argparse, numpy as np, pandas as pd

def list_npz(root):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".npz"):
                yield os.path.join(dirpath, f)

def main():
    ap = argparse.ArgumentParser(description="Crea un index.csv desde NPZ existentes, con opción de segmentar virtualmente (sin duplicar archivos).")
    ap.add_argument("--src", required=True, help="Carpeta raíz con .npz")
    ap.add_argument("--out", default="data/index.csv", help="Ruta de salida para index.csv")
    ap.add_argument("--seg-window", type=float, default=None, help="Tamaño de ventana (seg). Si se omite, una fila por archivo.")
    ap.add_argument("--seg-step", type=float, default=None, help="Paso entre ventanas (seg). Default = seg-window (no overlap)")
    args = ap.parse_args()

    rows = []
    for path in list_npz(args.src):
        try:
            with np.load(path) as d:
                if "signal" not in d.files or "fs" not in d.files:
                    print("Saltando (faltan keys) ->", path); continue
                sig = d["signal"]
                fs = float(d["fs"])
                leads = d["leads"] if "leads" in d.files else np.array([f"L{i+1}" for i in range(sig.shape[1])]) if sig.ndim > 1 else np.array(["L1"])
                if sig.ndim == 1:
                    sig = sig[:, None]
                n_samples, n_leads = sig.shape
                record_id = os.path.splitext(os.path.basename(path))[0]
        except Exception as e:
            print("ERROR leyendo", path, "->", e)
            continue

        if args.seg_window is None:
            rows.append({
                "segment_id": record_id,
                "record_id": record_id,
                "npz_path": path,
                "fs": fs,
                "n_samples": n_samples,
                "n_leads": n_leads,
                "leads": ",".join([str(x) for x in leads.tolist()]),
                "start_s": 0.0,
                "end_s": n_samples / fs
            })
        else:
            win = int(round(args.seg_window * fs))
            step = int(round((args.seg_step if args.seg_step else args.seg_window) * fs))
            if win <= 0 or step <= 0:
                print("Ventana/paso inválidos."); continue
            start = 0
            seg_idx = 0
            while start < n_samples:
                end = start + win
                if end > n_samples:
                    break
                start_s = start / fs
                end_s = end / fs
                seg_id = f"{record_id}_t{int(start_s)}-{int(end_s)}"
                rows.append({
                    "segment_id": seg_id,
                    "record_id": record_id,
                    "npz_path": path,
                    "fs": fs,
                    "n_samples": n_samples,
                    "n_leads": n_leads,
                    "leads": ",".join([str(x) for x in leads.tolist()]),
                    "start_s": float(start_s),
                    "end_s": float(end_s)
                })
                start += step
                seg_idx += 1

    if not rows:
        print("No se generaron filas. Revisa --src y el contenido de los .npz.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Index guardado en", args.out, "con", len(df), "filas.")

if __name__ == "__main__":
    main()