import os, argparse, numpy as np

ap = argparse.ArgumentParser(description="Valida NPZ: keys, shapes, fs/leads.")
ap.add_argument("--src", required=True, help="Carpeta con .npz")
args = ap.parse_args()

ok, bad = 0, 0
for dirpath, _, files in os.walk(args.src):
    for fname in files:
        if not fname.lower().endswith(".npz"): continue
        path = os.path.join(dirpath, fname)
        try:
            with np.load(path) as d:
                assert "signal" in d and "fs" in d, "Faltan 'signal' o 'fs'"
                sig = d["signal"]; fs = float(d["fs"])
                if sig.ndim == 1:
                    shape = (sig.shape[0], 1)
                else:
                    shape = sig.shape
                leads = d["leads"].tolist() if "leads" in d.files else ["L"+str(i+1) for i in range(shape[1])]
            print("OK:", fname, "| shape:", shape, "| fs:", fs, "| leads:", ",".join(map(str, leads)))
            ok += 1
        except Exception as e:
            print("ERROR:", fname, "->", e)
            bad += 1

print(f"Resumen: {ok} OK, {bad} con errores.")