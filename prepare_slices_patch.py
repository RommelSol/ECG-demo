# prepare_slices_patch.py — corrige headers con base_date '/' y reintenta
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample
from fractions import Fraction
from scipy.signal import resample_poly

def get_meta(meta, key, default=None):
    return meta.get(key, default) if isinstance(meta, dict) else getattr(meta, key, default)

def to_millivolts(sig, units):
    u = (units or "mV").lower()
    if u in ["mv","millivolt","millivolts"]: return sig
    if u in ["v","volt","volts"]:            return sig*1000.0
    if u in ["uv","microvolt","microvolts"]: return sig/1000.0
    return sig

def read_record_safe(rec_path: Path):
    """Lee un registro; si falla por fecha inválida en .hea, crea .hea '_fixed' con 01/01/1970 y reintenta."""
    rec_str = str(rec_path)
    try:
        return wfdb.rdsamp(rec_str)
    except Exception as e:
        msg = str(e)
        if "time data '/'" not in msg and "does not match format" not in msg:
            raise
        hea = rec_path.with_suffix(".hea")
        if not hea.exists(): raise

        fixed = rec_path.with_name(rec_path.name + "_fixed").with_suffix(".hea")
        txt = hea.read_text(encoding="utf-8", errors="ignore")

        # Si la línea base_date no tiene dígitos (o es '/'), reemplaza por 01/01/1970
        def _fix_line(m):
            prefix, tail = m.group(1), m.group(2).strip()
            return f"{prefix}01/01/1970" if not re.search(r'\d', tail) else m.group(0)

        txt_fixed = re.sub(r"(?mi)^(?:\s*)(base_date\s+)(.*)$", _fix_line, txt)
        fixed.write_text(txt_fixed, encoding="utf-8")

        # Reintentar con el nuevo header (misma base de nombre, sin extensión)
        return wfdb.rdsamp(str(fixed.with_suffix("")))

def resample_safe(x, fs_in, fs_out):
    if abs(fs_in - fs_out) < 1e-9: 
        return x
    frac = Fraction(fs_out, fs_in).limit_denominator(64)  # p.ej. 500/360 -> 25/18
    up, down = frac.numerator, frac.denominator
    return resample_poly(x, up, down)  # FIR, sin ringing visible

def main(args):
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("data/manifest.csv")

    bases = sorted([p.with_suffix("") for p in raw_dir.glob("**/*.hea")])
    if not bases:
        print(f"[!] No se encontraron headers .hea en {raw_dir}"); return

    rows, seg_count = [], 0
    for i, rec in enumerate(bases, 1):
        if args.max_segments and seg_count >= args.max_segments: break
        try:
            samp, meta = read_record_safe(rec)
        except Exception as e:
            print(f"SKIP {rec} | {e}"); continue

        fs = float(get_meta(meta, "fs", 500))
        ch_names = list(get_meta(meta, "sig_name", [])) or []
        units_all = get_meta(meta, "units", None)
        if fs <= 0 or not ch_names:
            print(f"SKIP {rec} | fs={fs} | sin canales"); continue

        try:
            lead_idx = ch_names.index(args.lead)
        except ValueError:
            lead_idx = 0

        sig_raw = samp[:, lead_idx].astype(float)
        units = None
        units_all = get_meta(meta, "units", None)
        if units_all is not None:
            try: units = units_all[lead_idx]
            except Exception: units = None
        sig = to_millivolts(sig_raw, units or "mV")

        if args.fs_out and abs(fs - args.fs_out) > 1e-6:
            n_out = int(len(sig) * args.fs_out / fs)
            sig = resample_safe(sig, fs, args.fs_out)
            fs = float(args.fs_out)

        win = int(args.win_s * fs); n = len(sig)
        if win <= 0 or n < win: continue
        starts = np.arange(0, n - win + 1, win)

        for st in starts:
            if args.max_segments and seg_count >= args.max_segments: break
            ed = int(st + win)
            seg = sig[st:ed].astype(np.float32)
            if seg.size < win: continue

            seg_id = f"{rec.name}_{ch_names[lead_idx]}_{st}_{ed}"
            np.savez_compressed(out_dir / f"{seg_id}.npz", signal=seg, fs=fs, lead=ch_names[lead_idx])
            rows.append({
                "segment_id": seg_id, "record_id": rec.name, "lead": ch_names[lead_idx],
                "fs": fs, "start_s": float(st/fs), "end_s": float(ed/fs),
            })
            seg_count += 1

        if i % 50 == 0: print(f"[{i}/{len(bases)}] segmentos: {seg_count}")

    if not rows:
        print("[!] No se generaron segmentos. Revisa --raw_dir / --win_s / --lead."); return

    mf_new = pd.DataFrame(rows)
    if manifest_path.exists():
        mf_old = pd.read_csv(manifest_path)
        mf = pd.concat([mf_old, mf_new], ignore_index=True).drop_duplicates(subset=["segment_id"])
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        mf = mf_new
    mf.to_csv(manifest_path, index=False)
    print(f"Listo. Segmentos creados: {seg_count}. Manifest: {manifest_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", default="data/slices")
    p.add_argument("--lead", default="II")
    p.add_argument("--win_s", type=float, default=15.0)
    p.add_argument("--fs_out", type=float, default=500.0)
    p.add_argument("--max_segments", type=int, default=200)
    args = p.parse_args()
    main(args)
