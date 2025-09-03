
from pathlib import Path
import numpy as np,pandas as pd
DATA_DIR=Path("data")
def load_manifest():
    mf_path=DATA_DIR/"manifest.csv"
    if not mf_path.exists():
        return pd.DataFrame([{"segment_id":"synthetic_01","record_id":"synthetic","lead":"II","fs":500.0,
        "start_s":0.0,"end_s":15.0,"path":str(DATA_DIR/"slices"/"synthetic_01.npz")}])
    mf=pd.read_csv(mf_path)
    mf["path"]=mf["segment_id"].apply(lambda s:str(DATA_DIR/"slices"/f"{s}.npz"))
    return mf
def load_slice(path):
    with np.load(path) as f:
        return f["signal"].astype(float),float(f["fs"]),str(f["lead"])
