
import numpy as np,neurokit2 as nk
def ecg_hr(sig,fs):
    clean=nk.ecg_clean(sig,sampling_rate=fs)
    _,info=nk.ecg_peaks(clean,sampling_rate=fs)
    rpeaks=info.get("ECG_R_Peaks",None)
    if rpeaks is None: r_idx=np.array([])
    else: r_idx=np.where(rpeaks==1)[0]
    if r_idx.size<2: return {"r_idx":r_idx,"hr_series":np.array([]),"hr_avg":np.nan}
    rr=np.diff(r_idx)/fs
    hr=60.0/rr
    return {"r_idx":r_idx,"hr_series":hr,"hr_avg":float(np.median(hr))}
