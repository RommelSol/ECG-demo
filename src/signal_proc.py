
from scipy.signal import butter,filtfilt
def bandpass_filter(sig,fs,low=5.0,high=15.0,order=3):
    nyq=0.5*fs
    b,a=butter(order,[low/nyq,high/nyq],btype="band")
    return filtfilt(b,a,sig)
def detrend_baseline(sig,fs,cutoff=0.5,order=2):
    nyq=0.5*fs
    b,a=butter(order,cutoff/nyq,btype="high")
    return filtfilt(b,a,sig)
