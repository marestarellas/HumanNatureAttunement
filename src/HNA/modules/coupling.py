import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

# --------------------------
# Utils: NaN handling & zscore
# --------------------------
def _nan_interp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return x
    mask = np.isnan(x)
    if not mask.any():
        return x
    idx = np.arange(n)
    # forward/backward fill edges
    if mask[0]:
        first = np.flatnonzero(~mask)
        if len(first) == 0:
            raise ValueError("All values are NaN.")
        x[:first[0]] = x[first[0]]
    if mask[-1]:
        last = np.flatnonzero(~mask)
        x[last[-1]+1:] = x[last[-1]]
    # linear interp interior
    mask = np.isnan(x)
    x[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return x

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.mean(x)
    s = np.std(x)
    return (x - m) / (s + 1e-12)

# --------------------------
# 1) Windowed Cross-Correlation with Lag
# --------------------------
@dataclass
class XCorrResult:
    times_s: np.ndarray          # center time per window
    lags_s: np.ndarray           # lag axis (seconds)
    xcorr: Optional[np.ndarray]  # shape (n_windows, n_lags) if return_matrix=True else None
    peak_r: np.ndarray           # max correlation per window
    peak_lag_s: np.ndarray       # lag (s) at which correlation peaks per window

def windowed_xcorr(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    win_sec: float = 180.0,
    step_sec: float = 10.0,
    max_lag_sec: float = 30.0,
    detrend: bool = True,
    zscore: bool = True,
    return_matrix: bool = False,
) -> XCorrResult:
    s1 = _nan_interp(s1)
    s2 = _nan_interp(s2)
    if detrend:
        s1 = signal.detrend(s1, type='linear')
        s2 = signal.detrend(s2, type='linear')
    if zscore:
        s1, s2 = _zscore(s1), _zscore(s2)

    N = len(s1)
    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    L = int(round(max_lag_sec * fs))
    if W <= 1 or W > N:
        raise ValueError("Window size must be >1 and <= length of signals.")

    starts = np.arange(0, N - W + 1, H)
    lags = np.arange(-L, L + 1)
    lags_s = lags / fs

    peak_r = []
    peak_lag_s = []
    xcorr_mat = [] if return_matrix else None
    times_s = []

    for st in starts:
        seg1 = s1[st:st+W]
        seg2 = s2[st:st+W]
        seg1 -= seg1.mean(); seg2 -= seg2.mean()
        denom = (np.std(seg1) * np.std(seg2) * len(seg1) + 1e-12)
        full = signal.correlate(seg1, seg2, mode='full') / denom
        # extract only desired lag range
        center = len(full) // 2
        corr = full[center - L:center + L + 1]
        if return_matrix:
            xcorr_mat.append(corr)
        i_max = np.argmax(corr)
        peak_r.append(corr[i_max])
        peak_lag_s.append(lags_s[i_max])
        times_s.append((st + W/2) / fs)

    return XCorrResult(
        times_s=np.asarray(times_s),
        lags_s=lags_s,
        xcorr=np.asarray(xcorr_mat) if return_matrix else None,
        peak_r=np.asarray(peak_r),
        peak_lag_s=np.asarray(peak_lag_s),
    )

# --------------------------
# 2) Coherence (Welch) + windowed band-avg coherence
# --------------------------
@dataclass
class CoherenceResult:
    f: np.ndarray
    Cxy: np.ndarray              # global magnitude-squared coherence
    peak_f: float
    peak_coh: float
    band_avg_coh: float
    times_s: Optional[np.ndarray] = None          # for windowed coherence
    band_avg_coh_win: Optional[np.ndarray] = None # time series of band-avg coherence

def band_coherence(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    fmin: float = 0.05,
    fmax: float = 0.5,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    detrend: str = 'constant',
    windowed: bool = True,
    win_sec: float = 180.0,
    step_sec: float = 30.0,
) -> CoherenceResult:
    s1 = _nan_interp(s1); s2 = _nan_interp(s2)
    if nperseg is None:
        # choose long segment for fine LF resolution (cap to signal length)
        nperseg = min(len(s1), int(round(fs * 300)))  # ~5 min
    if noverlap is None:
        noverlap = nperseg // 2

    f, Cxy = signal.coherence(s1, s2, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    band_mask = (f >= fmin) & (f <= fmax)
    if not np.any(band_mask):
        raise ValueError("No frequencies inside requested band. Adjust fmin/fmax or nperseg.")

    band_f = f[band_mask]; band_C = Cxy[band_mask]
    i_peak = np.argmax(band_C)
    peak_f, peak_coh = float(band_f[i_peak]), float(band_C[i_peak])
    band_avg = float(np.mean(band_C))

    times_s = None
    band_avg_win = None
    if windowed:
        W = int(round(win_sec * fs))
        H = int(round(step_sec * fs))
        if W < nperseg:
            W = nperseg  # ensure Welch can run
        starts = np.arange(0, len(s1) - W + 1, H)
        ts, vals = [], []
        for st in starts:
            seg1 = s1[st:st+W]; seg2 = s2[st:st+W]
            f_w, C_w = signal.coherence(seg1, seg2, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
            m = (f_w >= fmin) & (f_w <= fmax)
            vals.append(np.mean(C_w[m]) if np.any(m) else np.nan)
            ts.append((st + W/2)/fs)
        times_s = np.asarray(ts); band_avg_win = np.asarray(vals)

    return CoherenceResult(f=f, Cxy=Cxy, peak_f=peak_f, peak_coh=peak_coh, band_avg_coh=band_avg,
                           times_s=times_s, band_avg_coh_win=band_avg_win)

# --------------------------
# 3) Phase Synchrony (PLV) with Hilbert phases
# --------------------------
@dataclass
class PLVResult:
    f0: float
    band: Tuple[float, float]
    plv: float
    mean_phase_diff: float       # radians, in (-pi, pi]
    preferred_lag_s: float       # seconds (interpreting mean phase at f0)

def _butter_bandpass(lo, hi, fs, order=4):
    nyq = fs * 0.5
    lo = max(1e-6, lo)
    hi = min(hi, nyq * 0.99)
    sos = signal.butter(order, [lo/nyq, hi/nyq], btype='bandpass', output='sos')
    return sos

def _dominant_freq(x: np.ndarray, fs: float, fmin=0.05, fmax=0.5) -> float:
    # Welch PSD and find peak in the respiratory band
    nper = min(len(x), int(round(fs * 300)))
    f, Pxx = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper//2, detrend='constant')
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        raise ValueError("No bins in requested f-range for dominant freq detection.")
    return float(f[m][np.argmax(Pxx[m])])

def plv_phase_sync(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    f0: Optional[float] = None,          # set if you already know resp peak (Hz)
    bw_hz: float = 0.12,                  # total bandwidth around f0
    fmin_search: float = 0.05,
    fmax_search: float = 0.5,
    order: int = 4
) -> PLVResult:
    s1 = _nan_interp(s1); s2 = _nan_interp(s2)
    if f0 is None:
        f0 = _dominant_freq(s2, fs, fmin=fmin_search, fmax=fmax_search)  # detect from respiration by default
    half = bw_hz / 2.0
    lo, hi = max(1e-3, f0 - half), max(f0 + half, f0 + 1e-3)

    sos = _butter_bandpass(lo, hi, fs, order=order)
    x1 = signal.sosfiltfilt(sos, s1.astype(float))
    x2 = signal.sosfiltfilt(sos, s2.astype(float))

    phi1 = np.angle(signal.hilbert(x1))
    phi2 = np.angle(signal.hilbert(x2))
    dphi = np.angle(np.exp(1j*(phi1 - phi2)))  # wrap to (-pi, pi]

    plv = np.abs(np.mean(np.exp(1j*dphi)))
    mean_phase = np.angle(np.mean(np.exp(1j*dphi)))

    # interpret mean phase (radians) as time lag at f0
    preferred_lag_s = mean_phase / (2*np.pi * f0)

    return PLVResult(f0=f0, band=(lo, hi), plv=float(plv),
                     mean_phase_diff=float(mean_phase),
                     preferred_lag_s=float(preferred_lag_s))


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# --- windowed PLV (per-window time series) ---
def windowed_plv(s1, s2, fs, win_sec=180.0, step_sec=30.0,
                 f0=None, bw_hz=0.12, fmin_search=0.05, fmax_search=0.5):
    def _nan_interp(x):
        x = np.asarray(x, float)
        if not np.isnan(x).any(): return x
        idx = np.arange(len(x)); good = ~np.isnan(x)
        if not good.any(): raise ValueError("All-NaN segment.")
        x[:np.argmax(good)] = x[good][0]
        x[len(x)-1-np.argmax(good[::-1])+1:] = x[good][-1]
        bad = np.isnan(x); x[bad] = np.interp(idx[bad], idx[good], x[good])
        return x

    def _bandpass_sos(fs, lo, hi, order=4):
        ny = 0.5*fs
        lo = max(1e-6, lo/ny); hi = min(0.99, hi/ny)
        return signal.butter(order, [lo, hi], btype="bandpass", output="sos")

    def _dominant_freq(x, fs, fmin=0.05, fmax=0.5):
        nper = min(len(x), int(fs*300))
        f, Pxx = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper//2, detrend="constant")
        m = (f>=fmin)&(f<=fmax)
        return float(f[m][np.argmax(Pxx[m])]) if np.any(m) else (fmin+fmax)/2

    s1 = _nan_interp(s1); s2 = _nan_interp(s2)
    W = int(round(win_sec*fs)); H = int(round(step_sec*fs))
    starts = np.arange(0, max(len(s1)-W+1, 0), H)

    times, plv, mean_phi, lag_s = [], [], [], []
    for st in starts:
        seg1, seg2 = s1[st:st+W], s2[st:st+W]
        f0w = f0 if f0 is not None else _dominant_freq(seg2, fs, fmin_search, fmax_search)
        half = bw_hz/2
        sos = _bandpass_sos(fs, max(1e-3, f0w-half), f0w+half)
        x1 = signal.sosfiltfilt(sos, seg1)
        x2 = signal.sosfiltfilt(sos, seg2)
        dphi = np.angle(np.exp(1j*(np.angle(signal.hilbert(x1)) - np.angle(signal.hilbert(x2)))))
        e = np.exp(1j*dphi)
        plv.append(np.abs(np.mean(e)))
        mphi = np.angle(np.mean(e))
        mean_phi.append(mphi)
        lag_s.append(mphi/(2*np.pi*f0w) if f0w>0 else np.nan)
        times.append((st+W/2)/fs)

    return {"times_s": np.asarray(times),
            "plv": np.asarray(plv),
            "mean_phase_diff": np.asarray(mean_phi),
            "preferred_lag_s": np.asarray(lag_s)}

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import signal

# ---------- wPLI (global) ----------
@dataclass
class WPLIResult:
    f0: float
    band: Tuple[float, float]
    wpli: float

def _wpli_from_phase(dphi: np.ndarray) -> float:
    # dphi in radians; use imaginary part of unit phasors (sin)
    s = np.sin(dphi.astype(float))
    den = np.mean(np.abs(s)) + 1e-12
    num = np.abs(np.mean(s))
    return float(num / den) if np.isfinite(den) and den > 0 else np.nan

def wpli_phase_sync(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    f0: Optional[float] = None,
    bw_hz: float = 0.12,
    fmin_search: float = 0.05,
    fmax_search: float = 0.5,
    order: int = 4
) -> WPLIResult:
    s1 = _nan_interp(np.asarray(s1, float))
    s2 = _nan_interp(np.asarray(s2, float))
    if f0 is None:
        # detect from s2 (resp) by default
        nper = min(len(s2), int(round(fs * 300)))
        f, Pxx = signal.welch(s2, fs=fs, nperseg=nper, noverlap=nper//2, detrend="constant")
        m = (f >= fmin_search) & (f <= fmax_search)
        if not np.any(m):
            raise ValueError("No bins in requested f-range for dominant freq detection.")
        f0 = float(f[m][np.argmax(Pxx[m])])

    half = bw_hz / 2.0
    lo, hi = max(1e-3, f0 - half), max(f0 + half, f0 + 1e-3)
    sos = _butter_bandpass(lo, hi, fs, order=order)
    x1 = signal.sosfiltfilt(sos, s1)
    x2 = signal.sosfiltfilt(sos, s2)

    phi1 = np.angle(signal.hilbert(x1))
    phi2 = np.angle(signal.hilbert(x2))
    dphi = np.angle(np.exp(1j * (phi1 - phi2)))  # wrap to (-pi, pi]

    return WPLIResult(f0=f0, band=(lo, hi), wpli=_wpli_from_phase(dphi))

# ---------- wPLI (windowed time series) ----------
def windowed_wpli(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    win_sec: float = 180.0,
    step_sec: float = 30.0,
    f0: Optional[float] = None,
    bw_hz: float = 0.12,
    fmin_search: float = 0.05,
    fmax_search: float = 0.5,
    order: int = 4
):
    s1 = _nan_interp(np.asarray(s1, float))
    s2 = _nan_interp(np.asarray(s2, float))

    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    if W <= 1 or W > len(s1):
        raise ValueError("Window size must be >1 and <= length of signals.")

    starts = np.arange(0, len(s1) - W + 1, H)

    # Either fixed f0 for all windows, or auto-detect per window from s2
    def _dominant_freq(x, fs, fmin=fmin_search, fmax=fmax_search):
        nper = min(len(x), int(round(fs * 300)))
        f, Pxx = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper//2, detrend="constant")
        m = (f >= fmin) & (f <= fmax)
        return float(f[m][np.argmax(Pxx[m])]) if np.any(m) else (fmin + fmax) / 2.0

    wpli_vals, times_s = [], []
    lohi_cache: Optional[Tuple[float, float]] = None

    for st in starts:
        seg1 = s1[st:st+W]
        seg2 = s2[st:st+W]

        if f0 is None:
            f0_w = _dominant_freq(seg2, fs)
            half = bw_hz / 2.0
            lo, hi = max(1e-3, f0_w - half), max(f0_w + half, f0_w + 1e-3)
        else:
            f0_w = f0
            if lohi_cache is None:
                half = bw_hz / 2.0
                lohi_cache = (max(1e-3, f0 - half), max(f0 + half, f0 + 1e-3))
            lo, hi = lohi_cache

        sos = _butter_bandpass(lo, hi, fs, order=order)
        x1 = signal.sosfiltfilt(sos, seg1)
        x2 = signal.sosfiltfilt(sos, seg2)

        phi1 = np.angle(signal.hilbert(x1))
        phi2 = np.angle(signal.hilbert(x2))
        dphi = np.angle(np.exp(1j * (phi1 - phi2)))

        wpli_vals.append(_wpli_from_phase(dphi))
        times_s.append((st + W/2) / fs)

    return {
        "times_s": np.asarray(times_s),
        "wpli": np.asarray(wpli_vals),
        "band": (lo, hi) if lohi_cache is not None else None,  # if fixed f0
    }






# --- quick plotting helper ---
def plot_coupling_over_time(xc, coh, plv_win):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # 1) cross-corr peak + lag
    ax = axes[0]
    ax.plot(xc.times_s, xc.peak_r, lw=1.6, label="XCorr peak r")
    ax.set_ylabel("peak r"); ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(xc.times_s, xc.peak_lag_s, lw=1.1, ls="--", label="XCorr lag")
    ax2.set_ylabel("lag (s)")
    ax.set_title("Windowed cross-correlation")

    # 2) band-avg coherence (if windowed series available)
    ax = axes[1]
    if coh.times_s is not None and coh.band_avg_coh_win is not None:
        ax.plot(coh.times_s, coh.band_avg_coh_win, lw=1.6)
    ax.set_ylabel("band-avg coherence"); ax.grid(True, alpha=0.3)
    ax.set_title(f"Welch coherence (band avg ~ {getattr(coh,'band_avg_coh',np.nan):.2f}, global peak {coh.peak_coh:.2f} @ {coh.peak_f:.3f} Hz)")

    # 3) PLV + preferred lag
    ax = axes[2]
    ax.plot(plv_win["times_s"], plv_win["plv"], lw=1.6, label="PLV")
    ax.set_ylabel("PLV"); ax.grid(True, alpha=0.3)
    #ax2 = ax.twinx()
    #ax2.plot(plv_win["times_s"], plv_win["preferred_lag_s"], lw=1.1, ls="--", label="PLV lag")
    #ax2.set_ylabel("lag (s)")
    #ax.set_title("Windowed PLV")

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


import numpy as np
from scipy import signal

def _coh_choose_nperseg(W, fs, target_sec=60, min_segments=4):
    """Pick nperseg/noverlap so each window has enough Welch segments."""
    nper = int(fs * min(target_sec, max(8, W / 2)))  # not larger than half the window
    nper = max(32, nper)  # floor
    while True:
        nover = nper // 2
        step = nper - nover
        nseg = 1 + max(0, (W - nper) // max(1, step))
        if nseg >= min_segments or nper <= 32:
            return nper, nover
        nper = int(nper * 0.8)  # shrink until we have enough segments

def band_coherence_windowed(
    s1, s2, fs,
    fmin=0.08, fmax=0.35,
    win_sec=180.0, step_sec=30.0,
    detrend="constant"
):
    s1 = np.asarray(s1, float); s2 = np.asarray(s2, float)
    N = len(s1); assert N == len(s2)
    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    starts = np.arange(0, max(N - W + 1, 0), H)

    # global coherence (for reference)
    nper_g, nover_g = _coh_choose_nperseg(N, fs, target_sec=60, min_segments=8)
    f, Cxy = signal.coherence(s1, s2, fs=fs, nperseg=nper_g, noverlap=nover_g, detrend=detrend)
    m = (f >= fmin) & (f <= fmax)
    peak_f = float(f[m][np.argmax(Cxy[m])]) if np.any(m) else np.nan
    peak_coh = float(np.max(Cxy[m])) if np.any(m) else np.nan
    band_avg = float(np.mean(Cxy[m])) if np.any(m) else np.nan

    # windowed coherence
    times, band_avg_win = [], []
    for st in starts:
        seg1 = s1[st:st+W]; seg2 = s2[st:st+W]
        nper, nover = _coh_choose_nperseg(W, fs, target_sec=60, min_segments=4)
        fw, Cw = signal.coherence(seg1, seg2, fs=fs, nperseg=nper, noverlap=nover, detrend=detrend)
        mw = (fw >= fmin) & (fw <= fmax)
        # require at least ~4 bins in the band to be stable
        if np.sum(mw) >= 4 and np.isfinite(Cw[mw]).any():
            band_avg_win.append(float(np.nanmean(Cw[mw])))
        else:
            band_avg_win.append(np.nan)
        times.append((st + W/2) / fs)

    return {
        "f": f, "Cxy": Cxy,
        "peak_f": peak_f, "peak_coh": peak_coh, "band_avg_coh": band_avg,
        "times_s": np.asarray(times), "band_avg_coh_win": np.asarray(band_avg_win),
        "params": dict(fmin=fmin, fmax=fmax, win_sec=win_sec, step_sec=step_sec)
    }

import numpy as np
import matplotlib.pyplot as plt

def plot_coherence_results(coh, band=None, title="Coherence summary"):
    """
    Plot global coherence spectrum + windowed band-avg coherence over time.

    Parameters
    ----------
    coh : dict or object
        Must expose the following attributes/keys:
          - f (1D array): frequency axis
          - Cxy (1D array): global magnitude-squared coherence spectrum
          - peak_f (float): frequency of global peak within band
          - peak_coh (float): value of global peak within band
          - band_avg_coh (float): mean coherence within band (global)
          - times_s (1D array): window centers in seconds (optional)
          - band_avg_coh_win (1D array): time series of band-avg coherence (optional)
    band : tuple or None
        (fmin, fmax) to shade the band on the spectrum plot. Optional.
    title : str
        Figure title.
    """
    # allow both dicts and dataclasses/objects
    def get(name, default=None):
        if isinstance(coh, dict):
            return coh.get(name, default)
        return getattr(coh, name, default)

    f = get("f"); Cxy = get("Cxy")
    peak_f = get("peak_f", np.nan)
    peak_coh = get("peak_coh", np.nan)
    band_avg = get("band_avg_coh", np.nan)
    times = get("times_s", None)
    band_ts = get("band_avg_coh_win", None)

    if f is None or Cxy is None:
        raise ValueError("`coh` must contain 'f' and 'Cxy'.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    fig.suptitle(title)

    # --- Top: global spectrum
    ax = axes[0]
    ax.plot(f, Cxy, lw=1.8)
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    if band is not None:
        fmin, fmax = band
        ax.axvspan(fmin, fmax, color="0.9", zorder=0)
    if np.isfinite(peak_f) and np.isfinite(peak_coh):
        ax.plot([peak_f], [peak_coh], "o")
        ax.text(peak_f, min(1.0, peak_coh+0.03), f"peak {peak_coh:.2f}@{peak_f:.3f} Hz",
                ha="center", va="bottom")
    if np.isfinite(band_avg):
        ax.text(0.01, 0.95, f"band avg ≈ {band_avg:.2f}", transform=ax.transAxes,
                ha="left", va="top")

    # --- Bottom: windowed band-avg over time (if available)
    ax = axes[1]
    if times is not None and band_ts is not None:
        ax.plot(times, band_ts, lw=1.8)
        ax.set_ylabel("Band-avg coherence")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No windowed series in `coh`.\nProvide times_s & band_avg_coh_win.",
                ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
