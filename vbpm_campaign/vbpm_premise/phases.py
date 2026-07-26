"""Bar-phase constructors + wrapped-increment utilities (labels only)."""
import sys, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_final')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
from audit_common import ideal_barphase, ideal_beatlinear_barphase, FPS
TWO_PI = 2*np.pi

def wrap(x):
    return (x + np.pi) % TWO_PI - np.pi

def frame_t(T, fps=FPS):
    return (np.arange(T)+0.5)/fps

def inside_mask(s, T, fps=FPS):
    t = frame_t(T, fps)
    if len(s["downs"]) < 2: return np.zeros(T, bool)
    return (t >= s["downs"][0]) & (t < s["downs"][-1])

def bar_knots(s):
    """Per bar: (t_start, t_end, beat times inside)."""
    d = s["downs"]; b = s["beats"]
    out = []
    for i in range(len(d)-1):
        a, e = d[i], d[i+1]
        inb = b[(b >= a-1e-6) & (b < e-1e-6)]
        out.append((a, e, inb))
    return out

def phase_beatlinear(s, T):
    return ideal_beatlinear_barphase(s["beats"], s["downs"], T, FPS)

def phase_downlinear(s, T):
    return ideal_barphase(s["downs"], T, FPS, mode="extrap")

def phase_pchip(s, T):
    """UNWRAPPED cumulative bar-phase through beat knots with monotone-cubic (PCHIP)
    interpolation = a smooth, continuously-varying tempo consistent with the same knots."""
    from scipy.interpolate import PchipInterpolator
    d, b = s["downs"], s["beats"]
    kt, kv = [], []
    cum = 0.0
    for i in range(len(d)-1):
        a, e = d[i], d[i+1]
        inb = b[(b >= a-1e-6) & (b < e-1e-6)]
        m = max(len(inb), 1)
        if len(inb) == 0: inb = np.array([a])
        for j, bt in enumerate(inb):
            kt.append(bt); kv.append(cum + TWO_PI*j/m)
        cum += TWO_PI
    kt.append(d[-1]); kv.append(cum)
    kt = np.asarray(kt, float); kv = np.asarray(kv, float)
    ok = np.concatenate([[True], np.diff(kt) > 1e-6])
    kt, kv = kt[ok], kv[ok]
    if len(kt) < 4: return None
    f = PchipInterpolator(kt, kv)
    t = frame_t(T)
    t = np.clip(t, kt[0], kt[-1])
    return f(t)          # UNWRAPPED
