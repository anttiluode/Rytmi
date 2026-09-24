"""
rytmi_core — shared pieces for Rytmi, carried over from TATWATASW (E3, E4).

  make_rhythm   theta phase with wandering frequency + a noisy membrane oscillation
  leaky         first-order leaky trace (vectorised)
  iq_phase      phase of a rhythm read from TWO leaky traces (fast - slow, slow) with a fixed 2x2 readout
  stdp          pairwise additive STDP (+/-20 ms), optional per-spike gate
  replay        rate network with adaptation; cue one cell, see who follows
  replay_detail 165-setting grid: forward-only in order / forward out of order / any backward / none
"""
import numpy as np
from scipy.signal import lfilter

FDT = 0.0005                     # fine time step (s)
F0 = 8.0                         # nominal theta (Hz)
TAU_STDP = 0.02


def ou(rng, n, tau, sd):
    a = np.exp(-FDT / tau); b = sd * np.sqrt(1 - a * a)
    e = b * rng.standard_normal(n); e[0] = sd * rng.standard_normal()
    return lfilter([1.0], [1.0, -a], e)


def make_rhythm(rng, n, f_sd=0.6, amp_sd=0.4, noise=0.3):
    """Theta phase Phi (radians) with OU-wandering frequency around F0; amplitude A(t) (log-normal drift);
    membrane oscillation v = A cos(Phi) + white noise (what a synapse could read locally)."""
    f = F0 + ou(rng, n, 0.5, f_sd)
    phi = 2 * np.pi * np.cumsum(f) * FDT + rng.uniform(0, 2 * np.pi)
    amp = np.exp(ou(rng, n, 1.0, amp_sd))
    v = amp * np.cos(phi) + noise * rng.standard_normal(n)
    return phi, amp, v


def leaky(x, tau):
    a = FDT / tau
    return lfilter([a], [1.0, -(1.0 - a)], x, axis=-1)


def iq_phase(v, ratio=9.0, w=2 * np.pi * F0):
    tf = 1.0 / (w * np.sqrt(ratio)); ts = ratio * tf
    fast, slow = leaky(v, tf), leaky(v, ts)
    Hc = 1 / (1 + 1j * w * tf) - 1 / (1 + 1j * w * ts); Hs = 1 / (1 + 1j * w * ts)
    M = np.array([[Hc.real, -Hc.imag], [Hs.real, -Hs.imag]])
    xy = np.linalg.solve(M, np.vstack([fast - slow, slow]))
    return np.arctan2(xy[1], xy[0])


def stdp(st, sid, n, weight=None, W=None):
    if W is None:
        W = np.zeros((n, n))
    if weight is None:
        weight = np.ones(len(st))
    for k in range(len(st)):
        if weight[k] == 0:
            continue
        lo, hi = np.searchsorted(st, st[k] - 5 * TAU_STDP), np.searchsorted(st, st[k] + 5 * TAU_STDP)
        for m in range(lo, hi):
            if m == k or sid[m] == sid[k] or weight[m] == 0:
                continue
            d = st[k] - st[m]
            W[sid[k], sid[m]] += weight[k] * weight[m] * (np.exp(-d / TAU_STDP) if d > 0 else -np.exp(d / TAU_STDP))
    return W


RDT = 0.005


def replay(W, cue, T=1.5, tau=0.02, tau_a=0.25, g_a=1.5, gain=2.2, thr=0.3):
    n = W.shape[0]; steps = int(T / RDT)
    r = np.zeros(n); a = np.zeros(n); out = np.zeros((n, steps))
    for k in range(steps):
        I = np.zeros(n)
        if k * RDT < 0.05:
            I[cue] = 1.5
        drive = gain * W @ r + I - a - thr
        r += RDT / tau * (-r + np.clip(drive, 0, 1))
        a += RDT / tau_a * (-a + g_a * r)
        out[:, k] = r
    return out


def replay_detail(Wp, cue):
    n = Wp.shape[0]
    out = {'forward_only_in_order': 0, 'forward_out_of_order': 0, 'any_backward': 0, 'none': 0}
    for gain in np.linspace(1.5, 4.0, 11):
        for thr in np.linspace(0.1, 0.8, 15):
            rr = replay(Wp, cue, gain=gain, thr=thr)
            act = [i for i in range(n) if i != cue and rr[i].max() > 0.2]
            pk = {i: np.argmax(rr[i]) for i in act}
            fwd = sorted(i for i in act if i > cue); bwd = [i for i in act if i < cue]
            if not act: out['none'] += 1
            elif bwd: out['any_backward'] += 1
            elif all(pk[a] < pk[b] for a, b in zip(fwd, fwd[1:])): out['forward_only_in_order'] += 1
            else: out['forward_out_of_order'] += 1
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def score(W, n_spk):
    n = W.shape[0]; iu = np.triu_indices(n, 1)
    net = W.T[iu] - W[iu]
    d = np.array([j - i for i, j in zip(*iu)]); pos = np.maximum(net, 0)
    Wp = np.maximum(W, 0); Wp /= max(Wp.max(), 1e-12)
    return dict(asymmetry_index=float(net.sum() / (np.abs(W).sum() + 1e-9)),
                forward_drive_per_spike=float(net.sum() / max(n_spk, 1)),
                mean_forward_reach_items=float((d * pos).sum() / (pos.sum() + 1e-9)),
                replay=replay_detail(Wp, cue=n // 2))


def circ_lin_fit(x, ph):
    """Phase-position fit: slope a (cycles per unit x) maximising |mean exp(i 2pi (ph - a x))| (Kempter et al. 2012).
    Returns slope, resultant length R and the circular-linear correlation rho."""
    best = (-1, 0.0)
    for a in np.linspace(-3, 3, 1201):
        R = np.abs(np.mean(np.exp(1j * 2 * np.pi * (ph - a * x))))
        if R > best[0]:
            best = (R, a)
    R, a = best
    th = (2 * np.pi * a * x) % (2 * np.pi); pc = 2 * np.pi * ph
    tb = np.angle(np.mean(np.exp(1j * th))); pb = np.angle(np.mean(np.exp(1j * pc)))
    rho = np.sum(np.sin(pc - pb) * np.sin(th - tb)) / np.sqrt(np.sum(np.sin(pc - pb) ** 2) * np.sum(np.sin(th - tb) ** 2))
    return dict(slope_cycles_per_unit=float(a), resultant=float(R), rho=float(rho))
