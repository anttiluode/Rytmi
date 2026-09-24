"""
E4 — DELETE THE CLOCK: can two local leaky traces stand in for the imposed theta phase?

AnotherOddThing v5 showed a fast-minus-slow trace separates "entering" from "leaving" at matched level.
The frequency-domain reason: two leaky traces form a band-pass H(w) = 1/(1+iw tf) - 1/(1+iw ts), peaked at
w* = 1/sqrt(tf ts), where the contrast (fast - slow) and the slow trace sit at a FIXED phase offset.
A fixed 2x2 linear readout of (contrast, slow) therefore recovers the phase of a rhythm near w*:
a local, amplitude-invariant phase estimate made of two state variables. No global phase variable.

E4a  LOCAL DEAD TIME.
     E3's failing condition: precession spread over the WHOLE theta cycle (no dead time) -> wrap-around
     pairs cancel the order. Here nothing in the spike trains changes. Instead, plasticity is gated
     by a phase estimate: a spike pair only writes if both spikes fall in the "open" part of the cycle.
       oracle gate   : true theta phase (a global clock; upper reference)
       local I/Q gate: phase read from two leaky traces of a noisy membrane theta oscillation whose
                       frequency (OU around 8 Hz) and amplitude (slow log-normal drift) wander.
                       The window is fixed a priori to the oracle's; nothing is tuned on the result.
       one-state gate: a single leaky trace thresholded (best of a tau x threshold grid, chosen ON the
                       result -- deliberately generous). It can make a window, but its width depends on
                       amplitude, which it cannot know.
       wrong-rhythm  : the same I/Q machinery fed an independent rhythm realization (same statistics).
                       Same open fraction, no information -> must NOT rescue.
E4b  NO RHYTHM AT ALL.
     Poisson spikes from the same place fields, no theta, no precession. At each post spike, the synapse
     reads the PRE cell's traces at FIELD timescale.
       STDP +/-20 ms         : E3's rate-only baseline
       one-state (pre trace) : dW[i,j] += s_j(t_post)             (plain trace Hebbian)
       two-state contrast    : dW[i,j] += -(f_j - s_j)(t_post)    (post fires while pre is FALLING -> pre came
                                                                   first -> potentiate). This is differential
                                                                   Hebbian learning (Kosko 1986; Klopf 1988),
                                                                   known to relate to STDP (Rao & Sejnowski 2001).
       shuffled contrast     : same rule, contrast taken from a random other cell.
"""
import json
import numpy as np
from scipy.signal import lfilter
from tatwatasw import replay_grid, replay, t as T_REPLAY

FDT = 0.0005                       # fine grid for rhythm and traces (s)
N, SPACING, SIGMA, RATE, LAPS = 20, 0.2, 0.25, 25.0, 20
CENTERS = 0.3 + SPACING * np.arange(N)
T_TOTAL = CENTERS[-1] + 0.5
TF = np.arange(0, T_TOTAL, FDT)
F0 = 8.0
TAU_STDP = 0.02


# ---------------------------------------------------------------------------------------------- rhythm
def ou(rng, n, tau, sd):
    a = np.exp(-FDT / tau); b = sd * np.sqrt(1 - a * a)
    e = b * rng.standard_normal(n); e[0] = sd * rng.standard_normal()
    return lfilter([1.0], [1.0, -a], e)


def make_rhythm(rng, f_sd=0.6, amp_sd=0.4, noise=0.3):
    """Theta phase Phi(t) with wandering frequency, and a noisy membrane oscillation v(t) = A(t) cos Phi + noise."""
    f = F0 + ou(rng, len(TF), 0.5, f_sd)
    phi = 2 * np.pi * np.cumsum(f) * FDT + rng.uniform(0, 2 * np.pi)
    amp = np.exp(ou(rng, len(TF), 1.0, amp_sd))
    v = amp * np.cos(phi) + noise * rng.standard_normal(len(TF))
    return phi, v


def leaky(x, tau):
    a = FDT / tau                                                         # y[k] = y[k-1] + a (x[k] - y[k-1])
    return lfilter([a], [1.0, -(1.0 - a)], x, axis=-1)


def iq_phase(v, ratio=9.0, w=2 * np.pi * F0):
    """Two leaky traces (tf, ts) with sqrt(tf ts) = 1/w. Fixed 2x2 readout of (contrast, slow) -> phase."""
    tf = 1.0 / (w * np.sqrt(ratio)); ts = ratio * tf
    fast, slow = leaky(v, tf), leaky(v, ts)
    c = fast - slow
    Hc = 1 / (1 + 1j * w * tf) - 1 / (1 + 1j * w * ts); Hs = 1 / (1 + 1j * w * ts)
    M = np.array([[Hc.real, -Hc.imag], [Hs.real, -Hs.imag]])     # [c; s] = M [x; y],  z = x + i y ~ A e^{i Phi}
    xy = np.linalg.solve(M, np.vstack([c, slow]))
    return np.arctan2(xy[1], xy[0]), (tf, ts)


# ---------------------------------------------------------------------------------------------- spikes
def precession_spikes(rng, phi, full_cycle=True):
    """E3's precession spikes, but locked to the wandering rhythm: a spike at cycle phase p is placed where Phi = 2 pi (k + p)."""
    per = 1.0 / F0
    k0, k1 = int(np.ceil(phi[0] / (2 * np.pi))), int(np.floor(phi[-1] / (2 * np.pi))) - 1
    st, sid = [], []
    for k in range(k0, k1):
        mid = np.interp(2 * np.pi * (k + 0.5), phi, TF)
        for i in range(N):
            p_fire = min(RATE * per * np.exp(-(mid - CENTERS[i]) ** 2 / (2 * SIGMA ** 2)) / 3, 1.0)
            if rng.random() < p_fire:
                rel = np.clip((CENTERS[i] - mid) / (2.5 * SIGMA), -0.5, 0.5)
                ph = (0.5 + rel) if full_cycle else (0.40 + 0.5 * rel)
                ph = np.clip(ph, 0.0, 0.999)
                st.append(np.interp(2 * np.pi * (k + ph), phi, TF) + rng.normal(0, 0.003)); sid.append(i)
    o = np.argsort(st)
    return np.array(st)[o], np.array(sid)[o]


def stdp(st, sid, weight=None, W=None):
    """E3's pairwise additive STDP (+/-20 ms). weight[k] in [0,1] gates each spike's participation."""
    if W is None:
        W = np.zeros((N, N))
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


def replay_detail(Wp, cue=N // 2):
    """Same 165-setting grid as replay_grid, but splits E3's 'both_directions' bucket into
    'any_backward' (a cell behind the cue fired) and 'forward_out_of_order' (only forward cells fired, not in order)."""
    out = {'forward_only_in_order': 0, 'forward_out_of_order': 0, 'any_backward': 0, 'none': 0}
    for gain in np.linspace(1.5, 4.0, 11):
        for thr in np.linspace(0.1, 0.8, 15):
            rr = replay(Wp, cue, gain=gain, thr=thr)
            act = [i for i in range(N) if i != cue and rr[i].max() > 0.2]
            pk = {i: T_REPLAY[np.argmax(rr[i])] for i in act}
            fwd = sorted(i for i in act if i > cue); bwd = [i for i in act if i < cue]
            if not act: out['none'] += 1
            elif bwd: out['any_backward'] += 1
            elif all(pk[a] < pk[b] for a, b in zip(fwd, fwd[1:])): out['forward_only_in_order'] += 1
            else: out['forward_out_of_order'] += 1
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def score(W, n_spk):
    iu = np.triu_indices(N, 1)
    net = W.T[iu] - W[iu]
    d = np.array([j - i for i, j in zip(*iu)]); pos = np.maximum(net, 0)
    Wp = np.maximum(W, 0); Wp /= max(Wp.max(), 1e-12)
    return dict(asymmetry_index=float(net.sum() / (np.abs(W).sum() + 1e-9)),
                forward_drive_per_spike=float(net.sum() / max(n_spk, 1)),
                mean_forward_reach_items=float((d * pos).sum() / (pos.sum() + 1e-9)),
                replay_outcome_fractions=replay_grid(Wp, cue=N // 2), replay_detail=replay_detail(Wp))


def in_window(cyc_phase, closed):
    """open part of the cycle = everything except a band of width `closed` (fraction of cycle) around the boundary."""
    return ((cyc_phase > closed / 2) & (cyc_phase < 1 - closed / 2)).astype(float)


# ---------------------------------------------------------------------------------------------- E4a
def e4a(closed_fracs=(0.1, 0.15, 0.2, 0.3, 0.4, 0.5), seed=1):
    rng = np.random.default_rng(seed)
    laps = []
    for _ in range(LAPS):
        phi, v = make_rhythm(rng)
        _, v_other = make_rhythm(rng)                                     # independent rhythm, same statistics
        st, sid = precession_spikes(rng, phi, full_cycle=True)
        st_w, sid_w = precession_spikes(rng, phi, full_cycle=False)       # E3's passing condition, as reference
        ph_hat, taus = iq_phase(v); ph_wrong, _ = iq_phase(v_other)
        laps.append(dict(phi=phi, v=v, st=st, sid=sid, st_w=st_w, sid_w=sid_w, ph_hat=ph_hat, ph_wrong=ph_wrong))

    def cyc(ph_arr, times):                                               # cycle phase in [0,1) at spike times
        return (np.interp(times, TF, np.unwrap(ph_arr)) / (2 * np.pi)) % 1.0

    # phase-estimate quality (local I/Q vs truth), in cycle fraction
    err = []
    for L in laps:
        e = np.angle(np.exp(1j * (L['ph_hat'] - L['phi'])))[int(0.3 / FDT):]
        err.append(e)
    err = np.concatenate(err)
    R = dict(iq_taus_ms=[1000 * x for x in taus],
             iq_phase_error=dict(circular_mean_cycles=float(np.angle(np.mean(np.exp(1j * err))) / (2 * np.pi)),
                                 circular_sd_cycles=float(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * err))))) / (2 * np.pi))))

    def run(gate_fn):
        W = np.zeros((N, N)); n = 0
        for L in laps:
            g = gate_fn(L); W = stdp(L['st'], L['sid'], g, W); n += len(L['st'])
        return score(W, n), float(np.mean(np.concatenate([gate_fn(L) for L in laps])))

    s, _ = run(lambda L: np.ones(len(L['st']))); R['no_gate_full_cycle'] = s
    Ww = np.zeros((N, N)); nw = 0
    for L in laps:
        Ww = stdp(L['st_w'], L['sid_w'], None, Ww); nw += len(L['st_w'])
    R['reference_windowed_spikes'] = score(Ww, nw)

    R['sweep'] = {}
    for cf in closed_fracs:
        row = {}
        for name, key in [('oracle', 'phi'), ('local_iq', 'ph_hat'), ('wrong_rhythm_iq', 'ph_wrong')]:
            s, frac = run(lambda L, key=key: in_window(cyc(L[key], L['st']), cf))
            s['fraction_spikes_open'] = frac; row[name] = s
        # one-state attacker: single leaky trace of v, gate open when sign*trace > thr; best of grid ON THE RESULT
        best = None
        for tau in [0.001, 0.003, 0.006, 0.012, 0.02]:
            tr = [leaky(L['v'], tau) for L in laps]
            allv = np.concatenate([x[int(0.3 / FDT):] for x in tr])
            for sign in [1, -1]:
                thr = np.quantile(sign * allv, cf)                        # same open fraction of TIME on average
                Wt = np.zeros((N, N)); n = 0; opens = []
                for L, x in zip(laps, tr):
                    g = (sign * np.interp(L['st'], TF, x) > thr).astype(float); opens.append(g)
                    Wt = stdp(L['st'], L['sid'], g, Wt); n += len(L['st'])
                s = score(Wt, n); s['fraction_spikes_open'] = float(np.mean(np.concatenate(opens)))
                s['tau_ms'] = tau * 1000; s['sign'] = sign
                key = s['replay_outcome_fractions']['forward_only_in_order'] + 1e-3 * s['forward_drive_per_spike']
                if best is None or key > best[0]:
                    best = (key, s)
        row['one_state_best_of_grid'] = best[1]
        R['sweep'][str(cf)] = row
    return R


# ---------------------------------------------------------------------------------------------- E4b
def e4b(seed=2, tf=0.08, ts=0.32, tau_one=0.2):
    rng = np.random.default_rng(seed)
    rates = RATE / 3 * np.exp(-(TF[None, :] - CENTERS[:, None]) ** 2 / (2 * SIGMA ** 2))  # matched mean rate to E3
    W = {k: np.zeros((N, N)) for k in ['stdp', 'one_state', 'two_state_contrast', 'shuffled_contrast']}
    n_spk = 0
    for _ in range(LAPS):
        spk = rng.random(rates.shape) < rates * FDT
        x = spk.astype(float) / FDT
        fast, slow, one = leaky(x, tf), leaky(x, ts), leaky(x, tau_one)
        contrast = fast - slow
        perm = rng.permutation(N)
        while np.any(perm == np.arange(N)):
            perm = rng.permutation(N)
        ks, ids = np.nonzero(spk.T)                                        # time-ordered
        st = TF[ks]; sid = ids; n_spk += len(st)
        W['stdp'] = stdp(st, sid, None, W['stdp'])
        for k, i in zip(ks, ids):
            W['one_state'][i] += one[:, k]
            W['two_state_contrast'][i] += -contrast[:, k]
            W['shuffled_contrast'][i] += -contrast[perm, k]
        for key in ['one_state', 'two_state_contrast', 'shuffled_contrast']:
            np.fill_diagonal(W[key], 0)
    R = {k: score(v, n_spk) for k, v in W.items()}
    R['taus_ms'] = dict(fast=tf * 1000, slow=ts * 1000, one_state=tau_one * 1000,
                        band_centre_hz=float(1 / (2 * np.pi * np.sqrt(tf * ts))))
    return R


def seed_check(seeds=(11, 12, 13)):
    """Repeat the deciding conditions on fresh seeds (new rhythms, new spikes)."""
    out = {'E4a': {}, 'E4b': {}}
    for sd in seeds:
        a = e4a(closed_fracs=(0.15, 0.2, 0.3), seed=sd)
        out['E4a'][sd] = {cf: {k: row[k]['replay_outcome_fractions']['forward_only_in_order'] for k in row}
                          for cf, row in a['sweep'].items()}
        out['E4a'][sd]['no_gate'] = a['no_gate_full_cycle']['replay_outcome_fractions']['forward_only_in_order']
        b = e4b(seed=sd)
        out['E4b'][sd] = {k: dict(asym=b[k]['asymmetry_index'], **b[k]['replay_detail']) for k in
                          ['stdp', 'one_state', 'two_state_contrast', 'shuffled_contrast']}
    return out


def main():
    R = {'E4a': e4a(), 'E4b': e4b()}
    a = R['E4a']
    print('E4a I/Q taus ms', [round(x, 2) for x in a['iq_taus_ms']], 'phase error', a['iq_phase_error'])
    fmt = lambda s: dict(fwd=round(s['forward_drive_per_spike'], 3), fwd_only=round(s['replay_outcome_fractions']['forward_only_in_order'], 2),
                         both=round(s['replay_outcome_fractions']['both_directions'], 2), none=round(s['replay_outcome_fractions']['none'], 2),
                         open=round(s.get('fraction_spikes_open', 1.0), 2))
    print('E4a no gate, full cycle     ', fmt(a['no_gate_full_cycle']), a['no_gate_full_cycle']['replay_detail'])
    print('E4a reference windowed spikes', fmt(a['reference_windowed_spikes']), a['reference_windowed_spikes']['replay_detail'])
    for cf, row in a['sweep'].items():
        for k, s in row.items():
            print('E4a closed', cf, k.ljust(24), fmt(s), {kk: s[kk] for kk in ['tau_ms', 'sign'] if kk in s})
    for k, s in R['E4b'].items():
        if k != 'taus_ms':
            print('E4b', k.ljust(20), fmt(s), 'asym', round(s['asymmetry_index'], 3), 'reach', round(s['mean_forward_reach_items'], 2),
                  {kk: round(vv, 2) for kk, vv in s['replay_detail'].items()})
    print('E4b taus', R['E4b']['taus_ms'])
    R['seed_check'] = seed_check()
    for sd, v in R['seed_check']['E4a'].items():
        print('seed', sd, 'E4a forward-only-in-order', {cf: ({k: round(x, 2) for k, x in row.items()} if isinstance(row, dict) else round(row, 2)) for cf, row in v.items()})
    for sd, v in R['seed_check']['E4b'].items():
        print('seed', sd, 'E4b', {k: {kk: round(x, 2) for kk, x in row.items()} for k, row in v.items()})
    json.dump(R, open('results_e4.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
