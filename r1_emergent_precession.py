"""
Rytmi R1 — let phase precession EMERGE instead of writing it in by formula.

TATWATASW E3/E4 imposed precession: spike phases were set by (centre - position). Here nothing sets a phase.

  1. FIELD (tuft). 70 upstream place inputs. Each of 20 target cells gets ONE dendritic plateau at its own
     location; BTSP writes its input weights (TATWATASW E1 rule: window reaches back 1.0 s, forward 0.35 s).
     Result: an asymmetric excitatory drive that ramps up slowly and falls quickly.
  2. RHYTHM. Rhythmic inhibition A (1 + cos Phi)/2, Phi = theta with wandering frequency and drifting amplitude.
  3. CELL. Leaky integrate-and-fire with spike-triggered adaptation and membrane noise.
     Nothing tells the cell a phase. It fires when drive beats inhibition; stronger drive crosses earlier in the cycle.
  4. WRITE. Pairwise STDP (+/-20 ms, balanced) on the emergent spikes, no gate.
  5. READ. Cue the middle cell in the TATWATASW replay network; 165 gain/threshold settings.

Conditions (same cells, same laps):
  asymmetric   BTSP window reaches back further (E1)          -> prediction: precession, forward order
  symmetric    BTSP window symmetric                          -> precession then recession
  no_theta     constant inhibition, level set to MATCH the spike count of the asymmetric condition
  shuffled     asymmetric spikes, each moved to a random phase inside its OWN theta cycle
               (same spikes per cell per cycle; within-cycle timing destroyed)

Known mechanism, stated up front: drive-to-phase conversion by a ramp against oscillating inhibition is the
Mehta, Lee & Wilson (2002) account (also Kamondi et al. 1998; Harris et al. 2002; Magee 2001), and
experience-dependent field asymmetry is Mehta et al. (2000). What R1 tests is the chain in one machine:
plateau-written field -> emergent precession -> STDP -> ordered replay, with nothing phase-shaped imposed.
"""
import json, time
import numpy as np
import rytmi_core as C

N = 20
CENTERS = 0.3 + 0.2 * np.arange(N)
T_LAP = CENTERS[-1] + 0.6
TF = np.arange(0, T_LAP, C.FDT)
UP = np.linspace(-0.3, T_LAP + 0.3, 70); SIG_UP = 0.15
X = np.exp(-(TF[None, :] - UP[:, None]) ** 2 / (2 * SIG_UP ** 2))
CELL = dict(P=2.6, A=2.4, tau=0.01, tau_a=0.05, ja=3.0, sd=0.15)
NO_THETA_LEVEL = 0.6                    # constant inhibition giving the asymmetric condition's spike count (checked below)


def kernel(d, kind):
    """d = t_plateau - t_input. Same shapes as TATWATASW E1."""
    if kind == 'asymmetric':
        return np.where(d >= 0, np.exp(-d / 1.0), np.exp(d / 0.35))
    if kind == 'symmetric':
        return np.exp(-np.abs(d) / 0.6)
    raise ValueError(kind)


def drive(kind):
    W = np.array([(X * kernel(tp - TF, kind)[None, :]).sum(1) * C.FDT for tp in CENTERS])   # one plateau per cell
    I = W @ X
    return I / I.max(1, keepdims=True)


def lif(I, phi, amp, rng, P, A, tau, tau_a, ja, sd, const_inh=None):
    inh = A * (1 + np.cos(phi)) / 2 * amp if const_inh is None else np.full(len(phi), const_inh)
    v = np.zeros(N); a = np.zeros(N); st, sid = [], []
    noise = sd * np.sqrt(C.FDT / tau) * rng.standard_normal((len(phi), N))
    for k in range(len(phi)):
        v += C.FDT / tau * (-v + P * I[:, k] - inh[k] - a) + noise[k]
        a -= C.FDT / tau_a * a
        f = np.nonzero(v > 1.0)[0]
        if len(f):
            st += [C.FDT * k] * len(f); sid += list(f)
            v[f] = 0.0; a[f] += ja
    return np.array(st), np.array(sid)


def run(kind='asymmetric', theta=True, shuffle=False, laps=20, seed=0, cell=CELL, keep_spikes=False, gate_closed=None):
    rng = np.random.default_rng(seed); I = drive(kind)
    W = np.zeros((N, N)); n = 0; rec = []
    for _ in range(laps):
        phi, amp, v_mem = C.make_rhythm(rng, len(TF))
        st, sid = lif(I, phi, amp, rng, **cell, const_inh=None if theta else NO_THETA_LEVEL)
        ph = np.interp(st, TF, phi)
        if shuffle:
            cyc = np.floor(ph / (2 * np.pi)); ph = 2 * np.pi * (cyc + rng.random(len(st)))
            st = np.interp(ph, phi, TF)
        o = np.argsort(st); st, sid, ph = st[o], sid[o], ph[o]
        g = None
        if gate_closed is not None:     # TATWATASW E4a: plasticity gated by the phase read from two leaky traces of v_mem
            ph_hat = np.interp(st, TF, np.unwrap(C.iq_phase(v_mem)))
            cp = (ph_hat / (2 * np.pi)) % 1.0          # boundary (phase 0) = peak of inhibition
            g = ((cp > gate_closed / 2) & (cp < 1 - gate_closed / 2)).astype(float)
        W = C.stdp(st, sid, N, g, W); n += len(st)
        if keep_spikes:
            rec.append((st, sid, ph))
    out = C.score(W, n); out['spikes_per_cell_per_lap'] = n / laps / N
    return (out, rec) if keep_spikes else out


def precession_stats(rec):
    """First spike per cell per theta cycle: phase vs position relative to the cell's plateau."""
    x1, p1, allp = [], [], []
    for st, sid, ph in rec:
        seen = set(); allp += list((ph / (2 * np.pi)) % 1)
        for s, i, p in zip(st, sid, ph):
            key = (i, int(np.floor(p / (2 * np.pi))))
            if key in seen:
                continue
            seen.add(key); x1.append(s - CENTERS[i]); p1.append((p / (2 * np.pi)) % 1)
    x1, p1, allp = np.array(x1), np.array(p1), np.array(allp)
    edges = np.arange(-1.0, 0.61, 0.2); prof = []
    for lo in edges[:-1]:
        m = (x1 >= lo) & (x1 < lo + 0.2)
        prof.append(float((np.angle(np.mean(np.exp(2j * np.pi * p1[m]))) / (2 * np.pi)) % 1) if m.sum() > 10 else None)
    # dead time: widest arc of the cycle where the spike density is below 10% of uniform (40 bins, wraps around)
    h, _ = np.histogram(allp, bins=40, range=(0, 1)); low = h < 0.1 * h.mean()
    low = np.concatenate([low, low]); best = cur = 0
    for b in low:
        cur = cur + 1 if b else 0; best = max(best, cur)
    fit = C.circ_lin_fit(x1[(x1 > -0.9) & (x1 < -0.1)], p1[(x1 > -0.9) & (x1 < -0.1)])
    return dict(profile_bins_s=[float(e) for e in edges[:-1]], first_spike_mean_phase=prof,
                entering_fit=fit, empty_arc_fraction=min(best, 40) / 40,
                phase_histogram=(h / h.sum()).tolist())


def main():
    t0 = time.time(); R = {'cell': CELL, 'no_theta_constant_inhibition': NO_THETA_LEVEL}
    conds = {'asymmetric': dict(kind='asymmetric'), 'symmetric': dict(kind='symmetric'),
             'no_theta': dict(kind='asymmetric', theta=False), 'shuffled': dict(kind='asymmetric', shuffle=True)}
    R['main'] = {}
    for name, kw in conds.items():
        s, rec = run(**kw, keep_spikes=True)
        if name in ('asymmetric', 'symmetric'):
            s['precession'] = precession_stats(rec)
        R['main'][name] = s
        p = s.get('precession', {})
        print(f"R1 {name:11s} spk/cell/lap {s['spikes_per_cell_per_lap']:.1f}  asym {s['asymmetry_index']:.3f}  "
              f"fwd/spike {s['forward_drive_per_spike']:.3f}  reach {s['mean_forward_reach_items']:.2f}  "
              f"{ {k: round(v, 2) for k, v in s['replay'].items()} }")
        if p:
            print('    first-spike phase by position', [None if v is None else round(v, 2) for v in p['first_spike_mean_phase']],
                  'entering-half rho %.2f slope %.2f cyc/s' % (p['entering_fit']['rho'], p['entering_fit']['slope_cycles_per_unit']),
                  'empty arc %.2f' % p['empty_arc_fraction'])
    R['seeds'] = {}
    for sd in [1, 2, 3]:
        R['seeds'][sd] = {}
        for name, kw in conds.items():
            s = run(**kw, seed=sd)
            R['seeds'][sd][name] = dict(forward_only_in_order=s['replay']['forward_only_in_order'],
                                        any_backward=s['replay']['any_backward'], asymmetry=s['asymmetry_index'],
                                        spikes_per_cell_per_lap=s['spikes_per_cell_per_lap'])
        print('seed', sd, {k: (round(v['forward_only_in_order'], 2), round(v['spikes_per_cell_per_lap'], 1)) for k, v in R['seeds'][sd].items()})
    R['robustness'] = {}
    for P in [2.2, 2.6, 3.0]:
        for A in [1.6, 2.4, 3.2]:
            cell = dict(CELL, P=P, A=A); row = {}
            for kind in ['asymmetric', 'symmetric']:
                s, rec = run(kind=kind, laps=12, seed=7, cell=cell, keep_spikes=True)
                row[kind] = dict(forward_only_in_order=s['replay']['forward_only_in_order'],
                                 any_backward=s['replay']['any_backward'],
                                 spikes_per_cell_per_lap=s['spikes_per_cell_per_lap'],
                                 empty_arc_fraction=precession_stats(rec)['empty_arc_fraction'])
            s = run(kind='asymmetric', laps=12, seed=7, cell=cell, gate_closed=0.3)
            row['asymmetric_local_gate_0.3'] = dict(forward_only_in_order=s['replay']['forward_only_in_order'],
                                                    any_backward=s['replay']['any_backward'])
            R['robustness'][f'P{P}_A{A}'] = row
            print('robust', f'P {P} A {A}', {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in row.items()})
    json.dump(R, open('results_r1.json', 'w'), indent=1)
    print('seconds', round(time.time() - t0))


if __name__ == '__main__':
    main()
