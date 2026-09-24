"""
Rytmi G0 — chaining. How long a sequence does one cue replay, and how do errors compound?

The plan was to re-cue on replay, because TATWATASW E3's weights reached only 1.2-1.4 items forward.
R1's emergent weights reach ~3 items, so first measure whether re-cueing is needed at all.

Part A  LENGTH. Learn R1's machine (asymmetric plateau fields, rhythmic inhibition, emergent precession,
        plain STDP) on laps of n = 20, 40 and 80 items. Cue item 0 once, no noise, and let the replay run.
        Score over the full 165-setting replay grid (same grid as TATWATASW / R1):
          full_in_order  every item 1..n-1 reached, with peaks strictly in order, nothing behind the cue
          reach          the last item of the longest in-order run starting at the cue
        Controls at n = 40: the three R1 controls (symmetric field, rate-matched no theta, phase-shuffled).
        Also: replay speed (ms per item) against experience (200 ms per item).

Part B  NOISE AND COMPOUNDING. On the settings where noiseless replay of the n = 20 lap is fully in order
        (the working regime), add white noise to every cell's drive at every step (sd = 0.1 ... 0.4) and
        compare two ways of reading the chain:
          free       one cue, the replay runs on by itself (a travelling wave)
          theta      re-cue every 125 ms: 75 ms open, 50 ms dead time with all activity silenced;
                     the next cue is the last item that peaked in the previous cycle
        For each, the survival curve P(reach >= L) and the per-item hazard early vs late in the 80-item chain.

Part C  WHAT THE ERRORS ARE. At the first out-of-order item: is it a local swap at the front of the replay,
        or a second wave ignited somewhere else? Start at item 0 and at item 40 (where everything behind the
        cue has never been adapted).
"""
import json, time
import numpy as np
import rytmi_core as C
import r1_emergent_precession as R1

GRID = [(g, t) for g in np.linspace(1.5, 4.0, 11) for t in np.linspace(0.1, 0.8, 15)]
DT = 0.005
MS_PER_ITEM_EXPERIENCE = 200.0


def configure(n):
    """Re-point R1's module-level lap geometry to n items (R1 itself is unchanged; n=20 reproduces it)."""
    R1.N = n; R1.CENTERS = 0.3 + 0.2 * np.arange(n); R1.T_LAP = R1.CENTERS[-1] + 0.6
    R1.TF = np.arange(0, R1.T_LAP, C.FDT)
    R1.UP = np.linspace(-0.3, R1.T_LAP + 0.3, int(round(70 * (R1.T_LAP + 0.6) / 5.3)))
    R1.X = np.exp(-(R1.TF[None, :] - R1.UP[:, None]) ** 2 / (2 * R1.SIG_UP ** 2))


def learn_W(kind='asymmetric', theta=True, shuffle=False, laps=20, seed=0):
    rng = np.random.default_rng(seed); I = R1.drive(kind); n = R1.N; W = np.zeros((n, n))
    for _ in range(laps):
        phi, amp, _ = C.make_rhythm(rng, len(R1.TF))
        st, sid = R1.lif(I, phi, amp, rng, **R1.CELL, const_inh=None if theta else R1.NO_THETA_LEVEL)
        if shuffle:
            ph = np.interp(st, R1.TF, phi); cyc = np.floor(ph / (2 * np.pi))
            st = np.interp(2 * np.pi * (cyc + rng.random(len(st))), phi, R1.TF)
        o = np.argsort(st); W = C.stdp(st[o], sid[o], n, None, W)
    Wp = np.maximum(W, 0)
    return Wp / Wp.max()


def replay_free(W, cue, T, rng, sd=0.0, gain=2.2, thr=0.3, tau=0.02, tau_a=0.25, g_a=1.5):
    """TATWATASW's replay network, plus optional drive noise. Returns activity (n x steps)."""
    n = W.shape[0]; steps = int(T / DT); r = np.zeros(n); a = np.zeros(n); out = np.zeros((n, steps))
    for k in range(steps):
        I = np.zeros(n)
        if k * DT < 0.05:
            I[cue] = 1.5
        drive = gain * W @ r + I - a - thr
        if sd:
            drive = drive + sd * rng.standard_normal(n)
        r += DT / tau * (-r + np.clip(drive, 0, 1)); a += DT / tau_a * (-a + g_a * r); out[:, k] = r
    return out


def seq_from_activity(out, cue):
    n = out.shape[0]
    act = [i for i in range(n) if i != cue and out[i].max() > 0.2]
    pk = {i: np.argmax(out[i]) for i in act}
    return [cue] + sorted(act, key=lambda i: pk[i]), pk


def replay_theta(W, start, rng, sd=0.0, gain=2.2, thr=0.3, tau=0.02, tau_a=0.25, g_a=1.5,
                 per=0.125, open_frac=0.6, max_cycles=None):
    n = W.shape[0]; r = np.zeros(n); a = np.zeros(n); seq = [start]; cue = start
    spc = int(round(per / DT)); nopen = int(round(open_frac * spc))
    for _ in range(max_cycles or 2 * n):
        out = np.zeros((n, nopen))
        for k in range(spc):
            I = np.zeros(n)
            if k * DT < 0.05:
                I[cue] = 1.5
            drive = gain * W @ r + I - a - thr
            if sd:
                drive = drive + sd * rng.standard_normal(n)
            if k >= nopen:
                drive = np.full(n, -1.0)                    # dead time: everything silenced
            r += DT / tau * (-r + np.clip(drive, 0, 1)); a += DT / tau_a * (-a + g_a * r)
            if k < nopen:
                out[:, k] = r
        order, _ = seq_from_activity(out, cue); order = order[1:]
        if not order:
            break
        seq += order; cue = order[-1]
        if cue == n - 1 or cue < seq[0]:
            break
    return seq


def reach_in_order(seq):
    """Last item of the longest run that keeps increasing from the cue (a skip still counts as forward)."""
    last = seq[0]
    for i in seq[1:]:
        if i > last:
            last = i
        else:
            break
    return last - seq[0]


def classify(seq, n):
    return dict(full_in_order=(reach_in_order(seq) == n - 1 - seq[0] and all(i >= seq[0] for i in seq)
                               and len(set(seq)) == len(seq)),
                reach=reach_in_order(seq), backward=any(i < seq[0] for i in seq))


def part_a(ns=(20, 40, 80)):
    R = {}; Ws = {}
    for n in ns:
        configure(n); t0 = time.time(); W = learn_W(); Ws[n] = W
        rng = np.random.default_rng(0); full, reach = [], []
        for g, t in GRID:
            out = replay_free(W, 0, 0.03 * n + 0.6, rng, gain=g, thr=t)
            seq, pk = seq_from_activity(out, 0); c = classify(seq, n)
            full.append(c['full_in_order']); reach.append(c['reach'])
        out = replay_free(W, 0, 0.03 * n + 0.6, rng); seq, pk = seq_from_activity(out, 0)
        times = np.array([pk[i] for i in seq[1:]]) * DT * 1000
        ms_item = float(np.median(np.diff(times))) if len(times) > 2 else None
        R[n] = dict(full_in_order=float(np.mean(full)), median_reach=float(np.median(reach)),
                    ms_per_item_default_setting=ms_item,
                    compression=MS_PER_ITEM_EXPERIENCE / ms_item if ms_item else None,
                    forward_reach_of_weights=float(C.score(W, 1)['mean_forward_reach_items']),
                    learn_seconds=round(time.time() - t0, 1))
        print(f"G0-A n={n:3d} full in order {R[n]['full_in_order']:.2f}  median reach {R[n]['median_reach']:.0f}  "
              f"{ms_item} ms/item (x{R[n]['compression']:.1f} faster than experience)  weight reach {R[n]['forward_reach_of_weights']:.2f}")
    configure(40); ctrl = {}
    for name, kw in [('symmetric', dict(kind='symmetric')), ('no_theta', dict(theta=False)), ('shuffled', dict(shuffle=True))]:
        W = learn_W(**kw); rng = np.random.default_rng(0); full, reach, bwd = [], [], []
        for g, t in GRID:
            seq, _ = seq_from_activity(replay_free(W, 0, 0.03 * 40 + 0.6, rng, gain=g, thr=t), 0); c = classify(seq, 40)
            full.append(c['full_in_order']); reach.append(c['reach'])
        ctrl[name] = dict(full_in_order=float(np.mean(full)), median_reach=float(np.median(reach)))
        print(f"G0-A n= 40 control {name:10s} full in order {ctrl[name]['full_in_order']:.2f}  median reach {ctrl[name]['median_reach']:.0f}")
    R['controls_n40'] = ctrl
    return R, Ws


def part_b(Ws, sds=(0.0, 0.1, 0.2, 0.3, 0.4), noise_seeds=(1, 2, 3)):
    configure(20); W20 = Ws[20]; rng = np.random.default_rng(0)
    working = {  # each read-out mode gets its OWN working regime: settings where its noiseless n=20 replay is fully in order
        'free': [(g, t) for g, t in GRID
                 if classify(seq_from_activity(replay_free(W20, 0, 1.2, rng, gain=g, thr=t), 0)[0], 20)['full_in_order']],
        'theta': [(g, t) for g, t in GRID if classify(replay_theta(W20, 0, rng, gain=g, thr=t), 20)['full_in_order']]}
    R = dict(working_settings={k: len(v) for k, v in working.items()}, results={})
    print('G0-B working settings (of 165):', R['working_settings'])
    for n, W in Ws.items():
        R['results'][n] = {}
        for sd in sds:
            row = {}
            for mode in ['free', 'theta']:
                reach, full = [], []
                for ns in noise_seeds:
                    rng = np.random.default_rng(1000 * ns + n)
                    for g, t in working[mode]:
                        if mode == 'free':
                            seq, _ = seq_from_activity(replay_free(W, 0, 0.03 * n + 0.6, rng, sd=sd, gain=g, thr=t), 0)
                        else:
                            seq = replay_theta(W, 0, rng, sd=sd, gain=g, thr=t)
                        c = classify(seq, n); reach.append(c['reach']); full.append(c['full_in_order'])
                reach = np.array(reach)
                surv = [float(np.mean(reach >= L)) for L in range(n)]
                row[mode] = dict(full_in_order=float(np.mean(full)), median_reach=float(np.median(reach)), survival=surv)
            R['results'][n][str(sd)] = row
            print(f"G0-B n={n:3d} sd {sd:.1f}  free: full {row['free']['full_in_order']:.2f} median reach {row['free']['median_reach']:.0f}"
                  f"  |  theta re-cue: full {row['theta']['full_in_order']:.2f} median reach {row['theta']['median_reach']:.0f}")
    # per-item hazard in two stretches of the 80-item chain (launch failures excluded by starting at item 10)
    haz = {}
    for sd in sds:
        haz[str(sd)] = {}
        for mode in ['free', 'theta']:
            sv = np.array(R['results'][80][str(sd)][mode]['survival'])
            h = lambda a, b: (1 - (sv[b] / sv[a]) ** (1 / (b - a))) if sv[b] > 0 and sv[a] > 0 else None
            haz[str(sd)][mode] = dict(items_10_40=h(10, 40), items_40_79=h(40, 79), launched=float(sv[5]))
        print(f"G0-B n=80 sd {sd}: per-item hazard items 10-40 / 40-79  free {haz[str(sd)]['free']['items_10_40']:.4f} / {haz[str(sd)]['free']['items_40_79']:.4f}"
              f"   theta {haz[str(sd)]['theta']['items_10_40']:.4f} / {haz[str(sd)]['theta']['items_40_79']:.4f}")
    R['hazard_n80'] = haz
    R['_working'] = working
    return R


def part_c(Ws, working, sd=0.3, noise_seeds=(1, 2, 3)):
    """What the errors are. Start at item 0 or item 40 of the 80-item lap; at the first out-of-order item, where is it
    relative to the front of the replay?  adjacent = 1-5 items behind the front (a local swap);
    far_behind = more than 5 behind (a second wave ignited elsewhere); the rest are repeats of the front."""
    configure(80); W = Ws[80]; R = {}
    for start in [0, 40]:
        for mode in ['free', 'theta']:
            offs, reach = [], []
            for ns in noise_seeds:
                rng = np.random.default_rng(5000 + 100 * ns + start)
                for g, t in working[mode]:
                    if mode == 'free':
                        seq, _ = seq_from_activity(replay_free(W, start, 0.03 * (80 - start) + 0.6, rng, sd=sd, gain=g, thr=t), start)
                    else:
                        seq = replay_theta(W, start, rng, sd=sd, gain=g, thr=t)
                    reach.append(reach_in_order(seq)); last = start
                    for i in seq[1:]:
                        if i > last:
                            last = i
                        else:
                            offs.append(i - last); break
            offs, reach = np.array(offs), np.array(reach)
            sv = np.array([np.mean(reach >= L) for L in range(80 - start)])
            R[f'start{start}_{mode}'] = dict(
                errors=int(len(offs)), trials=int(len(reach)),
                adjacent_1_5=float(np.mean((offs < 0) & (offs >= -5))) if len(offs) else None,
                far_behind=float(np.mean(offs < -5)) if len(offs) else None,
                median_offset=float(np.median(offs)) if len(offs) else None,
                hazard_distance_10_39=(1 - (sv[39] / sv[10]) ** (1 / 29)) if sv[39] > 0 else None,
                reach_39_in_order=float(sv[39]))
            r = R[f'start{start}_{mode}']
            print(f"G0-C start {start:2d} {mode:5s} sd {sd}: errors {r['errors']}/{r['trials']}  adjacent {r['adjacent_1_5']:.2f}  far behind {r['far_behind']:.2f}"
                  f"  (median {r['median_offset']:.0f})  hazard over items 10-39 from the cue {r['hazard_distance_10_39']:.4f}")
    return R


def main():
    t0 = time.time()
    A, Ws = part_a()
    B = part_b(Ws)
    Cres = part_c(Ws, B.pop('_working'))
    json.dump(dict(A=A, B=B, C=Cres), open('results_g0.json', 'w'), indent=1)
    print('seconds', round(time.time() - t0))


if __name__ == '__main__':
    main()
