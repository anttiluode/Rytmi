import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import g0_chaining as G
R = json.load(open('results_g0.json')); A, B = R['A'], R['B']
fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))

a = ax[0]
G.configure(80); W = G.learn_W(); rng = np.random.default_rng(0)
out = G.replay_free(W, 0, 3.0, rng); seq, pk = G.seq_from_activity(out, 0)
a.plot([pk[i] * G.DT * 1000 for i in seq[1:]], seq[1:], 'k.', ms=4, label='one cue, no noise: peak time of each item')
seqt = G.replay_theta(W, 0, np.random.default_rng(0), sd=0.3, gain=2.2, thr=0.3)
a.set_xlabel('replay time (ms)'); a.set_ylabel('item'); a.legend(fontsize=7, loc='lower right')
a.set_title(f"G0-A one cue replays the 80-item lap as a wave\n~{A['80']['ms_per_item_default_setting']:.0f} ms/item vs 200 ms/item when learned "
            f"(x{A['80']['compression']:.0f} compression)", fontsize=9)
a.text(0.02, 0.97, 'full in order, all 165 settings:\n' + '\n'.join(f"n={n}: {A[n]['full_in_order']:.2f}" for n in ['20', '40', '80']) +
       '\ncontrols (n=40): symmetric 0.00,\nno theta 0.00, shuffled 0.00', transform=a.transAxes, va='top', fontsize=7.5)

a = ax[1]; sds = sorted(B['results']['20'], key=float); cols = {'20': '#2a6', '40': '#36c', '80': '#c33'}
for n in ['20', '40', '80']:
    for mode, ls in [('free', '--'), ('theta', '-')]:
        a.plot([float(s) for s in sds], [B['results'][n][s][mode]['full_in_order'] for s in sds], ls, marker='o', ms=4, c=cols[n],
               label=f"n={n} {'one cue (free wave)' if mode == 'free' else 're-cue each theta cycle'}")
a.set_xlabel('noise on every drive, every 5 ms step (sd)'); a.set_ylabel('full chain in order (working settings, 3 noise seeds)')
a.set_ylim(0, 1.02); a.legend(fontsize=6.5); a.set_title('G0-B under noise, re-cueing each cycle holds the chain\n(the dead-time reset clears accumulated noise)', fontsize=9)

a = ax[2]; H = B['hazard_n80']; sds3 = ['0.2', '0.3', '0.4']; x = np.arange(len(sds3)); w = 0.2
for k, (mode, part, col, lab) in enumerate([('free', 'items_10_40', '#ccc', 'one cue, items 10-40'), ('free', 'items_40_79', '#777', 'one cue, items 40-79'),
                                           ('theta', 'items_10_40', '#f3b0a8', 're-cue, items 10-40'), ('theta', 'items_40_79', '#c33', 're-cue, items 40-79')]):
    a.bar(x + (k - 1.5) * w, [H[s_][mode][part] for s_ in sds3], w, color=col, label=lab)
a.set_xticks(x); a.set_xticklabels([f'noise sd {s_}' for s_ in sds3]); a.set_ylabel('per-item error hazard (80-item lap)'); a.legend(fontsize=7, loc='upper left')
Cc = R['C']
a.text(0.02, 0.62, "cue at item 40 (untouched cells behind it), sd 0.3:\n"
       f"  errors that are a wave far behind: {Cc['start40_free']['far_behind']:.0%} (median {Cc['start40_free']['median_offset']:.0f} items)\n"
       f"  hazard, one cue {Cc['start40_free']['hazard_distance_10_39']:.4f} vs re-cue {Cc['start40_theta']['hazard_distance_10_39']:.4f}",
       transform=a.transAxes, fontsize=7, va='top')
a.set_title('the error is a second wave igniting behind the front;\nit grows as the wave leaves recovered cells behind; dead time suppresses it', fontsize=9)
fig.tight_layout(); fig.savefig('g0.png', dpi=100); print('ok')
