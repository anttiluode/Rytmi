import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import r1_emergent_precession as R1
R = json.load(open('results_r1.json'))
fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))

# 1) the only inputs: plateau-written drive (no phase anywhere) and rhythmic inhibition
a = ax[0]; i = 10
for kind, c in [('asymmetric', '#c33'), ('symmetric', '#36c')]:
    a.plot(R1.TF - R1.CENTERS[i], R1.drive(kind)[i], c=c, label=f'{kind} BTSP window -> drive')
_, rec = R1.run(kind='asymmetric', laps=20, seed=0, keep_spikes=True)
xs, ps = [], []
for st, sid, ph in rec:
    m = sid == i; xs += list(st[m] - R1.CENTERS[i]); ps += list((ph[m] / (2 * np.pi)) % 1)
a2 = a.twinx(); a2.scatter(xs, ps, s=5, c='k', alpha=.35, label='emergent spike phase (asymmetric)'); a2.set_ylim(0, 1); a2.set_ylabel('theta phase of spike (0 = inhibition peak)')
a.set_xlim(-1.3, 0.7); a.set_xlabel('time relative to the cell\'s plateau (s)'); a.set_ylabel('excitatory drive (norm.)')
a.axvline(0, c='#999', ls='--', lw=1); a.legend(fontsize=7, loc='upper left'); a2.legend(fontsize=7, loc='lower left')
a.set_title('R1 inputs: one plateau per cell + rhythmic inhibition\nno phase is given; spikes precess as drive ramps', fontsize=9)

# 2) first-spike phase profile
a = ax[1]
for kind, c in [('asymmetric', '#c33'), ('symmetric', '#36c')]:
    p = R['main'][kind]['precession']; x = np.array(p['profile_bins_s']) + 0.1
    y = [np.nan if v is None else v for v in p['first_spike_mean_phase']]
    a.plot(x, y, 'o-', c=c, label=f"{kind} (entering half rho {p['entering_fit']['rho']:.2f})")
a.set_xlabel('position relative to plateau (s)'); a.set_ylabel('mean phase of first spike per cycle'); a.legend(fontsize=7)
a.set_title('both field shapes precess while entering;\nthe symmetric one precesses back just as long', fontsize=9)

# 3) replay outcomes
a = ax[2]; keys = ['asymmetric', 'symmetric', 'no_theta', 'shuffled']
cats = [('forward_only_in_order', '#2a6'), ('forward_out_of_order', '#e9b'), ('any_backward', '#c64'), ('none', '#ccc')]
bottom = np.zeros(len(keys))
for c, col in cats:
    h = np.array([R['main'][k]['replay'][c] for k in keys]); a.bar(range(len(keys)), h, bottom=bottom, color=col, label=c.replace('_', ' ')); bottom += h
for sd, v in R['seeds'].items():
    a.plot(range(len(keys)), [v[k]['forward_only_in_order'] for k in keys], 'k.', ms=7)
a.set_xticks(range(len(keys))); a.set_xticklabels(['asymmetric\nfield', 'symmetric\nfield', 'no theta\n(rate matched)', 'phase shuffled\nin own cycle'], fontsize=8)
a.set_ylabel('fraction of 165 replay settings (dots: forward in order, other seeds)'); a.set_ylim(0, 1.02); a.legend(fontsize=7, loc='upper right')
a.set_title('emergent timing + plain STDP -> ordered replay\nonly with plateau asymmetry AND the rhythm', fontsize=9)

# 4) robustness
a = ax[3]; Ps, As = [2.2, 2.6, 3.0], [1.6, 2.4, 3.2]
M = np.array([[R['robustness'][f'P{P}_A{A}']['asymmetric']['forward_only_in_order'] for A in As] for P in Ps])
im = a.imshow(M, vmin=0, vmax=1, cmap='Greens', origin='lower')
for r, P in enumerate(Ps):
    for c, A in enumerate(As):
        rb = R['robustness'][f'P{P}_A{A}']
        a.text(c, r, f"{M[r, c]:.2f}\nsym {rb['symmetric']['forward_only_in_order']:.2f}\nempty arc {rb['asymmetric']['empty_arc_fraction']:.2f}", ha='center', va='center', fontsize=7)
a.set_xticks(range(3)); a.set_xticklabels(As); a.set_yticks(range(3)); a.set_yticklabels(Ps)
a.set_xlabel('inhibition swing A'); a.set_ylabel('peak excitation P (threshold 1)')
a.set_title('asymmetric: forward-only in order (12 laps)\nfails when excitation outruns inhibition (P - 1 >= A)', fontsize=9)
fig.tight_layout(); fig.savefig('r1.png', dpi=100); print('ok')
