import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import e4_local_phase as E

R = json.load(open('results_e4.json'))
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))

# 1) what the synapse sees: a noisy, wandering membrane theta, and the phase read from two leaky traces
rng = np.random.default_rng(5); phi, v = E.make_rhythm(rng); ph_hat, _ = E.iq_phase(v)
k0, k1 = int(1.0 / E.FDT), int(1.6 / E.FDT); tt = E.TF[k0:k1]
a = ax[0]
a.plot(tt, v[k0:k1] / 3 + 1.4, c='#999', lw=.6, label='membrane theta seen by the synapse (noisy, drifting)')
a.plot(tt, (phi[k0:k1] % (2 * np.pi)) / (2 * np.pi), c='k', lw=1.4, label='true phase (the global clock E3 imposed)')
a.plot(tt, (ph_hat[k0:k1] % (2 * np.pi)) / (2 * np.pi), c='#c33', lw=1, ls='--', label='phase from 2 leaky traces (local)')
pe = R['E4a']['iq_phase_error']
a.set_xlabel('time (s)'); a.set_ylabel('cycle phase'); a.set_ylim(-0.05, 2.1)
a.set_title(f"E4a local I/Q phase: error sd {pe['circular_sd_cycles']*360:.0f} deg\n"
            f"taus {R['E4a']['iq_taus_ms'][0]:.1f} / {R['E4a']['iq_taus_ms'][1]:.1f} ms, band centre 8 Hz", fontsize=9)
a.legend(fontsize=7, loc='upper right')

# 2) dead time made by a local gate rescues E3's failing full-cycle condition
a = ax[1]; sw = R['E4a']['sweep']; cfs = sorted(sw, key=float); x = [float(c) for c in cfs]
for k, c, lab in [('oracle', 'k', 'oracle gate (true phase)'), ('local_iq', '#c33', 'local I/Q gate (2 traces, window fixed a priori)'),
                  ('one_state_best_of_grid', '#36c', '1 trace + threshold (best of grid, chosen on result)'),
                  ('wrong_rhythm_iq', '#aaa', 'I/Q of an unrelated rhythm (control)')]:
    a.plot(x, [sw[c][k]['replay_outcome_fractions']['forward_only_in_order'] for c in cfs], 'o-', c=c, label=lab)
for sd, v_ in R['seed_check']['E4a'].items():
    for k, c in [('local_iq', '#c33'), ('one_state_best_of_grid', '#36c')]:
        a.plot([float(cf) for cf in ['0.15', '0.2', '0.3']], [v_[cf][k] for cf in ['0.15', '0.2', '0.3']], '.', c=c, alpha=.5)
a.axhline(R['E4a']['no_gate_full_cycle']['replay_outcome_fractions']['forward_only_in_order'], ls=':', c='k', label='no gate (E3 full-cycle failure)')
a.axhline(R['E4a']['reference_windowed_spikes']['replay_outcome_fractions']['forward_only_in_order'], ls='--', c='#2a6', lw=1,
          label='spikes confined to half-cycle (E3 pass)')
a.axvline(0.16, c='#ccc', lw=6, zorder=0)
a.set_xlabel('closed part of the cycle around the boundary (fraction)\n(grey band: 20 ms STDP window at 8 Hz)')
a.set_ylabel('replay forward-only, in order'); a.set_ylim(0, 1.02); a.legend(fontsize=6.5, loc='lower right')
a.set_title('E4a same full-cycle spikes; plasticity gated by phase\n(dots: fresh seeds)', fontsize=9)

# 3) no rhythm at all
a = ax[2]; keys = ['stdp', 'one_state', 'two_state_contrast', 'shuffled_contrast']
cats = [('forward_only_in_order', '#2a6'), ('forward_out_of_order', '#e9b'), ('any_backward', '#c64'), ('none', '#ccc')]
bottom = np.zeros(len(keys))
for c, col in cats:
    h = np.array([R['E4b'][k]['replay_detail'][c] for k in keys]); a.bar(range(len(keys)), h, bottom=bottom, color=col, label=c.replace('_', ' ')); bottom += h
a.set_xticks(range(len(keys))); a.set_xticklabels(['STDP\n+/-20 ms', '1 trace\n(pre, 200 ms)', '2 traces\n-(fast-slow)', 'contrast of\nrandom cell'], fontsize=8)
for i, k in enumerate(keys):
    a.text(i, 1.02, f"asym {R['E4b'][k]['asymmetry_index']:.2f}", ha='center', fontsize=7)
a.set_ylim(0, 1.1); a.set_ylabel('fraction of 165 settings'); a.legend(fontsize=7, loc='lower right')
a.set_title('E4b no rhythm: 2 slow traces write DIRECTION (0 backward)\nbut not sharp ORDER (forward, out of order)', fontsize=9)
fig.tight_layout(); fig.savefig('e4.png', dpi=105); print('ok')
