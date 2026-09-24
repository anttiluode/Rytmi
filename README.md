# Rytmi

*Rytmi* is Finnish for rhythm. This repo is where the TATWATASW line becomes one machine: a brain-inspired sequence learner that works in time windows and phases instead of positions and indices. It learns order with local rules, in one pass, with a fixed-size state and no backprop.

Every step is measured separately and each has a control that should fail. Run `python r1_emergent_precession.py` (about a minute), then `python make_figure_r1.py`.

## The machine, and what is measured so far

| step | what it does | status |
|---|---|---|
| **Field** | One dendritic plateau per cell writes its input weights (BTSP). The written drive is asymmetric: slow rise, faster fall. | measured, TATWATASW E1 |
| **Rhythm → phase** | Rhythmic inhibition turns drive strength into spike phase. Stronger drive crosses threshold earlier in the cycle. | **measured here (R1): precession emerges, it is not imposed** |
| **Dead time** | The part of the cycle where nothing may be written, so that one compressed sequence doesn't wrap into the next. | measured: imposed on spikes (TATWATASW E3), made by a local two-trace gate (E4a), **made by the inhibition itself (R1)** |
| **Write** | Pairwise STDP writes the within-cycle order into cell-to-cell weights. | measured, TATWATASW E3 |
| **Local phase** | Two leaky traces of the local theta give each synapse its own phase estimate (fast − slow and slow, a fixed 2×2 readout). | measured, TATWATASW E4 and AnotherOddThing v5 |
| **Read** | Cue one cell; the written chain replays forward, in order. | measured (replay grid of 165 settings) |
| **Chain** | Re-cue on replay so a one-to-two-item reach becomes a long sequence. | next (G0) |

## R1 — phase precession emerges

![R1](r1.png)

**Setup.** Nothing in R1 sets a spike phase.

- **Inputs.** 70 upstream place inputs.
- **Fields.** Each of 20 target cells gets one plateau at its own location, and TATWATASW E1's BTSP rule writes its input weights. The resulting drive rises over 1.74 s and falls over 0.83 s (20%-of-peak points).
- **Rhythm.** Inhibition is A(1 + cos Φ)/2. Its frequency wanders around 8 Hz and its amplitude drifts, as in E4.
- **Cells.** Leaky integrate-and-fire, with spike-triggered adaptation and membrane noise.
- **Learning.** The spikes go into E3's plain STDP (±20 ms) with no gate, over 20 laps. Then the middle cell is cued on E3's 165-setting replay grid.

**Results.**

| condition | spikes / cell / lap | asymmetry index | replay forward-only, in order | 3 fresh seeds |
|---|---|---|---|---|
| **asymmetric field (BTSP window reaches back)** | 11.4 | **0.996** | **0.95** | 0.79–0.87 |
| symmetric field | 10.8 | 0.945 | 0.06 (0.78 forward but out of order) | 0.07–0.37 |
| no theta, constant inhibition set to match the spike count | 11.3 | 0.439 | 0.09 (0.75 reach a cell behind the cue) | 0.05–0.37 |
| asymmetric spikes, each moved to a random phase inside its **own** theta cycle | 11.3 | 0.175 | 0.00 | 0.00–0.04 |

- **Precession emerges from drive against inhibition.** The mean phase of the first spike per cycle moves from 0.51 to 0.33 of the cycle as the cell's drive ramps up (circular-linear ρ = 0.76 over the entering half). Then it recedes after the peak. A range of about 0.2 cycle (≈ 70°) is smaller than the up-to-360° seen in rats. It is emergent, but not full-size.
- **That emergent timing is enough for plain STDP to write a replayable sequence.** 0.95 forward-in-order, matching the 0.96 that TATWATASW E3 got with precession *written in by formula*.
- **The timing is what carries it.** The phase-shuffled control keeps every spike, every cell and every theta cycle, and only scrambles order inside each cycle. It drops to 0.00. The rate-matched no-theta control mostly replays backward too.
- **The rhythm's inhibition supplies its own dead time.** In the working regime, 45% of the cycle is effectively empty of spikes. So E4a's local phase gate is not needed here. Adding it (30% closed) changes 0.89 to 0.88.

**Why the plateau's asymmetry matters.** Both field shapes precess while the drive rises (ρ 0.76 and 0.80). The difference is how much firing happens on the way *down*. With the asymmetric field, 36% of spikes come after the drive peak; with the symmetric field it's 47%. Receding spikes run the within-cycle order backward. With a symmetric field there are enough of them to smear the order: the replay stays mostly forward (asymmetry 0.945) but loses sequence.

This corrects TATWATASW E2. There, the plateau's asymmetry could not write a sequence on its own. Here it turns out to be *necessary* for the rhythm to write one. The tuft and the rhythm are not alternatives; the tuft shapes the field so the rhythm's phase code comes out one-directional. Experience-dependent backward skew of place fields is a measured effect in rats (Mehta et al. 2000). R1 shows what it buys in this machine.

**Where it breaks.** The robustness grid covers peak excitation P ∈ {2.2, 2.6, 3.0} (threshold 1) and inhibition swing A ∈ {1.6, 2.4, 3.2}, with 12 laps each.

- **Asymmetric fields pass in 7 of 9 settings,** at 0.81–0.94.
- **Symmetric fields pass in none,** staying at 0.15 or below everywhere.
- **The two failures are where excitation outruns inhibition** (P − 1 ≥ A): 0.24 and 0.01. There, cells fire more often per cycle (13–16 spikes per lap against 9–12), the empty arc shrinks to 0.17 of the cycle, and the written links reach further. The replay stays forward (0% backward at P 2.6, A 1.6) but out of order, the same failure as TATWATASW E4b.
- **The E4a local gate does not rescue these cases** (0.25 and 0.00). The problem is too many spikes per cycle, not wrap-around. So the operating condition is plain: inhibition must be strong enough to hold each cell to about one burst per cycle.

## What R1 means for Rytmi

The chain **plateau → asymmetric field → rhythmic inhibition → emergent precession → STDP → ordered replay** now runs in one machine, with no phase variable anywhere. Each link has a control that breaks it:

- **A symmetric plateau** breaks the order.
- **No rhythm** breaks it.
- **Scrambled within-cycle timing** breaks it.
- **Too little inhibition** breaks it.

## Ledger

- **The mechanism is known.** Precession from a rising excitatory ramp against oscillating inhibition is the Mehta, Lee & Wilson (2002) account; related models include Kamondi et al. (1998), Harris et al. (2002) and Magee (2001). Theta sequences writing asymmetric recurrent weights through STDP is also established. What R1 adds is the *measured chain* in one small model, with matched controls for each link.
- **Still imposed:**
  - where each cell's plateau happens (one per cell, at its place, as if an instructive signal chose it)
  - the upstream place inputs
  - one theta rhythm shared by all cells
  - the cell parameters (the grid shows the working region, not a fit)
- **Precession is partial:** about 0.2 cycle, then recession. Real place cells can precess through a full cycle.
- **The spike count is higher than TATWATASW E3's** (11 against about 5 per cell per lap), so forward drive per spike (1.19 here, 0.61 there) is not a like-for-like efficiency comparison. The replay fractions are comparable.
- **"Empty arc"** is the widest part of the cycle where the spike density falls below 10% of uniform. It is a descriptive measure, not a model parameter.
- **The replay network and its 165-setting grid** are TATWATASW's, unchanged.

## Next gates

- **G0 — chaining.** Re-cue on replay. Does iterated one-step replay hold order over 8–16 items, and where does it compound into errors?
- **G1 — one-shot capacity.** Show K new sequences once each; recall versus K at a fixed state size. Baseline: an asymmetric Hopfield sequence memory (Sompolinsky & Kanter 1986).
- **G2 — plateaus that choose themselves.** Replace "one plateau per cell at its place" with plateaus triggered by local surprise (prediction error). That turns the field step into learning instead of an input.

## Lineage

AnotherOddThing v5 (fast − slow separates entering from leaving at the same level) → TATWATASW E1–E3 (plateau writes predictive fields; rhythm plus dead time writes order) → TATWATASW E4 (two traces as a local phase; dead time made locally; direction without rhythm) → **Rytmi R1** (precession emerges from plateau-written drive against rhythmic inhibition, and plain STDP writes ordered replay from it).

## References

- Mehta, M. R., Quirk, M. C. & Wilson, M. A. (2000). Experience-dependent asymmetric shape of hippocampal receptive fields. *Neuron* 25, 707–715.
- Mehta, M. R., Lee, A. K. & Wilson, M. A. (2002). Role of experience and oscillations in transforming a rate code into a temporal code. *Nature* 417, 741–746.
- Kamondi, A., Acsády, L., Wang, X.-J. & Buzsáki, G. (1998). Theta oscillations in somata and dendrites of hippocampal pyramidal cells in vivo. *Hippocampus* 8, 244–261.
- Harris, K. D. et al. (2002). Spike train dynamics predicts theta-related phase precession in hippocampal pyramidal cells. *Nature* 417, 738–741.
- Magee, J. C. (2001). Dendritic mechanisms of phase precession in hippocampal CA1 pyramidal neurons. *J. Neurophysiol.* 86, 528–532.
- Bittner, K. C. et al. (2017). Behavioral time scale synaptic plasticity underlies CA1 place fields. *Science* 357, 1033–1036.
- Kempter, R., Leibold, C., Buzsáki, G., Diba, K. & Schmidt, R. (2012). Quantifying circular–linear associations. *J. Neurosci. Methods* 207, 113–124.
- Sompolinsky, H. & Kanter, I. (1986). Temporal association in asymmetric neural networks. *Phys. Rev. Lett.* 57, 2861–2864.
