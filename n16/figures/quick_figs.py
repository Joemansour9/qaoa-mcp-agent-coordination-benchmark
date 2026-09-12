"""
Quick Figure Generator — n=16
===============================
Regenerates fig_n16_approx_ratio.png and fig_n16_sim_vs_hw.png directly
from the verified n=16 SINGLE-RUN results (Table tab:results16). No live
simulation or hardware run needed.

Both figures are explicitly captioned in the paper as "single-run data"
(see Figure fig:ratio_n16's caption: Sim r=0.967-1.000, HW r=0.974-1.000),
which is Table tab:results16, NOT the repeated/verified dataset behind
Table tab:utility. Values below are taken directly from
supplementary/audit_trail/results_n16_singlerun_depthprofile_PREAUDIT.json
and independently cross-checked against every cell of the printed
tab:results16 table in paper_draft.tex (all 40 sim/hw values, plus
greedy and optimal_cut, match exactly).

Usage:
  python quick_figs.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Known results from the verified n=16 single-run dataset (Table tab:results16) ──
RESULTS = [
    {'instance': 'I1_full_graph',
     'optimal_cut': 34158.70542921973, 'greedy_cut': 31617.55928042445, 'greedy_ratio': 0.9256076564710338,
     'qaoa_sim': {1: {'approximation_ratio': 0.9723099049907167}, 2: {'approximation_ratio': 1.0},
                  3: {'approximation_ratio': 0.9904665485119499}, 4: {'approximation_ratio': 0.9853288767949449}},
     'qaoa_hw':  {1: {'approximation_ratio': 0.9857483301382749}, 2: {'approximation_ratio': 0.9904665485119499},
                  3: {'approximation_ratio': 0.9904665485119499}, 4: {'approximation_ratio': 0.9853288767949449}}},
    {'instance': 'I2_simple_tasks',
     'optimal_cut': 20042.494112764583, 'greedy_cut': 18551.486005550523, 'greedy_ratio': 0.925607656471034,
     'qaoa_sim': {1: {'approximation_ratio': 0.9904665485119498}, 2: {'approximation_ratio': 0.9728884479220352},
                  3: {'approximation_ratio': 0.9851087060780819}, 4: {'approximation_ratio': 0.9851087060780819}},
     'qaoa_hw':  {1: {'approximation_ratio': 0.9929034586431367}, 2: {'approximation_ratio': 0.9929034586431367},
                  3: {'approximation_ratio': 0.985328876794945}, 4: {'approximation_ratio': 1.0}}},
    {'instance': 'I3_complex_tasks',
     'optimal_cut': 57514.08877729828, 'greedy_cut': 53235.48092722206, 'greedy_ratio': 0.9256076564710339,
     'qaoa_sim': {1: {'approximation_ratio': 1.0}, 2: {'approximation_ratio': 0.9929034586431367},
                  3: {'approximation_ratio': 0.9745166633903845}, 4: {'approximation_ratio': 0.966674886477515}},
     'qaoa_hw':  {1: {'approximation_ratio': 0.9929034586431367}, 2: {'approximation_ratio': 0.9740724565026777},
                  3: {'approximation_ratio': 0.978922450254433}, 4: {'approximation_ratio': 1.0}}},
    {'instance': 'I4_latency_reweighted',
     'optimal_cut': 34207.65817970167, 'greedy_cut': 32515.712679713295, 'greedy_ratio': 0.9505389848349118,
     'qaoa_sim': {1: {'approximation_ratio': 0.9886032890146994}, 2: {'approximation_ratio': 0.9904004111932898},
                  3: {'approximation_ratio': 1.0}, 4: {'approximation_ratio': 0.9904004111932898}},
     'qaoa_hw':  {1: {'approximation_ratio': 0.9912730090189855}, 2: {'approximation_ratio': 1.0},
                  3: {'approximation_ratio': 1.0}, 4: {'approximation_ratio': 0.9973302799957142}}},
    {'instance': 'I5_random_baseline',
     'optimal_cut': 35788.53551968903, 'greedy_cut': 34040.92885433187, 'greedy_ratio': 0.9511685337223225,
     'qaoa_sim': {1: {'approximation_ratio': 0.9825060995131254}, 2: {'approximation_ratio': 0.9883478374074394},
                  3: {'approximation_ratio': 0.9883478374074394}, 4: {'approximation_ratio': 0.9883478374074394}},
     'qaoa_hw':  {1: {'approximation_ratio': 0.9825060995131254}, 2: {'approximation_ratio': 0.9883478374074394},
                  3: {'approximation_ratio': 1.0}, 4: {'approximation_ratio': 1.0}}},
]

P_VALUES    = [1, 2, 3, 4]
SHORT_NAMES = ['I1: Full', 'I2: Simple', 'I3: Complex',
               'I4: Reweighted', 'I5: Random']
OUTPUT_DIR  = '.'

colors = plt.cm.tab10(np.linspace(0, 0.6, 5))

# ── Figure 1: Approximation ratio vs depth ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, mode_key, label in zip(
        axes,
        ['qaoa_sim', 'qaoa_hw'],
        ['Simulation (Ideal — Qiskit Statevector)',
         'Hardware (IBM Marrakesh)']):
    for i, r in enumerate(RESULTS):
        ratios = [r[mode_key][p]['approximation_ratio'] for p in P_VALUES]
        ax.plot(P_VALUES, ratios, 'o-', color=colors[i],
                label=SHORT_NAMES[i], linewidth=2, markersize=7)
    gm = np.mean([r['greedy_ratio'] for r in RESULTS])
    ax.axhline(gm, color='red', linestyle='--', linewidth=1.5,
               label=f'Greedy (mean={gm:.3f})')
    ax.axhline(1.0, color='green', linestyle=':', linewidth=1,
               label='Optimal (r=1.0)')
    ax.set_xlabel('Circuit Depth $p$', fontsize=12)
    ax.set_ylabel('Approximation Ratio $r$', fontsize=12)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_ylim(0.5, 1.05)
    ax.set_xticks(P_VALUES)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
plt.suptitle('QAOA Approximation Ratio vs Circuit Depth — n=16\n'
             'MCP-Derived Agent Coordination Instances (16 qubits)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig_n16_approx_ratio.png',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved: fig_n16_approx_ratio.png")

# ── Figure 2: Sim vs Hardware gap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(P_VALUES)); w = 0.35
sm = [np.mean([r['qaoa_sim'][p]['approximation_ratio'] for r in RESULTS]) for p in P_VALUES]
hm = [np.mean([r['qaoa_hw'][p]['approximation_ratio']  for r in RESULTS]) for p in P_VALUES]
ss = [np.std([r['qaoa_sim'][p]['approximation_ratio']   for r in RESULTS]) for p in P_VALUES]
hs = [np.std([r['qaoa_hw'][p]['approximation_ratio']    for r in RESULTS]) for p in P_VALUES]

ax.bar(x - w/2, sm, w, yerr=ss, label='Simulation',
       color='steelblue', alpha=0.85, capsize=5)
ax.bar(x + w/2, hm, w, yerr=hs, label='IBM Marrakesh Hardware',
       color='coral', alpha=0.85, capsize=5)
ax.set_xlabel('Circuit Depth $p$', fontsize=12)
ax.set_ylabel('Mean Approximation Ratio', fontsize=12)
ax.set_title('Simulation vs Hardware — n=16\n'
             '(Mean $\\pm$ Std across 5 instances)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'$p={p}$' for p in P_VALUES])
ax.set_ylim(0.5, 1.10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig_n16_sim_vs_hw.png',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved: fig_n16_sim_vs_hw.png")

print("\nDone.")
