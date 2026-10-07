"""Measurement-outcome histogram of the example n = 8 hardware job (vector PDF), from the raw counts saved in
n8/job_manifest/job_d8gt5elv8cos73f3mmb0_counts.json (all 256 outcomes). The two optimal bitstrings of I1 are highlighted
(the job is untagged, so its instance cannot be recovered; I2 and I3 have the same optimal partition as I1).
Also prints the statistics quoted in the paper. Run:  python make_histogram_figure.py"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8, 'axes.labelsize': 9})
J = json.load(open(os.path.join(PKG, 'n8', 'job_manifest', 'job_d8gt5elv8cos73f3mmb0_counts.json')))
cnt = J['counts']
n = 8
W = pd.read_csv(os.path.join(PKG, 'n8', 'cost_matrices', 'mcp_agent_data_cost_matrix.csv'), index_col=0).values.astype(float)
W = (W + W.T) / 2
allb = [format(k, '08b') for k in range(256)]


def cut(bs):
    a = [int(bs[n - 1 - i]) for i in range(n)]                     # little-endian: rightmost character = qubit 0
    return sum(W[i][j] for i in range(n) for j in range(i + 1, n) if a[i] != a[j])


cv = {b: cut(b) for b in allb}
opt = max(cv.values())
optb = [b for b in allb if abs(cv[b] - opt) < 1e-6]
y = np.array([cnt[b] for b in allb])
top = allb[int(np.argmax(y))]
print('outcomes %d, counts min %d max %d mean %.1f; in 73..125: %d' % (len(y), y.min(), y.max(), y.mean(), int(((y >= 73) & (y <= 125)).sum())))
print('most frequent %s (%d counts), cut/C* = %.3f under I1' % (top, y.max(), cv[top] / opt))
print('I1 optimal bitstrings %s with counts %s (uniform expectation %.0f)' % (optb, [cnt[b] for b in optb], 8192 / 256))
print('E[C]/C*: hardware %.4f, uniform %.4f (I1)' % (sum(cnt[b] * cv[b] for b in allb) / 8192 / opt, np.mean(list(cv.values())) / opt))

fig, ax = plt.subplots(figsize=(6.9, 2.5))
colors = ['#D55E00' if b in optb else '#56B4E9' for b in allb]   # darker bars for the optimal strings (distinct in greyscale)
ax.bar(range(256), y, width=0.85, color=colors, linewidth=0)
opt_idx = [allb.index(b) for b in optb]
ax.plot(opt_idx, [y[i] + 0.04 * y.max() for i in opt_idx], 'v', color='k', ms=4, zorder=5)   # triangle marker as well as colour
ax.axhline(8192 / 256, color='k', ls='--', lw=0.8)
ax.text(255, 108, 'dashed line: uniform expectation (32)', ha='right', va='bottom', fontsize=7, bbox=dict(fc='w', ec='none', pad=1.5))
ax.set_xlim(-1, 256)
ticks = list(range(0, 256, 32)) + [255]
ax.set_xticks(ticks)
ax.set_xticklabels([allb[t] for t in ticks], rotation=45, ha='right', fontsize=7)
ax.set_xlabel('Measurement outcome (bitstring, qubit 0 rightmost)')
ax.set_ylabel('Counts (8,192 shots)')
ax.grid(True, alpha=0.25, lw=0.5, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_measurement_hist.pdf'))
print('wrote fig_measurement_hist.pdf')
