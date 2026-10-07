"""Schematic of the logical QAOA circuit used at n = 8, p = 1 (vector PDF; matplotlib only).
Same construction as the experiment scripts: Hadamard layer, one RZZ(gamma) per edge of the instance graph (in the
experiment the angle is scaled by the edge weight), RX(beta) mixer layer, measurement. The edges are those of I1 (22 for n = 8),
packed greedily into layers of non-overlapping gates. Run:  python make_circuit_figure.py"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8})
OKABE = {'h': '#56B4E9', 'zz': '#D55E00', 'rx': '#009E73'}

W = pd.read_csv(os.path.join(PKG, 'n8', 'cost_matrices', 'mcp_agent_data_cost_matrix.csv'), index_col=0).values.astype(float)
W = (W + W.T) / 2
n = W.shape[0]
edges = [(i, j) for i in range(n) for j in range(i + 1, n) if W[i][j] > 0]
layers = []                                  # greedy packing into columns whose vertical spans are disjoint with a one-wire gap,
for e in sorted(edges, key=lambda t: -(t[1] - t[0])):   # so that no two gates in a column touch or can be misread as one
    for L in layers:
        if all(e[1] + 1 < x[0] or x[1] + 1 < e[0] for x in L):
            L.append(e)
            break
    else:
        layers.append([e])
nl = len(layers)
print('edges', len(edges), 'cost-layer columns', nl)

col_h, col_zz0 = 0, 1.2
col_rx = col_zz0 + nl + 0.5
col_m = col_rx + 1.2
fig, ax = plt.subplots(figsize=(6.9, 2.9))
for q in range(n):
    y = n - 1 - q
    ax.plot([-0.6, col_m + 0.6], [y, y], color='0.55', lw=0.8, zorder=0)
    ax.text(-0.85, y, '$q_%d$' % q, ha='right', va='center', fontsize=8)
box = 0.40
for q in range(n):
    y = n - 1 - q
    ax.add_patch(Rectangle((col_h - box, y - box), 2 * box, 2 * box, fc=OKABE['h'], ec='k', lw=0.6, zorder=3))
    ax.text(col_h, y, 'H', ha='center', va='center', zorder=4)
    ax.add_patch(Rectangle((col_rx - box, y - box), 2 * box, 2 * box, fc=OKABE['rx'], ec='k', lw=0.6, zorder=3))
    ax.text(col_rx, y, r'$R_X(\beta)$', ha='center', va='center', fontsize=6.5, zorder=4)
    ax.add_patch(Rectangle((col_m - box, y - box), 2 * box, 2 * box, fc='w', ec='k', lw=0.6, zorder=3))
    ax.add_patch(Arc((col_m, y - 0.08), 0.42, 0.34, theta1=0, theta2=180, lw=0.7, zorder=4))
    ax.plot([col_m, col_m + 0.14], [y - 0.08, y + 0.1], color='k', lw=0.7, zorder=4)
for k, L in enumerate(layers):
    x = col_zz0 + k
    for (i, j) in L:
        y1, y2 = n - 1 - i, n - 1 - j
        ax.plot([x, x], [y1, y2], color=OKABE['zz'], lw=1.3, zorder=2)
        for y in (y1, y2):
            ax.add_patch(Circle((x, y), 0.09, fc=OKABE['zz'], ec='k', lw=0.4, zorder=4))
ax.annotate('', xy=(col_zz0 - 0.3, n - 0.35), xytext=(col_zz0 + nl - 0.7, n - 0.35), arrowprops=dict(arrowstyle='<->', lw=0.7))
ax.text(col_zz0 + (nl - 1) / 2 - 0.2, n - 0.15, r'cost layer: $R_{ZZ}(\gamma w_{ij})$ on each of the %d edges' % len(edges), ha='center', va='bottom', fontsize=8)
ax.set_xlim(-1.4, col_m + 0.9)
ax.set_ylim(-0.7, n + 0.5)
ax.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_circuit_schematic.pdf'))
print('wrote fig_circuit_schematic.pdf')
