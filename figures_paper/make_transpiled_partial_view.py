"""Partial view of the transpiled circuit of the depth-4 n = 8 job d8gt5elv8cos73f3mmb0 on ibm_marrakesh (vector PDF).
Reads the saved native-gate circuit (n8/job_manifest/job_d8gt5elv8cos73f3mmb0_transpiled_circuit.json) and draws the first
N_LAYERS layers (as-soon-as-possible scheduling per physical qubit) on the eight physical qubits the job used.
RZ: blue boxes with the angle in units of pi; sqrt(X): vermilion boxes; CZ: vertical lines with dots.
Run:  python make_transpiled_partial_view.py"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
N_LAYERS = 16
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8})
C = json.load(open(os.path.join(PKG, 'n8', 'job_manifest', 'job_d8gt5elv8cos73f3mmb0_transpiled_circuit.json')))
ops = [o for o in C['ops'] if o['name'] != 'measure']
active = sorted({q for o in C['ops'] for q in o['qubits']})
row = {q: len(active) - 1 - k for k, q in enumerate(active)}           # lowest physical qubit at the top

last, placed = {}, []
for o in ops:
    L = max(last.get(q, -1) for q in o['qubits']) + 1
    for q in o['qubits']:
        last[q] = L
    placed.append((L, o))
shown = [(L, o) for L, o in placed if L < N_LAYERS]
n_hidden = len(ops) - len(shown)
print('active qubits', active, '| layers shown', N_LAYERS, '| gates shown', len(shown), 'of', len(ops))

fig, ax = plt.subplots(figsize=(6.9, 3.0))
for q in active:
    ax.plot([-0.7, N_LAYERS - 0.3], [row[q], row[q]], color='0.6', lw=0.8, zorder=0)
    ax.text(-0.85, row[q], 'q[%d]' % q, ha='right', va='center', fontsize=8)
b = 0.36
for L, o in shown:
    if o['name'] == 'cz':
        y1, y2 = row[o['qubits'][0]], row[o['qubits'][1]]
        ax.plot([L, L], [y1, y2], color='k', lw=1.2, zorder=2)
        for y in (y1, y2):
            ax.add_patch(Circle((L, y), 0.1, fc='k', zorder=4))
    else:
        y = row[o['qubits'][0]]
        if o['name'] == 'rz':
            ax.add_patch(Rectangle((L - b, y - b), 2 * b, 2 * b, fc='#56B4E9', ec='k', lw=0.5, zorder=3))
            ax.text(L, y + 0.12, 'RZ', ha='center', va='center', fontsize=6, zorder=4)
            ax.text(L, y - 0.14, '%+.2fπ' % (o['params'][0] / 3.141592653589793), ha='center', va='center', fontsize=5.2, zorder=4)
        elif o['name'] == 'sx':
            ax.add_patch(Rectangle((L - b, y - b), 2 * b, 2 * b, fc='#D55E00', ec='k', lw=0.5, zorder=3))
            ax.text(L, y, '$\\sqrt{X}$', ha='center', va='center', fontsize=7, color='w', zorder=4)
ax.text(N_LAYERS - 0.05, (len(active) - 1) / 2, '…', ha='left', va='center', fontsize=14)
ax.set_xlim(-1.9, N_LAYERS + 0.6)
ax.set_ylim(-0.7, len(active) - 0.3)
ax.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_circuit_transpiled_partial.pdf'))
print('wrote fig_circuit_transpiled_partial.pdf')
