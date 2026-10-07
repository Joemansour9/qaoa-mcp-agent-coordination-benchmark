"""Paper figures as vector PDF (fonts embedded), sized to the journal's text width, no in-figure titles.

Data are read from the package, not retyped:
  - n=8 and n=16 single-run values: the RESULTS table inside n8/figures/quick_figs.py and n16/figures/quick_figs.py
    (n=16 values are the single-run data of Table tab:results16; see n16/results/results_n16_singlerun_depthprofile.json).
  - n=8 cost matrix: n8/cost_matrices/mcp_agent_data_cost_matrix.csv.
  - retention figure: normalised_retention/{pilot,expansion}/ (frozen analysis outputs).
Run:  python make_paper_figures.py      (writes the PDFs next to this script)
"""
import ast, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8,
                     'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 7.5, 'axes.linewidth': 0.8, 'lines.linewidth': 1.4, 'lines.markersize': 5})
W_FULL, W_80 = 6.9, 5.5          # inches (174 mm and about 139 mm)
OKABE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']      # colour-blind-safe palette
MARK = ['o', 's', '^', 'D', 'v']
LSTY = ['-', '--', '-.', ':', (0, (5, 1))]
NAMES = ['I1: Full', 'I2: Simple', 'I3: Complex', 'I4: Reweighted', 'I5: Random']
P = [1, 2, 3, 4]


def load_results(path):
    for node in ast.parse(open(path, encoding='utf-8').read()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'RESULTS':
            return ast.literal_eval(node.value)
    raise ValueError(path)


def panel_letter(ax, s):
    ax.text(0.0, 1.02, s, transform=ax.transAxes, va='bottom', ha='left', fontsize=10, fontweight='bold')


def depth_figure(results, ylim, out):
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.9), sharey=True)
    gm = np.mean([r['greedy_ratio'] for r in results])
    off = np.linspace(-0.12, 0.12, 5)
    for ax, key, letter in zip(axes, ['qaoa_sim', 'qaoa_hw'], ['(a)', '(b)']):
        for i, r in enumerate(results):
            y = [r[key][p]['approximation_ratio'] for p in P]
            ax.plot(np.array(P) + off[i], y, marker=MARK[i], ls=LSTY[i], color=OKABE[i], label=NAMES[i], mfc='none' if i % 2 else OKABE[i])
        ax.axhline(gm, color='0.35', ls=(0, (6, 2)), lw=1.0, label='Greedy (mean %.3f)' % gm)
        ax.axhline(1.0, color='k', ls=':', lw=0.8, label='Optimum ($r=1$)')
        ax.set_xlabel('Circuit depth $p$')
        ax.set_xticks(P)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25, lw=0.5)
        panel_letter(ax, letter)
    axes[0].set_ylabel('Best-of-shots approximation ratio $r$')
    axes[1].legend(loc='lower right', frameon=True, framealpha=0.9, ncol=1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def gap_figure(results, ylim, out):
    fig, ax = plt.subplots(figsize=(W_80, 3.0))
    x = np.arange(len(P))
    for key, lab, c, m, d in (('qaoa_sim', 'Simulation', OKABE[0], 'o', -0.08), ('qaoa_hw', 'IBM Marrakesh hardware', OKABE[4], 's', 0.08)):
        mean = [np.mean([r[key][p]['approximation_ratio'] for r in results]) for p in P]
        sd = [np.std([r[key][p]['approximation_ratio'] for r in results]) for p in P]
        ax.errorbar(x + d, mean, yerr=sd, fmt=m, color=c, capsize=3, label=lab)
    ax.axhline(1.0, color='k', ls=':', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['$p=%d$' % p for p in P])
    ax.set_xlabel('Circuit depth $p$')
    ax.set_ylabel('Mean best-of-shots ratio $r$ (error bars: s.d. over 5 instances)')
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25, lw=0.5, axis='y')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def cost_matrix_figure(out):
    W = pd.read_csv(os.path.join(PKG, 'n8', 'cost_matrices', 'mcp_agent_data_cost_matrix.csv'), index_col=0).values.astype(float)
    W = (W + W.T) / 2
    mats = [W, W * (2063 / 3516), W * (5920 / 3516)]
    tools = ['fs_read', 'fs_write', 'ws_search', 'ws_fetch', 'db_query', 'db_insert', 'an_analyse', 'an_report']
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 3.3))
    for ax, M, letter in zip(axes, mats, ['(a)', '(b)', '(c)']):
        im = ax.imshow(M, cmap='viridis', aspect='auto')
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels(tools, rotation=60, ha='right', fontsize=7)
        ax.set_yticklabels(tools, fontsize=7)
        for pos in (1.5, 3.5, 5.5):
            ax.axhline(pos, color='w', lw=0.6, ls='--', alpha=0.8)
            ax.axvline(pos, color='w', lw=0.6, ls='--', alpha=0.8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label('Token cost', fontsize=7)
        cb.ax.tick_params(labelsize=6)
        panel_letter(ax, letter)
    axes[0].set_ylabel('Tool node')
    for ax in axes:
        ax.set_xlabel('Tool node')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def retention_figure(out):
    rows = []                                           # (label, p, U, ideal, hw, rho, lo, hi, pilot)
    P1 = json.load(open(os.path.join(PKG, 'normalised_retention', 'pilot', 'pilot_results.json')))['results']
    for p in (1, 2):
        r = P1[str(p)]
        rows.append(('I1', p, r['U'], r['E_ideal'], r['E_hw_mean'], r['retention'], *r['retention_CI95'], True))
    E = json.load(open(os.path.join(PKG, 'normalised_retention', 'expansion', 'expansion_results.json')))['cells']
    for inst in ('I7', 'I8', 'I10', 'I11'):
        for p in (1, 2):
            r = E[inst][str(p)]
            rows.append((inst, p, r['U'], r['E_ideal'], r['E_hw_mean'], r['retention'], *r['retention_CI95'], False))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 3.4), gridspec_kw={'width_ratios': [1, 1]})
    pos = []
    for k, (lab, p, U, ide, hw, rho, lo, hi, pil) in enumerate(rows):
        x = k + 0.5 * (k // 2)
        pos.append(x)
        a.plot([x, x], [U, ide], color='0.6', lw=1.0, zorder=1)
        a.scatter([x], [U], marker='_', s=90, color='0.25', zorder=3, label='Uniform' if k == 0 else None)
        a.scatter([x], [ide], marker='D', s=26, facecolor='none', edgecolor=OKABE[0], zorder=3, label='Noiseless' if k == 0 else None)
        a.scatter([x], [hw], marker='o', s=26, color=OKABE[4], zorder=4, label='Hardware' if k == 0 else None)
        c = OKABE[1] if pil else OKABE[2]
        b.errorbar([x], [rho], yerr=[[rho - lo], [hi - rho]], fmt='o' if not pil else 's', color=c, capsize=2.5, mfc='none' if pil else c,
                   label=('I1 pilot' if pil else 'Second study') if k in (0, 2) else None)
    labels = ['%s\n$p$=%d' % (r[0], r[1]) for r in rows]
    for ax in (a, b):
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(True, alpha=0.25, lw=0.5, axis='y')
    a.set_ylabel('Expected cut ratio $E[C]/C^*$')
    a.set_ylim(0.63, 0.96)
    a.legend(loc='upper center', fontsize=7, ncol=3, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.3)
    b.legend(loc='upper center', fontsize=7, ncol=2, columnspacing=1.0, handletextpad=0.4)
    b.set_ylabel(r'Retention $\rho$ (95% CI, shot noise only)')
    b.axhline(0.25, color='0.4', ls='--', lw=0.8)
    b.axhline(0.05, color='0.4', ls=':', lw=0.8)
    b.text(pos[-1] + 0.4, 0.255, 'A', fontsize=7, color='0.3', ha='right', va='bottom')
    b.text(pos[-1] + 0.4, 0.055, 'B', fontsize=7, color='0.3', ha='right', va='bottom')
    b.set_ylim(0, 0.9)
    panel_letter(a, '(a)')
    panel_letter(b, '(b)')
    fig.tight_layout()
    fig.canvas.draw()
    wa, wb = [ax.get_window_extent().width / fig.dpi for ax in (a, b)]
    ha, hb = [ax.get_window_extent().height / fig.dpi for ax in (a, b)]
    print('retention panel (a) %.2f x %.2f in, panel (b) %.2f x %.2f in' % (wa, ha, wb, hb))
    fig.savefig(out)
    plt.close(fig)


if __name__ == '__main__':
    r8 = load_results(os.path.join(PKG, 'n8', 'figures', 'quick_figs.py'))
    r16 = load_results(os.path.join(PKG, 'n16', 'figures', 'quick_figs.py'))
    depth_figure(r8, (0.90, 1.01), os.path.join(HERE, 'fig_n8_approx_ratio.pdf'))
    gap_figure(r8, (0.95, 1.02), os.path.join(HERE, 'fig_n8_sim_vs_hw.pdf'))
    cost_matrix_figure(os.path.join(HERE, 'fig_cost_matrices.pdf'))
    depth_figure(r16, (0.92, 1.005), os.path.join(HERE, 'fig_n16_approx_ratio.pdf'))
    gap_figure(r16, (0.96, 1.01), os.path.join(HERE, 'fig_n16_sim_vs_hw.pdf'))
    retention_figure(os.path.join(HERE, 'fig_retention.pdf'))
    print('wrote 6 PDFs to', HERE)
