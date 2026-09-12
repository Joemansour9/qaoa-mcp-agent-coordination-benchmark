"""
QAOA Benchmarking for MCP-Derived Agent Coordination Instances
==============================================================
Paper: "Benchmarking QAOA on MCP-Derived Agent Coordination
        Instances Under Realistic IBM Hardware Noise"

Problem: Max-Cut on 8-node MCP tool-interaction graph
  - 8 nodes = 4 MCP servers × 2 tools each
  - Edge weights = empirically measured token costs between tools
  - Objective: partition tools into 2 groups maximizing cut weight

Usage:
  pip install qiskit qiskit-aer qiskit-ibm-runtime numpy scipy matplotlib pandas
  python qaoa_qap_experiment.py --mode simulate
  python qaoa_qap_experiment.py --mode hardware --token YOUR_IBM_TOKEN
  python qaoa_qap_experiment.py --mode both --token YOUR_IBM_TOKEN
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import time
import pandas as pd
import os

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

N_QUBITS = 8

TOOL_NAMES = [
    'fs_read',        # 0 filesystem_read_file    (FileSystem)
    'fs_write',       # 1 filesystem_write_file   (FileSystem)
    'ws_search',      # 2 websearch_search_web    (WebSearch)
    'ws_fetch',       # 3 websearch_fetch_url     (WebSearch)
    'db_query',       # 4 database_query_db       (Database)
    'db_insert',      # 5 database_insert_record  (Database)
    'an_analyse',     # 6 analytics_run_analysis  (Analytics)
    'an_report',      # 7 analytics_generate_report (Analytics)
]

SERVER_LABELS = ['FileSystem', 'FileSystem',
                 'WebSearch',  'WebSearch',
                 'Database',   'Database',
                 'Analytics',  'Analytics']


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD MCP COST MATRIX AND BUILD INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

def load_mcp_matrix(csv_path='mcp_agent_data_cost_matrix.csv'):
    """Load the empirically derived 8x8 cost matrix from MCP data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Cost matrix not found at '{csv_path}'.\n"
            f"Run mcp_data_generator.py first to generate it.")
    df = pd.read_csv(csv_path, index_col=0)
    W = df.values.astype(float)
    # Symmetrize: use mean of W[i][j] and W[j][i] for undirected Max-Cut
    W_sym = (W + W.T) / 2
    return W_sym


def build_instances(W_full):
    """
    Build 5 problem instances from the MCP cost matrix.
    All instances are 8x8 — same graph, different edge weight subsets.

    I1: Full graph — all 150 tasks
    I2: Simple tasks subgraph — scale weights down by observed ratio
    I3: Complex tasks subgraph — scale weights up by observed ratio
    I4: Latency-reweighted — shuffle weights to break server clustering
    I5: Random baseline — same density and mean, random weights (control)
    """
    instances = []

    # I1: Full graph (your real data)
    instances.append((W_full.copy(), 'I1_full_graph'))

    # I2: Simple tasks — token costs lower (simple tasks mean=2063 vs overall)
    # Scale factor: 2063/3516 ≈ 0.587 (simple mean / overall mean)
    scale_simple = 2063.0 / 3516.0
    W_simple = W_full * scale_simple
    instances.append((W_simple, 'I2_simple_tasks'))

    # I3: Complex tasks — token costs higher (complex mean=5920 vs overall)
    # Scale factor: 5920/3516 ≈ 1.684
    scale_complex = 5920.0 / 3516.0
    W_complex = W_full * scale_complex
    instances.append((W_complex, 'I3_complex_tasks'))

    # I4: Latency-reweighted — permute weights to break server-cluster structure
    rng = np.random.default_rng(42)
    W_latency = W_full.copy()
    nonzero_idx = [(i, j) for i in range(8) for j in range(i+1, 8)
                   if W_full[i][j] > 0]
    nonzero_vals = [W_full[i][j] for i, j in nonzero_idx]
    shuffled_vals = rng.permutation(nonzero_vals)
    for (i, j), v in zip(nonzero_idx, shuffled_vals):
        W_latency[i][j] = v
        W_latency[j][i] = v
    instances.append((W_latency, 'I4_latency_reweighted'))

    # I5: Random baseline — same edge density and mean as I1 (control)
    n_edges = sum(1 for i in range(8) for j in range(i+1, 8)
                  if W_full[i][j] > 0)
    mean_w = W_full[W_full > 0].mean()
    std_w  = W_full[W_full > 0].std()
    W_random = np.zeros((8, 8))
    all_pairs = [(i, j) for i in range(8) for j in range(i+1, 8)]
    rng2 = np.random.default_rng(99)
    chosen = rng2.choice(len(all_pairs), size=n_edges, replace=False)
    for idx in chosen:
        i, j = all_pairs[idx]
        w = abs(rng2.normal(mean_w, std_w))
        W_random[i][j] = w
        W_random[j][i] = w
    instances.append((W_random, 'I5_random_baseline'))

    return instances


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MAX-CUT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cut_value(bitstring, adjacency):
    """Compute the cut value for a given bitstring partition."""
    n = len(bitstring)
    cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if bitstring[i] != bitstring[j]:
                cut += adjacency[i][j]
    return cut


def brute_force_optimal(adjacency):
    """Find optimal Max-Cut by exhaustive search (feasible for n=8)."""
    n = adjacency.shape[0]
    best_cut = 0.0
    best_partition = None
    for k in range(1, 2 ** n):
        bits = [(k >> i) & 1 for i in range(n)]
        cut = compute_cut_value(bits, adjacency)
        if cut > best_cut:
            best_cut = cut
            best_partition = bits[:]
    return best_cut, best_partition


def greedy_max_cut(adjacency):
    """Greedy baseline: sequential node assignment maximizing immediate cut."""
    n = adjacency.shape[0]
    assignment = np.zeros(n, dtype=int)
    for i in range(1, n):
        cut_0 = sum(adjacency[i][j] for j in range(i) if assignment[j] != 0)
        cut_1 = sum(adjacency[i][j] for j in range(i) if assignment[j] != 1)
        assignment[i] = 1 if cut_1 > cut_0 else 0
    return compute_cut_value(assignment, adjacency), assignment


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  QAOA CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════════

def build_qaoa_circuit(adjacency, p):
    """Build QAOA circuit for weighted Max-Cut at depth p."""
    n = adjacency.shape[0]
    gamma = ParameterVector('γ', p)
    beta  = ParameterVector('β', p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for i in range(n):
            for j in range(i + 1, n):
                w = adjacency[i][j]
                if w > 0:
                    qc.rzz(2 * gamma[layer] * w, i, j)
        for i in range(n):
            qc.rx(2 * beta[layer], i)
    qc.measure_all()
    return qc


def evaluate_circuit(qc, params_gamma, params_beta, adjacency,
                     backend, shots=8192):
    """Bind parameters, run circuit, return expected and best cut value."""
    p = len(params_gamma)
    param_dict = {}
    for k in range(p):
        param_dict[qc.parameters[k]] = params_gamma[k]
    for k in range(p):
        param_dict[qc.parameters[p + k]] = params_beta[k]

    bound = qc.assign_parameters(param_dict)
    job = backend.run(
        transpile(bound, backend, optimization_level=1), shots=shots)
    counts = job.result().get_counts()

    total_shots = sum(counts.values())
    expected_cut = 0.0
    best_cut = 0.0
    best_bitstring = None

    for bitstring, count in counts.items():
        bits = [int(b) for b in reversed(bitstring)]
        cut = compute_cut_value(bits, adjacency)
        expected_cut += cut * count / total_shots
        if cut > best_cut:
            best_cut = cut
            best_bitstring = bits[:]

    return expected_cut, best_cut, best_bitstring, counts


def optimize_qaoa(qc, adjacency, p, backend, shots=8192, max_iter=80):
    """Optimize QAOA parameters with COBYLA. Returns (gamma, beta, history)."""
    history = []

    def objective(params):
        gamma = params[:p]
        beta  = params[p:]
        expected_cut, _, _, _ = evaluate_circuit(
            qc, gamma, beta, adjacency, backend, shots)
        history.append(expected_cut)
        return -expected_cut

    rng = np.random.default_rng(0)
    x0 = rng.uniform(0, np.pi, 2 * p)
    result = minimize(objective, x0, method='COBYLA',
                      options={'maxiter': max_iter, 'rhobeg': 0.5})
    return result.x[:p], result.x[p:], history


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FULL EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(mode='simulate', ibm_token=None,
                   p_values=[1, 2, 3, 4], shots=8192,
                   matrix_path='mcp_agent_data_cost_matrix.csv'):

    W_full = load_mcp_matrix(matrix_path)
    instances = build_instances(W_full)

    print(f"\nLoaded cost matrix from: {matrix_path}")
    print(f"Non-zero edges: {int((W_full > 0).sum() // 2)}/28")
    print(f"Weight range: {W_full[W_full>0].min():.0f}"
          f"–{W_full[W_full>0].max():.0f} tokens")

    sim_backend = AerSimulator(method='statevector')

    hw_backend = None
    if mode in ('hardware', 'both'):
        if ibm_token is None:
            raise ValueError("Pass --token YOUR_IBM_TOKEN for hardware mode.")
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(
            channel='ibm_quantum', token=ibm_token)
        hw_backend = service.least_busy(
            operational=True, simulator=False,
            min_num_qubits=N_QUBITS)
        print(f"Hardware backend: {hw_backend.name}")

    results = []

    for inst_idx, (adjacency, inst_name) in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"Instance {inst_idx+1}/5: {inst_name}")
        print(f"{'='*60}")

        t0 = time.time()
        opt_cut, opt_partition = brute_force_optimal(adjacency)
        print(f"  Optimal cut:  {opt_cut:.1f}  ({time.time()-t0:.2f}s)")

        greedy_cut, _ = greedy_max_cut(adjacency)
        greedy_ratio  = greedy_cut / opt_cut if opt_cut > 0 else 0
        print(f"  Greedy cut:   {greedy_cut:.1f}  (r={greedy_ratio:.3f})")

        inst_result = {
            'instance':     inst_name,
            'optimal_cut':  float(opt_cut),
            'greedy_cut':   float(greedy_cut),
            'greedy_ratio': float(greedy_ratio),
            'qaoa_sim':     {},
            'qaoa_hw':      {}
        }

        for p in p_values:
            print(f"\n  ── p={p} ──")
            qc = build_qaoa_circuit(adjacency, p)

            if mode in ('simulate', 'both'):
                print(f"    Simulating...", end=' ', flush=True)
                opt_g, opt_b, hist = optimize_qaoa(
                    qc, adjacency, p, sim_backend, shots)
                _, sim_best, _, _ = evaluate_circuit(
                    qc, opt_g, opt_b, adjacency, sim_backend, shots)
                sim_r = sim_best / opt_cut if opt_cut > 0 else 0
                inst_result['qaoa_sim'][p] = {
                    'best_cut':           float(sim_best),
                    'approximation_ratio': float(sim_r),
                    'opt_gamma':          opt_g.tolist(),
                    'opt_beta':           opt_b.tolist()
                }
                print(f"cut={sim_best:.1f}, r={sim_r:.3f}")

            if mode in ('hardware', 'both') and hw_backend:
                print(f"    Hardware...", end=' ', flush=True)
                opt_g_hw = inst_result['qaoa_sim'][p]['opt_gamma']
                opt_b_hw = inst_result['qaoa_sim'][p]['opt_beta']
                _, hw_best, _, _ = evaluate_circuit(
                    qc, opt_g_hw, opt_b_hw, adjacency, hw_backend, shots)
                hw_r = hw_best / opt_cut if opt_cut > 0 else 0
                inst_result['qaoa_hw'][p] = {
                    'best_cut':           float(hw_best),
                    'approximation_ratio': float(hw_r)
                }
                print(f"cut={hw_best:.1f}, r={hw_r:.3f}")

        results.append(inst_result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results, p_values=[1, 2, 3, 4], output_dir='.',
                 matrix_path='mcp_agent_data_cost_matrix.csv'):
    """Generate all four paper-ready figures."""
    short_names = ['I1: Full', 'I2: Simple', 'I3: Complex',
                   'I4: Reweighted', 'I5: Random']
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(results)))

    # ── Figure 1: Approximation ratio vs depth ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mode_key, label in zip(
            axes,
            ['qaoa_sim', 'qaoa_hw'],
            ['Simulation (Ideal — Qiskit Statevector)',
             'Hardware (IBM Eagle)']):

        if not any(r[mode_key] for r in results):
            ax.text(0.5, 0.5, f'{label}\n(not run)',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(label)
            continue

        for i, r in enumerate(results):
            if not r[mode_key]:
                continue
            ps     = [p for p in p_values if p in r[mode_key]]
            ratios = [r[mode_key][p]['approximation_ratio'] for p in ps]
            ax.plot(ps, ratios, 'o-', color=colors[i],
                    label=short_names[i], linewidth=2, markersize=7)

        greedy_mean = np.mean([r['greedy_ratio'] for r in results])
        ax.axhline(greedy_mean, color='red', linestyle='--', linewidth=1.5,
                   label=f'Greedy (mean={greedy_mean:.3f})')
        ax.axhline(1.0, color='green', linestyle=':', linewidth=1,
                   label='Optimal (r=1.0)')
        ax.set_xlabel('Circuit Depth $p$', fontsize=12)
        ax.set_ylabel('Approximation Ratio $r$', fontsize=12)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.12)
        ax.set_xticks(p_values)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('QAOA Approximation Ratio vs Circuit Depth\n'
                 'MCP-Derived Agent Coordination Instances ($n=8$ qubits)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{output_dir}/fig1_approx_ratio_vs_depth.{ext}',
                    bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Figure 1")

    # ── Figure 2: Sim vs Hardware gap ────────────────────────────────────────
    has_hw = any(r['qaoa_hw'] for r in results)
    has_sim = any(r['qaoa_sim'] for r in results)
    if has_sim and has_hw:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(p_values))
        w = 0.35
        sim_m, hw_m, sim_s, hw_s = [], [], [], []
        for p in p_values:
            sr = [r['qaoa_sim'][p]['approximation_ratio']
                  for r in results if p in r['qaoa_sim']]
            hr = [r['qaoa_hw'][p]['approximation_ratio']
                  for r in results if p in r['qaoa_hw']]
            sim_m.append(np.mean(sr) if sr else 0)
            hw_m.append(np.mean(hr) if hr else 0)
            sim_s.append(np.std(sr) if sr else 0)
            hw_s.append(np.std(hr) if hr else 0)

        ax.bar(x - w/2, sim_m, w, yerr=sim_s, label='Simulation',
               color='steelblue', alpha=0.85, capsize=5)
        ax.bar(x + w/2, hw_m,  w, yerr=hw_s, label='IBM Eagle Hardware',
               color='coral', alpha=0.85, capsize=5)
        ax.set_xlabel('Circuit Depth $p$', fontsize=12)
        ax.set_ylabel('Mean Approximation Ratio', fontsize=12)
        ax.set_title('Simulation vs Hardware Approximation Ratio\n'
                     '(Mean $\\pm$ Std across 5 MCP instances)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'$p={p}$' for p in p_values])
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        for ext in ('pdf', 'png'):
            plt.savefig(f'{output_dir}/fig2_sim_vs_hardware_gap.{ext}',
                        bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved Figure 2")

    # ── Figure 3: MCP cost matrix heatmaps ───────────────────────────────────
    W_full = load_mcp_matrix(matrix_path)
    inst_matrices = build_instances(W_full)
    plot_ids    = [0, 1, 2]   # I1, I2, I3
    plot_titles = ['I1: Full Graph\n(All 150 Tasks)',
                   'I2: Simple Tasks\n(Scaled)',
                   'I3: Complex Tasks\n(Scaled)']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, idx, title in zip(axes, plot_ids, plot_titles):
        W = inst_matrices[idx][0]
        im = ax.imshow(W, cmap='YlOrRd', aspect='auto')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Tool Node', fontsize=10)
        ax.set_ylabel('Tool Node', fontsize=10)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(TOOL_NAMES, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(TOOL_NAMES, fontsize=7)
        # Draw server boundary lines
        for pos in [1.5, 3.5, 5.5]:
            ax.axhline(pos, color='blue', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.axvline(pos, color='blue', linewidth=0.8, linestyle='--', alpha=0.5)
        plt.colorbar(im, ax=ax, label='Token Cost')
    plt.suptitle('MCP Tool-Interaction Cost Matrices\n'
                 '(Blue dashed lines = MCP server boundaries)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{output_dir}/fig3_cost_matrices.{ext}',
                    bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Figure 3")

    # ── Figure 4: Summary table ───────────────────────────────────────────────
    best_p = max(p_values)
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis('off')
    table_data = []
    for i, r in enumerate(results):
        sim_r = r['qaoa_sim'].get(best_p, {}).get('approximation_ratio', 'N/A')
        hw_r  = r['qaoa_hw'].get(best_p,  {}).get('approximation_ratio', 'N/A')
        sim_r = f"{sim_r:.3f}" if isinstance(sim_r, float) else sim_r
        hw_r  = f"{hw_r:.3f}"  if isinstance(hw_r,  float) else hw_r
        delta = ('N/A' if 'N/A' in (sim_r, hw_r)
                 else f"{float(sim_r)-float(hw_r):.3f}")
        table_data.append([
            short_names[i],
            f"{r['optimal_cut']:.0f}",
            f"{r['greedy_ratio']:.3f}",
            sim_r, hw_r, delta
        ])
    col_labels = ['Instance', 'Optimal Cut', 'Greedy $r$',
                  f'Sim $r$ ($p={best_p}$)',
                  f'HW $r$ ($p={best_p}$)',
                  '$\\Delta r$']
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2.2)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title(
        f'Summary: Approximation Ratios Across All Instances ($p={best_p}$)',
        fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{output_dir}/fig4_summary_table.{ext}',
                    bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Figure 4")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='QAOA MCP Agent Coordination Benchmark')
    parser.add_argument('--mode',
                        choices=['simulate', 'hardware', 'both'],
                        default='simulate')
    parser.add_argument('--token',   type=str, default=None)
    parser.add_argument('--p_max',   type=int, default=4)
    parser.add_argument('--shots',   type=int, default=8192)
    parser.add_argument('--output',  type=str, default='.')
    parser.add_argument('--matrix',  type=str,
                        default='mcp_agent_data_cost_matrix.csv',
                        help='Path to MCP cost matrix CSV')
    args = parser.parse_args()

    p_values = list(range(1, args.p_max + 1))

    print(f"\n{'='*60}")
    print(f"QAOA MCP Agent Coordination Benchmark")
    print(f"Mode:   {args.mode}")
    print(f"Depths: p=1..{args.p_max}")
    print(f"Shots:  {args.shots}")
    print(f"Matrix: {args.matrix}")
    print(f"{'='*60}")

    results = run_experiment(
        mode=args.mode,
        ibm_token=args.token,
        p_values=p_values,
        shots=args.shots,
        matrix_path=args.matrix
    )

    # Save raw results
    out_json = f'{args.output}/results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Generate figures
    plot_results(results, p_values=p_values, output_dir=args.output,
                 matrix_path=args.matrix)

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Instance':<22} {'Optimal':>8} {'Greedy':>8} "
          f"{'Sim p='+str(max(p_values)):>10} {'HW p='+str(max(p_values)):>10}")
    print('-' * 62)
    for r in results:
        bp = max(p_values)
        sim_r = r['qaoa_sim'].get(bp, {}).get('approximation_ratio', float('nan'))
        hw_r  = r['qaoa_hw'].get(bp,  {}).get('approximation_ratio', float('nan'))
        sim_s = f"{sim_r:.3f}" if not np.isnan(sim_r) else 'N/A'
        hw_s  = f"{hw_r:.3f}"  if not np.isnan(hw_r)  else 'N/A'
        print(f"{r['instance']:<22} {r['optimal_cut']:>8.0f} "
              f"{r['greedy_ratio']:>8.3f} {sim_s:>10} {hw_s:>10}")
