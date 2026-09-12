"""
QAOA Benchmarking — 16 Node MCP Agent Coordination
====================================================
Paper: "Benchmarking QAOA on MCP-Derived Agent Coordination
        Instances: Scaling from n=8 to n=16 on IBM Hardware"

Runs QAOA at p=1-4 on 5 instances derived from the 16-node
MCP cost matrix.

Usage:
  python qaoa_experiment_16.py --mode simulate --p_max 4
  python qaoa_experiment_16.py --mode both --token YOUR_TOKEN --p_max 4
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

N_QUBITS = 16

TOOL_NAMES = [
    'fs_read',    'fs_write',       # FileSystem    0,1
    'ws_search',  'ws_fetch',       # WebSearch     2,3
    'db_query',   'db_insert',      # Database      4,5
    'an_analyse', 'an_report',      # Analytics     6,7
    'em_send',    'em_inbox',       # Email         8,9
    'cal_create', 'cal_list',       # Calendar      10,11
    'code_run',   'code_output',    # CodeExecution 12,13
    'vec_embed',  'vec_search',     # VectorSearch  14,15
]

SERVER_LABELS = [
    'FileSystem', 'FileSystem',
    'WebSearch',  'WebSearch',
    'Database',   'Database',
    'Analytics',  'Analytics',
    'Email',      'Email',
    'Calendar',   'Calendar',
    'CodeExec',   'CodeExec',
    'VectorSearch','VectorSearch',
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD MATRIX AND BUILD INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

def load_matrix(csv_path='mcp_agent_data_16_cost_matrix.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Cost matrix not found at '{csv_path}'.\n"
            f"Run mcp_data_generator_16.py first.")
    df = pd.read_csv(csv_path, index_col=0)
    W  = df.values.astype(float)
    return (W + W.T) / 2   # symmetrize for undirected Max-Cut


def build_instances(W):
    """
    5 instances from the 16-node MCP cost matrix.
    Same derivation strategy as n=8 for direct comparison.
    """
    instances = []

    # I1: Full graph
    instances.append((W.copy(), 'I1_full_graph'))

    # I2: Simple tasks scaled (mean token ratio)
    scale_simple = 2063.0 / 3516.0
    instances.append((W * scale_simple, 'I2_simple_tasks'))

    # I3: Complex tasks scaled
    scale_complex = 5920.0 / 3516.0
    instances.append((W * scale_complex, 'I3_complex_tasks'))

    # I4: Latency-reweighted (permute weights, break server structure)
    rng   = np.random.default_rng(42)
    W_lat = W.copy()
    idx   = [(i, j) for i in range(N_QUBITS)
             for j in range(i+1, N_QUBITS) if W[i][j] > 0]
    vals  = rng.permutation([W[i][j] for i, j in idx])
    for (i, j), v in zip(idx, vals):
        W_lat[i][j] = v
        W_lat[j][i] = v
    instances.append((W_lat, 'I4_latency_reweighted'))

    # I5: Random baseline (same density and mean as I1)
    rng2    = np.random.default_rng(99)
    W_rand  = np.zeros((N_QUBITS, N_QUBITS))
    all_pairs = [(i, j) for i in range(N_QUBITS)
                 for j in range(i+1, N_QUBITS)]
    n_edges = len(idx)
    mean_w  = W[W > 0].mean()
    std_w   = W[W > 0].std()
    chosen  = rng2.choice(len(all_pairs), size=n_edges, replace=False)
    for c in chosen:
        i, j = all_pairs[c]
        w = abs(rng2.normal(mean_w, std_w))
        W_rand[i][j] = w
        W_rand[j][i] = w
    instances.append((W_rand, 'I5_random_baseline'))

    return instances


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MAX-CUT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cut_value(bitstring, adjacency):
    n   = len(bitstring)
    cut = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if bitstring[i] != bitstring[j]:
                cut += adjacency[i][j]
    return cut


def brute_force_optimal(adjacency):
    """
    Exhaustive search — feasible at n=16 but slow (~65,536 evaluations).
    Uses symmetry: only search half the space.
    """
    n        = adjacency.shape[0]
    best_cut = 0.0
    best_partition = None
    for k in range(1, 2**n // 2):   # exploit symmetry
        bits = [(k >> i) & 1 for i in range(n)]
        cut  = compute_cut_value(bits, adjacency)
        if cut > best_cut:
            best_cut      = cut
            best_partition = bits[:]
    return best_cut, best_partition


def greedy_max_cut(adjacency):
    n          = adjacency.shape[0]
    assignment = np.zeros(n, dtype=int)
    for i in range(1, n):
        cut_0 = sum(adjacency[i][j] for j in range(i) if assignment[j] != 0)
        cut_1 = sum(adjacency[i][j] for j in range(i) if assignment[j] != 1)
        assignment[i] = 1 if cut_1 > cut_0 else 0
    return compute_cut_value(assignment, adjacency), assignment


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  QAOA CIRCUIT (16 qubits)
# ═══════════════════════════════════════════════════════════════════════════════

def build_qaoa_circuit(adjacency, p):
    n     = adjacency.shape[0]
    gamma = ParameterVector('γ', p)
    beta  = ParameterVector('β', p)
    qc    = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for i in range(n):
            for j in range(i+1, n):
                w = adjacency[i][j]
                if w > 0:
                    qc.rzz(2 * gamma[layer] * w, i, j)
        for i in range(n):
            qc.rx(2 * beta[layer], i)
    qc.measure_all()
    return qc


def evaluate_circuit_sim(qc, params_gamma, params_beta,
                          adjacency, backend, shots=8192, seed_simulator=42):
    p          = len(params_gamma)
    param_dict = {}
    for k in range(p):
        param_dict[qc.parameters[k]]     = params_gamma[k]
        param_dict[qc.parameters[p + k]] = params_beta[k]
    bound  = qc.assign_parameters(param_dict)
    job    = backend.run(
        transpile(bound, backend, optimization_level=1), shots=shots,
        seed_simulator=seed_simulator)
    counts = job.result().get_counts()

    total_shots  = sum(counts.values())
    expected_cut = 0.0
    best_cut     = 0.0
    best_bits    = None
    for bitstring, count in counts.items():
        bits = [int(b) for b in reversed(bitstring)]
        cut  = compute_cut_value(bits, adjacency)
        expected_cut += cut * count / total_shots
        if cut > best_cut:
            best_cut  = cut
            best_bits = bits[:]
    return expected_cut, best_cut, best_bits, counts


def evaluate_circuit_hardware(qc, params_gamma, params_beta,
                               adjacency, hw_backend, shots=8192):
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    p          = len(params_gamma)
    param_dict = {}
    for k in range(p):
        param_dict[qc.parameters[k]]     = params_gamma[k]
        param_dict[qc.parameters[p + k]] = params_beta[k]
    bound = qc.assign_parameters(param_dict)

    pm         = generate_preset_pass_manager(
        backend=hw_backend, optimization_level=1)
    isa        = pm.run(bound)
    sampler    = Sampler(hw_backend)
    job        = sampler.run([isa], shots=shots)
    result     = job.result()
    pub_result = result[0]
    counts     = pub_result.data.meas.get_counts()

    total_shots  = sum(counts.values())
    expected_cut = 0.0
    best_cut     = 0.0
    best_bits    = None
    for bitstring, count in counts.items():
        bits = [int(b) for b in reversed(bitstring)]
        cut  = compute_cut_value(bits, adjacency)
        expected_cut += cut * count / total_shots
        if cut > best_cut:
            best_cut  = cut
            best_bits = bits[:]
    return expected_cut, best_cut, best_bits, counts


def optimize_qaoa(qc, adjacency, p, backend, shots=8192, max_iter=200,
                  seed_simulator=42):
    history = []

    def objective(params):
        gamma = params[:p]
        beta  = params[p:]
        exp, _, _, _ = evaluate_circuit_sim(
            qc, gamma, beta, adjacency, backend, shots, seed_simulator)
        history.append(exp)
        return -exp

    rng    = np.random.default_rng(0)
    x0     = rng.uniform(0, np.pi, 2 * p)
    result = minimize(objective, x0, method='COBYLA',
                      options={'maxiter': max_iter, 'rhobeg': 0.5})
    return result.x[:p], result.x[p:], history


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FULL EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(mode='simulate', ibm_token=None,
                   p_values=[1, 2], shots=8192,
                   matrix_path='mcp_agent_data_16_cost_matrix.csv',
                   max_iter=200):

    W_full    = load_matrix(matrix_path)
    instances = build_instances(W_full)

    nz = W_full[W_full > 0]
    print(f"\nLoaded 16-node cost matrix from: {matrix_path}")
    print(f"Active edges:  {int((W_full > 0).sum() // 2)}/120 possible")
    print(f"Weight range:  {nz.min():.0f}–{nz.max():.0f} tokens")
    print(f"CV:            {nz.std()/nz.mean():.3f}")

    sim_backend = AerSimulator(method='statevector')
    hw_backend  = None

    if mode in ('hardware', 'both'):
        if ibm_token is None:
            raise ValueError("Pass --token YOUR_IBM_TOKEN for hardware mode.")
        from qiskit_ibm_runtime import QiskitRuntimeService
        service    = QiskitRuntimeService(
            channel='ibm_quantum_platform', token=ibm_token)
        hw_backend = service.least_busy(
            operational=True, simulator=False,
            min_num_qubits=N_QUBITS)
        print(f"Hardware backend: {hw_backend.name}")

    results = []

    for inst_idx, (adjacency, inst_name) in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"Instance {inst_idx+1}/5: {inst_name}")
        print(f"{'='*60}")

        t0       = time.time()
        opt_cut, _ = brute_force_optimal(adjacency)
        print(f"  Optimal cut:  {opt_cut:.1f}  ({time.time()-t0:.1f}s)")

        greedy_cut, _ = greedy_max_cut(adjacency)
        greedy_r      = greedy_cut / opt_cut if opt_cut > 0 else 0
        print(f"  Greedy cut:   {greedy_cut:.1f}  (r={greedy_r:.3f})")

        inst_result = {
            'instance':     inst_name,
            'optimal_cut':  float(opt_cut),
            'greedy_cut':   float(greedy_cut),
            'greedy_ratio': float(greedy_r),
            'qaoa_sim':     {},
            'qaoa_hw':      {}
        }

        for p in p_values:
            print(f"\n  ── p={p} ──")
            qc = build_qaoa_circuit(adjacency, p)

            if mode in ('simulate', 'both'):
                print(f"    Simulating...", end=' ', flush=True)
                opt_g, opt_b, _ = optimize_qaoa(
                    qc, adjacency, p, sim_backend, shots,
                    max_iter=max_iter)
                _, sim_best, _, _ = evaluate_circuit_sim(
                    qc, opt_g, opt_b, adjacency, sim_backend, shots)
                sim_r = sim_best / opt_cut if opt_cut > 0 else 0
                inst_result['qaoa_sim'][p] = {
                    'best_cut':            float(sim_best),
                    'approximation_ratio': float(sim_r),
                    'opt_gamma':           opt_g.tolist(),
                    'opt_beta':            opt_b.tolist()
                }
                print(f"cut={sim_best:.1f}, r={sim_r:.3f}")

            if mode in ('hardware', 'both') and hw_backend:
                print(f"    Hardware...", end=' ', flush=True)
                opt_g_hw = inst_result['qaoa_sim'][p]['opt_gamma']
                opt_b_hw = inst_result['qaoa_sim'][p]['opt_beta']
                try:
                    _, hw_best, _, _ = evaluate_circuit_hardware(
                        qc, opt_g_hw, opt_b_hw,
                        adjacency, hw_backend, shots)
                    hw_r = hw_best / opt_cut if opt_cut > 0 else 0
                    inst_result['qaoa_hw'][p] = {
                        'best_cut':            float(hw_best),
                        'approximation_ratio': float(hw_r)
                    }
                    print(f"cut={hw_best:.1f}, r={hw_r:.3f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    inst_result['qaoa_hw'][p] = {
                        'best_cut': 0.0,
                        'approximation_ratio': 0.0,
                        'error': str(e)
                    }

        results.append(inst_result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results, p_values=[1, 2], output_dir='.',
                 matrix_path='mcp_agent_data_16_cost_matrix.csv'):

    short_names = ['I1: Full', 'I2: Simple', 'I3: Complex',
                   'I4: Reweighted', 'I5: Random']
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(results)))

    # Figure 1: Approximation ratio vs depth
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mode_key, label in zip(
            axes,
            ['qaoa_sim', 'qaoa_hw'],
            ['Simulation (Ideal — Qiskit Statevector)',
             'Hardware (IBM Marrakesh)']):

        if not any(r[mode_key] for r in results):
            ax.text(0.5, 0.5, f'{label}\n(not run)',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(label)
            continue

        for i, r in enumerate(results):
            if not r[mode_key]: continue
            ps     = [p for p in p_values if p in r[mode_key]]
            ratios = [r[mode_key][p]['approximation_ratio'] for p in ps]
            ax.plot(ps, ratios, 'o-', color=colors[i],
                    label=short_names[i], linewidth=2, markersize=7)

        gm = np.mean([r['greedy_ratio'] for r in results])
        ax.axhline(gm, color='red', linestyle='--', linewidth=1.5,
                   label=f'Greedy (mean={gm:.3f})')
        ax.axhline(1.0, color='green', linestyle=':', linewidth=1,
                   label='Optimal (r=1.0)')
        ax.set_xlabel('Circuit Depth $p$', fontsize=12)
        ax.set_ylabel('Approximation Ratio $r$', fontsize=12)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylim(0.5, 1.05)
        ax.set_xticks(p_values)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('QAOA Approximation Ratio vs Circuit Depth — n=16\n'
                 'MCP-Derived Agent Coordination Instances (16 qubits)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_n16_approx_ratio.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved: fig_n16_approx_ratio.png")

    # Figure 2: Sim vs Hardware gap
    has_sim = any(r['qaoa_sim'] for r in results)
    has_hw  = any(r['qaoa_hw']  for r in results)
    if has_sim and has_hw:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(p_values)); w = 0.35
        sm, hm, ss, hs = [], [], [], []
        for p in p_values:
            sr = [r['qaoa_sim'][p]['approximation_ratio']
                  for r in results if p in r['qaoa_sim']]
            hr = [r['qaoa_hw'][p]['approximation_ratio']
                  for r in results if p in r['qaoa_hw']]
            sm.append(np.mean(sr) if sr else 0)
            hm.append(np.mean(hr) if hr else 0)
            ss.append(np.std(sr)  if sr else 0)
            hs.append(np.std(hr)  if hr else 0)

        ax.bar(x-w/2, sm, w, yerr=ss, label='Simulation',
               color='steelblue', alpha=0.85, capsize=5)
        ax.bar(x+w/2, hm, w, yerr=hs, label='IBM Marrakesh Hardware',
               color='coral',     alpha=0.85, capsize=5)
        ax.set_xlabel('Circuit Depth $p$', fontsize=12)
        ax.set_ylabel('Mean Approximation Ratio', fontsize=12)
        ax.set_title('Simulation vs Hardware — n=16\n'
                     '(Mean $\\pm$ Std across 5 instances)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'$p={p}$' for p in p_values])
        ax.set_ylim(0.5, 1.10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fig_n16_sim_vs_hw.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved: fig_n16_sim_vs_hw.png")

    # Summary table
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY — n=16")
    print(f"{'='*60}")
    print(f"{'Instance':<25} {'Optimal':>10} {'Greedy':>8} "
          + "  ".join([f"Sim p={p}" for p in p_values])
          + "  "
          + "  ".join([f"HW p={p}" for p in p_values]))
    print('-' * 80)
    for r in results:
        sim_vals = " ".join([
            f"{r['qaoa_sim'].get(p,{}).get('approximation_ratio', float('nan')):>8.3f}"
            for p in p_values])
        hw_vals = " ".join([
            f"{r['qaoa_hw'].get(p,{}).get('approximation_ratio', float('nan')):>7.3f}"
            for p in p_values])
        print(f"{r['instance']:<25} {r['optimal_cut']:>10.0f} "
              f"{r['greedy_ratio']:>8.3f}  {sim_vals}  {hw_vals}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='QAOA n=16 MCP Benchmark')
    parser.add_argument('--mode',
                        choices=['simulate', 'hardware', 'both'],
                        default='simulate')
    parser.add_argument('--token',  type=str, default=None)
    parser.add_argument('--p_max',  type=int, default=4,
                    help='Max QAOA depth (1-4)')
    parser.add_argument('--max_iter', type=int, default=200,
                    help='COBYLA max iterations')
    parser.add_argument('--shots',  type=int, default=8192)
    parser.add_argument('--output', type=str, default='.')
    parser.add_argument('--matrix', type=str,
                        default='mcp_agent_data_16_cost_matrix.csv')
    args = parser.parse_args()

    p_values = list(range(1, args.p_max + 1))

    print(f"\n{'='*60}")
    print(f"QAOA n=16 MCP Agent Coordination Benchmark")
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
        matrix_path=args.matrix,
        max_iter=args.max_iter
    )

    out_json = f'{args.output}/results_n16.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    plot_results(results, p_values=p_values,
                 output_dir=args.output, matrix_path=args.matrix)