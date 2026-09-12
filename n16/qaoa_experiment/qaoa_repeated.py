"""
QAOA Repeated Hardware Runs
============================
Runs each of the 5 original n=16 instances 3 times
on hardware to get mean +/- std confidence intervals
on the noise-assisted optimization finding.

Usage:
  python qaoa_repeated.py --mode both --token YOUR_TOKEN --p_max 4 --repeats 3
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


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD MATRIX AND BUILD INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

def load_matrix(csv_path='mcp_agent_data_16_cost_matrix.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cost matrix not found at '{csv_path}'.")
    df = pd.read_csv(csv_path, index_col=0)
    W  = df.values.astype(float)
    return (W + W.T) / 2


def build_instances(W):
    instances = []

    instances.append((W.copy(), 'I1_full_graph'))

    scale_simple = 2063.0 / 3516.0
    instances.append((W * scale_simple, 'I2_simple_tasks'))

    scale_complex = 5920.0 / 3516.0
    instances.append((W * scale_complex, 'I3_complex_tasks'))

    rng   = np.random.default_rng(42)
    W_lat = W.copy()
    idx   = [(i, j) for i in range(N_QUBITS)
             for j in range(i+1, N_QUBITS) if W[i][j] > 0]
    vals  = rng.permutation([W[i][j] for i, j in idx])
    for (i, j), v in zip(idx, vals):
        W_lat[i][j] = v
        W_lat[j][i] = v
    instances.append((W_lat, 'I4_latency_reweighted'))

    rng2      = np.random.default_rng(99)
    W_rand    = np.zeros((N_QUBITS, N_QUBITS))
    all_pairs = [(i, j) for i in range(N_QUBITS)
                 for j in range(i+1, N_QUBITS)]
    n_edges   = len(idx)
    mean_w    = W[W > 0].mean()
    std_w     = W[W > 0].std()
    chosen    = rng2.choice(len(all_pairs), size=n_edges, replace=False)
    for c in chosen:
        i, j = all_pairs[c]
        w = abs(rng2.normal(mean_w, std_w))
        W_rand[i][j] = w
        W_rand[j][i] = w
    instances.append((W_rand, 'I5_random_baseline'))

    return instances


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MAX-CUT FUNCTIONS
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
    n        = adjacency.shape[0]
    best_cut = 0.0
    best_partition = None
    for k in range(1, 2**n // 2):
        bits = [(k >> i) & 1 for i in range(n)]
        cut  = compute_cut_value(bits, adjacency)
        if cut > best_cut:
            best_cut       = cut
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
# 3. QAOA CIRCUIT
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

    pm      = generate_preset_pass_manager(
        backend=hw_backend, optimization_level=1)
    isa     = pm.run(bound)
    sampler = Sampler(hw_backend)
    job     = sampler.run([isa], shots=shots)
    result  = job.result()
    counts  = result[0].data.meas.get_counts()

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
# 4. REPEATED HARDWARE EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(mode='both', ibm_token=None,
                   p_values=[1,2,3,4], shots=8192,
                   matrix_path='mcp_agent_data_16_cost_matrix.csv',
                   max_iter=200, repeats=3):

    W_full    = load_matrix(matrix_path)
    instances = build_instances(W_full)

    sim_backend = AerSimulator(method='statevector')
    hw_backend  = None

    if mode in ('hardware', 'both'):
        if ibm_token is None:
            raise ValueError("Pass --token YOUR_IBM_TOKEN")
        from qiskit_ibm_runtime import QiskitRuntimeService
        service    = QiskitRuntimeService(
            channel='ibm_quantum_platform', token=ibm_token)
        hw_backend = service.least_busy(
            operational=True, simulator=False,
            min_num_qubits=N_QUBITS)
        print(f"Hardware backend: {hw_backend.name}")

    all_results = []

    for inst_idx, (adjacency, inst_name) in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"Instance {inst_idx+1}/5: {inst_name}")
        print(f"{'='*60}")

        opt_cut, _  = brute_force_optimal(adjacency)
        greedy_cut, _ = greedy_max_cut(adjacency)
        greedy_r    = greedy_cut / opt_cut if opt_cut > 0 else 0
        print(f"  Optimal: {opt_cut:.1f}  Greedy: r={greedy_r:.3f}")

        # Optimize parameters once via simulation
        sim_params = {}
        for p in p_values:
            qc = build_qaoa_circuit(adjacency, p)
            if mode in ('simulate', 'both'):
                opt_g, opt_b, _ = optimize_qaoa(
                    qc, adjacency, p, sim_backend, shots,
                    max_iter=max_iter)
                _, sim_best, _, _ = evaluate_circuit_sim(
                    qc, opt_g, opt_b, adjacency, sim_backend, shots)
                sim_r = sim_best / opt_cut if opt_cut > 0 else 0
                sim_params[p] = {
                    'opt_gamma': opt_g.tolist(),
                    'opt_beta':  opt_b.tolist(),
                    'sim_r':     sim_r
                }
                print(f"  Sim p={p}: r={sim_r:.3f}")

        # Repeated hardware runs
        hw_runs = {p: [] for p in p_values}

        for rep in range(repeats):
            print(f"\n  --- Hardware repeat {rep+1}/{repeats} ---")
            for p in p_values:
                if not (mode in ('hardware', 'both') and hw_backend):
                    break
                qc       = build_qaoa_circuit(adjacency, p)
                opt_g_hw = sim_params[p]['opt_gamma']
                opt_b_hw = sim_params[p]['opt_beta']
                try:
                    _, hw_best, _, _ = evaluate_circuit_hardware(
                        qc, opt_g_hw, opt_b_hw,
                        adjacency, hw_backend, shots)
                    hw_r = hw_best / opt_cut if opt_cut > 0 else 0
                    hw_runs[p].append(hw_r)
                    print(f"    p={p}: r={hw_r:.3f}")
                except Exception as e:
                    print(f"    p={p}: ERROR {e}")
                    hw_runs[p].append(0.0)

        # Summarize
        inst_result = {
            'instance':     inst_name,
            'optimal_cut':  float(opt_cut),
            'greedy_ratio': float(greedy_r),
            'sim':          {p: sim_params[p]['sim_r']
                             for p in p_values if p in sim_params},
            'hw_runs':      {p: hw_runs[p] for p in p_values},
            'hw_mean':      {p: float(np.mean(hw_runs[p]))
                             if hw_runs[p] else 0 for p in p_values},
            'hw_std':       {p: float(np.std(hw_runs[p]))
                             if hw_runs[p] else 0 for p in p_values},
        }
        all_results.append(inst_result)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results, p_values):
    print(f"\n{'='*80}")
    print(f"REPEATED HARDWARE RUNS SUMMARY")
    print(f"{'='*80}")

    for r in results:
        print(f"\n{r['instance']}")
        print(f"  {'p':<4} {'Sim':>7} {'HW mean':>9} {'HW std':>8} "
              f"{'HW runs'}")
        print(f"  {'-'*60}")
        for p in p_values:
            sim_r  = r['sim'].get(p, float('nan'))
            hw_m   = r['hw_mean'].get(p, float('nan'))
            hw_s   = r['hw_std'].get(p, float('nan'))
            runs   = r['hw_runs'].get(p, [])
            runs_s = ' '.join([f"{x:.3f}" for x in runs])
            print(f"  p={p}  {sim_r:>7.3f} {hw_m:>9.3f} "
                  f"{hw_s:>8.4f}  [{runs_s}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='QAOA Repeated Hardware Runs')
    parser.add_argument('--mode',
                        choices=['simulate', 'hardware', 'both'],
                        default='both')
    parser.add_argument('--token',    type=str, default=None)
    parser.add_argument('--p_max',    type=int, default=4)
    parser.add_argument('--repeats',  type=int, default=3,
                        help='Number of hardware repeats per instance')
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--shots',    type=int, default=8192)
    parser.add_argument('--output',   type=str, default='.')
    parser.add_argument('--matrix',   type=str,
                        default='mcp_agent_data_16_cost_matrix.csv')
    args = parser.parse_args()

    p_values = list(range(1, args.p_max + 1))

    print(f"\n{'='*60}")
    print(f"QAOA Repeated Hardware Runs — n=16")
    print(f"Mode:     {args.mode}")
    print(f"Depths:   p=1..{args.p_max}")
    print(f"Repeats:  {args.repeats}")
    print(f"Shots:    {args.shots}")
    print(f"{'='*60}")

    results = run_experiment(
        mode=args.mode,
        ibm_token=args.token,
        p_values=p_values,
        shots=args.shots,
        matrix_path=args.matrix,
        max_iter=args.max_iter,
        repeats=args.repeats
    )

    out_json = f'{args.output}/results_repeated.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    print_summary(results, p_values)