"""
qaoa_experiment_16_I6.py
=========================
QAOA Benchmarking — Instance I6 (topology-shifted, patent case P202602120)

Adapted from qaoa_experiment_16.py (same circuit construction, optimizer
configuration, and evaluation methodology — verified against real I1 data:
this script's symmetrization reproduces the paper's reported CV=0.344
exactly). The only substantive change: instead of build_instances()
deriving I2-I5 from I1 by scaling/permutation, this script runs QAOA on
I6 alone — a single instance measured from a genuinely different task
mix (see mcp_data_generator_16_I6.py), with its own distinct edge set.

Paper: "Benchmarking QAOA on MCP-Derived Agent Coordination
        Instances: Scaling from n=8 to n=16 on IBM Hardware"

Usage:
  # Phase 1 (free) — simulation only, get warm-start params + r_sim at each depth
  python qaoa_experiment_16_I6.py --mode simulate --p_max 4 \\
      --matrix mcp_agent_data_16_I6_cost_matrix.csv

  # Phase 2 (costs QPU time) — hardware run once warm-start params are ready.
  # NOTE: for the 7m53s budget, run p=2 ONLY first:
  python qaoa_experiment_16_I6.py --mode hardware --token YOUR_TOKEN \\
      --p_max 2 --matrix mcp_agent_data_16_I6_cost_matrix.csv \\
      --sim_results results_I6_simulate.json

  # If quota remains after Run 1, p=4 stretch run:
  python qaoa_experiment_16_I6.py --mode hardware --token YOUR_TOKEN \\
      --p_max 4 --p_min 4 --matrix mcp_agent_data_16_I6_cost_matrix.csv \\
      --sim_results results_I6_simulate.json
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
    I6 is a single, standalone instance — measured directly from a
    genuinely different task mix (see mcp_data_generator_16_I6.py),
    not derived from I1 by scaling/permutation the way I2-I5 were.
    Kept as a single-element list so run_experiment()'s loop structure
    below is otherwise unchanged from the original script.
    """
    return [(W.copy(), 'I6_topology_shifted')]


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
                          adjacency, backend, shots=8192):
    p          = len(params_gamma)
    param_dict = {}
    for k in range(p):
        param_dict[qc.parameters[k]]     = params_gamma[k]
        param_dict[qc.parameters[p + k]] = params_beta[k]
    bound  = qc.assign_parameters(param_dict)
    job    = backend.run(
        transpile(bound, backend, optimization_level=1), shots=shots)
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


def optimize_qaoa(qc, adjacency, p, backend, shots=8192, max_iter=200):
    history = []

    def objective(params):
        gamma = params[:p]
        beta  = params[p:]
        exp, _, _, _ = evaluate_circuit_sim(
            qc, gamma, beta, adjacency, backend, shots)
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
                   matrix_path='mcp_agent_data_16_I6_cost_matrix.csv',
                   max_iter=200, hw_depths=None, sim_results_path=None,
                   hardware_name='ibm_marrakesh'):

    W_full    = load_matrix(matrix_path)
    instances = build_instances(W_full)   # single-element list for I6

    # hw_depths controls which depths actually get submitted to hardware,
    # independent of p_values (which controls simulation depths). This is
    # the critical addition for I6's tight QPU budget: you can simulate
    # p=1..4 for free, but only ever submit p=2 (or whichever depths you
    # explicitly choose) to the real backend.
    if hw_depths is None:
        hw_depths = list(p_values)
    else:
        for p in hw_depths:
            if p not in p_values:
                raise ValueError(
                    f"--hw_depths includes p={p}, which is not in "
                    f"p_values={p_values}. Simulation results (for the "
                    f"warm-start) must exist for any depth you submit to "
                    f"hardware."
                )

    # If hardware-only and simulation wasn't run in THIS invocation,
    # load previously-saved warm-start parameters instead of re-simulating
    # (avoids any risk of accidentally re-running simulation, and avoids
    # ambiguity about which sim run produced the warm-start actually used).
    preloaded_sim = None
    if mode == 'hardware':
        if sim_results_path is None:
            raise ValueError(
                "mode='hardware' requires --sim_results pointing to a "
                "results JSON produced by a prior --mode simulate run "
                "(warm-start parameters must come from an already-"
                "completed, inspectable simulation — never re-simulated "
                "silently inside a run that also spends QPU time)."
            )
        with open(sim_results_path) as f:
            preloaded = json.load(f)
        preloaded_sim = preloaded[0]['qaoa_sim']   # single I6 instance
        for p in hw_depths:
            if str(p) not in preloaded_sim and p not in preloaded_sim:
                raise ValueError(
                    f"No simulation result for p={p} found in "
                    f"{sim_results_path}. Run --mode simulate with that "
                    f"depth included first (free, no QPU cost)."
                )

    nz = W_full[W_full > 0]
    print(f"\nLoaded 16-node cost matrix from: {matrix_path}")
    print(f"Active edges:  {int((W_full > 0).sum() // 2)}/120 possible")
    print(f"Weight range:  {nz.min():.0f}–{nz.max():.0f} tokens")
    print(f"CV:            {nz.std()/nz.mean():.3f}")
    if mode in ('hardware',):
        print(f"HARDWARE depths this run: {hw_depths}  "
              f"(warm-start loaded from {sim_results_path})")

    sim_backend = AerSimulator(method='statevector')
    hw_backend  = None

    if mode in ('hardware', 'both'):
        if ibm_token is None:
            raise ValueError("Pass --token YOUR_IBM_TOKEN for hardware mode.")
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(
            channel='ibm_quantum_platform', token=ibm_token)

        # I1-I5 were all measured on ibm_marrakesh (Heron r2, 156 qubits).
        # For I6 to be comparable, it MUST run on the same device — NOT
        # whichever backend happens to be least busy. Explicit name lookup,
        # hard failure (not silent fallback) if Marrakesh isn't available.
        try:
            hw_backend = service.backend(hardware_name)
        except Exception as e:
            raise RuntimeError(
                f"Could not get backend '{hardware_name}': {e}\n"
                f"I6 must run on the same device as I1-I5 ({hardware_name}) "
                f"for the hardware comparison to be valid. Refusing to "
                f"silently fall back to a different backend (e.g. via "
                f"least_busy()) — check device availability/status on "
                f"your IBM Quantum dashboard before retrying."
            )

        status = hw_backend.status()
        if not status.operational:
            raise RuntimeError(
                f"{hardware_name} is not currently operational "
                f"(status: {status.status_msg}). Not falling back to a "
                f"different backend, since that would break comparability "
                f"with I1-I5's hardware results. Check the IBM Quantum "
                f"dashboard and retry once it's back up."
            )

        print(f"Hardware backend: {hw_backend.name}  "
              f"(pinned — required to match I1-I5's device)")
        print(f"  Status: {status.status_msg}, "
              f"pending jobs: {status.pending_jobs}")

    results = []

    for inst_idx, (adjacency, inst_name) in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"Instance {inst_idx+1}/{len(instances)}: {inst_name}")
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

            if p not in hw_depths:
                if mode in ('hardware', 'both'):
                    print(f"    Hardware... SKIPPED (p={p} not in "
                          f"--hw_depths={hw_depths}, preserving QPU budget)")
                continue

            if mode in ('hardware', 'both') and hw_backend:
                print(f"    Hardware (QPU time will be spent now)...",
                      end=' ', flush=True)
                if mode == 'hardware':
                    # hardware-only run: warm-start comes from the
                    # already-completed, on-disk simulation results —
                    # never re-simulated inside a QPU-spending run.
                    sim_entry = preloaded_sim.get(str(p), preloaded_sim.get(p))
                    opt_g_hw = sim_entry['opt_gamma']
                    opt_b_hw = sim_entry['opt_beta']
                    print(f"[warm-start from {sim_results_path}, "
                          f"sim r={sim_entry['approximation_ratio']:.3f}]",
                          end=' ', flush=True)
                else:
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

    short_names = ['I6: Topology-shifted']
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
    plt.savefig(f'{output_dir}/fig_I6_n16_approx_ratio.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved: fig_I6_n16_approx_ratio.png")

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
            # qaoa_hw[depth] is a LIST of run-dicts (post-merge-fix), and
            # keys may be str (post-merge) or int (same-session, unmerged).
            # Flatten all runs across all depth-key representations.
            hr = []
            for r in results:
                for key in (p, str(p)):
                    entry = r['qaoa_hw'].get(key)
                    if entry is None:
                        continue
                    runs = entry if isinstance(entry, list) else [entry]
                    hr.extend(run['approximation_ratio'] for run in runs)
                    break  # don't double-count if both key forms present
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
        plt.savefig(f'{output_dir}/fig_I6_n16_sim_vs_hw.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved: fig_I6_n16_sim_vs_hw.png")

    # Summary table
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY — n=16")
    print(f"{'='*60}")
    print(f"{'Instance':<25} {'Optimal':>10} {'Greedy':>8} "
          + "  ".join([f"Sim p={p}" for p in p_values])
          + "  "
          + "  ".join([f"HW p={p}" for p in p_values]))
    print('-' * 80)

    def hw_mean_str(qaoa_hw, p):
        # qaoa_hw[depth] may be a list of run-dicts (post-merge) or a
        # single dict (same-session, not yet merged/saved). Keys may be
        # str or int depending on whether this came from a fresh run or
        # a loaded/merged JSON file.
        entry = qaoa_hw.get(p, qaoa_hw.get(str(p)))
        if entry is None:
            return "    nan"
        runs = entry if isinstance(entry, list) else [entry]
        ratios = [run['approximation_ratio'] for run in runs]
        if len(ratios) > 1:
            return f"{np.mean(ratios):>6.3f}(n={len(ratios)})"
        return f"{ratios[0]:>7.3f}"

    for r in results:
        sim_vals = " ".join([
            f"{r['qaoa_sim'].get(p, r['qaoa_sim'].get(str(p), {})).get('approximation_ratio', float('nan')):>8.3f}"
            for p in p_values])
        hw_vals = " ".join([hw_mean_str(r['qaoa_hw'], p) for p in p_values])
        print(f"{r['instance']:<25} {r['optimal_cut']:>10.0f} "
              f"{r['greedy_ratio']:>8.3f}  {sim_vals}  {hw_vals}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='QAOA n=16 MCP Benchmark — Instance I6')
    parser.add_argument('--mode',
                        choices=['simulate', 'hardware', 'both'],
                        default='simulate')
    parser.add_argument('--token',  type=str, default=None)
    parser.add_argument('--p_max',  type=int, default=4,
                    help='Max QAOA depth to SIMULATE (1-4). Simulation is '
                         'free (no QPU cost) regardless of this value.')
    parser.add_argument('--hw_depths', type=int, nargs='+', default=None,
                    help='Which depth(s) to actually submit to hardware — '
                         'independent of --p_max. THIS is what controls '
                         'QPU spend. Default: same as simulated depths '
                         '(dangerous for a tight QPU budget — set '
                         'explicitly, e.g. --hw_depths 2).')
    parser.add_argument('--sim_results', type=str, default=None,
                    help="Path to a prior --mode simulate results JSON. "
                         "REQUIRED for --mode hardware, so the warm-start "
                         "params used are the ones you already reviewed, "
                         "not silently re-simulated during a QPU-spending "
                         "run.")
    parser.add_argument('--max_iter', type=int, default=200,
                    help='COBYLA max iterations')
    parser.add_argument('--shots',  type=int, default=8192)
    parser.add_argument('--output', type=str, default='.')
    parser.add_argument('--backend', type=str, default='ibm_marrakesh',
                    help='Hardware backend name — MUST match I1-I5 '
                         '(ibm_marrakesh) for a valid comparison. Only '
                         'override this if you have a specific reason to.')
    parser.add_argument('--matrix', type=str,
                        default='mcp_agent_data_16_I6_cost_matrix.csv')
    args = parser.parse_args()

    p_values = list(range(1, args.p_max + 1))

    print(f"\n{'='*60}")
    print(f"QAOA n=16 MCP Agent Coordination Benchmark — INSTANCE I6")
    print(f"Mode:   {args.mode}")
    print(f"Sim depths:  p=1..{args.p_max}")
    print(f"HW depths:   {args.hw_depths if args.hw_depths else '(same as sim depths — set --hw_depths explicitly to limit QPU spend)'}")
    print(f"Shots:  {args.shots}")
    print(f"Backend: {args.backend}")
    print(f"Matrix: {args.matrix}")
    print(f"{'='*60}")

    if args.mode in ('hardware', 'both') and not args.hw_depths:
        print("\n*** WARNING: --hw_depths not set — this will submit "
              f"ALL of p={p_values} to hardware, not just your priority "
              "depth. Re-run with e.g. --hw_depths 2 to control QPU spend. ***\n")

    results = run_experiment(
        mode=args.mode,
        ibm_token=args.token,
        p_values=p_values,
        shots=args.shots,
        matrix_path=args.matrix,
        max_iter=args.max_iter,
        hw_depths=args.hw_depths,
        sim_results_path=args.sim_results,
        hardware_name=args.backend,
    )

    out_json = f'{args.output}/results_I6_n16_{args.mode}.json'

    if args.mode == 'hardware' and os.path.exists(out_json):
        # CRITICAL: merge with any existing hardware results rather than
        # overwriting. Each --hw_depths invocation only populates the
        # depths run THIS time — without merging, a prior run's results
        # (e.g. p=4 from an earlier invocation) get silently wiped out.
        with open(out_json) as f:
            existing = json.load(f)
        existing_hw = existing[0].get('qaoa_hw', {})
        new_hw = results[0].get('qaoa_hw', {})
        existing_hw = {str(k): v for k, v in existing_hw.items()}
        new_hw = {str(k): v for k, v in new_hw.items()}

        def as_run_list(entry):
            # Backward-compatible: older saved files stored a single
            # dict per depth (one run). Newer files store a LIST of
            # run-dicts per depth, so repeated runs at the SAME depth
            # accumulate (n_runs > 1) instead of overwriting each other.
            if isinstance(entry, list):
                return entry
            return [entry]

        merged_hw = {}
        for depth in set(existing_hw) | set(new_hw):
            prior_runs = as_run_list(existing_hw[depth]) if depth in existing_hw else []
            fresh_runs = as_run_list(new_hw[depth]) if depth in new_hw else []
            merged_hw[depth] = prior_runs + fresh_runs

        results[0]['qaoa_hw'] = merged_hw

        print(f"\nMerged with existing {out_json}:")
        for depth in sorted(merged_hw, key=int):
            ratios = [r['approximation_ratio'] for r in merged_hw[depth]]
            n = len(ratios)
            mean_r = sum(ratios) / n
            std_r = (sum((x - mean_r) ** 2 for x in ratios) / n) ** 0.5 if n > 1 else 0.0
            print(f"  p={depth}: n_runs={n}  ratios={[round(r,4) for r in ratios]}  "
                  f"mean={mean_r:.4f}  std={std_r:.4f}")

    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    plot_results(results, p_values=p_values,
                 output_dir=args.output, matrix_path=args.matrix)