"""
Standalone CP-SAT Max-Cut benchmark for I1-I11 (n=16).
No dependency on any other pipeline script -- reads cost matrices
directly from ../cost_matrices/, computes brute-force C* fresh (for
cross-check), then solves each instance with OR-Tools CP-SAT.

Self-contained: all paths are resolved relative to this script's own
location, so it can be run from any working directory as long as the
reproducibility_package/n16/ folder structure is intact.

Source for the n=16 Classical Baselines CP-SAT results reported in
Section 4.3 of the paper.
"""
import os
import time
import json
import csv
import ctypes
import importlib.util

# Workaround for a packaging bug in ortools 9.15.6755's Windows DLL loader:
# ortools/__init__.py looks for "utf8_validity.dll" but the wheel ships
# "libutf8_validity.dll" (lib-prefixed) -- silently skipped, which breaks a
# later DLL's dependency resolution ("WinError 127: procedure not found").
# On top of that, numpy/pandas load their own (incompatible) copy of some
# same-named shared dependency, which shadows ortools' if imported first.
# Fix: pre-load the correctly-named DLL, then import ortools/cp_model
# BEFORE numpy/pandas, so ortools' own DLLs win the process-wide binding.
_spec = importlib.util.find_spec('ortools')
_ortools_dir = os.path.dirname(_spec.origin)
_real_utf8_dll = os.path.join(_ortools_dir, '.libs', 'libutf8_validity.dll')
if os.path.exists(_real_utf8_dll):
    ctypes.WinDLL(_real_utf8_dll)

from ortools.sat.python import cp_model

import numpy as np
import pandas as pd

N_QUBITS = 16

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COST_MATRIX_DIR = os.path.join(SCRIPT_DIR, '..', 'cost_matrices')


def cost_matrix_path(filename):
    return os.path.join(COST_MATRIX_DIR, filename)


def load_and_symmetrize(path):
    df = pd.read_csv(path, index_col=0)
    W = df.values.astype(float)
    return (W + W.T) / 2

def build_adjacency_i1_i5(instance_key, W):
    if instance_key == 'I1':
        return W.copy()
    if instance_key == 'I2':
        return W * (2063.0 / 3516.0)
    if instance_key == 'I3':
        return W * (5920.0 / 3516.0)
    if instance_key == 'I4':
        rng = np.random.default_rng(42)
        W_lat = W.copy()
        idx = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS) if W[i][j] > 0]
        vals = rng.permutation([W[i][j] for i, j in idx])
        for (i, j), v in zip(idx, vals):
            W_lat[i][j] = v
            W_lat[j][i] = v
        return W_lat
    if instance_key == 'I5':
        rng2 = np.random.default_rng(99)
        W_rand = np.zeros((N_QUBITS, N_QUBITS))
        idx = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS) if W[i][j] > 0]
        all_pairs = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]
        n_edges = len(idx)
        mean_w = W[W > 0].mean(); std_w = W[W > 0].std()
        chosen = rng2.choice(len(all_pairs), size=n_edges, replace=False)
        for c in chosen:
            i, j = all_pairs[c]
            w = abs(rng2.normal(mean_w, std_w))
            W_rand[i][j] = w
            W_rand[j][i] = w
        return W_rand

def compute_cut_value(bits, adjacency):
    n = len(bits)
    cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if bits[i] != bits[j]:
                cut += adjacency[i][j]
    return cut

def brute_force_optimal(adjacency):
    n = adjacency.shape[0]
    best_cut = 0.0
    for k in range(1, 2 ** n // 2):
        bits = [(k >> i) & 1 for i in range(n)]
        cut = compute_cut_value(bits, adjacency)
        if cut > best_cut:
            best_cut = cut
    return best_cut

def edges_from_adjacency(adjacency):
    n = adjacency.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = adjacency[i][j]
            if w > 0:
                edges.append((i, j, w))
    return edges

def solve_cpsat(adjacency, time_limit_s=10.0):
    n = adjacency.shape[0]
    edges = edges_from_adjacency(adjacency)

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f'x_{i}') for i in range(n)]
    c = []
    for (i, j, w) in edges:
        cij = model.NewBoolVar(f'c_{i}_{j}')
        # channeling: cij == (x_i != x_j)
        model.Add(x[i] != x[j]).OnlyEnforceIf(cij)
        model.Add(x[i] == x[j]).OnlyEnforceIf(cij.Not())
        c.append(cij)

    # objective uses integer coefficients -- scale weights to preserve precision
    SCALE = 10**6
    model.Maximize(sum(int(round(w * SCALE)) * cij for (i, j, w), cij in zip(edges, c)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8

    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    status_name = solver.StatusName(status)
    obj_value = solver.ObjectiveValue() / SCALE if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    return obj_value, status_name, elapsed

# ---- build all 11 instances ----
W_base = load_and_symmetrize(cost_matrix_path('mcp_agent_data_16_cost_matrix.csv'))
adjacency_by_instance = {}
for inst in ['I1', 'I2', 'I3', 'I4', 'I5']:
    adjacency_by_instance[inst] = build_adjacency_i1_i5(inst, W_base)
for inst in ['I6', 'I7', 'I8', 'I9', 'I10', 'I11']:
    adjacency_by_instance[inst] = load_and_symmetrize(
        cost_matrix_path(f'mcp_agent_data_16_{inst}_cost_matrix.csv'))

instances_order = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11']

print(f"{'Instance':<10}{'C* (brute force)':>20}{'CP-SAT value':>18}{'r':>10}{'Solve time (s)':>16}{'Status':>12}")
print('-' * 86)

rows = []
for inst in instances_order:
    adj = adjacency_by_instance[inst]
    c_star = brute_force_optimal(adj)
    cpsat_val, status, elapsed = solve_cpsat(adj, time_limit_s=10.0)
    r = cpsat_val / c_star if (cpsat_val is not None and c_star > 0) else None
    rows.append((inst, c_star, cpsat_val, r, elapsed, status))
    r_str = f"{r:.6f}" if r is not None else "N/A"
    val_str = f"{cpsat_val:.4f}" if cpsat_val is not None else "N/A"
    print(f"{inst:<10}{c_star:>20.4f}{val_str:>18}{r_str:>10}{elapsed:>16.4f}{status:>12}")

print()
print("Copy-paste table (instance, C*, CP-SAT, r, solve_time_s, status):")
for inst, c_star, cpsat_val, r, elapsed, status in rows:
    print(f"{inst}\t{c_star:.4f}\t{cpsat_val:.4f}\t{r:.6f}\t{elapsed:.4f}\t{status}")

# ---- save results ----
csv_path = os.path.join(SCRIPT_DIR, 'cpsat_results.csv')
json_path = os.path.join(SCRIPT_DIR, 'cpsat_results.json')

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['instance', 'c_star', 'cpsat_value', 'r', 'solve_time_s', 'status'])
    for inst, c_star, cpsat_val, r, elapsed, status in rows:
        writer.writerow([inst, f'{c_star:.6f}', f'{cpsat_val:.6f}', f'{r:.6f}', f'{elapsed:.6f}', status])

json_rows = [
    {
        'instance': inst,
        'c_star': c_star,
        'cpsat_value': cpsat_val,
        'r': r,
        'solve_time_s': elapsed,
        'status': status,
    }
    for inst, c_star, cpsat_val, r, elapsed, status in rows
]
with open(json_path, 'w') as f:
    json.dump(json_rows, f, indent=2)

print()
print(f"Saved -> {csv_path}")
print(f"Saved -> {json_path}")
