"""
Restart-noise-floor analysis for the simulation-ruggedness metric R.

NOT part of the paper -- exploratory data check only.

R = (max_p r_sim - min_p r_sim) / mean_p r_sim is computed from a SINGLE
COBYLA run per depth (fixed seed for the initial point, x0_seed=0, in
qaoa_experiment_16.optimize_qaoa). That means R can't distinguish
"this instance's landscape is genuinely rugged across depth" from
"COBYLA started from a lucky/unlucky point at one particular depth".

This script re-runs each depth p=1..4 with N_RESTARTS independent COBYLA
optimizations from different random initial points (same seed_simulator,
same shots, same max_iter as the paper's protocol -- only x0 differs).
The spread of r_sim across restarts AT FIXED p is a noise floor: how much
r_sim moves purely from where COBYLA starts, with nothing about the
instance or depth changing.

Restricted to the extended, non-ceilinged set (I1, I7, I8, I10, I11) --
the same five instances used for the R vs Delta_r_bar correlation check
in the paper (analysis/i1_i6_i11_analysis.py), since that's the
correlation whose validity depends on R actually reflecting landscape
structure.

Paths below are resolved relative to this script's own location (not
the working directory), so it can be run from anywhere; this is the
packaged copy living at n16/restart_analysis/ -- cost matrices are
read from the sibling n16/cost_matrices/ folder and
qaoa_experiment_16.py from n16/qaoa_experiment/.

Usage:
    python restart_noise_floor.py
Writes/updates restart_noise_floor.json after every (instance, depth) so
partial progress survives if interrupted. This already-completed run's
output ships alongside this script; re-running from scratch takes
roughly 6 hours (5 instances x 4 depths x 30 restarts, ~25-40s/restart).
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "qaoa_experiment"))

from qaoa_experiment_16 import (
    build_qaoa_circuit, evaluate_circuit_sim, brute_force_optimal,
)
from qiskit_aer import AerSimulator

N_QUBITS = 16
P_VALUES = [1, 2, 3, 4]
RESTART_SEEDS = list(range(1, 31))   # deliberately avoid seed 0 (paper's original x0 seed)
SHOTS = 8192
MAX_ITER = 200
SEED_SIMULATOR = 42
OUT_PATH = os.path.join(SCRIPT_DIR, "restart_noise_floor.json")

COST_MATRIX_DIR = os.path.join(SCRIPT_DIR, "..", "cost_matrices")
INSTANCE_PATHS = {
    "I1":  os.path.join(COST_MATRIX_DIR, "mcp_agent_data_16_cost_matrix.csv"),
    "I7":  os.path.join(COST_MATRIX_DIR, "mcp_agent_data_16_I7_cost_matrix.csv"),
    "I8":  os.path.join(COST_MATRIX_DIR, "mcp_agent_data_16_I8_cost_matrix.csv"),
    "I10": os.path.join(COST_MATRIX_DIR, "mcp_agent_data_16_I10_cost_matrix.csv"),
    "I11": os.path.join(COST_MATRIX_DIR, "mcp_agent_data_16_I11_cost_matrix.csv"),
}


def load_adj(path):
    df = pd.read_csv(path, index_col=0)
    W = df.values.astype(float)
    return (W + W.T) / 2


def optimize_qaoa_seeded(qc, adjacency, p, backend, shots, max_iter, seed_simulator, x0_seed):
    def objective(params):
        gamma = params[:p]
        beta = params[p:]
        exp, _, _, _ = evaluate_circuit_sim(qc, gamma, beta, adjacency, backend, shots, seed_simulator)
        return -exp

    rng = np.random.default_rng(x0_seed)
    x0 = rng.uniform(0, np.pi, 2 * p)
    result = minimize(objective, x0, method="COBYLA", options={"maxiter": max_iter, "rhobeg": 0.5})
    return result.x[:p], result.x[p:]


def load_progress():
    try:
        with open(OUT_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main():
    backend = AerSimulator(method="statevector")
    out = load_progress()

    for inst, path in INSTANCE_PATHS.items():
        adj = load_adj(path)

        if inst not in out:
            t0 = time.time()
            opt_cut, _ = brute_force_optimal(adj)
            out[inst] = {"optimal_cut": float(opt_cut), "restarts": {}}
            print(f"{inst}: optimal_cut={opt_cut:.2f} ({time.time()-t0:.1f}s)")
            with open(OUT_PATH, "w") as f:
                json.dump(out, f, indent=2)

        opt_cut = out[inst]["optimal_cut"]

        for p in P_VALUES:
            pkey = str(p)
            if pkey in out[inst]["restarts"] and len(out[inst]["restarts"][pkey]) >= len(RESTART_SEEDS):
                continue  # already done

            qc = build_qaoa_circuit(adj, p)
            ratios = out[inst]["restarts"].get(pkey, [])
            done_seeds = len(ratios)

            for seed in RESTART_SEEDS[done_seeds:]:
                t0 = time.time()
                g, b = optimize_qaoa_seeded(qc, adj, p, backend, SHOTS, MAX_ITER, SEED_SIMULATOR, seed)
                _, best, _, _ = evaluate_circuit_sim(qc, g, b, adj, backend, SHOTS, SEED_SIMULATOR)
                r = best / opt_cut if opt_cut > 0 else 0.0
                ratios.append(float(r))
                print(f"{inst} p={p} x0_seed={seed} r_sim={r:.4f}  ({time.time()-t0:.1f}s)", flush=True)

                out[inst]["restarts"][pkey] = ratios
                with open(OUT_PATH, "w") as f:
                    json.dump(out, f, indent=2)

    print("\nDone ->", OUT_PATH)


if __name__ == "__main__":
    main()
