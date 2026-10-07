"""Shared helpers for the baseline analyses. Paths are relative to the package root.

I1 is the shipped base matrix; I2-I5 are rebuilt with the exact logic of build_instances() in
n16/qaoa_experiment/qaoa_experiment_16.py (I2, I3 scaled; I4 permuted weights; I5 random edge
positions with matched edge count and weight mean/std). I6-I11 are the shipped matrices."""
import os, json
import numpy as np
import pandas as pd

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CM = os.path.join(PKG, 'n16', 'cost_matrices')
RES = os.path.join(PKG, 'n16', 'results')
OUT = os.path.dirname(os.path.abspath(__file__))
N = 16
SHOTS = 8192
NAMES = ['I%d' % i for i in range(1, 12)]


def load_sym(path):
    W = pd.read_csv(path, index_col=0).values.astype(float)
    return (W + W.T) / 2


def build_I1_I5(W):
    inst = {'I1': W.copy(), 'I2': W * (2063.0 / 3516.0), 'I3': W * (5920.0 / 3516.0)}
    rng = np.random.default_rng(42)
    Wl = W.copy()
    idx = [(i, j) for i in range(N) for j in range(i + 1, N) if W[i][j] > 0]
    vals = rng.permutation([W[i][j] for i, j in idx])
    for (i, j), v in zip(idx, vals):
        Wl[i][j] = v
        Wl[j][i] = v
    inst['I4'] = Wl
    rng2 = np.random.default_rng(99)
    Wr = np.zeros((N, N))
    allp = [(i, j) for i in range(N) for j in range(i + 1, N)]
    mw, sw = W[W > 0].mean(), W[W > 0].std()
    for c in rng2.choice(len(allp), size=len(idx), replace=False):
        i, j = allp[c]
        w = abs(rng2.normal(mw, sw))
        Wr[i][j] = w
        Wr[j][i] = w
    inst['I5'] = Wr
    return inst


def get_instances():
    inst = build_I1_I5(load_sym(os.path.join(CM, 'mcp_agent_data_16_cost_matrix.csv')))
    for k in NAMES[5:]:
        inst[k] = load_sym(os.path.join(CM, 'mcp_agent_data_16_%s_cost_matrix.csv' % k))
    return inst


_K = np.arange(2 ** N, dtype=np.int64)
BITS = [((_K >> i) & 1).astype(np.int8) for i in range(N)]


def cut_values(A):
    cv = np.zeros(2 ** N)
    for i in range(N):
        for j in range(i + 1, N):
            if A[i][j] > 0:
                cv += A[i][j] * (BITS[i] ^ BITS[j])
    return cv


def expected_best_of_k(cv, prob, k=SHOTS):
    """Exact E[max cut over k i.i.d. draws] from a distribution over all 2^N bitstrings."""
    uniq, inv = np.unique(np.round(cv, 9), return_inverse=True)
    pm = np.bincount(inv, weights=prob)
    cdfk = np.minimum(np.cumsum(pm), 1.0) ** k
    return float((uniq * np.diff(np.concatenate([[0.0], cdfk]))).sum())


def save(name, obj):
    path = os.path.join(OUT, name)
    json.dump(obj, open(path, 'w'), indent=2)
    print('wrote', path)
