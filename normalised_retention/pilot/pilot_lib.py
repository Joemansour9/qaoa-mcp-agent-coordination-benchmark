"""Shared helpers for the normalised-weight hardware pilot (I1, p=1 and 2).
Parameters are bound BY NAME (gamma = cost angle, beta = mixer angle), never by position."""
import os, json, hashlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), 'reproducibility_package')
N = 16
SHOTS = 8192
INSTANCE = 'I1'
DEPTHS = [1, 2]
REPEATS = [1, 2, 3]
TAG_PREFIX = 'paper3-normpilot'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def load_normalised_I1():
    W = pd.read_csv(os.path.join(PKG, 'n16', 'cost_matrices', 'mcp_agent_data_16_cost_matrix.csv'), index_col=0).values.astype(float)
    W = (W + W.T) / 2
    return W / W.max()            # max-weight normalisation; ratios are scale invariant

_K = np.arange(2 ** N, dtype=np.int64)
BITS = [((_K >> i) & 1).astype(np.int8) for i in range(N)]

def cut_values(A):
    cv = np.zeros(2 ** N)
    for i in range(N):
        for j in range(i + 1, N):
            if A[i][j] > 0:
                cv += A[i][j] * (BITS[i] ^ BITS[j])
    return cv

def build_circuit(A, p):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    g = ParameterVector('gamma', p)
    b = ParameterVector('beta', p)
    qc = QuantumCircuit(N)
    qc.h(range(N))
    for l in range(p):
        for i in range(N):
            for j in range(i + 1, N):
                if A[i][j] > 0:
                    qc.rzz(2 * g[l] * A[i][j], i, j)
        for i in range(N):
            qc.rx(2 * b[l], i)
    qc.measure_all()
    return qc, g, b

def bind(qc, g, b, gam, bet):
    d = {}
    for k in range(len(gam)):
        d[g[k]] = float(gam[k])
        d[b[k]] = float(bet[k])
    return qc.assign_parameters(d)

def expected_best_of_k(cv, prob, k=SHOTS):
    uniq, inv = np.unique(np.round(cv, 9), return_inverse=True)
    pm = np.bincount(inv, weights=prob)
    cdf = np.minimum(np.cumsum(pm), 1.0) ** k
    return float((uniq * np.diff(np.concatenate([[0.0], cdf]))).sum())

def counts_to_vec(counts):
    c = np.zeros(2 ** N)
    for bs, cnt in counts.items():
        c[int(bs, 2)] += cnt
    return c
