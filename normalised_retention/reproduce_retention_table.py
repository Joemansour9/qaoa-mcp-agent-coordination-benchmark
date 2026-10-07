"""Recompute the retention table (paper Table tab:retention) from the raw hardware counts in the manifests.
Needs only numpy and pandas (no IBM access). Reads:
  pilot/manifest_pilot.json, pilot/reference.json            (I1 pilot)
  expansion/manifest_expansion.json, expansion/reference.json (I7, I8, I10, I11)
  ../n16/cost_matrices/                                        (cost matrices, max-normalised here)
and checks the recomputed retention against the frozen analysis outputs
(pilot/pilot_results.json, expansion/expansion_results.json). Prints the table rows."""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CM = os.path.join(os.path.dirname(HERE), 'n16', 'cost_matrices')
N, SHOTS = 16, 8192
_K = np.arange(2 ** N, dtype=np.int64)
BITS = [((_K >> i) & 1).astype(np.int8) for i in range(N)]


def normalised(inst):
    f = 'mcp_agent_data_16_cost_matrix.csv' if inst == 'I1' else 'mcp_agent_data_16_%s_cost_matrix.csv' % inst
    W = pd.read_csv(os.path.join(CM, f), index_col=0).values.astype(float)
    W = (W + W.T) / 2
    return W / W.max()


def cut_values(A):
    cv = np.zeros(2 ** N)
    for i in range(N):
        for j in range(i + 1, N):
            if A[i][j] > 0:
                cv += A[i][j] * (BITS[i] ^ BITS[j])
    return cv


def vec(counts):
    c = np.zeros(2 ** N)
    for bs, n in counts.items():
        c[int(bs, 2)] += n
    return c


def cell(jobs, cv, U, ideal):
    opt = cv.max()
    optmask = np.abs(cv - opt) < 1e-9
    vs = [vec(j['result_counts']) for j in jobs]
    assert all(int(v.sum()) == SHOTS for v in vs) and len(vs) == 3
    e = float(np.mean([(cv * v).sum() / v.sum() / opt for v in vs]))
    hits = int(sum(v[optmask].sum() for v in vs))
    return e, (e - U) / (ideal - U), hits


rows, worst = [], 0.0
studies = [('pilot', 'manifest_pilot.json', 'pilot_results.json', ['I1']),
           ('expansion', 'manifest_expansion.json', 'expansion_results.json', ['I7', 'I8', 'I10', 'I11'])]
for sub, man, res, insts in studies:
    M = json.load(open(os.path.join(HERE, sub, man)))
    R = json.load(open(os.path.join(HERE, sub, 'reference.json')))
    F = json.load(open(os.path.join(HERE, sub, res)))
    for inst in insts:
        cv = cut_values(normalised(inst))
        ref = R if sub == 'pilot' else R['instances'][inst]
        U = ref['uniform']['U'] if sub == 'pilot' else ref['U']
        PU = ref['uniform']['P_opt'] if sub == 'pilot' else ref['P_opt_uniform']
        for p in (1, 2):
            d = ref['depths'][str(p)]
            jobs = [j for j in M['jobs'] if j['depth'] == p and j.get('instance', 'I1') == inst]
            e, rho, hits = cell(jobs, cv, U, d['E_C_over_Cstar'])
            frozen = (F['results'][str(p)] if sub == 'pilot' else F['cells'][inst][str(p)])['retention']
            worst = max(worst, abs(rho - frozen))
            gates = (M['transpiled'][str(p)] if sub == 'pilot' else M['transpiled']['%s_p%d' % (inst, p)])['two_qubit_gates']
            tot = SHOTS * 3
            rows.append('%-4s %d | U %.3f ideal %.3f HW %.3f | rho %.2f | hits %2d | P(opt) HW/ideal %.0fx / %.0fx | CZ %d' % (
                inst, p, U, d['E_C_over_Cstar'], e, rho, hits, hits / tot / PU, d['P_opt'] / PU, gates))
print('\n'.join(rows))
print('max |recomputed rho - frozen analysis rho| = %.2e' % worst)
assert worst < 1e-9, 'retention does not match the frozen analysis'
print('OK: recomputed retention matches the frozen analysis outputs')
