"""Instance-level numbers quoted in the paper, recomputed from the shipped cost matrices.
  - Table tab:instances69 (I6-I11): off-diagonal edge count, density, CV (population), Jaccard overlap of the edge set with I1.
  - Table tab:nodes8 / tab:nodes16: mean cost of a node's incident edges (symmetrised matrix, nonzero off-diagonal entries).
  - Section 6.1: intra-server versus inter-server edge counts and mean costs; fraction of bitstrings within 1% and 5% of C*.
  - Section 5.4: heaviest edge of I7; share of I9's edges that avoid the WebSearch tools.
  - C* ranges of I1-I5 at n = 8 and n = 16.
Run:  python compute_instance_structure.py"""
import numpy as np
import pandas as pd
from common import *

N8 = os.path.join(PKG, 'n8', 'cost_matrices', 'mcp_agent_data_cost_matrix.csv')
out = {}


def sym(path):
    W = pd.read_csv(path, index_col=0).values.astype(float)
    return (W + W.T) / 2


def edges(S):
    n = S.shape[0]
    return {(i, j) for i in range(n) for j in range(i + 1, n) if S[i][j] > 0}


def cutvals(S):
    n = S.shape[0]
    K = np.arange(2 ** n, dtype=np.int64)
    cv = np.zeros(2 ** n)
    for i in range(n):
        for j in range(i + 1, n):
            if S[i][j] > 0:
                cv += S[i][j] * (((K >> i) & 1) ^ ((K >> j) & 1))
    return cv


S8 = sym(N8)
S16 = {k: sym(os.path.join(CM, 'mcp_agent_data_16_cost_matrix.csv' if k == 'I1' else 'mcp_agent_data_16_%s_cost_matrix.csv' % k))
       for k in ['I1'] + ['I%d' % i for i in range(6, 12)]}
E1 = edges(S16['I1'])

print('== Table instances69')
out['instances69'] = {}
for k in ['I6', 'I7', 'I8', 'I9', 'I10', 'I11']:
    S = S16[k]
    E = edges(S)
    w = np.array([S[i][j] for i, j in E])
    row = dict(edges=len(E), density=len(E) / 120, cv=float(w.std() / w.mean()), jaccard_vs_I1=len(E & E1) / len(E | E1))
    out['instances69'][k] = row
    print('  %-4s edges %2d density %.3f CV %.4f Jaccard vs I1 %.3f' % (k, row['edges'], row['density'], row['cv'], row['jaccard_vs_I1']))


def node_means(S):
    n = S.shape[0]
    return [float(np.mean([x for x in np.delete(S[i], i) if x > 0])) for i in range(n)]


print('== node mean cost (n=8 nodes 0-7; n=16 nodes 8-15)')
out['node_mean_cost_n8'] = [round(x) for x in node_means(S8)]
out['node_mean_cost_n16_nodes8_15'] = [round(x) for x in node_means(S16['I1'])[8:]]
print('  n=8 ', out['node_mean_cost_n8'])
print('  n=16', out['node_mean_cost_n16_nodes8_15'])

print('== server-level contrast and near-optimal bitstrings (I1)')
for label, S in (('n=8', S8), ('n=16', S16['I1'])):
    n = S.shape[0]
    intra = [S[i][j] for i in range(n) for j in range(i + 1, n) if i // 2 == j // 2 and S[i][j] > 0]
    inter = [S[i][j] for i in range(n) for j in range(i + 1, n) if i // 2 != j // 2 and S[i][j] > 0]
    pairs_intra = n // 2
    cv = cutvals(S)
    opt = cv.max()
    near = {thr: int((cv >= thr * opt).sum()) for thr in (0.99, 0.95)}
    out['structure_' + label] = dict(intra_edges=len(intra), intra_pairs=pairs_intra, intra_mean=float(np.mean(intra)), inter_edges=len(inter),
                                     inter_mean=float(np.mean(inter)), bitstrings=2 ** n, within_1pct=near[0.99], within_5pct=near[0.95])
    print('  %s intra: %d of %d pairs, mean %.0f | inter: %d edges, mean %.0f (intra %.1f%% lower) | within 1%% of C*: %d, within 5%%: %d of %d (%.2f%%)' % (
        label, len(intra), pairs_intra, np.mean(intra), len(inter), np.mean(inter), 100 * (1 - np.mean(intra) / np.mean(inter)), near[0.99], near[0.95], 2 ** n, 100 * near[0.95] / 2 ** n))

print('== I7 heaviest edge, I9 non-WebSearch share')
names = list(pd.read_csv(os.path.join(CM, 'mcp_agent_data_16_I7_cost_matrix.csv'), index_col=0).columns)
S7 = S16['I7']
top = sorted(((S7[i][j], i, j) for i, j in edges(S7)), reverse=True)[0]
out['I7_heaviest_edge'] = [names[top[1]], names[top[2]], float(top[0])]
print('  I7 heaviest edge:', out['I7_heaviest_edge'])
E9 = edges(S16['I9'])
nonweb = [(i, j) for i, j in E9 if i // 2 != 1 and j // 2 != 1]
out['I9_nonwebsearch_share'] = len(nonweb) / len(E9)
print('  I9: %d of %d edges (%.1f%%) have both endpoints outside the WebSearch tools' % (len(nonweb), len(E9), 100 * len(nonweb) / len(E9)))

print('== C* ranges (I1-I5)')
sd = json.load(open(os.path.join(RES, 'results_n16_simulate_seeded_2026-08-05.json')))
c16 = [e['optimal_cut'] for e in sd]
out['Cstar_n16_I1_I5'] = [round(c) for c in c16]
# n = 8: rebuild I1-I5 with the logic of build_instances() in n8/qaoa_experiment/qaoa_qap_experiment.py, brute force over all 2^8 assignments
n8 = 8
sc, cc = 2063 / 3516, 5920 / 3516
rng = np.random.default_rng(42)
Wl = S8.copy()
idx = [(i, j) for i in range(n8) for j in range(i + 1, n8) if S8[i][j] > 0]
vals = rng.permutation([S8[i][j] for i, j in idx])
for (i, j), v in zip(idx, vals):
    Wl[i][j] = v
    Wl[j][i] = v
rng2 = np.random.default_rng(99)
Wr = np.zeros((n8, n8))
allp = [(i, j) for i in range(n8) for j in range(i + 1, n8)]
mw, sw = S8[S8 > 0].mean(), S8[S8 > 0].std()
for c in rng2.choice(len(allp), size=len(idx), replace=False):
    i, j = allp[c]
    w = abs(rng2.normal(mw, sw))
    Wr[i][j] = w
    Wr[j][i] = w
c8 = [float(cutvals(A).max()) for A in (S8, S8 * sc, S8 * cc, Wl, Wr)]
out['Cstar_n8_I1_I5'] = [round(c) for c in c8]
print('  n=8: ', out['Cstar_n8_I1_I5'], '(I4 = %.2f)' % c8[3])
print('  n=16:', out['Cstar_n16_I1_I5'])
save('instance_structure.json', out)
