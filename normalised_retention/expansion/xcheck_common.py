import numpy as np, pandas as pd, json, gzip, os

PKG = r'C:\Users\381487616\Downloads\project4\reproducibility_package'
ROOT = r'C:\Users\381487616\Downloads\project4'
CM = os.path.join(PKG, 'n16', 'cost_matrices')
RES = os.path.join(PKG, 'n16', 'results')
N = 16
SHOTS = 8192


def load_sym(path):
    W = pd.read_csv(path, index_col=0).values.astype(float)
    return (W + W.T) / 2


def build_I1_I5(W):
    # exact copy of build_instances() in qaoa_experiment_16.py
    inst = {}
    inst['I1'] = W.copy()
    inst['I2'] = W * (2063.0 / 3516.0)
    inst['I3'] = W * (5920.0 / 3516.0)
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
    ne = len(idx)
    mw = W[W > 0].mean()
    sw = W[W > 0].std()
    ch = rng2.choice(len(allp), size=ne, replace=False)
    for c in ch:
        i, j = allp[c]
        w = abs(rng2.normal(mw, sw))
        Wr[i][j] = w
        Wr[j][i] = w
    inst['I5'] = Wr
    return inst


def get_instances():
    W1 = load_sym(os.path.join(CM, 'mcp_agent_data_16_cost_matrix.csv'))
    inst = build_I1_I5(W1)
    for k in ['I6', 'I7', 'I8', 'I9', 'I10', 'I11']:
        inst[k] = load_sym(os.path.join(CM, f'mcp_agent_data_16_{k}_cost_matrix.csv'))
    return inst


_K = np.arange(2 ** N, dtype=np.int64)
BITS = [((_K >> i) & 1).astype(np.int8) for i in range(N)]


def cut_values(A):
    cv = np.zeros(2 ** N)
    for i in range(N):
        for j in range(i + 1, N):
            w = A[i][j]
            if w > 0:
                cv += w * (BITS[i] ^ BITS[j])
    return cv


def greedy_ratio(A, opt):
    # exact copy of greedy_max_cut()
    a = np.zeros(N, dtype=int)
    for i in range(1, N):
        c0 = sum(A[i][j] for j in range(i) if a[j] != 0)
        c1 = sum(A[i][j] for j in range(i) if a[j] != 1)
        a[i] = 1 if c1 > c0 else 0
    k = int(sum(int(a[i]) << i for i in range(N)))
    return cut_values_at(A, a) / opt


def cut_values_at(A, a):
    cut = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if a[i] != a[j]:
                cut += A[i][j]
    return cut


def expected_best_of_k(cv, prob, k=SHOTS):
    """exact E[max cut over k iid draws from distribution prob over states] """
    order = np.argsort(cv)
    cs = cv[order]
    ps = prob[order]
    # aggregate equal cut values
    uniq, inv = np.unique(np.round(cs, 9), return_inverse=True)
    pm = np.bincount(inv, weights=ps)
    cdf = np.cumsum(pm)
    cdf = np.minimum(cdf, 1.0)
    cdfk = cdf ** k
    pmf = np.diff(np.concatenate([[0.0], cdfk]))
    return float((uniq * pmf).sum())


def qaoa_state(A, gam, bet):
    n = N
    cvv = cut_values(A)  # cut value; H_C phase uses sum w*Z_i Z_j = sum w - 2*cut
    p = len(gam)
    psi = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
    W_total = sum(A[i][j] for i in range(n) for j in range(i + 1, n) if A[i][j] > 0)
    zz = W_total - 2 * cvv  # sum_{edges} w * z_i z_j with z=+1 for bit 0
    for layer in range(p):
        psi = psi * np.exp(-1j * gam[layer] * zz)  # rzz(2*gamma*w) = exp(-i*gamma*w*ZZ)
        t = psi.reshape([2] * n)
        c, s = np.cos(bet[layer]), np.sin(bet[layer])
        for ax in range(n):
            a0 = np.take(t, 0, axis=ax)
            a1 = np.take(t, 1, axis=ax)
            n0 = c * a0 - 1j * s * a1
            n1 = -1j * s * a0 + c * a1
            t = np.stack([n0, n1], axis=ax)
        psi = t.reshape(-1)
    return np.abs(psi) ** 2  # in 'tensor axis' order, need reorder to qubit-index order


def probs_by_index(A, gam, bet):
    # reshape([2]*n) puts qubit axis 0 = most significant index bit.
    # Our state index k has qubit i = bit i (LSB = qubit 0). The tensor view axis 0 corresponds to
    # index bit (n-1). Therefore axis ax corresponds to qubit (n-1-ax). Mixer is identical on all
    # qubits and phase is applied in index space, so the evolution is consistent with index space.
    return qaoa_state(A, gam, bet)
