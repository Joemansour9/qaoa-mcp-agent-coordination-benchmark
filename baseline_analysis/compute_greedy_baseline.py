"""Classical greedy baselines for I1-I11.
(a) single-pass greedy: the paper's deterministic greedy (nodes visited in index order).
(b) randomized greedy: the same greedy rule with a random visiting order, restarted 8192 times
    (the same budget as the 8192 QAOA shots); reports the best and mean cut ratio over restarts."""
from common import *

inst = get_instances()
RESTARTS = SHOTS


def greedy_in_order(A, order):
    a = -np.ones(N, dtype=int)
    a[order[0]] = 0
    for t in range(1, N):
        i = order[t]
        c0 = sum(A[i][j] for j in order[:t] if a[j] != 0)
        c1 = sum(A[i][j] for j in order[:t] if a[j] != 1)
        a[i] = 1 if c1 > c0 else 0
    return sum(A[i][j] for i in range(N) for j in range(i + 1, N) if a[i] != a[j])


out = {}
print('inst | single-pass greedy r | randomized greedy mean r | best of %d restarts' % RESTARTS)
for n, k in enumerate(NAMES):
    A = inst[k]
    opt = cut_values(A).max()
    single = greedy_in_order(A, list(range(N))) / opt
    rng = np.random.default_rng(1000 + n)           # seed per instance
    vals = np.array([greedy_in_order(A, rng.permutation(N)) / opt for _ in range(RESTARTS)])
    out[k] = dict(single_pass=float(single), randomized_mean=float(vals.mean()), randomized_best=float(vals.max()),
                  restarts=RESTARTS, seed=1000 + n)
    print('%-4s | %.4f | %.4f | %.4f' % (k, single, vals.mean(), vals.max()))
save('greedy_baseline.json', out)
