"""Uniform-random-sampler reference for I1-I11 (exact, no sampling).
For each instance: U = E[C]/C* of a uniform sampler, its standard deviation, the number of optimal
bitstrings, P(opt), and the exact expected best-of-8192 ratio (the paper's metric r for a uniform sampler)."""
from common import *

inst = get_instances()
U = np.ones(2 ** N) / 2 ** N
out = {}
print('inst | #opt | U = E[C]/C* (SD) | P(opt) | E[best of 8192] | P(best = 1)')
for k in NAMES:
    cv = cut_values(inst[k])
    opt = cv.max()
    nopt = int((np.abs(cv - opt) < 1e-6).sum())
    r = cv / opt
    eb = expected_best_of_k(cv, U) / opt
    out[k] = dict(U=float(r.mean()), sd=float(r.std()), n_opt=nopt, P_opt=nopt / 2 ** N,
                  E_best_of_8192=eb, P_best_equals_1=1 - (1 - nopt / 2 ** N) ** SHOTS)
    print('%-4s | %d | %.4f (%.4f) | %.2e | %.4f | %.3f' % (k, nopt, out[k]['U'], out[k]['sd'], out[k]['P_opt'], eb, out[k]['P_best_equals_1']))
save('uniform_baseline.json', out)
