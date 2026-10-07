import sys, json, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'baseline_analysis'))
from common import *
from xcheck_common import probs_by_index
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

OUT = os.path.dirname(os.path.abspath(__file__))
INSTS = ['I1', 'I7', 'I8', 'I10', 'I11']
_inst = None
_cv = {}

def _init():
    global _inst, _cv
    if _inst is None:
        _inst = get_instances()

def getA(k):
    _init()
    A = _inst[k]
    return A / A.max()          # max-weight normalisation (scale only)

def getcv(k):
    if k not in _cv:
        _cv[k] = cut_values(getA(k))     # ratio-invariant: same optimum partition
    return _cv[k]

def probs(A, x, p):
    return probs_by_index(A, x[:p], x[p:])   # x[:p] = cost angles gamma, x[p:] = mixer angles beta (correct binding)

def run_task(args):
    k, p, protocol, start = args
    A = getA(k); cv = getcv(k); opt = cv.max()
    kopt = np.abs(cv - opt) < 1e-9
    if start == 0:
        x0 = np.random.default_rng(0).uniform(0, np.pi, 2 * p)      # exactly the paper's x0 rule
    else:
        x0 = np.random.default_rng(1000 + start).uniform(0, np.pi, 2 * p)
    if protocol == 'shot':
        def f(x):
            pr = probs(A, x, p); pr = pr / pr.sum()
            counts = np.random.default_rng(42).multinomial(SHOTS, pr)    # seed, as seed_simulator=42
            return -(counts * cv).sum() / SHOTS / opt
    else:
        def f(x):
            pr = probs(A, x, p)
            return -(cv * pr).sum() / opt
    r = minimize(f, x0, method='COBYLA', options={'maxiter': 200, 'rhobeg': 0.5})
    pr = probs(A, r.x, p); pr = pr / pr.sum()
    return dict(inst=k, p=p, protocol=protocol, start=start, obj=float(-r.fun),
                meanr=float((cv * pr).sum() / opt), popt=float(pr[kopt].sum()),
                eb=float(expected_best_of_k(cv, pr) / opt), x=[float(v) for v in r.x])

if __name__ == '__main__':
    tasks = []
    for k in INSTS:
        for p in [1, 2, 3, 4]:
            tasks.append((k, p, 'shot', 0))       # protocol A (faithful): paper objective, one start
            tasks.append((k, p, 'exact', 0))      # protocol A with exact objective, same start
            for s in range(1, 7):
                tasks.append((k, p, 'exact', s))  # protocol B: extra starts
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8, initializer=_init) as ex:
        res = list(ex.map(run_task, tasks, chunksize=2))
    print('done %d tasks in %.0fs' % (len(res), time.time() - t0))
    json.dump(res, open(os.path.join(OUT, 'norm_study.json'), 'w'))
