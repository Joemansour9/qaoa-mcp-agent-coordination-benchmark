"""Compute and save the simulator reference for I7, I8, I10, I11 BEFORE any hardware submission.
Run once. Writes reference.json (its sha256 is recorded in PREREGISTRATION.md)."""
import json, datetime
from exp_lib import *
from qiskit.quantum_info import Statevector
from xcheck_common import probs_by_index      # independent NumPy simulator, cross-check only

STUDY = os.path.join(HERE, 'params_source_norm_study.json')
res = json.load(open(STUDY))
unif = np.ones(2 ** N) / 2 ** N
ref = dict(
    created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    weights='max-normalised (w / max w); cut ratios are scale invariant',
    params_source=dict(file='params_source_norm_study.json', sha256=sha256(STUDY),
                       selection='per (instance, depth): best exact-objective value among 7 COBYLA starts (maxiter 200, rhobeg 0.5), x[:p]=gamma (cost), x[p:]=beta (mixer)'),
    shots_per_job=SHOTS, jobs_per_cell=len(REPEATS), instances={})
for inst in INSTANCES:
    A = load_normalised(inst)
    cv = cut_values(A)
    opt = cv.max()
    optmask = np.abs(cv - opt) < 1e-9
    U = float((cv / opt).mean()); sd = float((cv / opt).std())
    d = dict(U=U, sd=sd, n_opt=int(optmask.sum()), P_opt_uniform=int(optmask.sum()) / 2 ** N,
             E_best8192_uniform=expected_best_of_k(cv, unif) / opt,
             SE_mean_of_3_jobs=sd / np.sqrt(SHOTS * len(REPEATS)), depths={})
    for p in DEPTHS:
        cands = [r for r in res if r['inst'] == inst and r['p'] == p and r['protocol'] == 'exact']
        assert len(cands) == 7
        best = max(cands, key=lambda r: r['obj'])
        gam, bet = best['x'][:p], best['x'][p:]
        qc, g, b = build_circuit(A, p)
        pr = Statevector(bind(qc.remove_final_measurements(inplace=False), g, b, gam, bet)).probabilities()
        pn = probs_by_index(A, np.array(gam), np.array(bet))
        tvd = float(0.5 * np.abs(pr - pn).sum())
        assert tvd < 1e-9, ('simulator cross-check failed', inst, p, tvd)
        d['depths'][str(p)] = dict(start=best['start'], gamma=[float(v) for v in gam], beta=[float(v) for v in bet],
                                   E_C_over_Cstar=float((cv * pr).sum() / opt), P_opt=float(pr[optmask].sum()),
                                   E_best8192=expected_best_of_k(cv, pr) / opt, crosscheck_tvd_qiskit_vs_numpy=tvd)
        d['depths'][str(p)]['ideal_margin'] = d['depths'][str(p)]['E_C_over_Cstar'] - U
    ref['instances'][inst] = d
out = os.path.join(HERE, 'reference.json')
json.dump(ref, open(out, 'w'), indent=2)
for inst in INSTANCES:
    d = ref['instances'][inst]
    print(inst, 'U %.4f sd %.4f n_opt %d' % (d['U'], d['sd'], d['n_opt']),
          ' | '.join('p=%s ideal %.4f margin %+.4f P_opt %.2e tvd %.1e' % (p, v['E_C_over_Cstar'], v['ideal_margin'], v['P_opt'], v['crosscheck_tvd_qiskit_vs_numpy']) for p, v in d['depths'].items()))
print('sha256(reference.json) =', sha256(out))
