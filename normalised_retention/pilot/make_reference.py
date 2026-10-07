"""Compute and save the simulator reference BEFORE any hardware submission.
Run once. Writes reference.json (its sha256 is recorded in PREREGISTRATION.md)."""
import sys, os, json, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
from pilot_lib import *
from qiskit.quantum_info import Statevector

STUDY = os.path.join(HERE, 'norm_study.json')  # multi-start study output (not shipped; see README)
A = load_normalised_I1()
cv = cut_values(A)
opt = cv.max()
optmask = np.abs(cv - opt) < 1e-9
n_opt = int(optmask.sum())
U = float((cv / opt).mean())
sd = float((cv / opt).std())
unif = np.ones(2 ** N) / 2 ** N
ref = dict(
    created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    instance=INSTANCE,
    weights='max-normalised (w / max w); cut ratios are scale invariant',
    params_source=dict(file='scratchpad/audit/norm_study.json', sha256=sha256(STUDY),
                       selection='best exact-objective value among 7 COBYLA starts (maxiter 200, rhobeg 0.5), x[:p]=gamma (cost), x[p:]=beta (mixer)'),
    uniform=dict(U=U, sd=sd, n_opt=n_opt, P_opt=n_opt / 2 ** N, E_best8192=expected_best_of_k(cv, unif) / opt),
    shots_per_job=SHOTS, jobs_per_depth=len(REPEATS), depths={})
ref['uniform']['SE_mean_of_3_jobs'] = sd / np.sqrt(SHOTS * len(REPEATS))

# parameters
res = json.load(open(STUDY))
# independent numpy simulator used only as a cross-check of qiskit
from common import probs_by_index
for p in DEPTHS:
    cands = [r for r in res if r['inst'] == INSTANCE and r['p'] == p and r['protocol'] == 'exact']
    best = max(cands, key=lambda r: r['obj'])
    x = best['x']; gam, bet = x[:p], x[p:]
    qc, g, b = build_circuit(A, p)
    qc_nm = qc.remove_final_measurements(inplace=False)
    pr = Statevector(bind(qc_nm, g, b, gam, bet)).probabilities()
    pn = probs_by_index(A, np.array(gam), np.array(bet))
    tvd = float(0.5 * np.abs(pr - pn).sum())
    assert tvd < 1e-9, ('simulator cross-check failed', p, tvd)
    ref['depths'][str(p)] = dict(
        start=best['start'], gamma=[float(v) for v in gam], beta=[float(v) for v in bet],
        E_C_over_Cstar=float((cv * pr).sum() / opt), P_opt=float(pr[optmask].sum()),
        E_best8192=expected_best_of_k(cv, pr) / opt, crosscheck_tvd_qiskit_vs_numpy=tvd)
out = os.path.join(HERE, 'reference.json')
json.dump(ref, open(out, 'w'), indent=2)
print(json.dumps(ref, indent=2))
print('sha256(reference.json) =', sha256(out))
