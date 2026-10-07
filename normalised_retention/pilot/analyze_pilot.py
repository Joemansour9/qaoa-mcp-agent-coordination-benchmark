"""Pre-registered analysis for the normalised-weight hardware pilot (I1, p=1 and p=2, 3 jobs each).
Primary metric: retention rho = (E_hw[C]/C* - U) / (E_ideal[C]/C* - U).
Secondary: P(optimum). Best-of-8192 is descriptive only.
Usage:  python analyze_pilot.py manifest_pilot.json      |      python analyze_pilot.py --selftest
"""
import sys, json
from pilot_lib import *
from scipy.stats import binomtest

REF = json.load(open(os.path.join(HERE, 'reference.json')))
A = load_normalised_I1()
CV = cut_values(A)
OPT = CV.max()
OPTMASK = np.abs(CV - OPT) < 1e-9
U = REF['uniform']['U']
SD = REF['uniform']['sd']
PU = REF['uniform']['P_opt']

def category(rho, excess_se):
    if rho >= 0.25 and excess_se > 5:
        return 'A'          # substantial survival
    if 0.05 < rho < 0.25 and excess_se > 5:
        return 'B'          # partial survival
    return 'C'              # not detected

def analyze(jobs, nboot=2000):
    out = {}
    for p in DEPTHS:
        js = [j for j in jobs if j['depth'] == p and j.get('status') == 'DONE']
        assert len(js) == len(REPEATS), ('expected 3 completed jobs at depth', p, len(js))
        vecs = [counts_to_vec(j['result_counts']) for j in js]
        for v in vecs:
            assert int(v.sum()) == SHOTS
        e_jobs = [float((CV * v).sum() / v.sum() / OPT) for v in vecs]
        mean = float(np.mean(e_jobs))
        se = SD / np.sqrt(SHOTS * len(js))
        ex_se = (mean - U) / se
        ideal = REF['depths'][str(p)]['E_C_over_Cstar']
        rho = (mean - U) / (ideal - U)
        rng = np.random.default_rng(12345)
        probs = [v / v.sum() for v in vecs]
        boots = []
        for _ in range(nboot):
            m = [float((CV * rng.multinomial(SHOTS, pr)).sum() / SHOTS / OPT) for pr in probs]
            boots.append((np.mean(m) - U) / (ideal - U))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        hits = int(sum(v[OPTMASK].sum() for v in vecs))
        tot = SHOTS * len(js)
        bt = binomtest(hits, tot, PU, alternative='greater')
        best = [float(CV[v > 0].max() / OPT) for v in vecs]
        out[p] = dict(E_hw_jobs=e_jobs, E_hw_mean=mean, E_ideal=ideal, U=U, excess_in_SE=ex_se,
                      retention=rho, retention_CI95=[float(lo), float(hi)], category=category(rho, ex_se),
                      P_opt_hw=hits / tot, hits=hits, shots=tot, P_opt_ideal=REF['depths'][str(p)]['P_opt'], P_opt_uniform=PU,
                      P_opt_ratio_vs_uniform=hits / tot / PU, P_opt_ratio_ideal_vs_uniform=REF['depths'][str(p)]['P_opt'] / PU,
                      binom_p_vs_uniform=float(bt.pvalue), best_of_8192_jobs=best,
                      E_best8192_ideal=REF['depths'][str(p)]['E_best8192'], E_best8192_uniform=REF['uniform']['E_best8192'])
    cats = [out[p]['category'] for p in DEPTHS]
    good = [c in ('A', 'B') for c in cats]
    verdict = ('EXPAND: both depths show surviving signal' if all(good) else
               'INCONCLUSIVE: signal at one depth only, decide with the user' if any(good) else
               'DO NOT EXPAND: no surviving signal detected; report as a hardware noise limit')
    return out, verdict

def report(out, verdict):
    print('pilot: I1, normalised weights, shared references from reference.json')
    for p in DEPTHS:
        r = out[p]
        print('\n p=%d' % p)
        print('  E_hw[C]/C* per job: %s  mean %.4f | uniform %.4f | ideal %.4f' % (', '.join('%.4f' % x for x in r['E_hw_jobs']), r['E_hw_mean'], r['U'], r['E_ideal']))
        print('  excess over uniform: %+.1f SE | RETENTION rho = %.3f (95%% CI %.3f to %.3f) | category %s' % (r['excess_in_SE'], r['retention'], r['retention_CI95'][0], r['retention_CI95'][1], r['category']))
        print('  P(opt): hw %.5f (%d hits of %d) | ideal %.5f | uniform %.5f | hw/uniform %.1fx, ideal/uniform %.1fx | binomial p vs uniform %.2g' % (
            r['P_opt_hw'], r['hits'], r['shots'], r['P_opt_ideal'], r['P_opt_uniform'], r['P_opt_ratio_vs_uniform'], r['P_opt_ratio_ideal_vs_uniform'], r['binom_p_vs_uniform']))
        print('  best-of-8192 r per job (descriptive): %s | ideal E[best] %.4f | uniform E[best] %.4f' % (', '.join('%.4f' % x for x in r['best_of_8192_jobs']), r['E_best8192_ideal'], r['E_best8192_uniform']))
    print('\nPRE-REGISTERED VERDICT:', verdict)

def selftest():
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(7)
    ideal_probs = {}
    for p in DEPTHS:
        qc, g, b = build_circuit(A, p)
        qc = qc.remove_final_measurements(inplace=False)
        d = REF['depths'][str(p)]
        ideal_probs[p] = Statevector(bind(qc, g, b, d['gamma'], d['beta'])).probabilities()
    unif = np.ones(2 ** N) / 2 ** N
    def fake(mix):   # mix = weight of the ideal state in a mixture with uniform
        jobs = []
        for p in DEPTHS:
            pr = mix * ideal_probs[p] + (1 - mix) * unif
            for r in REPEATS:
                cnt = rng.multinomial(SHOTS, pr / pr.sum())
                jobs.append(dict(depth=p, repeat=r, status='DONE', result_counts={format(i, '016b'): int(c) for i, c in enumerate(cnt) if c > 0}))
        return jobs
    for mix, expect in [(1.0, 'A'), (0.5, 'A'), (0.35, 'A'), (0.15, 'B'), (0.0, 'C')]:
        out, verdict = analyze(fake(mix), nboot=300)
        print('selftest mixture weight %.2f -> rho p1 %.3f p2 %.3f, categories %s/%s (expected %s) | %s' % (
            mix, out[1]['retention'], out[2]['retention'], out[1]['category'], out[2]['category'], expect, verdict))
        assert out[1]['category'] == expect and out[2]['category'] == expect, 'selftest category mismatch'
    print('selftest passed')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        selftest()
    else:
        man = json.load(open(sys.argv[1]))
        out, verdict = analyze(man['jobs'])
        report(out, verdict)
        json.dump(dict(results={str(k): v for k, v in out.items()}, verdict=verdict), open(os.path.join(HERE, 'pilot_results.json'), 'w'), indent=2, default=float)
