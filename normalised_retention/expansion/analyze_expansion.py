"""Pre-registered analysis for the normalised-weight hardware expansion (I7, I8, I10, I11; p=1,2; 3 jobs per cell).
Per cell (instance, depth): retention rho = (E_hw[C]/C* - U) / (E_ideal[C]/C* - U), using that instance's own U, sd and ideal value.
Category thresholds are identical to the pilot (A: rho >= 0.25 and excess > 5 SE; B: 0.05 < rho < 0.25 and excess > 5 SE; C otherwise).
Aggregation: a depth 'survives' if at least 3 of the 4 instances are category A or B at that depth.
Usage:  python analyze_expansion.py manifest_expansion.json      |      python analyze_expansion.py --selftest
"""
import sys, json
from exp_lib import *
from scipy.stats import binomtest

REF = json.load(open(os.path.join(HERE, 'reference.json')))
MIN_SURVIVING = 3          # of len(INSTANCES) = 4

_cache = {}
def inst_data(inst):
    if inst not in _cache:
        A = load_normalised(inst)
        cv = cut_values(A)
        opt = cv.max()
        _cache[inst] = (cv, opt, np.abs(cv - opt) < 1e-9)
    return _cache[inst]

def category(rho, excess_se):
    if rho >= 0.25 and excess_se > 5:
        return 'A'          # substantial survival
    if 0.05 < rho < 0.25 and excess_se > 5:
        return 'B'          # partial survival
    return 'C'              # not detected

def analyze_cell(jobs, inst, p, nboot=2000):
    cv, opt, optmask = inst_data(inst)
    R = REF['instances'][inst]
    U, SD, PU = R['U'], R['sd'], R['P_opt_uniform']
    d = R['depths'][str(p)]
    js = [j for j in jobs if j['instance'] == inst and j['depth'] == p and j.get('status') == 'DONE']
    assert len(js) == len(REPEATS), ('expected 3 completed jobs in cell', inst, p, len(js))
    vecs = [counts_to_vec(j['result_counts']) for j in js]
    for v in vecs:
        assert int(v.sum()) == SHOTS
    e_jobs = [float((cv * v).sum() / v.sum() / opt) for v in vecs]
    mean = float(np.mean(e_jobs))
    se = SD / np.sqrt(SHOTS * len(js))
    ex_se = (mean - U) / se
    ideal = d['E_C_over_Cstar']
    rho = (mean - U) / (ideal - U)
    rng = np.random.default_rng(12345)
    probs = [v / v.sum() for v in vecs]
    boots = []
    for _ in range(nboot):
        m = [float((cv * rng.multinomial(SHOTS, pr)).sum() / SHOTS / opt) for pr in probs]
        boots.append((np.mean(m) - U) / (ideal - U))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    hits = int(sum(v[optmask].sum() for v in vecs))
    tot = SHOTS * len(js)
    bt = binomtest(hits, tot, PU, alternative='greater')
    return dict(E_hw_jobs=e_jobs, E_hw_mean=mean, E_ideal=ideal, U=U, excess_in_SE=ex_se, retention=rho,
                retention_CI95=[float(lo), float(hi)], category=category(rho, ex_se),
                P_opt_hw=hits / tot, hits=hits, shots=tot, P_opt_ideal=d['P_opt'], P_opt_uniform=PU,
                P_opt_ratio_vs_uniform=hits / tot / PU, P_opt_ratio_ideal_vs_uniform=d['P_opt'] / PU,
                binom_p_vs_uniform=float(bt.pvalue),
                best_of_8192_jobs=[float(cv[v > 0].max() / opt) for v in vecs],
                E_best8192_ideal=d['E_best8192'], E_best8192_uniform=R['E_best8192_uniform'], _boots=boots)

def analyze(jobs, nboot=2000):
    cells = {inst: {p: analyze_cell(jobs, inst, p, nboot) for p in DEPTHS} for inst in INSTANCES}
    depth_summary = {}
    for p in DEPTHS:
        cats = [cells[i][p]['category'] for i in INSTANCES]
        rhos = [cells[i][p]['retention'] for i in INSTANCES]
        pooled = np.mean([np.array(cells[i][p]['_boots']) for i in INSTANCES], axis=0)
        depth_summary[p] = dict(categories=dict(zip(INSTANCES, cats)), n_AB=sum(c in ('A', 'B') for c in cats),
                                survives=sum(c in ('A', 'B') for c in cats) >= MIN_SURVIVING,
                                median_retention=float(np.median(rhos)), mean_retention=float(np.mean(rhos)),
                                mean_retention_CI95=[float(x) for x in np.percentile(pooled, [2.5, 97.5])])
    surv = [depth_summary[p]['survives'] for p in DEPTHS]
    verdict = ('CONFIRMED: signal survives at both depths across instances' if all(surv) else
               'PARTIAL: signal survives at one depth only' if any(surv) else
               'NOT CONFIRMED: signal does not survive across instances at either depth')
    # descriptive only: does p=2 retain less than p=1? (n=4 instances, a sign test cannot reach p<0.05 one-sided)
    lower = sum(cells[i][2]['retention'] < cells[i][1]['retention'] for i in INSTANCES)
    depth_summary['p2_below_p1_instances'] = lower
    for i in INSTANCES:
        for p in DEPTHS:
            cells[i][p].pop('_boots')
    return cells, depth_summary, verdict

def results_table(cells):
    """The principal result: all 8 cells in one table, expectation and P(optimum) side by side."""
    hdr = ('| Instance | p | E[C]/C* uniform | E[C]/C* ideal | E[C]/C* hardware | P(opt) uniform | P(opt) ideal | P(opt) hardware (hits/shots) | rho (95% CI) | Category |\n'
           '|---|---|---|---|---|---|---|---|---|---|\n')
    rows = []
    for i in INSTANCES:
        for p in DEPTHS:
            r = cells[i][p]
            rows.append('| %s | %d | %.4f | %.4f | %.4f | %.2e | %.2e | %.2e (%d/%d) | %.3f (%.3f to %.3f) | %s |' % (
                i, p, r['U'], r['E_ideal'], r['E_hw_mean'], r['P_opt_uniform'], r['P_opt_ideal'], r['P_opt_hw'], r['hits'], r['shots'],
                r['retention'], r['retention_CI95'][0], r['retention_CI95'][1], r['category']))
    return hdr + '\n'.join(rows) + '\n'

def report(cells, ds, verdict):
    print('expansion: normalised weights, references from reference.json')
    print('\nPRIMARY RESULT: all 8 cells\n')
    print(results_table(cells))
    for i in INSTANCES:
        for p in DEPTHS:
            r = cells[i][p]
            print('\n %s p=%d' % (i, p))
            print('  E_hw[C]/C* per job: %s  mean %.4f | uniform %.4f | ideal %.4f' % (', '.join('%.4f' % x for x in r['E_hw_jobs']), r['E_hw_mean'], r['U'], r['E_ideal']))
            print('  excess over uniform: %+.1f SE | RETENTION rho = %.3f (95%% CI %.3f to %.3f) | category %s' % (r['excess_in_SE'], r['retention'], r['retention_CI95'][0], r['retention_CI95'][1], r['category']))
            print('  P(opt): hw %.5f (%d hits of %d) | ideal %.5f | uniform %.5f | hw/uniform %.1fx, ideal/uniform %.1fx | binomial p vs uniform %.2g' % (
                r['P_opt_hw'], r['hits'], r['shots'], r['P_opt_ideal'], r['P_opt_uniform'], r['P_opt_ratio_vs_uniform'], r['P_opt_ratio_ideal_vs_uniform'], r['binom_p_vs_uniform']))
            print('  best-of-8192 r per job (descriptive): %s | ideal E[best] %.4f | uniform E[best] %.4f' % (', '.join('%.4f' % x for x in r['best_of_8192_jobs']), r['E_best8192_ideal'], r['E_best8192_uniform']))
    print()
    for p in DEPTHS:
        s = ds[p]
        print('p=%d: categories %s | %d of %d in A/B | median rho %.3f | mean rho %.3f (95%% CI %.3f to %.3f) | survives: %s' % (
            p, s['categories'], s['n_AB'], len(INSTANCES), s['median_retention'], s['mean_retention'], s['mean_retention_CI95'][0], s['mean_retention_CI95'][1], s['survives']))
    print('descriptive: p=2 retention below p=1 in %d of %d instances' % (ds['p2_below_p1_instances'], len(INSTANCES)))
    print('\nSECONDARY VERDICT (convenience summary; the per-cell table above is the result):', verdict)

def selftest():
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(7)
    ideal_probs = {}
    for inst in INSTANCES:
        A = load_normalised(inst)
        for p in DEPTHS:
            qc, g, b = build_circuit(A, p)
            d = REF['instances'][inst]['depths'][str(p)]
            ideal_probs[(inst, p)] = Statevector(bind(qc.remove_final_measurements(inplace=False), g, b, d['gamma'], d['beta'])).probabilities()
    unif = np.ones(2 ** N) / 2 ** N
    def fake(mixes):   # mixes[inst] = weight of the ideal state in a mixture with uniform
        jobs = []
        for inst in INSTANCES:
            for p in DEPTHS:
                pr = mixes[inst] * ideal_probs[(inst, p)] + (1 - mixes[inst]) * unif
                for r in REPEATS:
                    cnt = rng.multinomial(SHOTS, pr / pr.sum())
                    jobs.append(dict(instance=inst, depth=p, repeat=r, status='DONE',
                                     result_counts={format(k, '016b'): int(c) for k, c in enumerate(cnt) if c > 0}))
        return jobs
    uniform_mix = lambda w: {i: w for i in INSTANCES}
    for w, expect in [(1.0, 'A'), (0.5, 'A'), (0.35, 'A'), (0.15, 'B'), (0.0, 'C')]:
        cells, ds, verdict = analyze(fake(uniform_mix(w)), nboot=200)
        cats = {cells[i][p]['category'] for i in INSTANCES for p in DEPTHS}
        print('selftest weight %.2f -> categories %s (expected %s) | %s' % (w, sorted(cats), expect, verdict))
        assert cats == {expect}, 'selftest category mismatch'
        assert results_table(cells).count('\n') == 2 + len(INSTANCES) * len(DEPTHS), 'selftest table mismatch'
        assert verdict.startswith('NOT CONFIRMED' if expect == 'C' else 'CONFIRMED'), 'selftest verdict mismatch'
    # aggregation rule: 3 of 4 surviving -> survives; 2 of 4 -> does not
    mix3 = dict(zip(INSTANCES, [0.5, 0.5, 0.5, 0.0]))
    mix2 = dict(zip(INSTANCES, [0.5, 0.5, 0.0, 0.0]))
    _, ds3, v3 = analyze(fake(mix3), nboot=200)
    _, ds2, v2 = analyze(fake(mix2), nboot=200)
    print('selftest 3-of-4 ->', v3, '| 2-of-4 ->', v2)
    assert v3.startswith('CONFIRMED') and v2.startswith('NOT CONFIRMED'), 'selftest aggregation mismatch'
    print('selftest passed')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        selftest()
    else:
        man = json.load(open(sys.argv[1]))
        cells, ds, verdict = analyze(man['jobs'])
        report(cells, ds, verdict)
        open(os.path.join(HERE, 'expansion_results_table.md'), 'w').write(results_table(cells))
        json.dump(dict(cells={i: {str(p): cells[i][p] for p in DEPTHS} for i in INSTANCES},
                       depth_summary={str(k): v for k, v in ds.items()}, verdict=verdict),
                  open(os.path.join(HERE, 'expansion_results.json'), 'w'), indent=2, default=float)
