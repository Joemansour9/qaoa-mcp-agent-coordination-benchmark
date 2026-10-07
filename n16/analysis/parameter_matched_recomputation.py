"""Parameter-matched recomputation of Delta_r_bar and the R-vs-Delta_r_bar correlations (paper Sections 5.4 and 6.2).

Purpose: the hardware runs for I6-I9 executed the parameters of the unseeded (July) optimiser run, while the
seeded Sim columns and R come from the seeded run. The I5 p=2 hardware runs are stored with an unseeded
optimiser run (simulated ratio 0.9557); the seeded pairing uses the seeded value (0.9887).
This script computes each quantity with the simulated ratio of the run whose parameters were executed.

Inputs (all inside this package):
  n16/results/original_unseeded/results_I{6-9}_n16_simulate_ORIGINAL.json   (unseeded simulation)
  n16/results/results_I{6-11}_n16_simulate.json                             (seeded simulation)
  n16/results/results_I{6-11}_n16_hardware.json                             (tagged repeated hardware runs)
  n16/results/results_I1_I4_rerun_2026-08-05.json, results_repeated_optionB.json,
  n16/results/results_n16_simulate_seeded_2026-08-05.json

Prints: (1) the I6-I9 cell-by-cell comparison, published pairing versus matched; (2) the I1/I7/I8/I10/I11 correlations
(variant A as published, B matched Delta only, C matched Delta and R); (3) the I1-I5 correlations the same way.
Spearman p-values are exact (all n! permutations); the asymptotic value is shown for comparison."""
import os, json, itertools
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'results')
CEIL = 0.99999999


def load_sim(path):
    return {int(p): v['approximation_ratio'] for p, v in json.load(open(path))[0]['qaoa_sim'].items()}


def load_hw(inst):
    d = json.load(open(os.path.join(RES, 'results_%s_n16_hardware.json' % inst)))[0]['qaoa_hw']
    return {int(p): [r['approximation_ratio'] for r in v] for p, v in d.items()}


orig = {k: load_sim(os.path.join(RES, 'original_unseeded', 'results_%s_n16_simulate_ORIGINAL.json' % k)) for k in ['I6', 'I7', 'I8', 'I9']}
seeded = {k: load_sim(os.path.join(RES, 'results_%s_n16_simulate.json' % k)) for k in ['I6', 'I7', 'I8', 'I9', 'I10', 'I11']}
hw = {k: load_hw(k) for k in seeded}

print('== I6-I9: HW mean r (n=3) against the published (seeded) Sim and against the Sim of the run whose parameters HW executed')
print('inst p | HW mean | published sim | published dr | matched sim | matched dr | published cell | matched cell')
cnt = dict(pub=[0, 0, 0], mat=[0, 0, 0])      # uncensored, positive, negative
for k in ['I6', 'I7', 'I8', 'I9']:
    for p in [1, 2, 3, 4]:
        m = np.mean(hw[k][p])
        sp, so = seeded[k][p], orig[k][p]
        dp, dl = m - sp, m - so
        for key, s, d in (('pub', sp, dp), ('mat', so, dl)):
            if s < CEIL:
                cnt[key][0] += 1
                cnt[key][1] += d > 1e-9
                cnt[key][2] += d < -1e-9
        print('%s %d | %.4f | %.4f | %+.4f | %.4f | %+.4f | %s | %s' % (
            k, p, m, sp, dp, so, dl, 'CENSORED' if sp >= CEIL else 'uncens', 'CENSORED' if so >= CEIL else 'uncens'))
print('published pairing: %d uncensored cells, %d positive, %d negative' % tuple(cnt['pub']))
print('matched pairing  : %d uncensored cells, %d positive, %d negative' % tuple(cnt['mat']))


def exact_p(x, y):
    rho, pa = spearmanr(x, y)
    n, hit, tot = len(x), 0, 0
    for perm in itertools.permutations(range(n)):
        r, _ = spearmanr(x, [y[i] for i in perm])
        tot += 1
        hit += abs(r) >= abs(rho) - 1e-12
    return rho, hit / tot, pa


def R_of(vals):
    return (max(vals) - min(vals)) / np.mean(vals)


i1i4 = json.load(open(os.path.join(RES, 'results_I1_I4_rerun_2026-08-05.json')))['instances']
names = {'I1_full_graph': 'I1', 'I2_simple_tasks': 'I2', 'I3_complex_tasks': 'I3', 'I4_latency_reweighted': 'I4', 'I5_random_baseline': 'I5'}
seeded_all = {names[e['instance']]: {int(p): v['approximation_ratio'] for p, v in e['qaoa_sim'].items()}
              for e in json.load(open(os.path.join(RES, 'results_n16_simulate_seeded_2026-08-05.json')))}
seeded_all.update(seeded)
hw2 = {k: hw[k][2] for k in hw}
for k in ['I1', 'I2', 'I3', 'I4']:
    hw2[k] = i1i4[k]['hw_runs']['2']
defs = {'mean': np.mean, 'median': np.median, 'best3': max}


def variant(label, S, simref, Rsrc):
    x = [R_of([Rsrc[k][p] for p in [1, 2, 3, 4]]) for k in S]
    out = []
    for dn, f in defs.items():
        y = [f(hw2[k]) - simref[k] for k in S]
        rho, pe, pa = exact_p(x, y)
        out.append('%s: r_s=%+.3f exact p=%.3f (asym %.3f)' % (dn, rho, pe, pa))
    print('%-34s | %s' % (label, ' | '.join(out)))


print('\n== I1, I7, I8, I10, I11: R vs delta-r-bar at p=2')
S = ['I1', 'I7', 'I8', 'I10', 'I11']
pub_ref = {k: seeded_all[k][2] for k in S}
mat_ref = dict(pub_ref, I7=orig['I7'][2], I8=orig['I8'][2])
R_pub = {k: seeded_all[k] for k in S}
R_mat = dict(R_pub, I7=orig['I7'], I8=orig['I8'])
print('p=2 simulated ratio, published:', {k: round(v, 4) for k, v in pub_ref.items()})
print('p=2 simulated ratio, matched  :', {k: round(v, 4) for k, v in mat_ref.items()})
variant('A as published', S, pub_ref, R_pub)
variant('B matched delta, R as published', S, mat_ref, R_pub)
variant('C matched delta and R', S, mat_ref, R_mat)

print('\n== I1-I5: R vs delta-r-bar at p=2')
S = ['I1', 'I2', 'I3', 'I4', 'I5']
i5 = [e for e in json.load(open(os.path.join(RES, 'results_repeated_optionB.json'))) if e['instance'].startswith('I5')][0]
hw2['I5'] = i5['hw_runs']['2']
pub_ref = {k: seeded_all[k][2] for k in S}
mat_ref = dict(pub_ref, I5=i5['sim']['2'])
R_pub = {k: seeded_all[k] for k in S}
R_mat = dict(R_pub, I5={int(p): v for p, v in i5['sim'].items()})
print('I5 p=2: published simulated ratio %.5f ; ratio of the run stored with the hardware runs %.5f' % (pub_ref['I5'], mat_ref['I5']))
variant('A as published', S, pub_ref, R_pub)
variant('B matched delta (I5 only)', S, mat_ref, R_pub)
variant('C matched delta and R (I5)', S, mat_ref, R_mat)


# ---- Censoring-consistent subsets (paper Appendix A, Table tab:extended_predictor) ----------------------------------
# A cell is censored when its simulated ratio is 1.000 (hardware cannot exceed it). Cells are dropped per pairing.
# With n = 4 the smallest attainable two-sided exact p is 2/24 = 0.083 (n = 5: 2/120 = 0.017).
print('\n== Censoring-consistent subsets (cells with simulated ratio 1.000 at p=2 removed for the pairing used)')
S2 = ['I1', 'I7', 'I8', 'I10', 'I11']
pub2 = {k: seeded_all[k][2] for k in S2}
mat2 = dict(pub2, I7=orig['I7'][2], I8=orig['I8'][2])
Rpub2 = {k: seeded_all[k] for k in S2}
Rmat2 = dict(Rpub2, I7=orig['I7'], I8=orig['I8'])
for label, ref, Rs in (('second set, published pairing', pub2, Rpub2), ('second set, matched delta', mat2, Rpub2), ('second set, matched delta and R', mat2, Rmat2)):
    keep = [k for k in S2 if ref[k] < CEIL]
    print('  kept %s (n=%d)' % (keep, len(keep)))
    variant(label, keep, ref, Rs)
S1 = ['I1', 'I2', 'I3', 'I4', 'I5']
pub1 = {k: seeded_all[k][2] for k in S1}
mat1 = dict(pub1, I5=i5['sim']['2'])
Rpub1 = {k: seeded_all[k] for k in S1}
Rmat1 = dict(Rpub1, I5={int(p): v for p, v in i5['sim'].items()})
print('  I1-I5 simulated ratio at p=2 (published):', {k: round(v, 4) for k, v in pub1.items()})
for label, ref, Rs in (('I1-I5, published pairing', pub1, Rpub1), ('I1-I5, matched delta and R', mat1, Rmat1)):
    keep = [k for k in S1 if ref[k] < CEIL]
    print('  kept %s (n=%d)' % (keep, len(keep)))
    variant(label, keep, ref, Rs)
