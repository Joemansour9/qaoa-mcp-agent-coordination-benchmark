"""Which protocol change creates the noiseless margin over the uniform sampler?
For I1, I7, I8, I10, I11 at p = 1, 2 (the depths run on hardware), the margin E_ideal[C]/C* - U of the noiseless circuit under:
  (a) baseline protocol: raw weights, shot-based objective, single COBYLA start, parameters as shipped
      (ideal_margin.json, from compute_ideal_margin.py; run that script first);
  (b) normalised weights, shot-based objective, single start (the baseline optimisation rule on normalised weights);
  (c) normalised weights, exact objective, single start (same start as b);
  (d) normalised weights, exact objective, best of 7 starts (the protocol of the retention studies).
(b)-(d) come from the saved normalisation study ../normalised_retention/expansion/params_source_norm_study.json.
Margins use the uniform value U from uniform_baseline.json (run compute_uniform_baseline.py first)."""
import statistics
from common import *

U = json.load(open(os.path.join(OUT, 'uniform_baseline.json')))
A = {(r['inst'], r['p']): r['margin'] for r in json.load(open(os.path.join(OUT, 'ideal_margin.json')))['rows']}
S = json.load(open(os.path.join(PKG, 'normalised_retention', 'expansion', 'params_source_norm_study.json')))
INST = ['I1', 'I7', 'I8', 'I10', 'I11']


def margin(inst, p, protocol, starts):
    rows = [r for r in S if r['inst'] == inst and r['p'] == p and r['protocol'] == protocol and r['start'] in starts]
    assert rows, (inst, p, protocol)
    best = max(rows, key=lambda r: r['obj'])
    return best['meanr'] - U[inst]['U']


out = []
print('inst p | (a) baseline | (b) norm, shot, 1 start | (c) norm, exact, 1 start | (d) norm, exact, best of 7')
for k in INST:
    for p in (1, 2):
        row = dict(inst=k, p=p, original=A[(k, p)],
                   norm_shot_1start=margin(k, p, 'shot', {0}),
                   norm_exact_1start=margin(k, p, 'exact', {0}),
                   norm_exact_7starts=margin(k, p, 'exact', set(range(7))))
        out.append(row)
        print('%-4s %d | %+.4f | %+.4f | %+.4f | %+.4f' % (k, p, row['original'], row['norm_shot_1start'], row['norm_exact_1start'], row['norm_exact_7starts']))
summ = {c: dict(min=min(r[c] for r in out), max=max(r[c] for r in out), median=statistics.median(r[c] for r in out))
        for c in ('original', 'norm_shot_1start', 'norm_exact_1start', 'norm_exact_7starts')}
for c, v in summ.items():
    print('%-20s min %+.4f  median %+.4f  max %+.4f' % (c, v['min'], v['median'], v['max']))
save('protocol_decomposition.json', dict(rows=out, summary=summ))
