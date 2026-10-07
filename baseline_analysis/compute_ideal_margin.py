"""Noiseless-circuit margin over the uniform sampler at the simulation parameters shipped with the package.
For each instance and depth: ideal E[C]/C* (exact Qiskit Statevector) minus the uniform value U.
Requires qiskit.

Parameter binding follows the experiment scripts exactly: the circuit is built with
ParameterVector('gamma-glyph') and ParameterVector('beta-glyph'), and qc.parameters sorts the beta-glyph
parameters before the gamma-glyph ones, so the saved opt_gamma values end up on the mixer angles and
opt_beta on the cost angles. This is consistent between the shipped simulation and hardware runs, and is
reproduced here so the margins describe the circuits that were actually executed."""
import statistics
from common import *
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

GAMMA, BETA = chr(0x3b3), chr(0x3b2)       # the parameter names used by the experiment scripts


def build(adj, p):
    g, b = ParameterVector(GAMMA, p), ParameterVector(BETA, p)
    qc = QuantumCircuit(N)
    qc.h(range(N))
    for l in range(p):
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i][j] > 0:
                    qc.rzz(2 * g[l] * adj[i][j], i, j)
        for i in range(N):
            qc.rx(2 * b[l], i)
    return qc


def ideal_probs(adj, gam, bet):
    p = len(gam)
    qc = build(adj, p)
    d = {}
    for k in range(p):
        d[qc.parameters[k]] = gam[k]
        d[qc.parameters[p + k]] = bet[k]
    return Statevector(qc.assign_parameters(d)).probabilities()


inst = get_instances()
sim = {}
names15 = {'I1_full_graph': 'I1', 'I2_simple_tasks': 'I2', 'I3_complex_tasks': 'I3',
           'I4_latency_reweighted': 'I4', 'I5_random_baseline': 'I5'}
for e in json.load(open(os.path.join(RES, 'results_n16_simulate_seeded_2026-08-05.json'))):
    sim[names15[e['instance']]] = {int(p): v for p, v in e['qaoa_sim'].items()}
for k in NAMES[5:]:
    sim[k] = {int(p): v for p, v in json.load(open(os.path.join(RES, 'results_%s_n16_simulate.json' % k)))[0]['qaoa_sim'].items()}

rows = []
print('inst p | uniform U | ideal E[C]/C* | margin')
for k in NAMES:
    cv = cut_values(inst[k])
    opt = cv.max()
    U = float((cv / opt).mean())
    for p in sorted(sim[k]):
        pr = ideal_probs(inst[k], sim[k][p]['opt_gamma'], sim[k][p]['opt_beta'])
        ideal = float((cv * pr).sum() / opt)
        rows.append(dict(inst=k, p=p, U=U, ideal=ideal, margin=ideal - U, P_opt_ideal=float(pr[np.abs(cv - opt) < 1e-6].sum())))
        print('%-4s %d | %.4f | %.4f | %+.4f' % (k, p, U, ideal, ideal - U))
m15 = [r['margin'] for r in rows if int(r['inst'][1:]) <= 5]
m611 = [r['margin'] for r in rows if int(r['inst'][1:]) >= 6]
summary = dict(I1_I5_max=max(m15), I6_I11_min=min(m611), I6_I11_max=max(m611), I6_I11_median=statistics.median(m611))
print('I1-I5 max margin %.4f | I6-I11 min %.4f max %.4f median %.4f' % (summary['I1_I5_max'], summary['I6_I11_min'], summary['I6_I11_max'], summary['I6_I11_median']))
save('ideal_margin.json', dict(rows=rows, summary=summary))
