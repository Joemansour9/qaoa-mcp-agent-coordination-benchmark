"""Submit the 6 pre-registered pilot jobs. Modes:  --dry-run (no submission) | (default) submit+wait | --collect (fetch results of already submitted jobs)."""
import sys, json, time, datetime
from pilot_lib import *
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

EXPECTED = {
    'reference.json': 'f92f2df4dfbee48527806a198039bc87aaed6061ddf7c3138580eb21e982da97',
    'analyze_pilot.py': 'e215b7f2fab9e000556656599e876c93f1269e5723be742a140aebb9eadcfd96',
    'pilot_lib.py': 'd3908b4460daed40928d20b9095efc07f3c5aacd291f784fe11ec3fd8c874243',
    'PREREGISTRATION.md': 'b442bc5f44c99114a220572bd154272302de72c5545d198bd5f74b897d536c5b',
}
ACCOUNT = 'personal'
BACKEND = 'ibm_marrakesh'
MANIFEST = os.path.join(HERE, 'manifest_pilot.json')
ORDER = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]    # (depth, repeat), interleaved

def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def save(m):
    json.dump(m, open(MANIFEST, 'w'), indent=2)

def collect(svc, m):
    for j in m['jobs']:
        if j.get('status') == 'DONE':
            continue
        job = svc.job(j['job_id'])
        res = job.result()
        j['result_counts'] = {k: int(v) for k, v in res[0].data.meas.get_counts().items()}
        j['status'] = 'DONE'
        j['completed_at'] = now()
        try:
            j['usage_seconds'] = job.usage()
        except Exception:
            j['usage_seconds'] = None
        save(m)
        print('  DONE', j['tag'], 'shots', sum(j['result_counts'].values()), 'usage', j['usage_seconds'])

def main():
    mode = 'dry' if '--dry-run' in sys.argv else ('collect' if '--collect' in sys.argv else 'submit')
    for f, h in EXPECTED.items():
        got = sha256(os.path.join(HERE, f))
        assert got == h, ('LOCKED FILE CHANGED', f, got)
    print('locked-file hashes verified')
    ref = json.load(open(os.path.join(HERE, 'reference.json')))
    svc = QiskitRuntimeService(name=ACCOUNT)
    if mode == 'collect':
        m = json.load(open(MANIFEST)); collect(svc, m); return
    be = svc.backend(BACKEND)
    st = be.status()
    assert st.operational, 'backend not operational'
    usage = svc.usage()
    print('usage before:', usage.get('usage_consumed_seconds'), 'consumed;', usage.get('usage_remaining_seconds'), 'remaining')
    assert usage['usage_remaining_seconds'] >= 300, 'insufficient quota'
    pend = list(svc.jobs(limit=50, pending=True))
    print('pending/running jobs on this account:', len(pend))
    assert not pend, ('account has pending jobs; aborting to avoid cross-project interference', [(j.job_id(), list(j.tags or [])) for j in pend])
    A = load_normalised_I1()
    pm = generate_preset_pass_manager(backend=be, optimization_level=1, seed_transpiler=7)
    isa, tinfo = {}, {}
    for p in DEPTHS:
        qc, g, b = build_circuit(A, p)
        d = ref['depths'][str(p)]
        isa[p] = pm.run(bind(qc, g, b, d['gamma'], d['beta']))
        tinfo[p] = dict(depth=isa[p].depth(), two_qubit_gates=isa[p].num_nonlocal_gates(), ops={k: int(v) for k, v in isa[p].count_ops().items()})
        print('p=%d transpiled: depth %d, two-qubit gates %d, ops %s' % (p, tinfo[p]['depth'], tinfo[p]['two_qubit_gates'], tinfo[p]['ops']))
    if mode == 'dry':
        print('dry run complete: nothing submitted'); return
    m = dict(protocol='paper3 normalised-weight pilot, I1, p=1,2 x 3 repeats; see PREREGISTRATION.md',
             locked_hashes=EXPECTED, account=ACCOUNT, backend=BACKEND, shots=SHOTS, started_at=now(),
             usage_before=usage, transpile_seed=7, optimization_level=1, transpiled=tinfo, jobs=[])
    save(m)
    sampler = Sampler(mode=be)
    for (p, r) in ORDER:
        tag = '%s-N1_%s_p%d_rep%d' % (TAG_PREFIX, INSTANCE, p, r)
        d = ref['depths'][str(p)]
        job = sampler.run([isa[p]], shots=SHOTS)
        job.update_tags([tag])
        confirm = svc.job(job.job_id())
        ctags = list(confirm.tags or [])
        entry = dict(tag=tag, depth=p, repeat=r, job_id=job.job_id(), submitted_at=now(), status='SUBMITTED',
                     tag_confirmed=(tag in ctags), confirmed_tags=ctags, gamma=d['gamma'], beta=d['beta'])
        m['jobs'].append(entry); save(m)
        print('submitted', tag, job.job_id(), 'tag_confirmed', entry['tag_confirmed'])
        assert entry['tag_confirmed'], 'tag did not stick; stopping per protocol'
    print('all 6 submitted; waiting for results')
    collect(svc, m)
    m['usage_after'] = svc.usage(); m['finished_at'] = now(); save(m)
    print('usage after:', m['usage_after'].get('usage_consumed_seconds'))

if __name__ == '__main__':
    main()
