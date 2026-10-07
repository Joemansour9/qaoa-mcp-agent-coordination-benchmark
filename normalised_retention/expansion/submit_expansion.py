"""Submit the 24 pre-registered expansion jobs. Modes:  --dry-run (no submission) | (default) submit+wait | --collect (fetch results of already submitted jobs).
Refuses to run until EXPECTED holds the sha256 of every locked file (filled in at lock time, after the pre-registration is approved)."""
import sys, json, time, datetime
from exp_lib import *
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

EXPECTED = {
    'reference.json': '4fe1319e6d4da004b592e72f98b3dd8018e710ee8cf93b5eac992928b362ccbd',
    'params_source_norm_study.json': 'b52e78ccef38f48726186894e7dee222db67cef1a861b7521c2c3fac1d3fed02',
    'analyze_expansion.py': '1528d02142d0d8bef58c5b61cbe60971d65bfae1594f16306898c813ddf3b3b7',
    'exp_lib.py': '2edcf5079070628f43a525478248467e6ded331192a4c75adbce279f066ebcb5',
    'xcheck_common.py': 'e1383985895b04a61ccae12ebdfc4c7bdeef98313379a24471d32a6eaee54663',
    'make_reference.py': '917409ccd133bd1919a4dbcd4ce394f57b1e87ca99a7cb66bb69d0aa0c9b9f84',
    'PREREGISTRATION.md': '85a7020b3cb67df12e77a26ee3f6723b1ec264b186dc5286800a7653ffb322a2',
}
ACCOUNT = 'personal'
BACKEND = 'ibm_marrakesh'
MANIFEST = os.path.join(HERE, 'manifest_expansion.json')
# interleaved: repeat-major, then instance, then depth
ORDER = [(inst, p, r) for r in REPEATS for inst in INSTANCES for p in DEPTHS]

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
        assert got == h, ('LOCKED FILE CHANGED OR NOT YET LOCKED', f, got)
    lockpath = os.path.join(HERE, 'LOCK.json')
    lock = json.load(open(lockpath))
    for f, h in lock['files'].items():
        assert sha256(os.path.join(HERE, f)) == h, ('LOCK.json MISMATCH', f)
    lock_sha = sha256(lockpath)
    print('locked-file hashes verified (%d files in LOCK.json; sha256(LOCK.json) = %s)' % (len(lock['files']), lock_sha))
    ref = json.load(open(os.path.join(HERE, 'reference.json')))
    svc = QiskitRuntimeService(name=ACCOUNT)
    if mode == 'collect':
        m = json.load(open(MANIFEST)); collect(svc, m); return
    be = svc.backend(BACKEND)
    assert be.status().operational, 'backend not operational'
    usage = svc.usage()
    print('usage before:', usage.get('usage_consumed_seconds'), 'consumed;', usage.get('usage_remaining_seconds'), 'remaining')
    assert usage['usage_remaining_seconds'] >= 300, 'insufficient quota'
    pend = list(svc.jobs(limit=50, pending=True))
    print('pending/running jobs on this account:', len(pend))
    assert not pend, ('account has pending jobs; aborting to avoid cross-project interference', [(j.job_id(), list(j.tags or [])) for j in pend])
    recent = [dict(job_id=j.job_id(), tags=list(j.tags or []), created=str(j.creation_date)) for j in svc.jobs(limit=15)]
    print('most recent jobs on this account (recorded in manifest):')
    for r in recent:
        print('  ', r['job_id'], r['created'], r['tags'])
    pm = generate_preset_pass_manager(backend=be, optimization_level=1, seed_transpiler=7)
    isa, tinfo = {}, {}
    for inst in INSTANCES:
        A = load_normalised(inst)
        for p in DEPTHS:
            qc, g, b = build_circuit(A, p)
            d = ref['instances'][inst]['depths'][str(p)]
            c = pm.run(bind(qc, g, b, d['gamma'], d['beta']))
            isa[(inst, p)] = c
            tinfo['%s_p%d' % (inst, p)] = dict(depth=c.depth(), two_qubit_gates=c.num_nonlocal_gates(), ops={k: int(v) for k, v in c.count_ops().items()})
            print('%s p=%d transpiled: depth %d, two-qubit gates %d' % (inst, p, c.depth(), c.num_nonlocal_gates()))
    plan = [dict(tag='%s-%s_p%d_rep%d' % (TAG_PREFIX, i, p, r), instance=i, depth=p, repeat=r) for (i, p, r) in ORDER]
    assert len(plan) == 24 and len({x['tag'] for x in plan}) == 24
    assert {x['instance'] for x in plan} == {'I7', 'I8', 'I10', 'I11'} and {x['depth'] for x in plan} == {1, 2} and {x['repeat'] for x in plan} == {1, 2, 3}
    assert all(x['tag'].startswith('paper3-normexp-') for x in plan) and SHOTS == 8192
    for i in INSTANCES:
        for p in DEPTHS:
            assert sum(1 for x in plan if x['instance'] == i and x['depth'] == p) == 3
    print('plan: %d jobs, instances %s, depths %s, 3 repeats per cell, %d shots, tags %s ... %s' % (len(plan), INSTANCES, DEPTHS, SHOTS, plan[0]['tag'], plan[-1]['tag']))
    print('outputs: %s | expansion_results.json | expansion_results_table.md' % os.path.basename(MANIFEST))
    if mode == 'dry':
        print('dry run complete: nothing submitted'); return
    m = dict(protocol='paper3 normalised-weight expansion, I7 I8 I10 I11, p=1,2 x 3 repeats; see PREREGISTRATION.md',
             locked_hashes=EXPECTED, lock_json_files=lock['files'], lock_json_sha256=lock_sha, account=ACCOUNT, backend=BACKEND, shots=SHOTS, started_at=now(),
             usage_before=usage, recent_jobs_before=recent, transpile_seed=7, optimization_level=1, transpiled=tinfo, jobs=[])
    save(m)
    sampler = Sampler(mode=be)
    for (inst, p, r) in ORDER:
        tag = '%s-%s_p%d_rep%d' % (TAG_PREFIX, inst, p, r)
        d = ref['instances'][inst]['depths'][str(p)]
        job = sampler.run([isa[(inst, p)]], shots=SHOTS)
        job.update_tags([tag])
        ctags = list(svc.job(job.job_id()).tags or [])
        entry = dict(tag=tag, instance=inst, depth=p, repeat=r, job_id=job.job_id(), submitted_at=now(), status='SUBMITTED',
                     tag_confirmed=(tag in ctags), confirmed_tags=ctags, gamma=d['gamma'], beta=d['beta'])
        m['jobs'].append(entry); save(m)
        print('submitted', tag, job.job_id(), 'tag_confirmed', entry['tag_confirmed'])
        assert entry['tag_confirmed'], 'tag did not stick; stopping per protocol'
    print('all %d submitted; waiting for results' % len(ORDER))
    collect(svc, m)
    m['usage_after'] = svc.usage(); m['finished_at'] = now(); save(m)
    print('usage after:', m['usage_after'].get('usage_consumed_seconds'))

if __name__ == '__main__':
    main()
