"""Check the files of the two pre-registered studies against the sha256 hashes recorded before submission.
Pilot hashes: pilot/manifest_pilot.json ('locked_hashes'). Expansion hashes: expansion/LOCK.json ('files').
Run from this folder:  python verify_locks.py   (files must be byte-exact; see ../.gitattributes)"""
import hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
bad = 0
for study, manifest, key in (('pilot', 'manifest_pilot.json', 'locked_hashes'), ('expansion', 'LOCK.json', 'files')):
    recorded = json.load(open(os.path.join(HERE, study, manifest)))[key]
    for name, digest in recorded.items():
        actual = hashlib.sha256(open(os.path.join(HERE, study, name), 'rb').read()).hexdigest()
        ok = actual == digest
        bad += not ok
        print('%-9s %-32s %s' % (study, name, 'OK' if ok else 'MISMATCH'))
print('all locked files verify' if not bad else '%d mismatch(es)' % bad)
