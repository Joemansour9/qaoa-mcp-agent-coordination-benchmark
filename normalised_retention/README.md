# normalised_retention

The two pre-specified, hash-locked hardware studies behind Section 5.5 and Table `tab:retention` of the paper:
retention of the noiseless circuit's expected-cut advantage over a uniform sampler, with max-normalised
weights and hardware and simulation sharing exactly the same parameters (bound by name).

| Folder | Study | Jobs |
|---|---|---|
| `pilot/` | I1, p = 1 and 2, 3 repeats | 6 |
| `expansion/` | I7, I8, I10, I11, p = 1 and 2, 3 repeats | 24 |

Each folder holds the pre-specification document (`PREREGISTRATION.md`; the file name is historical, the paper calls the studies pre-specified because no external registry was used), the simulator reference written before any
job was submitted (`reference.json`), the frozen analysis script, the submit script, the job manifest (raw
counts, job IDs, tags, transpiled gate counts) and the frozen analysis output. `expansion/LOCK.json` records
the sha256 of every locked file, including `PREREGISTRATION.md` and `submit_expansion.py`; the scripts are
shipped unchanged, so those hashes verify. `verify_locks.py` re-checks every locked file of both studies (the pilot hashes are in `pilot/manifest_pilot.json`); the files are stored byte-exact (`.gitattributes` turns off line-ending conversion for `pilot/` and `expansion/`). The pre-registrations carry the author's own timestamps (pilot document 2026-10-07 08:16 AEST, first job 08:17; expansion locked 08:43, first job 08:46); they were not lodged with an external registry. `expansion/params_source_norm_study.json` holds the
best-of-7 multi-start parameters used by both studies.

**Reproduce the table without IBM access:**

```
python reproduce_retention_table.py
```

This recomputes retention, hits and gate counts from the raw counts in the manifests and the cost matrices in
`../n16/cost_matrices/`, and asserts that it matches the frozen analysis outputs.

**Path layout.** The shipped pilot and expansion scripts use the paths of the study folder layout (the study
folder next to this package, and, for `make_reference.py`, a folder holding the multi-start results), so those
paths do not resolve inside this package. `expansion/norm_study.py` runs
inside the package.

**Manifest contents.** The IBM Cloud instance and plan identifiers in `usage_before` and `usage_after` are
placeholders, and the manifests list the study jobs only (`recent_jobs_before` is absent). Job IDs, tags and
raw counts are as recorded.

**Scope.** Retention is relative to a noiseless simulation of the same circuit, on one device in one time
window. The confidence intervals resample shots only and do not include job-to-job variation.
