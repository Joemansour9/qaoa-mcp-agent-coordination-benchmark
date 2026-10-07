# normalised_retention

The two pre-registered hardware studies behind Section 5.5 and Table `tab:retention` of the paper:
retention of the noiseless circuit's expected-cut advantage over a uniform sampler, with max-normalised
weights and hardware and simulation sharing exactly the same parameters (bound by name).

| Folder | Study | Jobs |
|---|---|---|
| `pilot/` | I1, p = 1 and 2, 3 repeats | 6 |
| `expansion/` | I7, I8, I10, I11, p = 1 and 2, 3 repeats | 24 |

Each folder holds the pre-registration (`PREREGISTRATION.md`), the simulator reference written before any
job was submitted (`reference.json`), the frozen analysis script, the submit script, the job manifest (raw
counts, job IDs, tags, transpiled gate counts) and the frozen analysis output. `expansion/LOCK.json` records
the sha256 of every locked file, including `PREREGISTRATION.md` and `submit_expansion.py`; the scripts are
shipped unchanged, so those hashes still verify. `expansion/params_source_norm_study.json` holds the
best-of-7 multi-start parameters used by both studies.

**Reproduce the table without IBM access:**

```
python reproduce_retention_table.py
```

This recomputes retention, hits and gate counts from the raw counts in the manifests and the cost matrices in
`../n16/cost_matrices/`, and asserts that it matches the frozen analysis outputs.

**The shipped scripts do not run in place.** They were written for the original working layout (the study
folder next to this package, and, for `make_reference.py`, a scratch folder holding the multi-start results),
so their relative paths do not resolve inside this package. They are kept unchanged as the audit record.

**Manifest changes for publication.** The IBM Cloud instance and plan identifiers in `usage_before` and
`usage_after` are replaced by placeholders, and the listing of the 15 most recent account jobs recorded
before submission (`recent_jobs_before`) is removed, because it contained unrelated jobs. Job IDs, tags and
raw counts are untouched.

**Scope.** Retention is relative to a noiseless simulation of the same circuit, on one device in one time
window. The confidence intervals resample shots only and do not include job-to-job variation.
