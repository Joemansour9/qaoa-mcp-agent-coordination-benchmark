# Pre-registration: normalised-weight hardware expansion (Paper 3)

STATUS: LOCKED 2026-10-07T08:43:54+10:00. No expansion job had been submitted at lock time. Hashes below, and in LOCK.json.

Draft written 2026-10-07, after the pilot (pilot_norm_2026-10/) and before any expansion job.

## 1. Question
Does the surviving fraction of the ideal-state expectation advantage over a uniform sampler, seen on instance I1 in the pilot, also appear on four other n=16 MCP-derived instances on ibm_marrakesh, using max-normalised weights and multi-start simulation parameters?

This is not a test of whether hardware "looks better". Best-of-8192 does not drive any conclusion.

## 2. Pilot result this builds on (context, not part of the decision rule)
I1, 3 jobs per depth: retention 0.394 at p=1 (category A) and 0.113 at p=2 (category B). P(optimum) was not enriched at either depth (2 and 1 hits in 24,576 shots, indistinguishable from uniform). p=1 beat p=2 on hardware (457 vs 935 two-qubit gates). The pilot data are reported as a fifth, separately labelled row. They are excluded from the decision rule because the pilot result is what motivated this study.

## 3. Design (fixed)
- Instances: I7, I8, I10, I11. These are the four other instances covered by the simulation-only normalisation study (which examined I1, I7, I8, I10, I11). They were fixed before any normalised hardware data existed for them. I1 is already covered by the pilot. I2 to I6 and I9 are not studied here.
- Depths: p = 1 and p = 2. Depth 3 and above are dropped on cost grounds: the pilot shows hardware retention falling from p=1 to p=2 while the two-qubit gate count doubles, so deeper circuits are unlikely to retain anything measurable. The simulation reference at p >= 3 is therefore left unused, and this study says nothing about it.
- Repeats: 3 per cell. Total 4 x 2 x 3 = 24 jobs, 8192 shots each. Expected QPU time about 2 minutes.
- Circuit: normalised weights (w / max w), parameters bound by name (gamma = cost angle, beta = mixer angle). One transpiled circuit per (instance, depth) (optimization_level=1, seed_transpiler=7), reused for all 3 repeats. SamplerV2 defaults, no error mitigation.
- Parameters: per (instance, depth), the best of 7 COBYLA starts under the exact objective, taken from params_source_norm_study.json and written to reference.json before submission. Hardware is compared against these exact parameters, so there is no sim/hardware parameter mismatch by construction.
- Account: personal. Tags: paper3-normexp-<instance>_p<p>_rep<r>. Submission order interleaves repeats, instances and depths (repeat-major), so slow device drift is spread across cells. Tags are re-fetched after each submission to confirm.
- Pre-flight: abort if the account has any pending job or less than 300 s of quota. The 15 most recent jobs and their tags are recorded in the manifest, to keep other projects' usage separable.

## 4. Metrics
Reference (exact noiseless Statevector, in reference.json, cross-checked against an independent NumPy simulator to TVD < 1e-9): E[C]/C*, P(optimum), E[best of 8192], and the uniform-sampler values for each instance.
- PRIMARY, per cell (these 8 values are the scientific result; see section 5): retention rho = (mean over 3 jobs of E_hw[C]/C* - U) / (E_ideal[C]/C* - U), with that instance's own U, ideal value and uniform standard deviation. 95% CI from resampling shots (2000 resamples).
- SECONDARY, per cell: P(optimum) from raw counts as a multiple of uniform and of ideal, with a one-sided exact binomial test against uniform. Reported per cell and not pooled, since the optimum counts differ (8 optimal strings for I7, 2 for the others).
- DESCRIPTIVE ONLY: best-of-8192 r per job, and whether p=2 retention falls below p=1 retention (count of instances out of 4; with n=4 no sign test can reach one-sided p < 0.05, so it is not a test).

Ideal margins over uniform (E_ideal - U), from reference.json:

| Instance | U | p=1 margin | p=2 margin | Smallest detectable rho (5 SE), p=1 / p=2 |
|---|---|---|---|---|
| I7 | 0.6999 | 0.1303 | 0.1754 | 0.031 / 0.023 |
| I8 | 0.6660 | 0.1214 | 0.1719 | 0.029 / 0.020 |
| I10 | 0.6656 | 0.1521 | 0.2085 | 0.027 / 0.020 |
| I11 | 0.6532 | 0.1414 | 0.1929 | 0.027 / 0.020 |

The 5 SE condition is therefore never the binding one when rho exceeds 0.05. The category thresholds on rho decide.

## 5. Primary result, categories and summary verdict
PRIMARY RESULT: the per-cell retention values. The paper reports all 8 cells, every retention value and every confidence interval, in one table (written by the frozen script to expansion_results_table.md) with these columns for each cell: E[C]/C* for uniform, ideal and hardware; P(optimum) for uniform, ideal and hardware (with hits/shots); rho with its 95% CI; and the category. The pilot showed expectation shifting strongly while optimum probability did not, so both quantities stay visible side by side in every result, and no result is shown without the P(optimum) columns.

SECONDARY: the 3-of-4 depth verdict below. It is a convenience summary of the table and is never presented without the table.

Per cell (instance, depth), categories are identical to the pilot. Let excess = (mean E_hw - U) / SE, with SE = sd_uniform / sqrt(3 x 8192) for that instance.
- A (substantial survival): rho >= 0.25 and excess > 5 SE.
- B (partial survival): 0.05 < rho < 0.25 and excess > 5 SE.
- C (not detected): anything else.

Per depth, the signal "survives" if at least 3 of the 4 instances are A or B at that depth. The 3-of-4 requirement is fixed now so that one favourable instance cannot carry the result, and one unfavourable instance cannot erase it.

Summary verdict (the frozen script prints exactly one, after the table):
- CONFIRMED: signal survives at both depths.
- PARTIAL: signal survives at one depth only.
- NOT CONFIRMED: signal survives at neither depth.

What each verdict permits in the manuscript:
- CONFIRMED: the benchmark may report hardware retention on expectation metrics for 5 instances (4 pre-registered plus the I1 pilot), with per-cell categories and CIs. Retention is the headline hardware quantity, not best-of-8192.
- PARTIAL or NOT CONFIRMED: the results are reported in full as a hardware noise limit at n=16. The paper's hardware claims are then restricted to what the data support.
- In every case all 24 jobs are reported, and per-cell categories are shown whatever the verdict. Mean retention over instances (with a bootstrap CI) is reported per depth as a summary, not as the decision quantity.

Expectation stated before the data, so it can be checked: from the pilot, I expect mostly A or B at p=1 and B at p=2, with P(optimum) at the uniform level. This expectation is not used in any rule.

## 6. Procedure rules
- All 24 jobs are submitted once, regardless of interim results. No interim analysis is performed and nothing is changed after seeing partial data.
- No instance, depth, parameter, metric, threshold, aggregation rule or shot-count change after this document is locked.
- A job that fails for infrastructure reasons may be resubmitted once with the same tag plus suffix _retry. This is documented in the manifest. A job that completes is never replaced.
- analyze_expansion.py is frozen. It has a self-test on synthetic mixtures of the ideal state with uniform (weights 1.0, 0.5, 0.35, 0.15, 0.0 give A, A, A, B, C; a 3-of-4 mix confirms; a 2-of-4 mix does not), and the self-test passed before locking. Bug fixes only, documented.
- Raw counts, warm-start parameters, transpiled depth and two-qubit gate counts, job IDs and timestamps are stored in manifest_expansion.json. The frozen analysis writes expansion_results.json and expansion_results_table.md.
- The hashes of all locked files, including submit_expansion.py and this document, are recorded in LOCK.json. A file cannot contain its own hash, so submit_expansion.py and this document are covered by LOCK.json, and the hash of LOCK.json itself is stated to the user in chat at lock time and stored in the manifest.

## 7. What this study cannot show
Four instances, two depths, one device, one time window, three repeats per cell. Retention is relative to a noiseless simulation. It does not show an advantage over classical methods (randomized greedy already reaches r = 1.000 on all of I6 to I11 under best-of-8192 scoring, so no hardware result here can be framed as beating classical baselines), it does not cover I1 to I6 and I9 beyond the pilot, it does not cover p >= 3, and it does not generalise beyond n=16 MCP-derived instances. Retention is measured on the expectation, so a high value does not imply a higher probability of finding the optimum.

## 8. Locked files (sha256)
- reference.json: 4fe1319e6d4da004b592e72f98b3dd8018e710ee8cf93b5eac992928b362ccbd
- params_source_norm_study.json: b52e78ccef38f48726186894e7dee222db67cef1a861b7521c2c3fac1d3fed02
- analyze_expansion.py: 1528d02142d0d8bef58c5b61cbe60971d65bfae1594f16306898c813ddf3b3b7
- exp_lib.py: 2edcf5079070628f43a525478248467e6ded331192a4c75adbce279f066ebcb5
- xcheck_common.py: e1383985895b04a61ccae12ebdfc4c7bdeef98313379a24471d32a6eaee54663
- make_reference.py: 917409ccd133bd1919a4dbcd4ce394f57b1e87ca99a7cb66bb69d0aa0c9b9f84
- PREREGISTRATION.md, submit_expansion.py: covered by LOCK.json (a file cannot contain its own hash).
