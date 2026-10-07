# Pre-registration: normalised-weight hardware pilot (Paper 3)

Written 2026-10-07T08:16:28+10:00, before any pilot job has been submitted.

## 1. Question
What fraction of the ideal-state expectation advantage over a uniform sampler survives on ibm_marrakesh for instance I1, using max-normalised weights and multi-start simulation parameters?

This is not a test of whether hardware "looks better". Best-of-8192 does not drive any conclusion.

## 2. Design (fixed)
- Instance: I1 only. I1 is the founding full-graph instance. It was chosen before any normalised hardware data existed, and not from earlier hardware results. It has the smallest ideal margin of the five studied instances, so the pilot is conservative.
- Depths: p = 1 and p = 2. Repeats: 3 per depth. Total 6 jobs, 8192 shots each.
- Circuit: normalised weights, parameters bound by name (gamma = cost angle, beta = mixer angle). One transpiled circuit per depth (optimization_level=1, seed_transpiler=7), reused for all 3 repeats. SamplerV2 defaults, no error mitigation.
- Parameters: best of 7 COBYLA starts under the exact objective (see reference.json, params_source). Saved before submission. Hardware is compared to these exact parameters.
- Account: personal. Tags: paper3-normpilot-N1_I1_p<p>_rep<r>. Submission order interleaves depths. Tags are re-fetched after submission to confirm.

## 3. Metrics
Reference (exact, noiseless Statevector, in reference.json): E[C]/C*, P(optimum), E[best of 8192], and the uniform-sampler values (U = 0.7702, P(opt) = 3.05e-05).
- PRIMARY: retention rho = (mean over 3 jobs of E_hw[C]/C* - U) / (E_ideal[C]/C* - U), per depth, with a 95% CI from resampling shots.
  Ideal values: p=1 E[C]/C* = 0.8602; p=2 E[C]/C* = 0.9023.
- SECONDARY: P(optimum) from raw counts, as a multiple of uniform (3.05e-05) and of ideal, with a one-sided exact binomial test against uniform.
- DESCRIPTIVE ONLY: best-of-8192 r per job.

## 4. Outcome categories, per depth
Let excess = (mean E_hw - U) divided by SE, where SE = 0.0928/sqrt(3*8192) = 0.00059.
- A (substantial survival): rho >= 0.25 and excess > 5 SE.
- B (partial survival): 0.05 < rho < 0.25 and excess > 5 SE.
- C (not detected): anything else.

Decision rule:
- Both depths A or B: recommend expanding to the normalised benchmark.
- Exactly one depth A or B: inconclusive; decide with the user before any further jobs.
- Neither: do not expand; report as a hardware noise limit.

For context only, the earlier raw-weight Step 1 retention had median 0.24 on different circuits. It is not a threshold.

## 5. Procedure rules
- All 6 jobs are submitted once, regardless of interim results. No peeking is used to change anything.
- No instance, depth, parameter, metric, threshold or shot-count change after this document is locked.
- A job that fails for infrastructure reasons may be resubmitted once with the same tag plus suffix _retry. This is documented in the manifest.
- analyze_pilot.py is frozen. Bug fixes only, documented. All 6 jobs are reported whatever the outcome.
- Raw counts, warm-start parameters, transpiled depth and two-qubit gate counts, job IDs and timestamps are stored in manifest_pilot.json.

## 6. What the pilot cannot show
One instance, two depths, one device, one time window. Retention is relative to a noiseless simulation. It does not show an advantage over classical methods, and it does not generalise to other instances or depths.

## 7. Locked files (sha256)
- reference.json: f92f2df4dfbee48527806a198039bc87aaed6061ddf7c3138580eb21e982da97
- analyze_pilot.py: e215b7f2fab9e000556656599e876c93f1269e5723be742a140aebb9eadcfd96
- pilot_lib.py: d3908b4460daed40928d20b9095efc07f3c5aacd291f784fe11ec3fd8c874243
