# baseline_analysis

Scripts behind the baseline numbers quoted in Section 4.3 (Classical Baselines and Metrics), Section 5.5 and Section 6.1 of the paper.
Each script is self-contained and reads only files in this package. Run from this folder.

| Script | Output | Backs |
|---|---|---|
| `compute_uniform_baseline.py` | `uniform_baseline.json` | The uniform-random-sampler reference: `U` = E[C]/C\*, its standard deviation, the number of optimal bitstrings, P(opt), and the exact expected best-of-8192 ratio (0.988 on I1; this is the paper's metric `r` for a uniform sampler). Exact, no sampling. |
| `compute_greedy_baseline.py` | `greedy_baseline.json` | Single-pass greedy ratio (the paper's greedy baseline) and randomized greedy with 8192 restarts (best and mean). The best of 8192 restarts is 1.000 on every instance, including I6 to I11. Seed per instance. |
| `compute_ideal_margin.py` | `ideal_margin.json` | Noiseless expected cut ratio at the shipped seeded simulation parameters, minus `U`: at most 0.011 on I1 to I5, and 0.005 to 0.073 (median 0.022) on I6 to I11. Needs `qiskit`. |
| `compute_protocol_decomposition.py` | `protocol_decomposition.json` | Ablation of the retention protocol (paper Table `tab:protocol`), with components added cumulatively: the baseline protocol, plus weight normalisation, plus the exact objective, plus best of 7 starts (the retention studies' protocol). Reports the noiseless margin over uniform at each stage. Run `compute_uniform_baseline.py` and `compute_ideal_margin.py` first; it reads `../normalised_retention/expansion/params_source_norm_study.json`. Median margin: 0.022, 0.020, 0.020, 0.147. |
| `compute_instance_structure.py` | `instance_structure.json` | Instance-level numbers quoted in the paper: Table `tab:instances69` (edges, density, CV, Jaccard overlap with I1), the node mean costs of Tables `tab:nodes8` and `tab:nodes16`, the intra- versus inter-server edge costs and the share of near-optimal bitstrings at n = 8 and n = 16 (Section 6.1), I7's heaviest edge, the share of I9's edges avoiding WebSearch, and the C* of I1-I5 at both scales. |

`common.py` holds the shared helpers. I1 is the shipped matrix; I2 to I5 are rebuilt with the same logic as
`build_instances()` in `n16/qaoa_experiment/qaoa_experiment_16.py` (the matrices for I2 to I5 are not shipped).
I2 to I4 share I1's edge set; I5 places the same number of edges at random positions (edge-set Jaccard
overlap with I1: 0.47). I6 to I11 are the shipped cost matrices.

`compute_ideal_margin.py` reproduces the parameter binding of the experiment scripts: because
`qc.parameters` sorts alphabetically, the saved `opt_gamma` values are bound to the mixer angles and
`opt_beta` to the cost angles. This is consistent between the shipped simulation and hardware runs, and the
margins describe the circuits that were actually executed.
