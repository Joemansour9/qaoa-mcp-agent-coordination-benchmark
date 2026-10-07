# Reproducibility Package
## Benchmarking QAOA on MCP-Derived Agent Coordination Instances: Expectation-Based Evaluation and Hardware Retention on IBM Marrakesh

This package contains the data-generation scripts, cost matrices, QAOA
experiment code, results, hardware job manifests, and analysis scripts
backing every table and figure in the paper, organized by problem
scale (n=8, n=16), then by pipeline stage.

---

## n=8 (`n8/`)

| File | Backs |
|---|---|
| `data_generation/mcp_data_generator.py` | Section 4.1 — 4-server task generator |
| `data_generation/mcp_agent_data.csv` | 150-task execution log (token means: simple=2063.12, medium=3526.34, complex=5919.92) |
| `cost_matrices/mcp_agent_data_cost_matrix.csv` | 8×8 cost matrix (I1); Table `tab:nodes8`, Table `tab:instances` (n=8 column) |
| `qaoa_experiment/qaoa_qap_experiment.py` | QAOA circuit, `build_instances()` (I1→I2–I5 derivation), simulation/hardware execution; source for Table `tab:results8` |
| `job_manifest/n8_hardware_jobs.json` | 20 hardware jobs (`ibm_marrakesh`), taken from the IBM Quantum job history. Individual job-to-cell attribution is not meaningful at this scale (see Section 4.4) — every job's outcome saturates against all 5 candidate instances. Source for the aggregate r=1.000 result in Table `tab:results8`'s HW column. Each job also carries `quantum_seconds` (4 or 5 s, 90 s in total), the source of the "4--5 seconds QPU time" statement in Section 5.1. |
| `simulation_log/n8_execution_log.txt` | Console log of the n=8 noiseless simulation (all five instances, p = 1 to 4, r = 1.000 throughout), the source of the simulation column of Table `tab:results8`. Local file paths removed. |
| `job_manifest/job_d8gt5elv8cos73f3mmb0_counts.json` | Raw counts (all 256 outcomes, 8,192 shots), transpiled gate counts and physical qubits of the example job behind Figure `fig:measurement_hist`. The job is a depth-4 execution (inferred from its 1,802 transpiled gates; untagged, so its instance is not recoverable). |
| `job_manifest/job_d8gt5elv8cos73f3mmb0_transpiled_circuit.json` | The transpiled native-gate circuit of the same job (1,794 gates plus 8 measurements, program order, physical qubit indices). Lets the circuit figure be redrawn without IBM Quantum access. |
| `figures/quick_figs.py` | Plotting script only — regenerates figures from existing results, not a data source |

Figures used in the paper are in figures_paper/.

---

## n=16 (`n16/`)

### `data_generation/` — Section 4.1, task logs for I1–I11
150-task GPT-4o-mini execution logs and generator scripts for I1
(`mcp_agent_data_16.csv`) and I6–I11 (`mcp_agent_data_16_I*.csv`,
`mcp_data_generator_16_I*.py`).

**Edge counts.** The `Active edges` line printed by the generator scripts counts nonzero diagonal (self-loop) entries as well, so it overstates the edge count (for example 70 instead of 65 for I6). Self-loops never enter a cut. The paper's edge counts and densities are off-diagonal counts, as in `n16/analysis/density_cv.json`.

### `cost_matrices/` — Table `tab:instances`, Table `tab:instances69`
Symmetrized cost matrices for I1 (I2–I5 derived from it in-code, see
`qaoa_experiment/qaoa_experiment_16.py`'s `build_instances()`) and
I6–I11 (each independently measured).

### `qaoa_experiment/` — Section 4.2

| File | Backs |
|---|---|
| `qaoa_experiment_16.py`, `qaoa_experiment_16_I10.py`, `qaoa_experiment_16_I11.py` | Circuit construction, COBYLA optimization (`seed=0`, `seed_simulator=42`, 200 iterations), simulation, single-job hardware execution. |
| `qaoa_experiment_16_I6_seeded.py`–`_I9_seeded.py` | Same circuit construction and COBYLA protocol (`seed=0`, `seed_simulator=42`, 200 iterations), simulation and single-job hardware execution, for I6–I9. Default `--matrix` and `--output` paths resolve relative to this script's own location, so each runs standalone from inside this folder; running with `--mode simulate` reproduces `results_I6_n16_simulate.json`–`results_I9_n16_simulate.json` exactly. |
| `qaoa_repeated.py`, `submit_i5_p1_tagged.py`, `i1_i4_rerun_lib.py`, `run_i1_i4_batch.py` | The repeated (3×), individually-tagged hardware-run protocol used for the repeated dataset (Section 4.4). |

The n = 8 and n = 16 submission scripts that bind parameters by sorted name list the cost and mixer angles under swapped labels in their saved parameter files; the circuit is unchanged. The retention scripts bind parameters by name.

Some n = 8 and n = 16 submission scripts select the backend with `least_busy(...)`. Every job with a backend record ran on `ibm_marrakesh`.

### `results/` — Table `tab:results16`, Table `tab:utility`, Table `tab:results69`, Table `tab:extended_predictor`

| File | Backs |
|---|---|
| `results_I1_I4_rerun_2026-08-05.json` | I1–I4, all depths/repeats, tagged. Primary source for Table `tab:utility`'s I1–I4 rows |
| `results_repeated_optionB.json` | I5 hardware runs and simulation values. Primary source for Table `tab:utility`'s I5 row. The I5 p=2 runs have no job IDs or tags and come from an unseeded optimiser run (simulated ratio 0.9557 in this file, against 0.9887 in the seeded baseline used for the table); `analysis/multi_indicator_analysis.py` reproduces the seeded pairing. |
| `results_I6_n16_hardware.json`, `results_I7_n16_hardware.json`, `results_I8_n16_hardware.json`, `results_I9_n16_hardware.json` | Regenerated by `analysis/build_i6_i11_output.py` from the tagged rerun manifest. Source for Table `tab:results69`'s HW columns |
| `results_I6_n16_simulate.json`, `results_I7_n16_simulate.json`, `results_I8_n16_simulate.json`, `results_I9_n16_simulate.json` | Regenerated by `analysis/build_i6_i9_simulate_output.py`, or directly by running `qaoa_experiment/qaoa_experiment_16_I{6-9}_seeded.py --mode simulate` (`seed_simulator=42`). Source for Table `tab:results69`'s Sim columns |
| `results_I10_n16_hardware.json`, `results_I11_n16_hardware.json` | Regenerated by `build_i6_i11_output.py`, p=2 only. Source for Table `tab:extended_predictor`'s I10/I11 rows |
| `results_I10_n16_simulate.json`, `results_I11_n16_simulate.json` | Already `seed_simulator=42`-current. Source for R(I10), R(I11) |
| `results_n16_simulate_seeded_2026-08-05.json` | Seeded I1–I5 simulation baseline, used to compute R in Table `tab:utility` |
| `unseeded/results_I{6-9}_n16_simulate_unseeded.json` | The unseeded I6–I9 simulation runs, whose parameters the I6–I9 hardware runs executed. Used by `analysis/parameter_matched_recomputation.py` |
| `results_n16_singlerun_depthprofile.json` | Single run with unseeded simulation and untagged hardware jobs. Source of both the Sim and HW columns of Table 6 (`tab:results16`) and of Figures 4 and 5 (`fig:ratio_n16`, `fig:gap_n16`). Not reproducible: 10 of its 20 simulation cells differ by 0.01 or more (at most 0.031) from `results_n16_simulate_seeded_2026-08-05.json`. Used only to describe the depth profile |

### `job_manifest/` — Section 4.4, "Data provenance and verification"

| File | Backs |
|---|---|
| `i1_i4_rerun_manifest.json` | Job-by-job record (tag, job ID, timestamp, status, result) for all 48 I1–I4 hardware jobs, with job IDs and tags from the IBM Quantum job history. |
| `i6_i9_i10_i11_rerun_manifest.json.gz` | Same discipline, all 56 I6–I11 jobs (I6–I9: 4 depths × 3 repeats; I10/I11: p=2 only, 3 canonical repeats + 1 disambiguation job). Shipped gzipped (13MB → 1.5MB); `build_i6_i11_output.py` reads the `.gz` directly, no manual decompression needed (`gunzip -k i6_i9_i10_i11_rerun_manifest.json.gz` to inspect by hand). Primary source for Table `tab:results69` and Table `tab:extended_predictor`'s I10/I11 rows, via `analysis/build_i6_i11_output.py`. |
| `i5_p1_*_result.json` (×3) | The three individually tagged I5 p=1 jobs. Backs Table `tab:utility`'s I5 row. |
| `i4_p4_rep23_run_set_2_results.json` | The two I4 p=4 jobs (repeats 2 and 3). Backs Table `tab:utility`'s I4 row. |

**Privacy note:** IBM Cloud identifiers in the shipped files appear as redaction placeholders (`[REDACTED-...]`); they are not needed for reproducibility. Job IDs and tags are kept, since they are the basis for the package's provenance claims.

### `analysis/` — Table `tab:utility`, Table `tab:extended_predictor`, Table `tab:instances69`, Appendix A

| File | Backs |
|---|---|
| `multi_indicator_analysis.py` / `_2026-08-05.json` | R, Sim/HW p=2, and delta-r-bar for I1–I5, all three summary-statistic definitions, via both exact-permutation and asymptotic Spearman correlations (plus Pearson correlations). Reads directly from `results_n16_simulate_seeded_2026-08-05.json` (sim) and `results_I1_I4_rerun_2026-08-05.json` / `results_repeated_optionB.json` (I1–I4 / I5 hardware), nothing hardcoded. Source for Table `tab:utility` and every R/delta-r-bar number quoted in Appendix A.3's prose; reproduces all of them exactly (mean: r_s=-0.600, exact p=0.350, asymptotic p=0.285; median: r_s=0.000, exact p=1.000; best-of-3: r_s=+0.700, exact p=0.233, asymptotic p=0.188). |
| `i1_i6_i11_analysis.py` / `.json` | The extended-predictor analysis for I1, I7, I8, I10, I11 (n=5): R, density, and CV vs delta-r-bar (three definitions each), plus the CV-vs-R and density-vs-R pairwise redundancy checks, all via exact-permutation Spearman correlations. Density and CV are computed directly from the symmetrized cost matrices in `../cost_matrices/`, not hardcoded from the paper's table. Source for Table `tab:extended_predictor` and every R/density/CV correlation number quoted in Appendix A.3's prose; reproduces all of them exactly (R: r_s=-0.300/+0.100/-0.100; Density: r_s=-0.300/-0.100/-0.400; CV: r_s=+0.600/+0.200/+0.300; CV-vs-R: r_s=-0.900; Density-vs-R: r_s=+0.700). |
| `parameter_matched_recomputation.py` | Recomputes delta-r-bar and the R-vs-delta-r-bar correlations with the simulated ratio of the optimiser run whose parameters were actually executed on hardware, next to the seeded pairing (I6–I9 cell-by-cell censoring counts; I1/I7/I8/I10/I11 and I1–I5 correlations under variants A seeded pairing, B matched delta, C matched delta and R; then the same correlations with censored cells removed, n = 4: I1, I2, I4, I5 and I1, I7, I10, I11, as in paper Table `tab:extended_predictor`). Reads `n16/results/unseeded/` for the unseeded I6–I9 simulation runs. Source for every "parameter-matched recomputation" number in Section 5.4 and Appendix A; prints exact and asymptotic Spearman p-values. |
| `build_i1_i4_output.py` | Regenerates `results_I1_I4_rerun_2026-08-05.json` from `job_manifest/i1_i4_rerun_manifest.json`. |
| `build_i6_i11_output.py` | Regenerates `results_I{6,7,8,9,10,11}_n16_hardware.json` from `job_manifest/i6_i9_i10_i11_rerun_manifest.json.gz`. |
| `build_i6_i9_simulate_output.py` | Regenerates `results_I6_n16_simulate.json`–`results_I9_n16_simulate.json` from the seeded source in the top-level project's `seed_sensitivity_test/` folder (outside this package; the script errors clearly if that folder isn't found). Equivalent, fully self-contained alternative: run `qaoa_experiment/qaoa_experiment_16_I{6-9}_seeded.py --mode simulate` directly. |
| `density_cv.json` | Density and CV computed directly from symmetrized cost matrices, off-diagonal only. |

**Not included**: energy-gap and near-optimal-solution-density predictor tests — exploratory, not referenced in the submitted manuscript.

### `restart_analysis/` — Table `tab:restart_noise`, Appendix A.2

| File | Backs |
|---|---|
| `restart_noise_floor.py` | 30 independent COBYLA restarts per depth (p=1–4) for I1, I7, I8, I10, I11, isolating optimizer-initialization sensitivity from other noise sources. A full re-run takes roughly 6 hours; the completed output ships alongside. |
| `restart_noise_floor.json` | The 30-restart raw output (600 optimizations). |
| `analyze_noise_floor.py` | Computes the pooled per-depth noise floor (std, 90% bootstrap CI) against the denoised across-depth signal per instance. Source for Table `tab:restart_noise`; reproduces it exactly. |

### `cpsat_baseline/` — Section 4.3, Classical Baselines

| File | Backs |
|---|---|
| `cpsat_maxcut_benchmark.py` | Standalone OR-Tools CP-SAT solver for all eleven instances; paths resolve relative to the script's own location. |
| `cpsat_results.csv` / `.json` | Solver output for all eleven instances (CP-SAT reaches the proven optimum, r=1.000000, for every instance). |

### `figures/`

| File | Backs |
|---|---|
| `quick_figs.py` | Plotting script only — regenerates both figures from the single-run values of Table `tab:results16` (hardcoded in the script; single run with unseeded simulation and untagged hardware jobs, see `n16/results/results_n16_singlerun_depthprofile.json`), no live simulation or hardware run needed; not a data source |

---

## `figures_paper/` — all plotted figures of the paper, as vector PDF
`make_paper_figures.py` regenerates `fig_cost_matrices.pdf` (Figure `fig:heatmaps`), `fig_n8_approx_ratio.pdf` (`fig:ratio_vs_depth`), `fig_n8_sim_vs_hw.pdf` (`fig:sim_hw_gap`), `fig_n16_approx_ratio.pdf` (`fig:ratio_n16`), `fig_n16_sim_vs_hw.pdf` (`fig:gap_n16`) and `fig_retention.pdf` (`fig:retention`). The PDFs are sized to the journal text width, use embedded fonts, a colour-blind-safe palette and no in-figure titles. Data are read from the package (the RESULTS tables in `n8/figures/quick_figs.py` and `n16/figures/quick_figs.py`, the n=8 cost matrix, and the frozen outputs in `normalised_retention/`). The PNG scripts in `n8/figures/` and `n16/figures/` write PNG versions of the n = 8 and n = 16 figures. `make_transpiled_partial_view.py` draws `fig_circuit_transpiled_partial.pdf` (Figure `fig:circuit`), the first 16 layers of the job's real transpiled circuit. `make_circuit_figure.py` draws `fig_circuit_schematic.pdf`, a logical n = 8 schematic, kept as an alternative that the paper does not use. `make_histogram_figure.py` builds `fig_measurement_hist.pdf` from the saved counts of job `d8gt5elv8cos73f3mmb0` (all 256 outcomes) and prints the statistics quoted in Section 5.3.

---

## `baseline_analysis/` — Section 4.3, Section 5.5, Section 6.1
Uniform-sampler reference, greedy and randomized-greedy baselines, and the noiseless circuit's margin over the
uniform sampler, for I1–I11. See `baseline_analysis/README.md`.

## `normalised_retention/` — Section 5.5, Table `tab:retention`
Two pre-specified, hash-locked hardware studies (I1 pilot; I7, I8, I10, I11 expansion, 30 jobs in all) with their
pre-specification documents, references, frozen analyses, manifests and a script that recomputes the table from raw
counts. See `normalised_retention/README.md`.

---
