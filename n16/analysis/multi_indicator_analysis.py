"""
Founding-family simulation-ruggedness predictor analysis
(I1-I5; n=5) reported in Table tab:utility and Section
sec:disc:noise:i1i5.

This REPLACES the earlier ad hoc computation of
multi_indicator_analysis_2026-08-05.json, which stored only
scipy's asymptotic Spearman p-value (spearman_p) for each of the
three Delta_r_bar definitions (mean/median/best-of-3), with no
saved generating script and no exact-permutation cross-check.
That gap became a paper/repo inconsistency once the manuscript
was revised to report the exact permutation p-value alongside the
asymptotic one (matching the discipline already used for the
extended I1/I7/I8/I10/I11 analysis in i1_i6_i11_analysis.py).

Reads directly from source, nothing hardcoded:
  - Sim data (all depths, for R and for Sim p=2):
    results_n16_simulate_seeded_2026-08-05.json (seed_simulator=42
    baseline for I1-I5; this is also what R is computed from, per
    README.md). Note this deliberately does NOT use the "sim" field
    embedded in results_repeated_optionB.json for I5, which is
    stale pre-seed-fix data (I5 sim p=2 = 0.955675 there, vs the
    seeded/current 0.988689 used in the paper's printed table).
  - Hardware p=2 data (3 tagged repeats each):
    results_I1_I4_rerun_2026-08-05.json (I1-I4) and
    results_repeated_optionB.json (I5; canonical source per
    README.md, including its _data_integrity_fix record).

Reproduces the paper's exact printed numbers for R, Sim p=2, and
HW p=2 runs for all five instances (Table tab:utility), and for
Delta_r_bar vs R under all three summary-statistic definitions:
  mean:      r_s=-0.600 (exact p=0.350; asymptotic p=0.285)
  median:    r_s=0.000  (exact p=1.000; asymptotic p=1.000)
  best-of-3: r_s=+0.700 (exact p=0.233; asymptotic p=0.188)
"""

import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr, pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

INSTANCES = ["I1", "I2", "I3", "I4", "I5"]

# ---- Sim data (approximation ratio, p=1..4), seeded/current baseline ----
with open(os.path.join(RESULTS_DIR, "results_n16_simulate_seeded_2026-08-05.json")) as f:
    seeded = json.load(f)

sim = {}
for entry in seeded:
    inst = entry["instance"].split("_")[0]
    if inst in INSTANCES:
        sim[inst] = {int(k): v["approximation_ratio"] for k, v in entry["qaoa_sim"].items()}

# ---- Hardware p=2 data (3 tagged repeats each) ----
with open(os.path.join(RESULTS_DIR, "results_I1_I4_rerun_2026-08-05.json")) as f:
    i1_i4 = json.load(f)["instances"]

with open(os.path.join(RESULTS_DIR, "results_repeated_optionB.json")) as f:
    option_b = json.load(f)
i5_entry = [e for e in option_b if e["instance"] == "I5_random_baseline"][0]

hw_p2 = {inst: i1_i4[inst]["hw_runs"]["2"] for inst in ["I1", "I2", "I3", "I4"]}
hw_p2["I5"] = i5_entry["hw_runs"]["2"]

# ---- R = (max_p - min_p) / mean_p, over p=1..4 ----
R = {}
for inst in INSTANCES:
    vals = [sim[inst][p] for p in [1, 2, 3, 4]]
    R[inst] = (max(vals) - min(vals)) / np.mean(vals)

# ---- Delta_r_bar at p=2, three definitions ----
def best_of_three(vals):
    return max(vals)


defs = {"mean": np.mean, "median": np.median, "best_of_3": best_of_three}
delta = {d: {} for d in defs}
for inst in INSTANCES:
    for dname, fn in defs.items():
        delta[dname][inst] = float(fn(hw_p2[inst])) - sim[inst][2]


def exact_and_asymptotic_p(x, y):
    obs_rho, asymptotic_p = spearmanr(x, y)
    n = len(x)
    count = 0
    total = 0
    for perm in itertools.permutations(range(n)):
        y_perm = [y[i] for i in perm]
        rho, _ = spearmanr(x, y_perm)
        total += 1
        if abs(rho) >= abs(obs_rho) - 1e-12:
            count += 1
    return obs_rho, count / total, asymptotic_p


print("=" * 90)
print("R (simulation ruggedness) and Sim/HW p=2, I1-I5")
print("=" * 90)
for inst in INSTANCES:
    print(f"{inst}: sim={[round(sim[inst][p], 4) for p in [1,2,3,4]]}  R={R[inst]:.6f}  "
          f"hw_p2={[round(v, 4) for v in hw_p2[inst]]}  sim_p2={sim[inst][2]:.4f}")

print()
print("=" * 90)
print("Spearman R vs Delta_r_bar, exact permutation AND asymptotic p-value (n=5)")
print("=" * 90)
results = {}
x = [R[i] for i in INSTANCES]
for dname in defs:
    y = [delta[dname][i] for i in INSTANCES]
    rho, p_exact, p_asymptotic = exact_and_asymptotic_p(x, y)
    rp, pp = pearsonr(x, y)
    results[dname] = {
        "spearman_r": rho,
        "exact_p": p_exact,
        "asymptotic_p": p_asymptotic,
        "pearson_r": rp,
        "pearson_p": pp,
    }
    print(f"{dname:10s}: r_s={rho:+.3f}  exact p={p_exact:.3f}  asymptotic p={p_asymptotic:.3f}   "
          f"(Pearson r={rp:+.3f}, p={pp:.3f})")

density_cv_note = (
    "density is IDENTICAL (77/120=0.641667) across all 5 instances -- zero "
    "variance, correlation undefined by construction. CV is degenerate: "
    "I1-I4 tie exactly (0.343921), only I5 differs (0.338034) -- effectively "
    "one bit of information, not a continuous feature. See density_cv_verified.json."
)

out = {
    "density_cv_note": density_cv_note,
    "R": R,
    "sim_p2": {i: sim[i][2] for i in INSTANCES},
    "hw_p2_runs": hw_p2,
    "delta_r_bar_by_definition": delta,
    "spearman_vs_delta_r_bar": results,
}
out_path = os.path.join(SCRIPT_DIR, "multi_indicator_analysis_2026-08-05.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nSaved -> {out_path}")
