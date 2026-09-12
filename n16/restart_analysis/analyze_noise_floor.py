"""
Compare COBYLA restart-noise floor (restart_noise_floor.json) against the
across-depth range that the simulation-ruggedness metric R is built from.

NOT part of the paper -- exploratory check only.

Requires N_RESTARTS=30 per (instance, depth) -- range alone is too fragile
an estimator at small n, so this reports std and IQR as the primary noise
statistics, with a bootstrap CI on std to make the "noise floor" number
itself defensible, and keeps range only as a secondary/legacy comparison.

For each instance, at each depth p:
  - std(p), iqr(p)     : spread of r_sim across 30 restarts at fixed p
                         (x0 varies, seed_simulator/shots/max_iter fixed)
  - range(p)           : max-min, kept for continuity with an earlier
                         n=5-restart pilot of this same analysis

Pooled across the 4 depths:
  - pooled_std         : mean of std(p) -- the noise floor, primary number
  - pooled_std_ci90    : bootstrap 90% CI on pooled_std (resample restarts
                         within each depth, 5000 resamples)
  - depth_range        : max_p(mean_r(p)) - min_p(mean_r(p)), i.e. R's
                         numerator computed on the 30-restart-averaged r
                         (a denoised estimate of the across-depth signal)

If depth_range does not clearly exceed pooled_std (and its CI), the
across-depth variation R is measuring is within the noise a single COBYLA
run produces at one depth -- R (as computed from single runs, as in the
paper) cannot be trusted to reflect real landscape ruggedness.

Reproduces Table tab:restart_noise in paper_draft.tex exactly.
"""

import json
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_RESTARTS_REQUIRED = 30
P_VALUES = [1, 2, 3, 4]
N_BOOTSTRAP = 5000
RNG = np.random.default_rng(0)

with open(os.path.join(SCRIPT_DIR, "restart_noise_floor.json")) as f:
    data = json.load(f)


def bootstrap_std_ci(per_depth_vals, n_boot=N_BOOTSTRAP, ci=0.90):
    """Resample restarts within each depth (with replacement), recompute
    pooled_std each time, return (lo, hi) percentile CI."""
    boots = []
    depths = list(per_depth_vals.keys())
    for _ in range(n_boot):
        stds = []
        for p in depths:
            vals = per_depth_vals[p]
            resample = RNG.choice(vals, size=len(vals), replace=True)
            stds.append(np.std(resample, ddof=1))
        boots.append(np.mean(stds))
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi


print("=" * 100)
print(f"Per-depth restart spread (n={N_RESTARTS_REQUIRED}) vs across-depth range (R's numerator)")
print("=" * 100)

summary_rows = []

for inst, node in data.items():
    restarts = node["restarts"]
    if not all(str(p) in restarts and len(restarts[str(p)]) >= N_RESTARTS_REQUIRED for p in P_VALUES):
        counts = {p: len(restarts.get(str(p), [])) for p in P_VALUES}
        print(f"{inst}: incomplete (have {counts}, need {N_RESTARTS_REQUIRED} each), skipping")
        continue

    print(f"\n--- {inst} ---")
    per_depth_vals = {}
    per_depth_mean = {}
    per_depth_std = {}
    per_depth_iqr = {}
    per_depth_range = {}

    for p in P_VALUES:
        vals = np.array(restarts[str(p)][:N_RESTARTS_REQUIRED])
        per_depth_vals[p] = vals
        per_depth_mean[p] = float(np.mean(vals))
        per_depth_std[p] = float(np.std(vals, ddof=1))
        q75, q25 = np.percentile(vals, [75, 25])
        per_depth_iqr[p] = float(q75 - q25)
        per_depth_range[p] = float(vals.max() - vals.min())
        print(f"  p={p}: mean={per_depth_mean[p]:.4f}  std={per_depth_std[p]:.4f}  "
              f"IQR={per_depth_iqr[p]:.4f}  range={per_depth_range[p]:.4f}")

    pooled_std = float(np.mean(list(per_depth_std.values())))
    pooled_iqr = float(np.mean(list(per_depth_iqr.values())))
    pooled_range = float(np.mean(list(per_depth_range.values())))
    ci_lo, ci_hi = bootstrap_std_ci(per_depth_vals)

    depth_vals = list(per_depth_mean.values())
    depth_range = max(depth_vals) - min(depth_vals)
    r_denoised = depth_range / np.mean(depth_vals)

    print(f"  pooled noise floor: std={pooled_std:.4f} (90% CI [{ci_lo:.4f}, {ci_hi:.4f}])  "
          f"IQR={pooled_iqr:.4f}  range={pooled_range:.4f}")
    print(f"  across-depth range of restart-averaged r (R's numerator): {depth_range:.4f}")
    print(f"  R computed on restart-averaged r (denoised, n=20/depth):  {r_denoised:.4f}")

    verdict = "signal > noise" if depth_range > ci_hi else "NOISE-DOMINATED (within 90% CI of noise floor)"
    print(f"  --> {verdict}")

    summary_rows.append((inst, pooled_std, ci_lo, ci_hi, depth_range,
                          depth_range / pooled_std if pooled_std > 0 else float("inf")))

print()
print("=" * 100)
print("Summary")
print("=" * 100)
print(f"{'ID':<5}{'pooled_std':<12}{'90% CI':<20}{'depth_range':<14}{'ratio':<8}{'verdict'}")
for inst, pooled_std, ci_lo, ci_hi, depth_range, ratio in summary_rows:
    verdict = "signal > noise" if depth_range > ci_hi else "NOISE-DOMINATED"
    ci_str = f"[{ci_lo:.4f},{ci_hi:.4f}]"
    print(f"{inst:<5}{pooled_std:<12.4f}{ci_str:<20}{depth_range:<14.4f}{ratio:<8.2f}{verdict}")
