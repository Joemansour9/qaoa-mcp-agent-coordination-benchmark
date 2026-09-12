"""
Regenerates results_I6-I11_n16_hardware.json directly from the tagged,
independently-verified job manifest, analogous to build_i1_i4_output.py
for I1-I4.

Why this exists: the results_I{6..11}_n16_hardware.json files
originally shipped in this package were built before the I6-I9/I10-I11
tagged re-run (job_manifest/i6_i9_i10_i11_rerun_manifest.json) and
still contain the earlier, untagged hardware measurements. This script
rebuilds those files from the manifest alone, so the package's
hardware-result files match what the manifest actually records (and,
in turn, what paper_draft.tex currently reports in Table 6
(tab:results69) and Table 10 (tab:extended_predictor)).

I10 and I11 hardware data exists at p=2 only (that is the only depth
these two instances were run on hardware for, matching the paper's
Section 5.5 description of I10/I11 as p=2-only additions to the
extended predictor set). Each has 4 tagged jobs, not 3: three
"_rep{1,2,3}" jobs plus one "_disambig" job used to resolve an account
job-history ambiguity during the audit. Per
n16_reseed_analysis/final_i10_i11_tagged_results.json, the canonical
3-repeat statistic uses the "_rep1/2/3" jobs only; "_disambig" is
retained in the manifest for audit purposes but is not one of the 3
nominal repeats reported in the paper.

Usage: run from this directory (n16/analysis/); paths below are
resolved relative to this script's own location, not the working
directory, so it can be invoked from anywhere.
"""

import gzip
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "..", "job_manifest", "i6_i9_i10_i11_rerun_manifest.json")
MANIFEST_PATH_GZ = MANIFEST_PATH + ".gz"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

INSTANCE_NAMES = {
    "I6": "I6_topology_shifted",
    "I7": "I7_email_calendar_weighted",
    "I8": "I8_vectorsearch_codeexec_weighted",
    "I9": "I9_websearch_hub_attempt",
    "I10": "I10_two_hubs",
    "I11": "I11_sequential_chain",
}

OPTIMAL_CUTS = {
    "I6": 32310.196559262615, "I7": 22654.8150696294,
    "I8": 25658.04154039808, "I9": 19313.744786198826,
    "I10": 19910.53015426533, "I11": 19271.769924638742,
}
GREEDY_CUTS = {
    "I6": 30493.516928532383, "I7": 19816.817286495916,
    "I8": 23106.01463282152, "I9": 18568.109352431948,
    "I10": 18105.427627458255, "I11": 17520.915824184456,
}
GREEDY_RATIOS = {
    "I6": 0.943773798237404, "I7": 0.8747287155330591,
    "I8": 0.9005369562771015, "I9": 0.9613935338785417,
    "I10": 0.9093393037341914, "I11": 0.9091492837813595,
}

if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
else:
    with gzip.open(MANIFEST_PATH_GZ, "rt") as f:
        manifest = json.load(f)

by_instance_depth = {}
for j in manifest["jobs"]:
    by_instance_depth.setdefault(j["instance"], {}).setdefault(j["depth"], []).append(j)


def build_hw_block(inst):
    depths = sorted(by_instance_depth[inst].keys())
    qaoa_hw = {}
    job_tags = {}
    for d in depths:
        jobs = by_instance_depth[inst][d]
        # I10/I11 carry an extra "_disambig" job at p=2; the canonical
        # 3-repeat statistic uses only the "_rep{1,2,3}" tags.
        canonical = [j for j in jobs if "_disambig" not in j["tag"]]
        qaoa_hw[str(d)] = [
            {"best_cut": j["result"]["best_cut"], "approximation_ratio": j["result"]["approximation_ratio"]}
            for j in canonical
        ]
        job_tags[str(d)] = {
            "canonical_repeats": [j["tag"] for j in canonical],
            "excluded_non_repeat_jobs": [j["tag"] for j in jobs if "_disambig" in j["tag"]],
            "job_ids": [j["job_id"] for j in canonical],
        }
    return qaoa_hw, job_tags


for inst in ["I6", "I7", "I8", "I9", "I10", "I11"]:
    qaoa_hw, job_tags = build_hw_block(inst)
    output = [{
        "instance": INSTANCE_NAMES[inst],
        "optimal_cut": OPTIMAL_CUTS[inst],
        "greedy_cut": GREEDY_CUTS[inst],
        "greedy_ratio": GREEDY_RATIOS[inst],
        "qaoa_sim": {},
        "qaoa_hw": qaoa_hw,
        "_provenance": {
            "source": "job_manifest/i6_i9_i10_i11_rerun_manifest.json",
            "generated_by": "analysis/build_i6_i11_output.py",
            "note": "Regenerated from the tagged rerun manifest; supersedes "
                    "the untagged pre-rerun version of this file.",
            "job_tags_by_depth": job_tags,
        },
    }]
    out_path = os.path.join(RESULTS_DIR, f"results_{inst}_n16_hardware.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"{inst}: wrote {out_path}")
    for d in sorted(qaoa_hw.keys(), key=int):
        ratios = [r["approximation_ratio"] for r in qaoa_hw[d]]
        mean = sum(ratios) / len(ratios)
        print(f"  p={d}: {[round(r, 4) for r in ratios]}  mean={mean:.3f}")
