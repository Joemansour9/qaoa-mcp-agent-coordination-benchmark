import json
import numpy as np

with open('i1_i4_rerun_manifest.json') as f:
    manifest = json.load(f)

with open('results_n16.json') as f:
    sim_data = json.load(f)

name_map = {
    'I1': 'I1_full_graph', 'I2': 'I2_simple_tasks',
    'I3': 'I3_complex_tasks', 'I4': 'I4_latency_reweighted',
}

OPT_CUTS = {'I1': 34158.70542921973, 'I2': 20042.494112764583,
            'I3': 57514.08877729828, 'I4': 34207.65817970167}

by_instance = {}
for j in manifest['jobs']:
    inst = j['instance']
    d = j['depth']
    r = j['repeat']
    by_instance.setdefault(inst, {}).setdefault(d, {})[r] = j

output = {
    "_meta": {
        "description": "I1-I4 hardware re-run, p=1-4, target 3 repeats each, ibm_marrakesh. "
                        "Every job individually tagged (I{n}_p{depth}_rep{k}) and independently "
                        "verifiable via IBM Quantum job history. Supersedes the unverified I1-I4 "
                        "portion of results_repeated_optionB.json.",
        "date": "2026-08-05",
        "backend": "ibm_marrakesh",
        "shots": 8192,
        "warm_start_source": "Freshly-seeded, reproducible simulation via qaoa_experiment_16.py "
                              "(seed_simulator=42 fix applied) -- see results_n16.json for exact "
                              "opt_gamma/opt_beta used for each instance/depth.",
        "submission_order": "Prioritized by depth: p=2, p=1, p=3, p=4 (user-specified), "
                             "instances I1-I4 in order, repeats 1-3 in order within each.",
        "total_planned": 48,
        "total_completed": 0,   # filled below
        "total_missing": 0,     # filled below
        "missing_tags": [],     # filled below
        "budget": {
            "starting_seconds": 226.0,
            "used_seconds": 0,  # filled below
            "remaining_seconds": None,
        },
        "stopped_reason": "Budget essentially exhausted per internal tracking against the "
                           "user-stated starting balance -- NOT a real IBM-side quota rejection. "
                           "No submission was actually rejected by the API; the script stopped "
                           "itself proactively before attempting a submission projected to exceed "
                           "the stated budget. The 'quota rejection happens at submission time' "
                           "assumption from the task brief was therefore NOT tested or confirmed "
                           "in this run. (Applies to the accountA portion, 46/48 jobs.)",
        "account_cutover_note": "Jobs I4_p4_rep2 and I4_p4_rep3 were completed via accountB, "
                                 "a second IBM Quantum account (CRN: crn:v1:bluemix:public:quantum-computing:"
                                 "us-east:a/[REDACTED-ACCOUNT-ID]:"
                                 "[REDACTED-INSTANCE-ID]::) on 2026-08-05, following "
                                 "exhaustion of the accountA budget; all other "
                                 "I1-I4 jobs (46/48) ran on accountA "
                                 "(CRN: crn:v1:bluemix:public:quantum-computing:us-east:"
                                 "a/[REDACTED-ACCOUNT-ID]:"
                                 "[REDACTED-INSTANCE-ID]::). Account CRNs redacted for privacy; "
                                 "see reproducibility_package/README.md.",
    },
    "instances": {}
}

total_completed = 0
missing_tags = []
usage_by_account = {'accountA': 0.0, 'accountB': 0.0}

for inst in ['I1', 'I2', 'I3', 'I4']:
    sim_entry = [r for r in sim_data if r['instance'] == name_map[inst]][0]
    sim_ratios = {str(d): sim_entry['qaoa_sim'][str(d)]['approximation_ratio'] for d in [1, 2, 3, 4]}

    hw_runs = {}
    hw_mean = {}
    hw_std = {}
    job_ids = {}
    job_accounts = {}
    for d in [1, 2, 3, 4]:
        vals = []
        ids = []
        accts = []
        for r in [1, 2, 3]:
            entry = by_instance.get(inst, {}).get(d, {}).get(r)
            tag = f"{inst}_p{d}_rep{r}"
            if entry is not None and entry.get('status') == 'DONE' and 'result' in entry:
                vals.append(entry['result']['approximation_ratio'])
                ids.append(entry['job_id'])
                acct = entry.get('account', 'accountA')
                accts.append(acct)
                total_completed += 1
                usage_by_account[acct] += entry['result']['usage_seconds']
            else:
                vals.append(None)
                ids.append(None)
                accts.append(None)
                missing_tags.append(tag)
        hw_runs[str(d)] = vals
        job_ids[str(d)] = ids
        job_accounts[str(d)] = accts
        real_vals = [v for v in vals if v is not None]
        hw_mean[str(d)] = float(np.mean(real_vals)) if real_vals else None
        hw_std[str(d)] = float(np.std(real_vals)) if real_vals else None

    output["instances"][inst] = {
        "optimal_cut": OPT_CUTS[inst],
        "sim": sim_ratios,
        "hw_runs": hw_runs,
        "hw_job_ids": job_ids,
        "hw_job_accounts": job_accounts,
        "hw_mean": hw_mean,
        "hw_std": hw_std,
        "hw_runs_note": "null entries = job not submitted (budget exhausted before reaching it), "
                        "NOT a failed/errored measurement. See _meta.missing_tags. "
                        "hw_job_accounts: 'accountA' or 'accountB' per _meta.account_cutover_note.",
    }

output["_meta"]["total_completed"] = total_completed
output["_meta"]["total_missing"] = len(missing_tags)
output["_meta"]["missing_tags"] = missing_tags
output["_meta"]["budget"] = {
    "accountA": {
        "starting_seconds": 226.0,
        "used_seconds": usage_by_account['accountA'],
        "remaining_seconds": 226.0 - usage_by_account['accountA'],
    },
    "accountB": {
        "starting_seconds": 600.0,
        "used_seconds": usage_by_account['accountB'],
        "remaining_seconds": 600.0 - usage_by_account['accountB'],
    },
}

out_path = 'results_I1_I4_rerun_2026-08-05.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Saved -> {out_path}")
print(f"Completed: {total_completed}/48")
print(f"Missing: {missing_tags}")
print(f"accountA used: {usage_by_account['accountA']}s  accountB used: {usage_by_account['accountB']}s")
