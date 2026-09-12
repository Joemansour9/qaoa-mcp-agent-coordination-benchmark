import sys
import time
import traceback
from qiskit_ibm_runtime import QiskitRuntimeService
from i1_i4_rerun_lib import (
    load_matrix, get_backend, submit_one, complete_one,
    load_manifest, save_manifest
)

INSTANCES = ['I1', 'I2', 'I3', 'I4']
DEPTH_PRIORITY = [2, 1, 3, 4]  # user-specified priority order
REPEATS = [1, 2, 3]
BATCH_SIZE = 6

BUDGET_TOTAL = 226.0  # seconds remaining as of right before this batch starts
SAFETY_MARGIN = 8.0   # stop submitting a new batch if projected to breach this margin

def build_submission_order():
    order = []
    for depth in DEPTH_PRIORITY:
        for instance in INSTANCES:
            for rep in REPEATS:
                order.append((instance, depth, rep))
    return order

def main():
    order = build_submission_order()
    print(f"Total planned jobs: {len(order)}")
    print(f"Submission order (first 12): {order[:12]}")

    svc = QiskitRuntimeService()
    hw_backend = get_backend(svc)
    W = load_matrix()
    manifest = load_manifest()

    already_done_tags = {j['tag'] for j in manifest['jobs'] if j.get('status') == 'DONE'}
    remaining_order = [o for o in order if f"{o[0]}_p{o[1]}_rep{o[2]}" not in already_done_tags]
    print(f"Already done (resuming a prior run, if any): {len(order) - len(remaining_order)}")
    print(f"Remaining to submit: {len(remaining_order)}")

    cumulative_used = sum(
        j['result']['usage_seconds'] for j in manifest['jobs']
        if j.get('status') == 'DONE' and 'result' in j
    )
    print(f"Cumulative usage already recorded in manifest: {cumulative_used}s")
    budget_remaining = BUDGET_TOTAL - cumulative_used
    print(f"Budget remaining at start of this run: {budget_remaining:.1f}s\n")

    stopped_reason = None
    idx = 0
    while idx < len(remaining_order):
        # Shrink batch size as we approach the edge, so we observe the actual
        # rejection boundary precisely instead of either overshooting in one
        # big batch or preemptively stopping before ever testing it.
        est_per_job = 5.0
        if budget_remaining < 3 * BATCH_SIZE * est_per_job:
            current_batch_size = 2
        else:
            current_batch_size = BATCH_SIZE

        batch = remaining_order[idx: idx + current_batch_size]
        print(f"\n{'='*70}")
        print(f"BATCH: jobs {idx+1}-{idx+len(batch)} of {len(remaining_order)}  "
              f"(budget remaining: {budget_remaining:.1f}s, batch_size={len(batch)})")
        print(f"{'='*70}")

        if budget_remaining < est_per_job:
            stopped_reason = (f"Budget essentially exhausted (remaining={budget_remaining:.1f}s, "
                               f"less than one job's estimated cost). Stopping without further attempts.")
            print(stopped_reason)
            break

        submitted_this_batch = []
        for (instance, depth, rep) in batch:
            tag = f"{instance}_p{depth}_rep{rep}"
            try:
                job, entry, adjacency = submit_one(svc, hw_backend, instance, depth, rep, W, manifest)
                submitted_this_batch.append((job, entry, adjacency, instance))
                print(f"  Submitted {tag} -> job {job.job_id()}  tag_confirmed={entry['tag_confirmed']}")
            except Exception as e:
                err_str = str(e)
                print(f"  SUBMISSION FAILED for {tag}: {err_str}")
                is_quota = any(kw in err_str.lower() for kw in
                                ['quota', 'exceed', 'limit', 'insufficient', '429', 'too many'])
                if is_quota:
                    stopped_reason = f"QUOTA REJECTION on {tag}: {err_str}"
                    print(f"\n*** {stopped_reason} ***")
                    print("Per protocol: stopping immediately, no retry, no backend switch.")
                else:
                    stopped_reason = f"UNEXPECTED ERROR on {tag}: {err_str}"
                    print(traceback.format_exc())
                break
        else:
            pass

        # Wait for whatever WAS submitted this batch, regardless of whether we broke out
        for (job, entry, adjacency, instance) in submitted_this_batch:
            try:
                ratio, usage = complete_one(job, entry, adjacency, instance, manifest)
                cumulative_used += usage
                budget_remaining = BUDGET_TOTAL - cumulative_used
                print(f"  DONE {entry['tag']}: ratio={ratio:.6f}  usage={usage}s  "
                      f"(cumulative={cumulative_used}s, remaining={budget_remaining:.1f}s)")
            except Exception as e:
                entry['status'] = 'ERROR'
                entry['error'] = str(e)
                save_manifest(manifest)
                print(f"  ERROR completing {entry['tag']}: {e}")

        idx += len(batch)
        if stopped_reason:
            break

    print(f"\n{'='*70}")
    print(f"BATCH RUN FINISHED")
    print(f"{'='*70}")
    print(f"Stopped reason: {stopped_reason if stopped_reason else 'all planned jobs submitted'}")
    print(f"Final cumulative usage: {cumulative_used}s")
    print(f"Final budget remaining: {budget_remaining:.1f}s")

    with open('batch_run_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"stopped_reason: {stopped_reason}\n")
        f.write(f"cumulative_used: {cumulative_used}\n")
        f.write(f"budget_remaining: {budget_remaining}\n")

if __name__ == '__main__':
    main()
