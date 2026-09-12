"""
Shared library for the I1-I4 tagged hardware re-run.
Every submission goes through submit_one() and complete_one() so the
manifest is always updated immediately, job-by-job, never reconstructed
after the fact.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

N_QUBITS = 16
SHOTS = 8192
MANIFEST_PATH = 'i1_i4_rerun_manifest.json'

OPT_CUTS = {
    'I1': 34158.70542921973,
    'I2': 20042.494112764583,
    'I3': 57514.08877729828,
    'I4': 34207.65817970167,
}

def load_matrix(csv_path='mcp_agent_data_16_cost_matrix.csv'):
    df = pd.read_csv(csv_path, index_col=0)
    W = df.values.astype(float)
    return (W + W.T) / 2

def build_adjacency(instance_key, W):
    if instance_key == 'I1':
        return W.copy()
    if instance_key == 'I2':
        return W * (2063.0 / 3516.0)
    if instance_key == 'I3':
        return W * (5920.0 / 3516.0)
    if instance_key == 'I4':
        rng = np.random.default_rng(42)
        W_lat = W.copy()
        idx = [(i, j) for i in range(N_QUBITS) for j in range(i+1, N_QUBITS) if W[i][j] > 0]
        vals = rng.permutation([W[i][j] for i, j in idx])
        for (i, j), v in zip(idx, vals):
            W_lat[i][j] = v
            W_lat[j][i] = v
        return W_lat
    raise ValueError(instance_key)

def compute_cut_value(bits, adjacency):
    n = len(bits)
    cut = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if bits[i] != bits[j]:
                cut += adjacency[i][j]
    return cut

def build_qaoa_circuit(adjacency, p):
    n = adjacency.shape[0]
    gamma = ParameterVector('γ', p)
    beta = ParameterVector('β', p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for i in range(n):
            for j in range(i+1, n):
                w = adjacency[i][j]
                if w > 0:
                    qc.rzz(2 * gamma[layer] * w, i, j)
        for i in range(n):
            qc.rx(2 * beta[layer], i)
    qc.measure_all()
    return qc

def load_warmstart(instance_key, depth, sim_results_path='results_n16.json'):
    name_map = {
        'I1': 'I1_full_graph', 'I2': 'I2_simple_tasks',
        'I3': 'I3_complex_tasks', 'I4': 'I4_latency_reweighted',
    }
    with open(sim_results_path) as f:
        data = json.load(f)
    entry = [r for r in data if r['instance'] == name_map[instance_key]][0]
    p_entry = entry['qaoa_sim'][str(depth)]
    return p_entry['opt_gamma'], p_entry['opt_beta']

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {'protocol': 'I1-I4 hardware re-run, p=1-4, 3 repeats, tagged',
            'backend': 'ibm_marrakesh', 'shots': SHOTS, 'jobs': []}

def save_manifest(manifest):
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

def get_backend(svc):
    return svc.backend('ibm_marrakesh')

def submit_one(svc, hw_backend, instance_key, depth, rep, W, manifest):
    tag = f"{instance_key}_p{depth}_rep{rep}"
    adjacency = build_adjacency(instance_key, W)
    gamma, beta = load_warmstart(instance_key, depth)

    qc = build_qaoa_circuit(adjacency, depth)
    param_dict = {}
    for k in range(depth):
        param_dict[qc.parameters[k]] = gamma[k]
        param_dict[qc.parameters[depth + k]] = beta[k]
    bound = qc.assign_parameters(param_dict)

    pm = generate_preset_pass_manager(backend=hw_backend, optimization_level=1)
    isa = pm.run(bound)

    sampler = Sampler(mode=hw_backend)
    job = sampler.run([isa], shots=SHOTS)
    job.update_tags([tag])

    # Read the tag back from job metadata to CONFIRM it was actually accepted
    # (re-fetch via job_id, not just trust the local object we already have)
    confirm_job = svc.job(job.job_id())
    confirmed_tags = list(confirm_job.tags) if confirm_job.tags else []
    tag_confirmed = tag in confirmed_tags

    entry = {
        'tag': tag,
        'instance': instance_key,
        'depth': depth,
        'repeat': rep,
        'job_id': job.job_id(),
        'submitted_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'status': 'SUBMITTED',
        'tag_confirmed': tag_confirmed,
        'confirmed_tags': confirmed_tags,
    }
    manifest['jobs'].append(entry)
    save_manifest(manifest)
    return job, entry, adjacency

def complete_one(job, entry, adjacency, instance_key, manifest):
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()

    best_cut = 0.0
    for bitstring, count in counts.items():
        bits = [int(b) for b in reversed(bitstring)]
        cut = compute_cut_value(bits, adjacency)
        if cut > best_cut:
            best_cut = cut
    ratio = best_cut / OPT_CUTS[instance_key]
    usage = job.usage()

    entry['status'] = 'DONE'
    entry['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    entry['result'] = {
        'best_cut': best_cut,
        'opt_cut': OPT_CUTS[instance_key],
        'approximation_ratio': ratio,
        'usage_seconds': usage,
    }
    save_manifest(manifest)
    return ratio, usage
