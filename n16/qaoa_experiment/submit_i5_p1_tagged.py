import sys
import json
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

N_QUBITS = 16
OPT_CUT_I5 = 35788.53551968903
GAMMA = [1.9944847296004526]
BETA = [0.7978071852762219]
SHOTS = 8192

def load_matrix(csv_path='mcp_agent_data_16_cost_matrix.csv'):
    df = pd.read_csv(csv_path, index_col=0)
    W = df.values.astype(float)
    return (W + W.T) / 2

def build_I5_adjacency(W):
    rng2 = np.random.default_rng(99)
    W_rand = np.zeros((N_QUBITS, N_QUBITS))
    idx = [(i, j) for i in range(N_QUBITS) for j in range(i+1, N_QUBITS) if W[i][j] > 0]
    all_pairs = [(i, j) for i in range(N_QUBITS) for j in range(i+1, N_QUBITS)]
    n_edges = len(idx)
    mean_w = W[W > 0].mean(); std_w = W[W > 0].std()
    chosen = rng2.choice(len(all_pairs), size=n_edges, replace=False)
    for c in chosen:
        i, j = all_pairs[c]
        w = abs(rng2.normal(mean_w, std_w))
        W_rand[i][j] = w
        W_rand[j][i] = w
    return W_rand

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

def main():
    rep_tag = sys.argv[1]  # e.g. I5_p1_rep1

    W = load_matrix()
    adjacency = build_I5_adjacency(W)

    qc = build_qaoa_circuit(adjacency, p=1)
    param_dict = {qc.parameters[0]: GAMMA[0], qc.parameters[1]: BETA[0]}
    bound = qc.assign_parameters(param_dict)

    print(f"Connecting to saved Qiskit Runtime account...")
    service = QiskitRuntimeService()
    hw_backend = service.backend('ibm_marrakesh')
    status = hw_backend.status()
    print(f"Backend: {hw_backend.name}  status={status.status_msg}  pending={status.pending_jobs}")
    if not status.operational:
        print("BACKEND NOT OPERATIONAL - ABORTING")
        sys.exit(1)

    pm = generate_preset_pass_manager(backend=hw_backend, optimization_level=1)
    isa = pm.run(bound)

    sampler = Sampler(mode=hw_backend)
    print(f"Submitting job tagged '{rep_tag}' (shots={SHOTS})...")
    job = sampler.run([isa], shots=SHOTS)
    job.update_tags([rep_tag])
    print(f"Job ID: {job.job_id()}  tags={job.tags}")

    print("Waiting for result...")
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()

    total_shots = sum(counts.values())
    best_cut = 0.0
    for bitstring, count in counts.items():
        bits = [int(b) for b in reversed(bitstring)]
        cut = compute_cut_value(bits, adjacency)
        if cut > best_cut:
            best_cut = cut
    ratio = best_cut / OPT_CUT_I5

    usage_seconds = job.usage()

    print(f"\n{'='*60}")
    print(f"RESULT — {rep_tag}")
    print(f"{'='*60}")
    print(f"job_id: {job.job_id()}")
    print(f"tags: {job.tags}")
    print(f"best_cut: {best_cut:.6f}")
    print(f"approximation_ratio: {ratio:.10f}")
    print(f"usage_seconds: {usage_seconds}")

    out = {
        'rep_tag': rep_tag,
        'job_id': job.job_id(),
        'tags': list(job.tags) if job.tags else [],
        'best_cut': best_cut,
        'opt_cut': OPT_CUT_I5,
        'approximation_ratio': ratio,
        'usage_seconds': usage_seconds,
        'shots': SHOTS,
        'gamma': GAMMA,
        'beta': BETA,
    }
    with open(f'i5_p1_{rep_tag}_result.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> i5_p1_{rep_tag}_result.json")

if __name__ == '__main__':
    main()
