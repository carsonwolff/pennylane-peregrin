# Benchmark 2026-11-05T10:37:07

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: owner-mri; precision: f64; layers: 1 at n=2, 2 at n=4; embedding: stateprep; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0, pennylane_peregrin 0.1.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 2 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| peregrin-qnode | **54,118** | **495,191** | **3,070,634** | **5,804,516** | 0.02 s |

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| peregrin-qnode | **24,464** | **291,827** | **1,372,987** | **1,914,058** | 0.05 s |