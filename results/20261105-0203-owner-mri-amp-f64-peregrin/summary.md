# Benchmark 2026-11-05T02:04:10

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: owner-mri; precision: f64; layers: 1 at n=2, 2 at n=4; embedding: amplitude; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0, pennylane_peregrin 0.1.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 2 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| peregrin-cpu | **5,589,090** | **8,867,921** | **12,380,788** | **23,064,297** | 0.00 s |

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| peregrin-cpu | **2,251,847** | **3,213,109** | **6,346,772** | **6,913,618** | 0.01 s |