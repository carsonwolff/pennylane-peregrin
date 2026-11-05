# Benchmark 2026-11-05T02:04:33

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: angle-trainable; precision: f64; layers: 1; embedding: angle; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0, pennylane_peregrin 0.1.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| peregrin-cpu | **3,783,364** | **5,289,620** | **7,214,156** | **12,840,110** | 0.01 s |