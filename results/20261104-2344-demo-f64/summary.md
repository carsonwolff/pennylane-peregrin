# Benchmark 2026-11-05T01:28:59

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: pennylane-demo; precision: f64; layers: 1; embedding: angle; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 26,070 | 138,765 | 189,612 | 228,052 | 0.48 s |
| dq-jax-cpu | 6,328 | 48,305 | 175,123 | 198,970 | 1.65 s |
| jax-hand-cpu | **317,711** | **530,235** | **572,040** | **510,680** | 0.68 s |
| lightning.qubit | 5,672 | 6,185 | 5,691 | 5,882 | 17.18 s |
| lightning.qubit+qjit | 14,263 | 15,858 | 15,372 | 13,948 | 7.93 s |
| qulacs | 761 | 718 | 719 | skipped | 14.18 s |