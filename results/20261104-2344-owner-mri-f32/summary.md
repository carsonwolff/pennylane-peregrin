# Benchmark 2026-11-05T00:30:22

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: owner-mri; precision: f32; layers: 1 at n=2, 2 at n=4; embedding: stateprep; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 2 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 31,677 | 237,967 | 573,281 | 680,607 | 0.15 s |
| dq-jax-cpu | 8,155 | 76,375 | 447,377 | **1,179,074** | 0.78 s |
| jax-hand-cpu | **421,091** | **652,334** | **741,426** | 788,142 | 0.41 s |
| lightning.qubit | 4,629 | 5,063 | 4,650 | 4,734 | 21.50 s |
| lightning.qubit+qjit | 14,125 | 15,791 | 14,906 | 13,409 | 7.81 s |
| qulacs | 577 | 563 | 571 | skipped | 17.65 s |

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 14,241 | 74,327 | 90,440 | 110,025 | 0.93 s |
| dq-jax-cpu | 3,458 | 29,831 | 104,282 | 200,839 | 1.78 s |
| jax-hand-cpu | **214,000** | **326,518** | **339,060** | **330,647** | 0.81 s |
| lightning.qubit | 2,279 | 2,251 | 2,293 | 2,337 | 43.28 s |
| lightning.qubit+qjit | 8,318 | 8,924 | 8,499 | 8,285 | 12.44 s |
| qulacs | 348 | 337 | 338 | skipped | 29.45 s |