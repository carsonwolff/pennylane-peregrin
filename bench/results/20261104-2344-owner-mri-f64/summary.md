# Benchmark 2026-11-04T23:44:11

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: owner-mri; precision: f64; layers: 1 at n=2, 2 at n=4; embedding: stateprep; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 2 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 22,046 | 222,783 | **651,320** | 703,654 | 0.20 s |
| dq-jax-cpu | 8,027 | 60,023 | 363,196 | **942,859** | 0.83 s |
| jax-hand-cpu | **363,525** | **316,832** | 510,585 | 603,553 | 0.43 s |
| lightning.qubit | 3,459 | 4,873 | 4,447 | 4,605 | 21.59 s |
| lightning.qubit+qjit | 13,890 | 15,477 | 14,739 | 13,107 | 7.99 s |
| qulacs | 551 | 517 | 537 | skipped | 18.69 s |

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 13,846 | 71,966 | 80,629 | 100,366 | 1.05 s |
| dq-jax-cpu | 3,446 | 27,022 | 91,326 | 112,770 | 2.48 s |
| jax-hand-cpu | **241,316** | **276,935** | **275,631** | **254,565** | 0.90 s |
| lightning.qubit | 2,275 | 2,174 | 2,225 | 2,250 | 43.37 s |
| lightning.qubit+qjit | 8,244 | 8,713 | 8,335 | 8,239 | 12.68 s |
| qulacs | 319 | 313 | 342 | skipped | 29.68 s |