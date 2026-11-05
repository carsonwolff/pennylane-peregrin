# Benchmark 2026-11-05T01:13:33

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: owner-mri; precision: f64; layers: 1 at n=2, 2 at n=4; embedding: amplitude; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 2 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 31,475 | 225,070 | 656,381 | 686,561 | 0.16 s |
| dq-jax-cpu | 8,375 | 78,148 | 391,448 | **772,998** | 1.01 s |
| jax-hand-cpu | **500,455** | **697,970** | **708,656** | 664,485 | 0.41 s |
| lightning.qubit | 3,534 | 3,835 | 3,538 | 3,651 | 27.73 s |
| lightning.qubit+qjit | 10,600 | 11,637 | 10,150 | 9,588 | 11.61 s |
| qulacs | 256 | 249 | 249 | skipped | 40.22 s |

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 13,976 | 71,793 | 90,251 | 107,985 | 0.94 s |
| dq-jax-cpu | 3,505 | 27,555 | 84,923 | 112,301 | 2.35 s |
| jax-hand-cpu | **252,228** | **303,326** | **296,705** | **276,166** | 0.80 s |
| lightning.qubit | 1,992 | 1,919 | 1,668 | 2,042 | 54.54 s |
| lightning.qubit+qjit | 5,056 | 5,208 | 4,750 | 4,553 | 24.79 s |
| qulacs | 97 | 100 | skipped | skipped | 10.16 s |