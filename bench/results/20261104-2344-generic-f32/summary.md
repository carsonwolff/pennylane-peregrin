# Benchmark 2026-11-05T00:43:22

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: generic; precision: f32; layers: 1; embedding: stateprep; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 2 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 43,186 | 359,904 | 1,240,355 | 1,340,201 | 0.08 s |
| dq-jax-cpu | 9,041 | 88,294 | 590,619 | 1,993,517 | 0.73 s |
| jax-hand-cpu | **2,443,315** | **6,198,494** | **16,988,009** | **25,525,936** | 0.31 s |
| lightning.qubit | 6,684 | 7,334 | 6,932 | 6,907 | 14.49 s |
| lightning.qubit+qjit | 14,184 | 14,910 | 13,055 | 12,591 | 7.90 s |
| qulacs | 665 | 657 | 654 | skipped | 15.21 s |

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 23,235 | 126,334 | 181,409 | 210,365 | 0.47 s |
| dq-jax-cpu | 4,737 | 42,275 | 155,052 | 312,908 | 1.62 s |
| jax-hand-cpu | **1,073,739** | **1,509,632** | **1,762,608** | **2,497,865** | 0.54 s |
| lightning.qubit | 3,182 | 4,506 | 4,315 | 4,303 | 23.21 s |
| lightning.qubit+qjit | 8,468 | 9,156 | 8,691 | 8,560 | 12.23 s |
| qulacs | 491 | 488 | 486 | skipped | 20.81 s |

### 6 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 13,493 | 37,783 | 38,218 | 41,279 | 2.43 s |
| dq-jax-cpu | 3,056 | 16,328 | 41,113 | 34,222 | 5.45 s |
| jax-hand-cpu | **194,584** | **494,177** | **407,359** | **356,336** | 0.88 s |
| lightning.qubit | 3,223 | 3,390 | 3,236 | 3,248 | 31.03 s |
| lightning.qubit+qjit | 5,871 | 6,207 | 5,868 | 5,688 | 17.89 s |
| qulacs | 381 | 387 | 386 | skipped | 26.03 s |

### 8 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 6,585 | 7,437 | 9,389 | 8,970 | 11.16 s |
| dq-jax-cpu | 1,904 | 6,150 | 9,884 | 4,302 | 22.98 s |
| jax-hand-cpu | **58,010** | **91,436** | **68,226** | **64,535** | 2.36 s |
| lightning.qubit | 2,568 | 2,666 | 2,602 | 2,529 | 39.62 s |
| lightning.qubit+qjit | 4,010 | 3,788 | 3,226 | 2,308 | 30.01 s |
| qulacs | 299 | 299 | 294 | skipped | 34.19 s |