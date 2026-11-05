# Benchmark 2026-11-04T23:57:43

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: generic; precision: f64; layers: 1; embedding: stateprep; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 2 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 44,834 | 344,094 | 918,228 | 1,073,893 | 0.10 s |
| dq-jax-cpu | 9,182 | 85,811 | 447,190 | 1,122,692 | 0.76 s |
| jax-hand-cpu | **1,586,710** | **8,100,938** | **11,592,738** | **12,396,752** | 0.27 s |
| lightning.qubit | 6,871 | 7,417 | 6,994 | 6,936 | 14.31 s |
| lightning.qubit+qjit | 14,469 | 15,027 | 13,004 | 12,432 | 7.99 s |
| qulacs | 380 | 660 | 664 | skipped | 15.05 s |

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 22,896 | 113,559 | 124,607 | 157,277 | 0.68 s |
| dq-jax-cpu | 4,843 | 36,868 | 104,461 | 169,388 | 1.98 s |
| jax-hand-cpu | **1,039,361** | **1,694,535** | **1,607,354** | **1,214,464** | 0.54 s |
| lightning.qubit | 3,933 | 4,532 | 4,337 | 4,357 | 22.90 s |
| lightning.qubit+qjit | 8,549 | 9,114 | 8,695 | 8,579 | 12.22 s |
| qulacs | 494 | 489 | 484 | skipped | 20.82 s |

### 6 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 12,681 | 23,063 | 29,034 | 27,471 | 3.61 s |
| dq-jax-cpu | 2,940 | 12,534 | 27,644 | 11,794 | 8.91 s |
| jax-hand-cpu | **180,077** | **395,127** | **257,183** | **217,116** | 1.00 s |
| lightning.qubit | 3,200 | 3,387 | 3,199 | 3,246 | 30.80 s |
| lightning.qubit+qjit | 5,910 | 6,184 | 5,851 | 5,791 | 17.97 s |
| qulacs | 394 | 387 | 384 | skipped | 25.97 s |

### 8 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 5,126 | 4,783 | 5,643 | 5,867 | 17.17 s |
| dq-jax-cpu | 1,480 | 3,886 | 4,243 | 2,397 | 42.60 s |
| jax-hand-cpu | **47,999** | **48,441** | **42,999** | **42,105** | 3.20 s |
| lightning.qubit | 2,513 | 2,643 | 2,595 | 2,570 | 38.91 s |
| lightning.qubit+qjit | 4,028 | 3,793 | 3,250 | 2,271 | 29.96 s |
| qulacs | 299 | 296 | 292 | skipped | 34.32 s |