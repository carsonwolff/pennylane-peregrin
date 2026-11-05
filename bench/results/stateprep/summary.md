# Benchmark 2026-11-04T21:12:38

- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz (2 threads); GPU: none
- Precision: f64; layers: 1; embedding: stateprep; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.3, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 2 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|
| default.qubit | 66,944 | 530,262 | 1,441,765 | 0.01 s |
| dq-jax-cpu | 13,702 | 144,140 | 568,903 | 0.72 s |
| jax-hand-cpu | **1,317,558** | **11,904,762** | **15,540,885** | 0.21 s |
| lightning.qubit | 12,685 | 13,033 | 12,605 | 0.74 s |
| lightning.qubit+qjit | 7,020 | 7,095 | 6,883 | 1.73 s |
| qulacs | 950 | 962 | 952 | 10.49 s |

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|
| default.qubit | 37,364 | 171,555 | 256,521 | 0.05 s |
| dq-jax-cpu | 5,718 | 46,937 | 203,897 | 1.21 s |
| jax-hand-cpu | **1,293,510** | **2,438,816** | **1,968,487** | 0.38 s |
| lightning.qubit | 7,017 | 7,368 | 7,079 | 1.68 s |
| lightning.qubit+qjit | 3,382 | 3,345 | 3,328 | 3.28 s |
| qulacs | 665 | 680 | 675 | 14.69 s |

### 8 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|
| default.qubit | 7,697 | 9,678 | 6,336 | 1.64 s |
| dq-jax-cpu | 1,899 | 5,849 | 5,963 | 4.55 s |
| jax-hand-cpu | **57,351** | **63,220** | **48,794** | 1.12 s |
| lightning.qubit | 4,042 | 4,512 | 4,271 | 2.34 s |
| lightning.qubit+qjit | 1,706 | 1,687 | 1,686 | 6.53 s |
| qulacs | 412 | 407 | 407 | 24.51 s |